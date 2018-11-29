import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import logging
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import gather

from dataloader import get_data_loader
from models import get_model
from utils.criterion import CriterionDSN, CriterionCrossEntropy
from utils.utils import AverageMeter, adjust_learning_rate
from utils.parallel import DataParallelModel, DataParallelCriterion
from utils.metrics import batch_pix_accuracy, batch_intersection_union
logger = logging.getLogger(__name__)

def train(opt, model, criterion, optimizer, loader):
    """
    Training the network in one epoch

    Args:
        opt (Namspace): training options
        model (LaneNet): a LaneNet model
        criterion: a CrossEntropyLoss criterion
        optimizer: optimizer (SGD, Adam, etc)
        loader: data loader

    Returns:
        None

    """

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(loader)

    for data in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        images, labels, _, _ = data
        images = Variable(images)
        labels = Variable(labels.long())

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        pbar.set_description(
            '>>> Training loss={:.6f}, i/o time={data_time.avg:.3f}s, gpu time={batch_time.avg:.3f}s'.format(
                loss.item(), data_time=data_time, batch_time=batch_time))
        end = time.time()


def test(opt, model, criterion, loader):
    """
    Validate the model at the current state

    Args:
        opt (Namspace): training options
        model (LaneNet): a LaneNet model
        criterion: a CrossEntropyLoss criterion
        loader: val data loader

    Returns:
        The average loss value on val data

    """

    val_loss = AverageMeter()
    model.eval()

    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    pbar = tqdm(loader)
    with torch.no_grad():
        for data in pbar:
            images, labels, _, _ = data

            images = Variable(images)
            labels = Variable(labels.long())

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            preds = model(images)
            loss = criterion(preds, labels)
            val_loss.update(loss.item())

            # compute pixAcc and mIoU at batch level
            preds = gather(preds, 0, dim=0)
            if 'dsn' in opt.model_type:
                preds = preds[-1]

            N_, C_, H_, W_ = images.shape
            preds = F.upsample(
                input=preds, size=(H_, W_),
                mode='bilinear', align_corners=True)

            correct, labeled = batch_pix_accuracy(preds.data, labels)
            inter, union = batch_intersection_union(preds.data, labels, opt.num_classes)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            pbar.set_description(
                '>>> Validating loss={:.6f}, pixAcc={:.6f}, mIoU={:.6f}'.format(
                    loss.item(), pixAcc, mIoU))

    return val_loss


def main(opt):

    # Set the random seed manually for reproducibility.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)

    train_loader = get_data_loader(opt,
                                   training=True,
                                   return_org_image=False,
                                   data_list=opt.train_data_list)

    val_loader = get_data_loader(opt,
                                 training=False,
                                 return_org_image=False,
                                 data_list=opt.val_data_list)

    logger.info('Building model...')

    model = get_model(opt, num_classes=opt.num_classes)

    # TODO: probably we can use standard pytorch pre-trained model here
    saved_state_dict = torch.load(opt.start_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc' and not i_parts[0] == 'last_linear' and not i_parts[0] == 'classifier':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model = DataParallelModel(model)

    if 'dsn' in opt.model_type:
        criterion = CriterionDSN(dsn_weight=float(opt.dsn_weight), use_weight=True)
    else:
        criterion = CriterionCrossEntropy()

    criterion = DataParallelCriterion(criterion)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

        # enabling benchmark mode in cudnn which is good whenever your input
        # sizes for your network do not vary. This way, cudnn will look for
        # the optimal set of algorithms for that particular configuration
        # (which takes some time). This usually leads to faster runtime.
        cudnn.benchmark = True

    logger.info("Start training...")
    best_loss = sys.maxsize
    best_epoch = 0

    for epoch in tqdm(range(opt.num_epochs), desc='Epoch: '):
        learning_rate = adjust_learning_rate(opt, optimizer, epoch)
        logger.info('===> Learning rate: %f: ', learning_rate)

        # train for one epoch
        train(
            opt,
            model,
            criterion,
            optimizer,
            train_loader)

        # validate at every val_step epoch
        if epoch % opt.val_step == 0:
            val_loss = test(
                opt,
                model,
                criterion,
                val_loader)
            logger.info('Val loss: %s\n', val_loss)

            loss = val_loss.avg
            if loss < best_loss:
                logger.info(
                    'Found new best loss: %.7f, previous loss: %.7f',
                    loss,
                    best_loss)
                best_loss = loss
                best_epoch = epoch

                logger.info('Saving new checkpoint to: %s', opt.output_file)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'opt': opt
                }, opt.output_file)

            else:
                logger.info(
                    'Current loss: %.7f, best loss is %.7f @ epoch %d',
                    loss,
                    best_loss,
                    best_epoch)

        if epoch - best_epoch > opt.max_patience:
            logger.info('Terminated by early stopping!')
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', type=str,
        help='path to the dataset directory')

    parser.add_argument(
        '--train_data_list',
        type=str,
        help='path to file contains list of image, label per line')

    parser.add_argument(
        '--val_data_list',
        type=str,
        help='path to file contains list of image, label per line')

    parser.add_argument(
        '--output_file',
        type=str,
        help='path to the output model file')

    parser.add_argument(
        '--dataset',
        default='cityscapes',
        choices=['cityscapes', 'bdd'],
        help='Name of dataset')

    # Model settings
    parser.add_argument(
        '--model_type',
        default='asp_oc_dsn',
        choices=['baseline', 'base_oc_dsn', 'pyramid_oc_dsn', 'asp_oc_dsn'],
        help='type of models (like ocnet, pspnet, deeplab, etc)')

    parser.add_argument(
        '--cnn_type',
        default='resnet101',
        choices=['resnet101'],
        help='The CNN backbone (e.g. vgg19, resnet152)')

    parser.add_argument(
        '--start_from',
        type=str,
        help='Path to the pre-trained model')

    # Optimization
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size')

    parser.add_argument(
        '--num_classes',
        type=int,
        default=19,
        help='number of classes')

    parser.add_argument(
        '--crop_size_h',
        type=int,
        default=769,
        help='cropped image height')

    parser.add_argument(
        '--crop_size_w',
        type=int,
        default=769,
        help='cropped image width')

    parser.add_argument(
        '--random_scale',
        default=False,
        action='store_true',
        help='random scaling at training')

    parser.add_argument(
        '--random_mirror',
        default=False,
        action='store_true',
        help='random mirroring at training')

    parser.add_argument(
        '--ignore_label',
        type=int,
        default=255,
        help='ignore classes having this label from training')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='learning rate')

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=30,
        help='max number of epochs to run the training')

    parser.add_argument('--lr_update', default=30, type=int,
                        help='Number of epochs to update the learning rate.')

    parser.add_argument(
        '--max_patience', type=int, default=5,
        help='max number of epoch to run since the minima is detected -- early stopping')

    # other options
    parser.add_argument(
        '--val_step',
        type=int,
        default=1,
        help='how often do we check the model (in terms of epoch)')

    parser.add_argument(
        '--num_workers', type=int, default=2,
        help='number of workers (each worker use a process to load a batch of data)')

    parser.add_argument(
        '--dsn_weight',
        type=float,
        default=0.4,
        help='weight of the auxilarily loss')

    parser.add_argument(
        '--log_step',
        type=int,
        default=20,
        help='How often to print training info (loss, system/data time, etc)')

    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='random number generator seed to use')

    opt = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(opt),
            sort_keys=True,
            indent=4))

    start = datetime.now()
    main(opt)
    logger.info('Time: %s', datetime.now() - start)
