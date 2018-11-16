import argparse
import os
import numpy as np
import numpy.ma as ma
import json
from datetime import datetime
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.nn.parallel.scatter_gather import gather

from dataloader import get_data_loader
from models import get_model
from utils.utils import AverageMeter
from utils.parallel import DataParallelModel

logger = logging.getLogger(__name__)
# warnings.filterwarnings('ignore')


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred_label] = label_count[cur_index]

    return confusion_matrix


def test(opt, model, loader):
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

    model.eval()

    run_time = AverageMeter()
    end = time.time()
    confusion_matrix = np.zeros((opt.num_classes, opt.num_classes))

    pbar = tqdm(loader)
    with torch.no_grad():
        for data in pbar:
            images, labels, sizes, _ = data
            sizes = sizes[0].numpy()

            images = Variable(images)
            N_, C_, H_, W_ = images.shape

            if torch.cuda.is_available():
                images = images.cuda()

            preds = model(images)
            preds = gather(preds, 0, dim=0)
            if 'dsn' in opt.model_type:
                preds = preds[-1]

            full_preds = F.upsample(
                input=preds, size=(H_, W_),
                mode='bilinear', align_corners=True)
            output = full_preds.cpu().data.numpy().transpose(0, 2, 3, 1)

            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            m_seg_pred = ma.masked_array(
                seg_pred, mask=torch.eq(
                    labels, opt.ignore_label))
            ma.set_fill_value(m_seg_pred, 20)
            seg_pred = m_seg_pred

            seg_gt = np.asarray(
                labels.numpy()[
                    :,
                    :sizes[0],
                    :sizes[1]],
                dtype=np.int)
            ignore_index = seg_gt != opt.ignore_label
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            confusion_matrix += get_confusion_matrix(
                seg_gt, seg_pred, opt.num_classes)

            # measure speed test
            bs = images.shape[0]
            run_time.update(time.time() - end)
            end = time.time()
            fps = bs/run_time.avg
            pbar.set_description(
                'Average run time: {fps:.3f} fps'.format(
                    fps=fps))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    print({'meanIU': mean_IU, 'IU_array': IU_array})
    return mean_IU


def main(opt):
    logger.info('Loading model: %s', opt.model_file)

    checkpoint = torch.load(opt.model_file)

    checkpoint_opt = checkpoint['opt']

    # Update/Overwrite some test options like batch size, location to metadata
    # file
    vars(checkpoint_opt).update(vars(opt))

    logger.info(
        'Updated input arguments: %s',
        json.dumps(
            vars(checkpoint_opt),
            sort_keys=True,
            indent=4))

    logger.info('Building model...')
    model = get_model(checkpoint_opt, num_classes=checkpoint_opt.num_classes)

    test_loader = get_data_loader(
        checkpoint_opt,
        split='test',
        data_list=opt.test_data_list)

    logger.info('Loading model parameters...')
    model = DataParallelModel(model)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model.cuda()

    logger.info('Start testing...')

    test(
        checkpoint_opt,
        model,
        test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_file',
        type=str,
        help='path to the model file')
    parser.add_argument(
        '--test_data_list',
        type=str,
        help='path to file contains list of image, label per line')
    parser.add_argument(
        '--output_file',
        type=str,
        help='path to the output file containing prediction info')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size')
    parser.add_argument(
        '--crop_size_h',
        type=int,
        default=1024,
        help='cropped image height')
    parser.add_argument(
        '--crop_size_w',
        type=int,
        default=2048,
        help='cropped image width')
    parser.add_argument(
        '--random-scale',
        default=False,
        action='store_true',
        help='random scaling at training')

    parser.add_argument(
        '--random-mirror',
        default=False,
        action='store_true',
        help='random mirroring at training')

    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of workers (each worker use a process to load a batch of data)')

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

    if not os.path.isfile(opt.model_file):
        logger.info('Model file does not exist: %s', opt.model_file)

    else:
        start = datetime.now()
        main(opt)
        logger.info('Time: %s', datetime.now() - start)
