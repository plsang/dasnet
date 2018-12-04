import argparse
import os
import numpy as np
import json
from datetime import datetime
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.nn.parallel.scatter_gather import gather
from utils.metrics import batch_pix_accuracy, batch_intersection_union
from PIL import Image
import cv2

from dataloader import get_data_loader
from models import get_model
from utils.utils import AverageMeter
from utils.parallel import DataParallelModel

logger = logging.getLogger(__name__)
# warnings.filterwarnings('ignore')

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Cf. https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)       # 0: 'road'
    palette[3:6] = (244, 35, 232)        # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)         # 2''building'
    palette[9:12] = (102, 102,156)       # 3 wall
    palette[12:15] =  (190, 153,153)     # 4 fence
    palette[15:18] = (153, 153,153)      # 5 pole
    palette[18:21] = (250, 170, 30)      # 6 'traffic light'
    palette[21:24] = (220, 220, 0)       # 7 'traffic sign'
    palette[24:27] = (107, 142, 35)      # 8 'vegetation'
    palette[27:30] = (152, 251,152)      # 9 'terrain'
    palette[30:33] = ( 70, 130,180)      # 10 sky
    palette[33:36] = (220, 20, 60)      # 11 person
    palette[36:39] = (255, 0, 0)        # 12 rider
    palette[39:42] = (0, 0, 142)        # 13 car
    palette[42:45] = (0, 0, 70)         # 14 truck
    palette[45:48] = (0, 60,100)        # 15 bus
    palette[48:51] = (0, 80,100)        # 16 train
    palette[51:54] = (0, 0,230)         # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)      # 18 'bicycle'
    # palette[57:60] = (105, 105, 105)
    palette[57:60] = (0, 0, 0)
    return palette


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
    palette = get_palette(20)

    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    pbar = tqdm(loader)
    with torch.no_grad():
        for data in pbar:
            images, labels, sizes, image_names, org_images = data
            sizes = sizes[0].numpy()

            images = Variable(images)
            N_, C_, H_, W_ = images.shape

            if torch.cuda.is_available():
                images = images.cuda()

            preds = model(images)
            preds = gather(preds, 0, dim=0)
            if 'dsn' in opt.model_type:
                preds = preds[-1]

            preds = F.upsample(
                input=preds, size=(H_, W_),
                mode='bilinear', align_corners=True)
            output = preds.cpu().data.numpy().transpose(0, 2, 3, 1)

            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)

            # store images
            if opt.store_output:
                for i in range(N_):
                    output_im = Image.fromarray(seg_pred[i])
                    output_im.putpalette(palette)
                    output_file = os.path.join(opt.output_dir, image_names[i]+'.png')
                    output_im.save(output_file)

                    src_img = org_images[i].data.numpy()

                    if opt.dataset == 'cityscapes':
                        drivable_img = np.where(seg_pred[i]==0, 0, 19).astype(np.uint8)
                        drivable_img = Image.fromarray(drivable_img)
                        drivable_img.putpalette(palette)
                        drivable_img = np.array(drivable_img.convert('RGB')).astype(src_img.dtype)
                        #overlay_img = cv2.addWeighted(src_img, 1.0, drivable_img, 1.0, 0)
                        src_img[drivable_img > 0] = 0
                    else:
                        drivable_img = seg_pred[i]
                        drivable_img[drivable_img == 0] = 19
                        drivable_img = Image.fromarray(drivable_img)
                        drivable_img.putpalette(palette)
                        drivable_img = np.array(drivable_img.convert('RGB')).astype(src_img.dtype)

                    overlay_img = cv2.add(src_img, drivable_img)
                    output_file = os.path.join(opt.output_dir, image_names[i] + '_drivable.png')
                    cv2.imwrite(output_file, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

            if len(labels) > 0:
                labels[labels == opt.ignore_label] = -1 #
                correct, labeled = batch_pix_accuracy(preds.data, labels)
                inter, union = batch_intersection_union(preds.data, labels, opt.num_classes)
                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

            # measure speed test
            run_time.update(time.time() - end)
            end = time.time()
            fps = N_/run_time.avg
            pbar.set_description(
                'Average run time: {:.3f} fps, pixAcc={:.6f}, mIoU={:.6f}'.format(
                    fps, pixAcc, mIoU))

    print({'meanIU': mIoU, 'IU_array': IoU})
    return mIoU


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
        training=False,
        return_org_image=True,
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
        '--image_ext',
        type=str,
        default='png',
        help='image extension')
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
        '--store_output',
        default=False,
        action='store_true',
        help='write output image to output directory')

    parser.add_argument(
        '--output_dir',
        type=str,
        help='output directory to write images to')

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
