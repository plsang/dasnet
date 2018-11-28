import argparse
import os
import json
from datetime import datetime
import glob

import logging
logger = logging.getLogger(__name__)


def cityscapes_get_files(data_root, data_type, split):
    # A map from data type to folder name that saves the data.
    _FOLDERS_MAP = {
        'image': 'leftImg8bit',
        'label': 'gtFine',
    }

    # A map from data type to filename postfix.
    _POSTFIX_MAP = {
        'image': '_leftImg8bit',
        'label': '_gtFine_labelIds',
    }

    # A map from data type to data format.
    _DATA_FORMAT_MAP = {
        'image': 'png',
        'label': 'png',
    }

    pattern = '*%s.%s' % (_POSTFIX_MAP[data_type], _DATA_FORMAT_MAP[data_type])
    search_files = os.path.join(
              data_root, _FOLDERS_MAP[data_type], split, '*', pattern)
    logger.info('Searching files with pattern: %s', pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)


def cityscapes(output_file, data_root, dataset_split):
    image_files = cityscapes_get_files(data_root, 'image', dataset_split)
    label_files = cityscapes_get_files(data_root, 'label', dataset_split)

    with open(output_file, 'w') as of:
        for image_file, label_file in zip(image_files, label_files):
            of.write('{}\t{}\n'.format(image_file, label_file))


def bdd(output_file, data_root, dataset_split):

    _LABEL_MAP = {
        'train': 'labels_new/bdd100k_labels_images_train.json',
        'val': 'labels_new/bdd100k_labels_images_val.json',
    }

    # A map from data type to filename postfix.
    _POSTFIX_MAP = {
        'image': '',
        'label': '_drivable_id',
    }

    # A map from data type to data format.
    _DATA_FORMAT_MAP = {
        'image': 'jpg',
        'label': 'png',
    }

    if dataset_split not in _LABEL_MAP:
        logger.warning('{} split data not found'.format(dataset_split))
        return

    json_file = os.path.join(data_root, _LABEL_MAP[dataset_split])
    logger.info('Loading %s', json_file)
    json_data = json.load(open(json_file))
    ids = [os.path.splitext(i['name'])[0] for i in json_data]

    image_files = [os.path.join(data_root, 'images', '100k', dataset_split, \
                                id + _POSTFIX_MAP['image'] + '.' + \
                                _DATA_FORMAT_MAP['image']) for id in ids]

    label_files = [os.path.join(data_root, 'drivable_maps', 'labels', dataset_split, \
                                id + _POSTFIX_MAP['label'] + '.' + \
                                _DATA_FORMAT_MAP['label']) for id in ids]

    with open(output_file, 'w') as of:
        for image_file, label_file in zip(image_files, label_files):
            of.write('{}\t{}\n'.format(image_file, label_file))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_file',
        type=str,
        help='path to the output file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        help='path to the root directory containing images and labels'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='path to the root directory containing images and labels'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cityscapes',
        choices=['cityscapes', 'bdd'],
        help='dataset name'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(args),
            sort_keys=True,
            indent=4))

    start = datetime.now()

    if args.dataset == 'cityscapes':
        cityscapes(args.output_file, args.data_root, args.split)
    elif args.dataset == 'bdd':
        bdd(args.output_file, args.data_root, args.split)

    logger.info('Saved output to %s', args.output_file)
    logger.info('Time: %s', datetime.now() - start)
