import os
import json
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import logging
import random
import glob
logger = logging.getLogger(__name__)


class BDDDataloader(data.Dataset):
    """
    Load raw images and labels
    """

    def __init__(self, opt, training=True, return_org_image=False, data_list=None):
        super(BDDDataloader, self).__init__()

        self.crop_h, self.crop_w = opt.crop_size_h, opt.crop_size_w
        self.random_scale = opt.random_scale
        self.random_mirror = opt.random_mirror
        self.ignore_label = opt.ignore_label
        self.cnn_type = opt.cnn_type
        self.training = training
        self.return_org_image = return_org_image

        # if data_list is a directory, load all images from that one
        # this is used at testing time where a directory is given,
        # otherwise, if data_list is a file, it should contain image paths
        if os.path.isdir(data_list):
            pattern = '{}/*.{}'.format(data_list, opt.image_ext)
            self.img_ids = [(os.path.basename(f), None) for f in glob.glob(pattern)]
            self.img_ids = sorted(self.img_ids, key=lambda x:x[0])
        else:
            self.img_ids = [i_id.strip().split() for i_id in open(data_list)]

        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(image_path))[0]
            img_file = os.path.join(opt.data_dir, image_path)
            if label_path:
                label_file = os.path.join(opt.data_dir, label_path)
            else:
                label_file = None
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })

        logger.info('{} images are loaded!'.format(len(self.img_ids)))

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        if self.return_org_image:
            org_image = image[:,:,::-1].copy()

        if datafiles["label"]:
            label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        else:
            label = {}
        size = image.shape
        name = datafiles["name"]

        if self.training and self.random_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        if self.cnn_type == "resnet101":
            mean = (102.9801, 115.9465, 122.7717)
            image = image[:,:,::-1]
            image -= mean

        if datafiles["label"]:
            # if image size < crop_size, then do padding with 0
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)

            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            # if image size > crop_size, then do cropping
            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        else:
            # if image size > crop_size, then do resizing
            if image.shape[0] > self.crop_h and image.shape[1] > self.crop_w:
                image = cv2.resize(image, (self.crop_w, self.crop_h), interpolation = cv2.INTER_LINEAR)
                if self.return_org_image:
                    org_image = cv2.resize(org_image, (self.crop_w, self.crop_h), interpolation = cv2.INTER_LINEAR)

        # get c x h x w images
        image = image.transpose((2, 0, 1))

        if self.training and self.random_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.return_org_image:
            return image.copy(), label.copy(), np.array(size), name, org_image
        else:
            return image.copy(), label.copy(), np.array(size), name


    def __len__(self):
        return len(self.files)


