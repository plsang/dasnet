import os
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import logging
import random
import glob
logger = logging.getLogger(__name__)


def get_image_transform(height=256, width=512):

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform


class CityscapesDataloader(data.Dataset):
    """
    Load raw images and labels
    """

    def __init__(self, opt, split='train', data_list=None):
        super(CityscapesDataloader, self).__init__()

        self.crop_h, self.crop_w = opt.crop_size_h, opt.crop_size_w
        self.random_scale = opt.random_scale
        self.random_mirror = opt.random_mirror
        self.ignore_label = opt.ignore_label
        self.cnn_type = opt.cnn_type
        self.id_to_trainid = self.__id_to_trainid()
        self.training = split == 'train'

        # if data_list is a directory, load all images from that one
        # this is used at testing time where a directory is given,
        # otherwise, if data_list is a file, it should contain image paths
        if os.path.isdir(data_list):
            pattern = '{}/*.{}'.format(data_list, opt.image_ext)
            self.img_ids = [(os.path.basename(f), None) for f in glob.glob(pattern)]
            self.img_ids = sorted(self.img_ids, key=lambda x:x[0])
            self.data_dir = data_list
        else:
            self.img_ids = [i_id.strip().split() for i_id in open(data_list)]
            self.data_dir = opt.data_dir

        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            #name = os.path.splitext(os.path.basename(label_path))[0]
            name = os.path.splitext(os.path.basename(image_path))[0]
            img_file = os.path.join(self.data_dir, image_path)
            if label_path:
                label_file = os.path.join(self.data_dir, label_path)
            else:
                label_file = None
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })

        logger.info('{} images are loaded!'.format(len(self.img_ids)))

    def __id_to_trainid(self):

        id_list = list(range(-1, 34))
        labelid_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        id_to_trainid = {}
        new_label = 0
        for id in id_list:
            if id in labelid_list:
                id_to_trainid[id] = new_label
                new_label += 1
            else:
                id_to_trainid[id] = self.ignore_label

        return id_to_trainid

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        org_image = image[:,:,::-1].copy()
        if datafiles["label"]:
            label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
            label = self.id2trainId(label)
        else:
            label = None
        size = image.shape
        name = datafiles["name"]

        if self.training and self.random_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        if self.cnn_type == "resnet101":
            mean = (102.9801, 115.9465, 122.7717)
            image = image[:,:,::-1]
            image -= mean

        if label is not None:
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
                org_image = cv2.resize(org_image, (self.crop_w, self.crop_h), interpolation = cv2.INTER_LINEAR)

        # get c x h x w images
        image = image.transpose((2, 0, 1))

        if self.training and self.random_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if label is not None:
            return image.copy(), label.copy(), np.array(size), name, org_image
        else:
            return image.copy(), {}, np.array(size), name, org_image


    def __len__(self):
        return len(self.files)


