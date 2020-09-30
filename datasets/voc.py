# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Code is adapted from : 
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
"""

import os
import random
import argparse
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
try:
    from utils import download_url, extract_file
except :
    from .utils import download_url, extract_file


DATASET_DICT =  {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': os.path.join('TrainVal', 'VOCdevkit', 'VOC2011')
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': os.path.join('VOCdevkit', 'VOC2009')
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': os.path.join('VOCdevkit', 'VOC2008')
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    },
    '2007-test': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    }
}

VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES)

class PascalVOC(Dataset):

    def __init__(self, args, root, transform=None, is_training=False):
        # Dataset year
        self.year = args.year
        if args.year == "2007" and is_training == False:
            year = "2007-test"

        # Dataset root directory
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok = True)

        # get the url
        self.url = DATASET_DICT[args.year]['url']
        self.filename = DATASET_DICT[args.year]['filename']
        self.md5 = DATASET_DICT[args.year]['md5']
        self.transforms = transform

        # download file if it is not yet available locally
        if args.download:
            file = download_url(DATASET_DICT[year]['url'],
                                    self.root,
                                    DATASET_DICT[year]['filename'],
                                    DATASET_DICT[year]['md5']
                                   )
            extract_file(os.path.join(self.root, file), self.root)
        
        voc_root = os.path.join(self.root, DATASET_DICT[year]['base_dir'])
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found/currepted. " \
                               "Check the path/ use download = True option")
        img_dir = os.path.join(voc_root, "JPEGImages")
        seg_dir = os.path.join(voc_root, "SegmentationClass")
        split_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
        if is_training:
            split = os.path.join(split_dir, "train" + ".txt")
        else:
            split = os.path.join(split_dir, "test" + ".txt")

        with open(split) as f:
            item_names = [x.rstrip("\n") for x in f.readlines()]

        self.images = [os.path.join(img_dir, x+".jpg") for x in item_names]
        self.segmask = [os.path.join(seg_dir, x+".png") for x in item_names]
        assert (len(self.images) == len(self.segmask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        mask = Image.open(self.segmask[index])

        if self.transforms == None:
            img, mask = self.transform(img, mask)
        else:
            img, mask = self.transforms(img, mask)

        #Transform to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        inputs = dict()
        inputs["image"] = img
        inputs["mask"] = mask

        return inputs

    def transform(self, image, mask):
        resize = transforms.Resize(size=(572, 572))
        image = resize(image)
        mask = resize(mask)

        #Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        #Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset class for PASCAL VOC dataset')
    parser.add_argument('--year', default=2, type=int, metavar="YEAR",
                    help='Year of the dataset')
    parser.add_argument('--download', default=True, metavar="DOWNLOAD",
                    help="ownload the dataset")
    args = parser.parse_args()
