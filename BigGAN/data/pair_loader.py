import os
import sys
import time

import random
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

import torch

class LoaderGANPair(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.path_img = os.path.join(self.path, 'img')
        self.path_z = os.path.join(self.path, 'z')

        self.categories = os.listdir(self.path_img)
        self.categories.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

    def _transform(self, sample):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        composed_transforms = transforms.Compose([
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        ])
        return composed_transforms(sample)

    def __getitem__(self, item):
        select_category = random.sample(self.categories, 1)[0]
        cat_path = os.path.join(self.path_img, select_category)
        instance_lists = os.listdir(cat_path)
        select_instance = random.sample(instance_lists, 1)[0]
        ins_path = os.path.join(cat_path, select_instance)

        target = self.category2id[select_category]

        sub_imglist = []
        for subroot, _, fnames in sorted(os.walk(ins_path, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(subroot, fname)
                sub_imglist.append(path)

        select_path = random.sample(sub_imglist, 1)[0]

        # path, target = self.filelist1[item % self.len1]  # need a shuffle to guarantee randomness
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(select_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
        sample = self._transform(img)
        img.close()

        np_cat = os.path.join(self.path_z, select_category)
        ins_path_z = os.path.join(np_cat, select_instance)
        z_select_path = os.path.join(ins_path_z, select_path.split('/')[-1].split('.')[0]+'.npy')

        noise = np.load(z_select_path)

        return sample, target, noise


    def __len__(self):
        return 10000000
