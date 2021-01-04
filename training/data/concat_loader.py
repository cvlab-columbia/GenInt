import os
import sys
import time

import random
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms as transforms

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Loader_Random(torch.utils.data.Dataset):
    def __init__(self, path, composed_transforms=None, get_pair=False):
        super().__init__()
        self.get_pair = get_pair
        self.path = path
        self.categories = os.listdir(path)
        self.categories.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

        self.transform = composed_transforms

    def __getitem__(self, item):

        select_category = random.sample(self.categories, 1)[0]
        cat_path = os.path.join(self.path, select_category)
        instance_lists = os.listdir(cat_path)
        select_instance = random.sample(instance_lists, 1)[0]
        ins_path = os.path.join(cat_path, select_instance)

        target = self.category2id[select_category]

        sub_imglist = []
        for subroot, _, fnames in sorted(os.walk(ins_path, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(subroot, fname)
                sub_imglist.append(path)

        if self.get_pair:
            select_path = random.sample(sub_imglist, 2)

            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(select_path[0], 'rb') as f:
                img = Image.open(f).convert("RGB")
            sample1 = self._transform(img)
            img.close()

            with open(select_path[1], 'rb') as f:
                img = Image.open(f).convert("RGB")
            sample2 = self._transform(img)
            img.close()
            # sample_imgnet, target_imgnet,
            return sample1, sample2, target
        else:
            select_path = random.sample(sub_imglist, 1)[0]

            # path, target = self.filelist1[item % self.len1]  # need a shuffle to guarantee randomness
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(select_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            sample = self._transform(img)
            img.close()
            # sample_imgnet, target_imgnet,
            return sample, target

    def __len__(self):
        return 10000000

    def _transform(self, sample):
        return self.transform(sample)





class LoaderConcat_split(torch.utils.data.Dataset):
    def __init__(self, path_1, path_2, composed_transforms=None, restrict1=None, restrict2=None, ratio=1, large=False):
        super().__init__()

        self.root_path = path_1
        self.restrict1 = restrict1
        self.categories = os.listdir(self.root_path)
        self.categories.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

        self.filelist1 = self._make_dataset(path_1, self.category2id, restrict=restrict1, large=large)
        self.imgnet_filelist2 = self._make_dataset(path_2, self.category2id, restrict=restrict2)
        self.ratio = ratio

        if ratio<1:
            length1 = int(len(self.filelist1) * ratio)
            select1 = random.sample(self.filelist1, length1)
            select1.shuffle()
        else:
            select1 = self.filelist1

        print('individual', len(select1), len(self.imgnet_filelist2))

        self.len1 = len(select1)
        self.len2 = len(self.imgnet_filelist2)

        self.transform = composed_transforms

    def reshuffle(self):
        self.filelist1 = self._make_dataset(self.root_path, self.category2id, restrict=self.restrict1, large=True)
        self.filelist1.shuffle()

    def _transform(self, sample):
        return self.transform(sample)

    def __getitem__(self, item):
        # path, target_imgnet = self.imgnet_filelist2[item]
        # # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        # with open(path, 'rb') as f:
        #     img = Image.open(f).convert("RGB")
        # sample_imgnet = self._transform(img)
        # img.close()

        path, target = self.filelist1[item % self.len1]  # need a shuffle to guarantee randomness
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f).convert("RGB")
        sample = self._transform(img)
        img.close()
        # sample_imgnet, target_imgnet,
        return sample, target

    def __len__(self):
        return self.len1

    # TODO: just pure random shold be fast

    @staticmethod
    def _make_dataset(path2data, class_to_idx, restrict=None, large=False):
        instances = []
        tt=0
        for target_class in sorted(class_to_idx.keys()):
            temp_ins = []
            tt+=1
            target_dir = os.path.join(path2data, target_class)
            class_index = class_to_idx[target_class]
            if not os.path.isdir(target_dir):
                continue

            cnt=0
            flag=False
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index

                    if restrict:
                        if restrict==cnt:
                            flag=True
                            print("restrict to ", restrict)
                            break

                    if large:
                        temp_ins.append(item)
                    else:
                        instances.append(item)
                    cnt += 1
                if flag:
                    break

            if large:
                temp_ins = random.sample(temp_ins, 1400)
                instances = instances + temp_ins
            # print(cnt)
            # if debug:
            print(target_class, tt, len(instances))

        return instances



class LoaderConcat(torch.utils.data.Dataset):
    def __init__(self, path_1, path_2, composed_transforms=None, restrict1=None, restrict2=None, ratio=1):
        super().__init__()

        self.root_path = path_1
        self.categories = os.listdir(self.root_path)
        self.categories.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

        self.filelist1 = self._make_dataset(path_1, self.category2id, restrict=restrict1)
        self.filelist2 = self._make_dataset(path_2, self.category2id, restrict=restrict2)

        if ratio<1:
            length1 = int(len(self.filelist1) * ratio)
            select1 = random.sample(self.filelist1, length1)
        else:
            select1 = self.filelist1

        print('individual', len(self.filelist1), len(self.filelist2))

        self.all_instances = select1 + self.filelist2
        print("", len(self.all_instances))

        self.transform = composed_transforms

    def _transform(self, sample):
        return self.transform(sample)

    def __getitem__(self, item):
        path, target = self.all_instances[item]
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f).convert("RGB")
        sample = self._transform(img)
        img.close()
        return sample, target

    def __len__(self):
        return len(self.all_instances)

    @staticmethod
    def _make_dataset(path2data, class_to_idx, restrict=None):
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(path2data, target_class)
            class_index = class_to_idx[target_class]
            if not os.path.isdir(target_dir):
                continue

            cnt=0
            flag=False
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index

                    if restrict:
                        if restrict==cnt:
                            flag=True
                            break

                    instances.append(item)
                    cnt+=1

                if flag:
                    break
            # print(cnt)
        return instances


if __name__ == "__main__":
    train_load = LoaderConcat(path_1="/local/vondrick/cz/GANdata/setting_50_16_sub",
                              path_2="/local/vondrick/cz/ImageNet-Data/train", restrict1=1000)


