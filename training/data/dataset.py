
from torch.utils import data
import numpy as np
import os
import torchvision.datasets as datasets

from tqdm import tqdm
from PIL import Image



# class ImgNetPerClass(data.Dataset):
#     def __init__(self, folder_path):
#
#         self.file_list = os.listdir(folder_path)

class MyImgFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(MyImgFolder, self).__init__(root, transform=transform)

    def __getitem__(self, item):
        return super(MyImgFolder, self).__getitem__(item), self.imgs[item]


class ClassLoader(data.Dataset):
    def __init__(self, root, transform=None):
        # start_time = time.time()

        self.image_path = []
        self.labels = []
        self.transform = transform
        self.load_data(root)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return (img, img_path)

    def load_data(self, root):
        images = os.listdir(root)
        images.sort()
        for i, img in enumerate(images):
            img_path = os.path.join(root, img)
            self.image_path.append(img_path)


class SpecifiedClassLoader(data.Dataset):
    def __init__(self, root, imglist, transform=None):
        # start_time = time.time()

        self.image_path = []
        self.labels = []
        self.transform = transform
        imglist.sort()
        self.image_path = [os.path.join(root, each) for each in imglist]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return (img, img_path)