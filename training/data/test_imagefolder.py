import torchvision.datasets as datasets
import torch
import torchvision.datasets as datasets
import os

class MyImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(MyImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



from PIL import Image

class ImageNetTrainOverlapObject(torch.utils.data.Dataset):
    # This loader will only load the images that are within the overlapping class between imagenet and objectnet
    def __init__(self, train_dir, transform):
        super().__init__()

        self.transform = transform

        self.train_dir = train_dir
        folder_list = os.listdir(self.train_dir)
        folder_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(folder_list)}

        from preprocessing.obj_img_nonoverlap_id import get_imagenet_overlap
        overlap, non_overlap = get_imagenet_overlap()

        self.folder_keep = []

        for i, each in enumerate(folder_list):
            if i in overlap:
                self.folder_keep.append(each)

        filelist = []
        labellist = []

        for eachclass in self.folder_keep:
            imglist = os.listdir(os.path.join(train_dir, eachclass))
            num_img = len(imglist)
            for cnt, imgname in enumerate(imglist):
                imgname = os.path.join(os.path.join(train_dir, eachclass), imgname)
                filelist.append(imgname)

            labellist.extend([self.category2id[eachclass]] * num_img)

        self.filelist = filelist
        self.labellist = labellist

    def __len__(self):
        return len(self.filelist)

    def _transform(self, sample):
        return self.transform(sample)

    def __getitem__(self, index):
        # print("load", self.filelist[index])
        img_path = self.filelist[index]
        img = Image.open(img_path).convert('RGB')
        # print("img find")

        if self.transform is not None:
            img = self.transform(img)

        label = self.labellist[index]

        return (img, label, img_path)




