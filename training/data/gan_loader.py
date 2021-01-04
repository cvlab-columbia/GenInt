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


class GAN_invariance(torch.utils.data.Dataset):
    def __init__(self, path, lenth_episode, composed_transforms=None, noaug=False):
        super().__init__()

        self.root_path = path
        self.noaug = noaug
        self.categories = os.listdir(self.root_path)
        self.categories.sort()
        self.transform = composed_transforms

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

        self.file_dict = self.get_variance_list()
        self.length_episode = lenth_episode

    def __getitem__(self, item):
        # one cannot have redundant category
        episode_categories = random.sample(self.categories, self.length_episode)
        random.shuffle(episode_categories)
        labels = []
        img_1 = []
        img_2 = []
        img_3 = []
        img_4 = []

        for c in episode_categories:
            c_path = os.path.join(self.root_path, c)
            instance_path = os.listdir(c_path)
            sampled_ins = random.sample(instance_path, 2)
            random.shuffle(sampled_ins)

            # choose a random image form one instance, the other instance will be more sophisticated
            individual_cat_path = os.path.join(c_path, sampled_ins[0])
            img0_path_ins0 = random.sample(self.file_dict["instance_list"][individual_cat_path], 1)[0]  # same category

            # the other instance, sampe a compoent of transformation
            individual_ins_path = os.path.join(c_path, sampled_ins[1])  # 10
            sampled_transf = random.sample(os.listdir(individual_ins_path), 2)  # pca's

            # One compoennet
            comp_selected_path = os.path.join(individual_ins_path, sampled_transf[0])
            comp_selected_path = os.path.join(comp_selected_path, 'activation') if random.random() > 0.5 else os.path.join(comp_selected_path, 'latent')
            # img1_path_comp0 = os.path.join(comp_selected_path, random.sample(self.file_dict["trans"][comp_selected_path], 1)[0]) # same instance
            # print(self.file_dict["trans"].keys())
            img1_path_comp0 = random.sample(self.file_dict["trans"][comp_selected_path], 1)[0] # same instance

            # One compoennet with two examples
            comp_selected_path = os.path.join(individual_ins_path, sampled_transf[1])
            comp_selected_path = os.path.join(comp_selected_path, 'activation') if random.random() > 0.5 else os.path.join(comp_selected_path, 'latent')

            img_l = random.sample(self.file_dict["trans"][comp_selected_path], 2)  # same instance
            # img1_path_comp1_0 = os.path.join(comp_selected_path, img_l[0])  # same instance
            img1_path_comp1_0 = img_l[0]  # same instance
            # img1_path_comp1_1 = os.path.join(comp_selected_path, img_l[0])  # same instance
            img1_path_comp1_1 = img_l[1]  # same instance

            ins1 = Image.open(img0_path_ins0).convert("RGB")
            ins1_tensor = self._notransform(ins1) if self.noaug else self._transform(ins1)
            ins1.close()
            ins1_tensor = ins1_tensor.unsqueeze(0)
            img_1.append(ins1_tensor)

            ins2_comp1 = Image.open(img1_path_comp0).convert("RGB")
            ins2_comp1_tensor = self._notransform(ins2_comp1) if self.noaug else self._transform(ins2_comp1)
            ins2_comp1.close()
            ins2_comp1_tensor = ins2_comp1_tensor.unsqueeze(0)
            img_2.append(ins2_comp1_tensor)

            ins2_comp2_1 = Image.open(img1_path_comp1_0).convert("RGB")
            ins2_comp2_1_tensor = self._notransform(ins2_comp2_1) if self.noaug else self._transform(ins2_comp2_1)
            ins2_comp2_1.close()
            ins2_comp2_1_tensor = ins2_comp2_1_tensor.unsqueeze(0)
            img_3.append(ins2_comp2_1_tensor)

            ins2_comp2_2 = Image.open(img1_path_comp1_1).convert("RGB")
            ins2_comp2_2_tensor = self._notransform(ins2_comp2_2) if self.noaug else self._transform(ins2_comp2_2)
            ins2_comp2_2.close()
            ins2_comp2_2_tensor = ins2_comp2_2_tensor.unsqueeze(0)
            img_4.append(ins2_comp2_2_tensor)

            labels.append(self.category2id[c])

        img_1 = torch.cat(img_1)
        img_2 = torch.cat(img_2)
        img_3 = torch.cat(img_3)
        img_4 = torch.cat(img_4)

        return {'ins1': img_1, 'ins2_comp1': img_2, 'ins2_comp2_1': img_3, 'ins2_comp2_2': img_4, 'labels': labels}

    def _notransform(self, sample):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        composed_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        return composed_transforms(sample)


    def _transform(self, sample):
        if self.transform is None:

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            composed_transforms = transforms.Compose([
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            ])

            return composed_transforms(sample)
        else:
            return self.transform(sample)

    def __len__(self):
        # pairs =
        # TODO: ???
        return 2000


    def get_variance_list(self):
        same_trans_list = {}
        instance_list = {} # same seed image, diff transformation
        category_list = {}

        for c in self.categories:
            c_path = os.path.join(self.root_path, c)
            instance_path = os.listdir(c_path)
            cat_oswalk=[]
            for ins in instance_path:
                individual_ins_path = os.path.join(c_path, ins)

                individual_ins_oswalk = []


                component_list = os.listdir(individual_ins_path)

                for comp in component_list:
                    com_path = os.path.join(individual_ins_path, comp)
                    for each in os.listdir(com_path):
                        folder = os.path.join(com_path, each)
                        folder_oswalk=[]
                        for img in os.listdir(folder):
                            img_path = os.path.join(folder, img)

                            folder_oswalk.append(img_path)
                            individual_ins_oswalk.append(img_path)
                            cat_oswalk.append(img_path)

                        same_trans_list[folder] = folder_oswalk
                instance_list[individual_ins_path] = individual_ins_oswalk
            category_list[c_path] = cat_oswalk

        return {"trans": same_trans_list, "instance_list": instance_list, "cat_list": category_list}

if __name__ == "__main__":

    length_episode = 5
    # split = 'train'
    split = 'test'

    # import socket
    # if socket.gethostname() == 'deep':
    #     train_path = '/mnt/md0/2020Spring/invariant_imagenet/train_examplar'
    # elif socket.gethostname() == 'amogh':
    #     train_path = '/home/amogh/data/datasets/exemplar/train'
    #     test_path = '/home/amogh/data/datasets/exemplar/test_exemplar'


    exemplar_loader = GAN_invariance(path = "/local/vondrick/cz/GANdata/setting_50_16_sub",
                                     lenth_episode = length_episode, noaug=True
                                      )

    def trans_back(img):
        mean = np.asarray([[[0.485, 0.456, 0.406]]])
        std = np.asarray([[[0.229, 0.224, 0.225]]])
        # print('img',img.shape)
        img = np.moveaxis(np.squeeze(img), 0, 2)
        # print('img', img.shape)
        img = img*std + mean
        # img[:,:,1] = img[:,:,1]*std[1] + mean[1]
        # img[:,:,2] = img[:,:,2]*std[2] + mean[2]
        return img



    dataloader = torch.utils.data.DataLoader(exemplar_loader, batch_size=1, shuffle=False, num_workers=1)
    for i in range(2):
        for ii, examples in enumerate(dataloader):

            img_1 = examples['ins1']
            img_2 = examples['ins2_comp1']
            img_3 = examples['ins2_comp2_1']
            img_4 = examples['ins2_comp2_2']
            target = examples['labels']

            # el = exemplar_loader.get_episode_length()
            # print(ii,el)
            # if ii == 1:
            #     exemplar_loader.update_length_episode(7)
            # print("sample, ", sample['episode_examples'].shape)

            f, axarr = plt.subplots(length_episode, 4)
            for category_num, category_axis in enumerate(axarr):
                i1 = img_1[0][category_num].numpy()
                # print(episode_example)
                # i1 = np.moveaxis(np.squeeze(i1), 0, 2)
                i1 = trans_back(i1)

                i2 = img_2[0][category_num].numpy()
                # print(episode_example)
                # i2 = np.moveaxis(np.squeeze(i2), 0, 2)
                i2 = trans_back(i2)

                i3 = img_3[0][category_num].numpy()
                # print(episode_example)
                # i3 = np.moveaxis(np.squeeze(i3), 0, 2)
                i3 = trans_back(i3)

                i4 = img_4[0][category_num].numpy()
                # print(episode_example)
                # i4 = np.moveaxis(np.squeeze(i4), 0, 2)
                i4 = trans_back(i4)
                # print("iimg shape", i4.shape)


                # episode_query = episode_queries[0][category_num].numpy()
                # episode_query = np.moveaxis(np.squeeze(episode_query), 0, 2)

                category_axis[0].imshow(i1)
                category_axis[1].imshow(i2)
                category_axis[2].imshow(i3)
                category_axis[3].imshow(i4)

            # plt.show()
            plt.savefig("ddebug{}.png".format(i))



