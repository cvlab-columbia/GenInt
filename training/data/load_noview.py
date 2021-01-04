import torch.utils.data as data

import os, shutil
import sys

import random
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms as transforms


class RandomExhaustLoader_GAN(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir,
                  length_episode,
                  mix_neg_length,
                  mirror_path,
                  GAN_solo_dir,
                  GAN_mix_dir,
                  transform=None):
        super().__init__()
        """Can either apply diff transformation to two same images, or get two"""
        """Here, solo will replace the corresponding class pair in some ratio, mixture will only appear as new categories"""
        """One concern for solo classes are that they are not variant enough, compared with real image, wondering how much addition info they bring"""
        " if treat the mixup as nuanace, then we can make the mixup not repel each other, but they still invariant to local change, and stay away from imagenet true category"
        self.transform = transform
        self.length_episode = length_episode
        self.mirror_path = mirror_path

        self.mix_neg_length = mix_neg_length
        self.GAN_mix_dir = GAN_mix_dir
        self.get_GAN_mix_categories()

        # self.data_path = os.path.join(base_dir, split) # stores the path where train/test/val is
        self.train_path = train_base_dir # stores the path where train/test/val is

        self.categories_list = os.listdir(self.train_path)
        # This mapping only working for the old version for matching network
        # self.ids2category = dict(list(enumerate(self.categories_list)))
        # self.category2id = { v : k for k,v in self.ids2category.items()}  #TODO: this is for training, how to guarantee same map for test set.

        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.create_pysude_mirror()
        # self.file_dict = {}
        # for each in self.categories_list:
        #     self.file_dict[each] = os.listdir(os.path.join(self.train_path, each))
        print("finish build")

    def get_GAN_mix_categories(self):
        Allfolders = os.listdir(self.GAN_mix_dir)

        self.mix_neg_list = []
        for each in Allfolders:
            if len(os.listdir(os.path.join(self.GAN_mix_dir, each)))>10:
                self.mix_neg_list.append(each)

    def create_pysude_mirror(self):
        """Following the data structure of the viewpoint and create a fake list of files, later if one of the file is used,
        It will be removed from the list, once all are exhausted, we will recrate this again."""
        print("Start creating mirrored sckleton folders")
        os.makedirs(os.path.join(self.mirror_path, 'new'), exist_ok=False)
        # os.makedirs(os.path.join(self.mirror_path, 'old'), exist_ok=False)

        self.new_root = os.path.join(self.mirror_path, 'new')
        for category in self.categories_list:
            new_category_path = os.path.join(self.new_root, category)
            os.makedirs(new_category_path, exist_ok=False)

            category_path = os.path.join(self.train_path, category)
            # for vv in os.listdir(category_path):
            #
            #     vv_path = os.path.join(new_category_path, vv)
            #     os.makedirs(vv_path)
            #
            #     example_viewpoint_path = os.path.join(category_path, vv)

            for ff in os.listdir(category_path):
                if 'jpeg' in ff.lower():
                    ff_name = ff.split('.')[0]
                    with open(os.path.join(new_category_path, ff_name), 'w') as filetmp:
                        filetmp.write("1")

                # # TODO: DEbug:
                # break

        print("Finish!")
        print("Finish!")
        print("Finish!")

    def check_finish(self):
        if len(os.listdir(self.new_root)) == 0:
            print("Finish One Whole Epoch!")
            shutil.rmtree(self.new_root)
            self.create_pysude_mirror()
        else:
            print("{} Number of Category Remains".format(len(os.listdir(self.new_root))))
            print("not finish yet")

    def remove_root(self):
        print("remove the mirror in the end")
        shutil.rmtree(self.new_root)
        print("remove finish")

    def __len__(self):
        # pairs =
        # TODO: ???
        return 500

    def __getitem__(self, index):

        # Choose a set of categories and sample length_episode
        # Choose a category and a viewpoint and return that, we will make the episode later
        episode_categories = random.sample(self.categories_list, self.length_episode)
        random.shuffle(episode_categories)  # It is inplace shuffle thus return None

        # print("episode length", len(episode_categories))
        # Initialise the empty arrays to which the image arrays will be appended
        episode_examples = []
        episode_queries = []
        labels = []

        # For each category, choose two viewpoints, one of which will be the query
        for category in episode_categories:
            category_path = os.path.join(self.train_path, category)
            # TODO: Amogh, optimize the speed by init all before hand
            # TODO: Can we shuffle each epoch? Prevent viewpoints rand time

            Flag_notused = True

            try:
                # exhaust the first image in the pair through trainset
                cat_path = os.path.join(self.new_root, category)
                existing_images = os.listdir(cat_path)
                sampled_img = random.sample(existing_images, 1)[0]

                remove_path = os.path.join(cat_path, sampled_img)
                os.remove(remove_path)

                if len(os.listdir(cat_path)) == 0:
                    shutil.rmtree(cat_path)

                load_img_path = os.path.join(category_path, sampled_img+".JPEG")
                example_image = Image.open(load_img_path).convert("RGB")
                example_image_tensor = self._transform(example_image)
                example_image.close()
                example_image_tensor = example_image_tensor.unsqueeze(0)
                episode_examples.append(example_image_tensor)

                # random generate paired images
                img_list = os.listdir(os.path.join(self.train_path, category))
                sampled2_img = random.sample(img_list, 1)[0]


                query_viewpoint_path = os.path.join(category_path, sampled2_img)
                query_image = Image.open(query_viewpoint_path).convert("RGB")
                query_image_tensor = self._transform(query_image)
                query_image.close()
                query_image_tensor = query_image_tensor.unsqueeze(0)
                episode_queries.append(query_image_tensor)
            except:
                Flag_notused = False

            if Flag_notused is False:
                files_list = os.listdir(category_path)
                # # files_list = self.file_dict[category]
                # length = len(self.file_dict[category])
                # # print('len', length)
                # r1 = random.randint(0, length-1)
                # while True:
                #     r2 = random.randint(0, length-1)
                #     if r2 != r1:
                #         break

                # Choose 2 viewpoints
                sampled_examples = random.sample(files_list, 2)

                # From each of the 2 viewpoints, choose 1 image each and put in stack
                # example = self.file_dict[category][r1]
                # query = self.file_dict[category][r2]
                example = sampled_examples[0]
                query = sampled_examples[1]

                # print('example', example, query)

                example_image_path = os.path.join(category_path, example)
                example_image = Image.open(example_image_path).convert("RGB")
                example_image_tensor = self._transform(example_image)
                example_image.close()
                example_image_tensor = example_image_tensor.unsqueeze(0)
                episode_examples.append(example_image_tensor)

                query_viewpoint_path = os.path.join(category_path, query)
                query_image = Image.open(query_viewpoint_path).convert("RGB")
                query_image_tensor = self._transform(query_image)
                query_image.close()
                query_image_tensor = query_image_tensor.unsqueeze(0)
                episode_queries.append(query_image_tensor)

            labels.append(self.category2id[category])
        episode_examples = torch.cat(episode_examples)
        episode_queries = torch.cat(episode_queries)

        # Sample the Mixture Negative
        mix_neg_categories = random.sample(self.mix_neg_list, self.mix_neg_length)
        mix_neg_1 = []
        mix_neg_2 = []
        for each in mix_neg_categories:
            category_path = os.path.join(self.GAN_mix_dir, each)
            img_list = os.listdir(category_path)
            img_list.sort()
            list_len = len(img_list)

            sampled_img_id = random.randint(1, list_len-2)  # -2 since there's ./ file
            img_name = img_list[sampled_img_id]
            img_name_prev = img_list[sampled_img_id-1]
            img_name_later = img_list[sampled_img_id+1]

            example_image = Image.open(os.path.join(category_path, img_name)).convert("RGB")
            example_image_tensor = self._transform(example_image)
            example_image.close()
            example_image_tensor = example_image_tensor.unsqueeze(0)
            mix_neg_1.append(example_image_tensor)

            diff1 = self.parse_img_name(img_name) - self.parse_img_name(img_name_prev)
            diff2 = self.parse_img_name(img_name_later) - self.parse_img_name(img_name)
            if diff1 < diff2:
                query = img_name_prev
            else:
                query = img_name_later

            query_image = Image.open(os.path.join(category_path, query)).convert("RGB")
            query_image_tensor = self._transform(query_image)
            query_image.close()
            query_image_tensor = query_image_tensor.unsqueeze(0)
            mix_neg_2.append(query_image_tensor)

        neg_1_examples = torch.cat(mix_neg_1, dim=0)
        neg_2_examples = torch.cat(mix_neg_2, dim=0)


        return {"episode_examples": episode_examples,
                "episode_queries": episode_queries,
                "labels": labels,
                "neg_1": neg_1_examples,
                "neg_2": neg_2_examples}

    def parse_img_name(self, input_name):
        splitname = input_name.split('_')
        return int(splitname[1])

    def update_length_episode(self, new_length):
        self.length_episode = new_length

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


class RandomExhaustLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir,
                  length_episode,
                  mirror_path,
                  transform=None):
        super().__init__()
        self.transform = transform
        self.length_episode = length_episode
        self.mirror_path = mirror_path

        # self.data_path = os.path.join(base_dir, split) # stores the path where train/test/val is
        self.train_path = train_base_dir # stores the path where train/test/val is

        self.categories_list = os.listdir(self.train_path)
        # This mapping only working for the old version for matching network
        # self.ids2category = dict(list(enumerate(self.categories_list)))
        # self.category2id = { v : k for k,v in self.ids2category.items()}  #TODO: this is for training, how to guarantee same map for test set.

        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.create_pysude_mirror()
        # self.file_dict = {}
        # for each in self.categories_list:
        #     self.file_dict[each] = os.listdir(os.path.join(self.train_path, each))
        print("finish build")

    def create_pysude_mirror(self):
        """Following the data structure of the viewpoint and create a fake list of files, later if one of the file is used,
        It will be removed from the list, once all are exhausted, we will recrate this again."""
        print("Start creating mirrored sckleton folders")
        os.makedirs(os.path.join(self.mirror_path, 'new'), exist_ok=False)
        # os.makedirs(os.path.join(self.mirror_path, 'old'), exist_ok=False)

        self.new_root = os.path.join(self.mirror_path, 'new')
        for category in self.categories_list:
            new_category_path = os.path.join(self.new_root, category)
            os.makedirs(new_category_path, exist_ok=False)

            category_path = os.path.join(self.train_path, category)
            # for vv in os.listdir(category_path):
            #
            #     vv_path = os.path.join(new_category_path, vv)
            #     os.makedirs(vv_path)
            #
            #     example_viewpoint_path = os.path.join(category_path, vv)

            for ff in os.listdir(category_path):
                if 'jpeg' in ff.lower():
                    ff_name = ff.split('.')[0]
                    with open(os.path.join(new_category_path, ff_name), 'w') as filetmp:
                        filetmp.write("1")

                # # TODO: DEbug:
                # break

        print("Finish!")
        print("Finish!")
        print("Finish!")

    def check_finish(self):
        if len(os.listdir(self.new_root)) == 0:
            print("Finish One Whole Epoch!")
            shutil.rmtree(self.new_root)
            self.create_pysude_mirror()
        else:
            print("{} Number of Category Remains".format(len(os.listdir(self.new_root))))
            print("not finish yet")

    def remove_root(self):
        print("remove the mirror in the end")
        shutil.rmtree(self.new_root)
        print("remove finish")

    def __len__(self):
        # pairs =
        # TODO: ???
        return 500

    def __getitem__(self, index):

        # Choose a set of categories and sample length_episode
        # Choose a category and a viewpoint and return that, we will make the episode later
        episode_categories = random.sample(self.categories_list, self.length_episode)
        random.shuffle(episode_categories)  # It is inplace shuffle thus return None

        # print("episode length", len(episode_categories))
        # Initialise the empty arrays to which the image arrays will be appended
        episode_examples = []
        episode_queries = []
        labels = []

        # For each category, choose two viewpoints, one of which will be the query
        for category in episode_categories:
            category_path = os.path.join(self.train_path, category)
            # TODO: Amogh, optimize the speed by init all before hand
            # TODO: Can we shuffle each epoch? Prevent viewpoints rand time

            Flag_notused = True

            try:
                # exhaust the first image in the pair through trainset
                cat_path = os.path.join(self.new_root, category)
                existing_images = os.listdir(cat_path)
                sampled_img = random.sample(existing_images, 1)[0]

                remove_path = os.path.join(cat_path, sampled_img)
                os.remove(remove_path)

                if len(os.listdir(cat_path)) == 0:
                    shutil.rmtree(cat_path)

                load_img_path = os.path.join(category_path, sampled_img+".JPEG")
                example_image = Image.open(load_img_path).convert("RGB")
                example_image_tensor = self._transform(example_image)
                example_image.close()
                example_image_tensor = example_image_tensor.unsqueeze(0)
                episode_examples.append(example_image_tensor)

                # random generate paired images
                img_list = os.listdir(os.path.join(self.train_path, category))
                sampled2_img = random.sample(img_list, 1)[0]


                query_viewpoint_path = os.path.join(category_path, sampled2_img)
                query_image = Image.open(query_viewpoint_path).convert("RGB")
                query_image_tensor = self._transform(query_image)
                query_image.close()
                query_image_tensor = query_image_tensor.unsqueeze(0)
                episode_queries.append(query_image_tensor)
            except:
                Flag_notused = False

            if Flag_notused is False:
                files_list = os.listdir(category_path)
                # # files_list = self.file_dict[category]
                # length = len(self.file_dict[category])
                # # print('len', length)
                # r1 = random.randint(0, length-1)
                # while True:
                #     r2 = random.randint(0, length-1)
                #     if r2 != r1:
                #         break

                # Choose 2 viewpoints
                sampled_examples = random.sample(files_list, 2)

                # From each of the 2 viewpoints, choose 1 image each and put in stack
                # example = self.file_dict[category][r1]
                # query = self.file_dict[category][r2]
                example = sampled_examples[0]
                query = sampled_examples[1]

                # print('example', example, query)

                example_image_path = os.path.join(category_path, example)
                example_image = Image.open(example_image_path).convert("RGB")
                example_image_tensor = self._transform(example_image)
                example_image.close()
                example_image_tensor = example_image_tensor.unsqueeze(0)
                episode_examples.append(example_image_tensor)

                query_viewpoint_path = os.path.join(category_path, query)
                query_image = Image.open(query_viewpoint_path).convert("RGB")
                query_image_tensor = self._transform(query_image)
                query_image.close()
                query_image_tensor = query_image_tensor.unsqueeze(0)
                episode_queries.append(query_image_tensor)

            labels.append(self.category2id[category])
        episode_examples = torch.cat(episode_examples)
        episode_queries = torch.cat(episode_queries)

        return {"episode_examples": episode_examples,
                "episode_queries": episode_queries,
                "labels": labels}


    def update_length_episode(self, new_length):
        self.length_episode = new_length

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



class ExemplarFreeLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir,
                  length_episode,
                  transform=None):
        super().__init__()
        self.transform = transform
        self.length_episode = length_episode

        # self.data_path = os.path.join(base_dir, split) # stores the path where train/test/val is
        self.train_path = train_base_dir # stores the path where train/test/val is

        self.categories_list = os.listdir(self.train_path)
        # This mapping only working for the old version for matching network
        # self.ids2category = dict(list(enumerate(self.categories_list)))
        # self.category2id = { v : k for k,v in self.ids2category.items()}  #TODO: this is for training, how to guarantee same map for test set.

        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}
        # self.file_dict = {}
        # for each in self.categories_list:
        #     self.file_dict[each] = os.listdir(os.path.join(self.train_path, each))
        print("finish build")

    def __getitem__(self, index):

        # Choose a set of categories and sample length_episode
        # Choose a category and a viewpoint and return that, we will make the episode later
        episode_categories = random.sample(self.categories_list, self.length_episode)
        random.shuffle(episode_categories)  # It is inplace shuffle thus return None

        # Initialise the empty arrays to which the image arrays will be appended
        episode_examples = []
        episode_queries = []
        labels = []

        # For each category, choose two viewpoints, one of which will be the query
        for category in episode_categories:
            category_path = os.path.join(self.train_path, category)
            # TODO: Amogh, optimize the speed by init all before hand
            # TODO: Can we shuffle each epoch? Prevent viewpoints rand time

            files_list = os.listdir(category_path)
            # # files_list = self.file_dict[category]
            # length = len(self.file_dict[category])
            # # print('len', length)
            # r1 = random.randint(0, length-1)
            # while True:
            #     r2 = random.randint(0, length-1)
            #     if r2 != r1:
            #         break

            # Choose 2 viewpoints
            sampled_examples = random.sample(files_list, 2)

            # From each of the 2 viewpoints, choose 1 image each and put in stack
            # example = self.file_dict[category][r1]
            # query = self.file_dict[category][r2]
            example = sampled_examples[0]
            query = sampled_examples[1]

            # print('example', example, query)

            example_image_path = os.path.join(category_path, example)
            example_image = Image.open(example_image_path).convert("RGB")
            example_image_tensor = self._transform(example_image)
            example_image.close()
            example_image_tensor = example_image_tensor.unsqueeze(0)
            episode_examples.append(example_image_tensor)

            query_viewpoint_path = os.path.join(category_path, query)
            query_image = Image.open(query_viewpoint_path).convert("RGB")
            query_image_tensor = self._transform(query_image)
            query_image.close()
            query_image_tensor = query_image_tensor.unsqueeze(0)
            episode_queries.append(query_image_tensor)

            labels.append(self.category2id[category])
        episode_examples = torch.cat(episode_examples)
        episode_queries = torch.cat(episode_queries)

        return {"episode_examples": episode_examples,
                "episode_queries": episode_queries,
                "labels": labels}

    def __len__(self):
        # pairs =
        # TODO: ???
        return 500

    def update_length_episode(self, new_length):
        self.length_episode = new_length

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

            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # tr.ToTensor()])
    #
    #     return composed_transforms(sample)


class TESTNOVIEWLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir,
                  test_base_dir,
                  length_episode):
        super().__init__()

        self.length_episode = length_episode

        # self.data_path = os.path.join(base_dir, split) # stores the path where train/test/val is
        self.train_path = train_base_dir # stores the path where train/test/val is
        self.test_path = test_base_dir

        self.categories_list = os.listdir(self.train_path)
        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        # self.ids2category = dict(list(enumerate(self.categories_list)))
        # self.category2id = { v : k for k,v in self.ids2category.items()}  #TODO: this is for training, how to guarantee same map for test set.

    def __getitem__(self, index):
        # Sample some categories from the
        # episode_categories = random.sample(self.categories_list, self.length_episode)
        episode_categories = random.sample(self.categories_list, self.length_episode) # TODO : the sequence still stable, THUS NOT ANOTHER PERMUTE!!!

        # From each category, sample 1 and append
        episode_queries = [] # Contains test query image
        episode_examples = [] # These form the examples
        labels = []

        # print('episode_categories', episode_categories)
        for category in episode_categories:

            # print("test category", category)

            # Get 1 image from category for example images
            train_category_path = os.path.join(self.train_path, category)
            # viewpoints_list = os.listdir(train_category_path)
            # example_viewpoint = random.sample(viewpoints_list, 1)[0]
            # example_viewpoint_path = os.path.join(train_category_path, example_viewpoint)
            # a = glob.glob(os.path.join(example_viewpoint_path,'/*'))  # ?


            example_image_path = random.sample(glob.glob(train_category_path + '/*'), 1)[0]
            example_image = Image.open(example_image_path).convert("RGB")
            example_image_tensor = self._transform(example_image)
            example_image.close()

            example_image_tensor = example_image_tensor.unsqueeze(0)
            episode_examples.append(example_image_tensor)

            # Get 1 image from category for query images from test folder path
            test_category_path = os.path.join(self.test_path, category)
            test_paths_list = glob.glob(test_category_path + '/*')

            query_image_path = random.sample(test_paths_list, 1)[0]  # TODO: wrong! TEST IS TRAVERSING ALL ONE BY ONE
            query_image = Image.open(query_image_path).convert("RGB")
            query_image_tensor = self._transform(query_image)
            query_image.close()

            query_image_tensor = query_image_tensor.unsqueeze(0)
            episode_queries.append(query_image_tensor)

            labels.append(self.category2id[category])

        episode_examples = torch.cat(episode_examples)
        episode_queries = torch.cat(episode_queries)

        # return [torch.Tensor([1,2,3]),], torch.Tensor([4,5,6])
        return {"episode_examples": episode_examples,
                "episode_queries": episode_queries,
                "labels": labels}



    def __len__(self):
        # pairs =
        #TODO: what is length for test?
        return 10

    def update_length_episode(self, new_length):
        self.length_episode = new_length

    def _transform(self, sample):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        composed_transforms = transforms.Compose([
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ])

        return composed_transforms(sample)


import json


class ObjectNOVIEWLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir,
                  object_dir,
                  obj_batchsize):
        super().__init__()

        self.obj_batchsize = obj_batchsize

        # self.data_path = os.path.join(base_dir, split) # stores the path where train/test/val is
        self.train_path = train_base_dir # stores the path where train/test/val is
        self.object_dir = object_dir

        self.categories_list = os.listdir(self.train_path)
        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        with open('preprocessing/obj2imgnet_id.txt') as f:
            self.dict_obj2imagenet_id = json.load(f)

        self.obj_categories_list = os.listdir(self.object_dir)
        self.obj_categories_list.sort()

        self.obj_file_lists = []
        self.obj_label_lists = []
        for each in self.obj_categories_list:
            obj_folder_path = os.path.join(self.object_dir, each)

            obj_files_names = os.listdir(obj_folder_path)

            for eachfile in obj_files_names:
                image_path = os.path.join(obj_folder_path, eachfile)
                self.obj_file_lists.append(image_path)
                self.obj_label_lists.append(self.dict_obj2imagenet_id[each])  #THese can be processed in separate list


        index_rand = [i for i in range(len(self.obj_file_lists))]
        random.shuffle(index_rand)

        self.obj_file_lists = [self.obj_file_lists[ii] for ii in index_rand]
        self.obj_label_lists = [self.obj_label_lists[ii] for ii in index_rand]

        # self.ids2category = dict(list(enumerate(self.categories_list)))
        # self.category2id = { v : k for k,v in self.ids2category.items()}  #TODO: this is for training, how to guarantee same map for test set.

    def __getitem__(self, index):
        # Sample some categories from the
        # episode_categories = random.sample(self.categories_list, self.length_episode)

        # From each category, sample 1 and append
        episode_queries = [] # Contains test query image
        train_support_set = [] # These form the examples
        labels_train = []

        # print('episode_categories', episode_categories)
        for category in self.categories_list:

            # print("test category", category)

            # Get 1 image from category for example images
            train_category_path = os.path.join(self.train_path, category)
            # viewpoints_list = os.listdir(train_category_path)
            # example_viewpoint = random.sample(viewpoints_list, 1)[0]
            # example_viewpoint_path = os.path.join(train_category_path, example_viewpoint)
            # a = glob.glob(os.path.join(example_viewpoint_path,'/*'))  # ?


            example_image_path = random.sample(glob.glob(train_category_path + '/*'), 1)[0]
            example_image = Image.open(example_image_path).convert("RGB")
            example_image_tensor = self._transform(example_image)
            example_image.close()

            example_image_tensor = example_image_tensor.unsqueeze(0)
            train_support_set.append(example_image_tensor)

            labels_train.append(self.category2id[category])


        object_file_selected = self.obj_file_lists[index * self.obj_batchsize: (index+1) * self.obj_batchsize]
        obj_labels = self.obj_label_lists[index * self.obj_batchsize: (index+1) * self.obj_batchsize]

        obj_loaded_images = []
        for each in object_file_selected:
            img = Image.open(each).convert("RGB")
            img_tensor = self._transform(img)
            img.close()

            img_tensor = img_tensor.unsqueeze(0)
            obj_loaded_images.append(img_tensor)


        train_support_set = torch.cat(train_support_set)
        obj_loaded_images = torch.cat(obj_loaded_images)

        # return [torch.Tensor([1,2,3]),], torch.Tensor([4,5,6])
        return {"train_set":        train_support_set,
                "train_labels":     labels_train,
                "objectnet_set":    obj_loaded_images,
                "objectnet_labels": obj_labels}



    def __len__(self):
        # pairs =
        #TODO: what is length for test?
        return len(self.obj_file_lists) // self.obj_batchsize

    def update_length_episode(self, new_length):
        self.length_episode = new_length

    def _transform(self, sample):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        composed_transforms = transforms.Compose([
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ])

        return composed_transforms(sample)
