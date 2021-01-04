'''
This file includes the dataloaders for ShapeNet.

'''

import torch.utils.data as data

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

class ShapeNetLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  root,
                  category_num=10,
                  im_per_model=3,
                  model_per_cat=2
                  ):

        super().__init__()
        
        start_time = time.time()
        
        self.root = root
        self.im_per_model = im_per_model
        self.model_per_cat = model_per_cat
        self.num_batches = 1000

        self.data_dict = {}
        self.batches = []

        self.dict_data, \
        self.dict_category2id, \
        self.dict_model2id, \
        self.dict_vp2id = self.getDictData(self.root)

        self.category_totalnum = category_num

        self.batches = self.getBatches(self.num_batches, self.dict_data, self.dict_category2id, self.dict_model2id, self.dict_vp2id)

        print("Dataloader initialised in {} seconds".format(time.time()-start_time),
              "\n Number of batches: ", self.num_batches,
              "\n dict_category2id: ", self.dict_category2id,
              "\n Number of viewpoints, dict_vp2id: ", len(self.dict_vp2id),
              )
        
    def getDictData(self, root):
        '''
        Returns dictionary of the form {cat1: dict_model_1: dict_viewpoint1:image_path}
        '''
        dict_data = {}
        categories = os.listdir(root)
        categories.sort()

        dict_category2id = {} # format {cat1_name: 0, cat2_name: 1, cat3_name: 2}
        dict_model2id = {} # format {cat1_name: {model1_name: 0}}, cat2_name: {model1_name: 0}}}
        dict_vp2id = {} # format {'c_1_x_0_y_37_z98': 3,'c_3_x_2_y_4_z9': 2 ....}
        vp_num = 0


        # Populate dict_data
        for category_num, category in enumerate(categories):

            dict_category2id[category] = category_num # format {cat1_name: 0, cat2_name: 1, cat3_name: 2}
            dict_model2id[category] = {} # format {cat1_name: {model1_name: 0}}, cat2_name: {model1_name: 0}}}

            path_category = os.path.join(root, category)
            list_models = os.listdir(path_category)
            list_models.sort()

            dict_model_to_dict_viewpoint = {}

            # Populate dict_model_to_dict_viewpoint
            for model_num, model in enumerate(list_models):

                dict_model2id[category][model] = model_num

                path_model_directory = os.path.join(path_category, model)
                list_viewpoint_image_paths = glob.glob(path_model_directory + '/*')
                list_viewpoint_image_paths.sort()
                
                dict_viewpoint_to_imagepath = {}

                # Populate dict_viewpoint_to_imagepath
                for viewpoint_image_path in list_viewpoint_image_paths:
                    viewpoint = "_".join(os.path.splitext(os.path.basename(viewpoint_image_path))[0].split("_")[1:]) # remove extension of path, remove model name and retain rotation information
                    dict_viewpoint_to_imagepath[viewpoint] = viewpoint_image_path

                    dict_vp2id[viewpoint] = vp_num
                    vp_num += 1

                dict_model_to_dict_viewpoint[model] = dict_viewpoint_to_imagepath

            dict_data[category] = dict_model_to_dict_viewpoint

        return dict_data, dict_category2id, dict_model2id, dict_vp2id

    def getBatches(self, num_batches, dict_data, dict_category2id, dict_model2id, dict_vp2id):
        '''
        Returns a schedule of batches.
        Currently: go through all categories, sample 2 models, sample 3 viewpoints from each of them.
        '''

        list_categories = dict_data.keys()
        batches = [] # Each batch is a dictionary

        for batch_num in range(num_batches):

            batch = {}
            batch['images'] = []
            batch['labels'] = []
            
            # Generate batches by going through each category
            for category in list_categories:

                dict_category_to_models = dict_data[category]
                all_models = dict_category_to_models.keys()

                # For this category, sample 2 models
                list_models = random.sample(all_models, self.model_per_cat)
                random.shuffle(list_models)

                images = []
                labels = []

                for model in list_models:
                    dict_model_to_viewpoints = dict_category_to_models[model]
                    all_viewpoints = dict_model_to_viewpoints.keys()
                    all_viewpoints.sort()
                    try:
                        list_viewpoints = random.sample(all_viewpoints, self.im_per_model-1)
                        random.shuffle(list_viewpoints)
                    except:
                        print("skipping model, category: ", model, category)
                        continue
                    list_viewpoints = [all_viewpoints[0]] + list_viewpoints   # Put the examplar on the front

                    list_viewpoints_paths = [dict_model_to_viewpoints[vp] for vp in list_viewpoints]
                    list_viewpoints_labels = [(category,model,vp, *self.getRotationsFromViewpoint(vp)) for vp in list_viewpoints]

                    # To save time during run time, mark the labels as ids from now

                    list_viewpoints_labels = [[dict_category2id[category], dict_model2id[category][model], dict_vp2id[vp],x,y,z] for (category,model,vp,x,y,z) in list_viewpoints_labels]


                    images.extend(list_viewpoints_paths)
                    labels.extend(list_viewpoints_labels)

                batch['images'].extend(images)
                batch['labels'].extend(labels)
            
            batches.append(batch)

        return batches

    def __getitem__(self, index):

        batch = self.batches[index]
        batch['images'] = [Image.open(path_image).convert('RGB') for path_image in batch['images']]
        # batch['labels'] = [(self.dict_category2id, )]
        batch['images'] = [self._transform(im) for im in batch['images']]
        batch['images'] = [im.unsqueeze(0) for im in batch['images']]
        batch['images'] = torch.cat(batch['images'])
        batch['labels'] = torch.LongTensor(batch['labels'])
        # batch['labels'] = torch.cat([l] for l in batch['labels'])
        return batch


    def __len__(self):
        return self.num_batches

    def _transform(self, sample):

#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

        composed_transforms = transforms.Compose([
            transforms.Compose([
#                 transforms.RandomResizedCrop(224),  # Due to ShapeNet dataset property
#                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ])
        ])

        return composed_transforms(sample)
    
    def shuffleBatches(self):
        self.batches = self.getBatches(self.dict_data, self.dict_category2id, self.dict_model2id, self.dict_vp2id)

    def getRotationsFromViewpoint(self, viewpoint):
        ''' Returns (x,y,z) from eg- "c_1_x_0_y_0_z0"'''
        split_viewpoint = viewpoint.split('_')
        x = int(split_viewpoint[3])
        y = int(split_viewpoint[5])
        z = int(split_viewpoint[6][1:])
        return x,y,z

class ExemplarTESTLoader(torch.utils.data.Dataset):
    pass

if __name__ == "__main__":
    split = 'train'

    import socket
    if socket.gethostname() == 'deep':
        train_path = '/mnt/md0/2020Spring/invariant_imagenet/train_examplar'
    elif socket.gethostname() == 'hulk':
        train_path = '/local/rcs/shared/examples/splited/train/'
    elif socket.gethostname() == 'amogh':
        train_path = '/home/amogh/data/shapenet_small/'
        # test_path = '/home/amogh/data/datasets/exemplar/test_exemplar'


    shapenet_train_loader = ShapeNetLoader(root = train_path)

    dataloader = torch.utils.data.DataLoader(shapenet_train_loader, batch_size=1, shuffle=False, num_workers=1)

    for ii, sample in enumerate(dataloader):
        check_sample = 1
    # for i in range(2):
    #     for ii, sample in enumerate(dataloader):
    #
    #         episode_examples = sample["episode_examples"]
    #         episode_queries = sample["episode_queries"]
    #         labels = sample["labels"]
    #
    #         el = exemplar_loader.get_episode_length()
    #         print(ii,el)
    #         if ii == 1:
    #             exemplar_loader.update_length_episode(7)
    #         print("sample, ", sample['episode_examples'].shape)
    #
    #         f, axarr = plt.subplots(length_episode, 2)
    #         for category_num, category_axis in enumerate(axarr):
    #             episode_example = episode_examples[0][category_num].numpy()
    #             # print(episode_example)
    #             episode_example = np.moveaxis(np.squeeze(episode_example), 0, 2)
    #             episode_query = episode_queries[0][category_num].numpy()
    #             episode_query = np.moveaxis(np.squeeze(episode_query), 0, 2)
    #
    #             category_axis[0].imshow(episode_example)
    #             category_axis[1].imshow(episode_query)
    #
    #         plt.show()

