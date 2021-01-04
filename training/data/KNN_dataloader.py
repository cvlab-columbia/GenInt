"""
KNN loader need to load images by class, to facilitate the following 1) class mean  2) closest vector in class to query

For objectnet, the loader need to load label lists, enable later accuracy calculation, remember to remove the red box.

"""

import torch, os
import numpy as np
from PIL import Image

import torch
import json
from torchvision import transforms
import torchvision.transforms as transforms


class ObjectNetClassWiseLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir
                  ):
        super().__init__()

        self.train_path = train_base_dir
        self.categories_list = os.listdir(self.train_path)
        self.categories_list.sort()

        with open('preprocessing/obj2imgnet_id.txt') as f:
            self.dict_obj2imagenet_id = json.load(f)

    def __len__(self):
        return len(self.categories_list)

    def _transform(self, sample):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        composed_transforms = transforms.Compose([
            transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ])

        return composed_transforms(sample)

    def __getitem__(self, item):
        folder_path = os.path.join(self.train_path, self.categories_list[item])

        files_names = os.listdir(folder_path)

        img_list = []
        labels = []
        path_list = []
        for eachfile in files_names:
            image_path = os.path.join(folder_path, eachfile)
            img = Image.open(image_path).convert("RGB")
            img_tensor = self._transform(img)
            img.close()

            img_tensor = img_tensor.unsqueeze(0)
            img_list.append(img_tensor)
            path_list.append(image_path)

            labels.append(self.dict_obj2imagenet_id[self.categories_list[item]])

        img_list = torch.cat(img_list)
        return {"images": img_list, "labels": labels, "path": path_list}


class ObjectNetLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir, few_test=None, composed_transform=None, center_crop=False
                  ):
        super().__init__()

        self.train_path = train_base_dir
        self.categories_list = os.listdir(self.train_path)
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.few_test = few_test
        if composed_transform is None:
             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

             composed_transform = transforms.Compose([
                transforms.Compose([
                transforms.Resize(256) if not center_crop else transforms.Resize(int(256 * 1.4)),
                transforms.CenterCrop(256 if not center_crop else 224),
                transforms.ToTensor(),
                normalize,
                ])
                ])
        self.composed_transforms=composed_transform

        with open('preprocessing/obj2imgnet_id.txt') as f:
            self.dict_obj2imagenet_id = json.load(f)

        for each in self.categories_list:
            folder_path = os.path.join(self.train_path, each)

            files_names = os.listdir(folder_path)

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                self.label_lists.append(self.dict_obj2imagenet_id[each]+[-1]*10)  # since the loader cutoff automatically on the
                # minimum length of labels, we make the minimum to be 11, by adding redundant 10 -1s.

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
            return len(self.label_lists)

    def _transform(self, sample):
        return self.composed_transforms(sample)

    def __getitem__(self, item):
        path_list=self.file_lists[item]
        img = Image.open(path_list).convert("RGB")

        img_tensor = self._transform(img)
        img.close()
        labels = self.label_lists[item]
        return {"images": img_tensor, "labels": labels, "path": path_list}

class ImagenetALoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir, few_test=None
                  ):
        super().__init__()

        self.train_path = train_base_dir
        self.categories_list = [f for f in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, f))]
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.few_test = few_test

        self.dict_id2int = generateImagenetClassID2Int(imagenet_path="/proj/vondrick/mcz/ImageNet-Data/train/")

        for each in self.categories_list:

            folder_path = os.path.join(self.train_path, each) # Category folder
            files_names = os.listdir(folder_path)
            label_id = self.dict_id2int[each]

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                self.label_lists.append([label_id] + [-1]*10)

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
            return len(self.label_lists)

    def _transform(self, sample):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        composed_transforms = transforms.Compose([
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize,
            ])
        ])

        return composed_transforms(sample)

    def __getitem__(self, item):
        path_list=self.file_lists[item]
        img = Image.open(path_list).convert("RGB")

        img_tensor = self._transform(img)
        img.close()
        labels = self.label_lists[item]
        return {"images": img_tensor, "labels": labels, "path": path_list}



class ImgNetClassWiseLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir
                  ):
        super().__init__()

        self.train_path = train_base_dir
        self.categories_list = os.listdir(self.train_path)
        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

    def __len__(self):
        return len(self.categories_list)

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

    def __getitem__(self, item):
        folder_path = os.path.join(self.train_path, self.categories_list[item])

        files_names = os.listdir(folder_path)
        img_list = []
        labels = []
        path_list = []
        for eachfile in files_names:
            image_path = os.path.join(folder_path, eachfile)
            img = Image.open(image_path).convert("RGB")
            img_tensor = self._transform(img)
            img.close()

            img_tensor = img_tensor.unsqueeze(0)
            img_list.append(img_tensor)
            path_list.append(image_path)

            labels.append(self.category2id[self.categories_list[item]])

        img_list = torch.cat(img_list)
        return {"images": img_list, "labels": labels, "path": path_list}

def generateImagenetClassID2Int(imagenet_path="/proj/vondrick/mcz/ImageNet-Data/train/"):
    '''
    Returns dictionary of classid --> Int. Eg - {n02119789: 23}
    '''
    filelist = sorted(os.listdir(imagenet_path))
    dict_id2int = {}

    for i, each in enumerate(filelist):
        dict_id2int[each] = i

    return dict_id2int

if __name__ == "__main__":

    import socket

    if socket.gethostname() == 'deep':
        train_path = '/mnt/md0/2020Spring/invariant_imagenet/train_examplar'
    elif socket.gethostname() == 'amogh':
        imageneta_valdir = '/media/amogh/Stuff/Data/natural-adversarial-examples-imageneta/imagenet-a/imagenet-a/'
    elif'cv' in socket.gethostname():
        imageneta_valdir = '/proj/vondrick/amogh/imagenet-a'
        obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test'
    
    visualize = False

    # val_loader = ImagenetALoader(train_base_dir=imageneta_valdir)
    val_dataset = ObjectNetLoader(train_base_dir=obj_valdir)

    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

    output_visualize = '/proj/vondrick/amogh/Invariant/visualize/images_objnet_loader/'

    for i in range(2):
        for ii, sample in enumerate(dataloader):

            images = sample["images"]
            labels = sample["labels"]
            path_list = sample["path"]
            
            array = images.numpy()
#            print(images.shape)

            if visualize:
                out_file = os.path.join(output_visualize, 'images_{}_{}.npy'.format(i, ii))
                np.save(out_file, array)
                print("Saved  at:  ",out_file)
            # print(ii, "\n", images, labels, path_list)

            if (ii == 5):
                break

            # f, axarr = plt.subplots(length_episode, 2)
            # for category_num, category_axis in enumerate(axarr):
            #     episode_example = episode_examples[0][category_num].numpy()
            #     # print(episode_example)
            #     episode_example = np.moveaxis(np.squeeze(episode_example), 0, 2)
            #     episode_query = episode_queries[0][category_num].numpy()
            #     episode_query = np.moveaxis(np.squeeze(episode_query), 0, 2)
            #
            #     category_axis[0].imshow(episode_example)
            #     category_axis[1].imshow(episode_query)
            #
            # plt.show()
        print("Dataloader debugging epoch done")













