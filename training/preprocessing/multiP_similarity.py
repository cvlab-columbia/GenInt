from utils import multiprocess, GPU_multiprocess
import os
import numpy as np
import pickle, random
import shutil
from sklearn import svm
from multiprocessing import Manager
import socket
from utils import fix_imagenet_temp, np_normalize_l2
from tqdm import tqdm
import csv
import time
import pandas as pd
import torch
from torchvision import transforms
import torchvision.transforms as transforms



class MultiP_model:
    def __init__(self, model, args, normalize, process_num, debug, class_list=None):
        print("start updating similarity matrix")
        start_time = time.time()
        self.process_num = process_num
        if 'cv' in socket.gethostname():
            data_root_path = '/proj/vondrick/mcz/ImageNet-Data'

        self.normalize = normalize
        self.args = args
        self.model = model
        self.debug = debug

        self.traindir = os.path.join(data_root_path, 'train')
        if class_list is None:
            self.class_list = os.listdir(self.traindir)
        else:
            self.class_list = class_list

        self.similarity_dict = {}

    def get_sim_dict(self):
        return self.similarity_dict

    def multi_process_generate(self):
        L = len(self.class_list)
        step = (L // self.process_num) + 1
        args = [(self.model_infer, self.class_list[i * step:min((i + 1) * step, L)], i) for i in
                range(self.process_num)]  # USing COpy
        GPU_multiprocess(self.iterate_f, args, self.process_num)

    @staticmethod
    def iterate_f(funct, filelist, process_id):
        for each in filelist:
            funct(each, process_id)

    def model_infer(self, each, process_id):
        if self.debug:
            print("process id {}".format(process_id))
        tmp_traindir = os.path.join(self.traindir, each)
        from data.dataset import ClassLoader
        t1 = time.time()
        train_dataset = ClassLoader(
            tmp_traindir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ]))
        num_examples = len(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=50, shuffle=False,
            num_workers=6, pin_memory=True)
        t2 = time.time()
        if self.debug:
            print("init loader {} seconds".format(t2 - t1))

        self.model.eval()
        with torch.no_grad():
            feature_arr = torch.tensor(np.zeros((num_examples, 2048)), dtype=torch.float32)

            cnt = 0
            path_list = []
            for i, (images, path) in enumerate(train_loader):
                for jj in path:
                    path_list.append(jj)
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                output, feature, norm = self.model(images)
                batchs = feature.size(0)
                feature_arr[cnt:cnt + batchs, :] = feature.data
                cnt = cnt + batchs
        t3 = time.time()
        if self.debug:
            print("forward time {} seconds".format(t3 - t2))
        # all = {'fea': feature, 'path': path_list}
        fea_vec = feature_arr
        similarity = torch.mm(fea_vec, fea_vec.t())  # num * fealen  * fealen * num = num * num
        similarity=similarity.cpu().numpy()
        sorted_id_s2l = np.argsort(similarity, axis=1)
        sorted_similarity = np.sort(similarity, axis=1)
        id = 0
        mapping_i2n = {}
        mapping_n2i = {}
        for each in path_list:
            # print(each)
            name = each.split('/')[-1]
            mapping_i2n[id] = name
            mapping_n2i[name] = id
            id += 1
        all_data = {'mapping_i2n': mapping_i2n, 'mapping_n2i': mapping_n2i, 'sorted_id_s2l': sorted_id_s2l,
                    'sorted_similarity': sorted_similarity}
        self.similarity_dict[each] = all_data
        t4 = time.time()
        if self.debug:
            print("sim matrix time {} seconds".format(t4 - t3))


class Similarity:
    def __init__(self, process_num, save_path, input_path='/proj/vondrick/mcz/ImageNet-Data/ResNet152features/train'):
        self.process_num = process_num

        self.path = input_path   # new ones
        self.cluster_path_root = save_path
        os.makedirs(save_path, exist_ok=True)

        self.class_list = os.listdir(self.path)

    def multi_process_generate(self):
        L = len(self.class_list)
        step = (L // self.process_num) + 1
        args = [(self.run_similarity, self.class_list[i * step:min((i + 1) * step, L)], i) for i in
                range(self.process_num)]  # USing COpy
        multiprocess(self.iterate_f, args, self.process_num)

    def run_similarity(self, filename, process_id):
        print("start process {}".format(process_id))
        start_time = time.time()
        fea_vec, img_path_list = self.load_single(self.path, filename)
        similarity = np.dot(fea_vec, fea_vec.T) # num * fealen  * fealen * num = num * num
        sorted_id_s2l = np.argsort(similarity, axis=1)

        sorted_similarity = np.sort(similarity, axis=1)

        path_name = img_path_list[0]
        path_split = path_name.split('/')
        class_folder = path_split[-2]  # Name of class. e.g., 'n01751748'

        id=0
        mapping_i2n={}
        mapping_n2i={}
        for each in img_path_list:
            # print(each)
            name = each.split('/')[-1]
            mapping_i2n[id] = name
            mapping_n2i[name] = id
            id += 1

        all_data={'mapping_i2n': mapping_i2n, 'mapping_n2i': mapping_n2i, 'sorted_id_s2l': sorted_id_s2l, 'sorted_similarity': sorted_similarity}

        with open(os.path.join(self.cluster_path_root, 'class_sim_{}.pkl'.format(class_folder)), 'wb') as f:
            pickle.dump(all_data, f)

        print("finish process {} in {} seconds".format(process_id, time.time() - start_time))


    @staticmethod
    def iterate_f(funct, filelist, process_id):
        for each in filelist:
            funct(each, process_id)

    @staticmethod
    def load_single(path_to, filename):
        '''Loading one category from a folder'''
        with open(os.path.join(path_to, filename), 'rb') as f:
            fea = pickle.load(f)
        fea_vec = fea['fea']
        img_path_list = fea['path']
        return fea_vec, img_path_list

from data.ShapeNet_Loader import getAngles
class SimilarityShapenet:
    def __init__(self, process_num, save_path):
        self.process_num = process_num

        self.path = '/proj/vondrick/mcz/ShapeNetRender/splited_128/train'  # new ones
        self.cluster_path_root = save_path
        os.makedirs(save_path, exist_ok=True)

        self.class_list = os.listdir(self.path)

    def multi_process_generate(self):
        L = len(self.class_list)
        step = (L // self.process_num) + 1
        args = [(self.run_similarity, self.class_list[i * step:min((i + 1) * step, L)], i) for i in
                range(self.process_num)]  # USing COpy
        multiprocess(self.iterate_f, args, self.process_num)

    def run_similarity(self, filename, process_id):
        print("start process {}".format(process_id))
        start_time = time.time()

        filepath = os.path.join(self.path, filename)
        obj_list = os.listdir(filepath)
        print("{} objects inside class {}".format(len(obj_list), filename))

        file_list = []
        for each in obj_list:
            img_list = os.listdir(os.path.join(filepath, each))
            file_list.extend([os.path.join(each, imgpath) for imgpath in img_list])

        if len(file_list)< 3000:
            x_ar = np.zeros((len(file_list), 1))
            y_ar = np.zeros((len(file_list), 1))
            z_ar = np.zeros((len(file_list), 1))

            print("done init")

            for cnt, each in enumerate(file_list):
                vp = "_".join(os.path.splitext(os.path.basename(each))[0].split("_")[1:])
                x_ar[cnt], y_ar[cnt], z_ar[cnt] = getAngles(vp)

            print('done angle')

            x_diff = np.remainder(np.abs(x_ar - x_ar.T), 180)
            y_diff = np.remainder(np.abs(y_ar - y_ar.T), 180)
            z_diff = np.remainder(np.abs(z_ar - z_ar.T), 180)

            print('calculate diff')

            print(x_diff)
            all_diff = x_diff + y_diff + z_diff

            similarity = -all_diff
            sorted_id_s2l = np.argsort(similarity, axis=1)
            sorted_similarity = np.sort(similarity, axis=1)
        else:
            # Matrix is too huge to do pair-wise comparision
            c_direction_id = np.zeros((len(file_list), 1))


        # print("{} images per obj".format(len(file_list)*1.0/len(obj_list)))

        class_folder = filename  # Name of class. e.g., 'n01751748'
        #
        id=0
        mapping={}
        for each in file_list:
            # print(each)
            mapping[id] = each

        all_data={'mapping': mapping, 'sorted_id_s2l': sorted_id_s2l, 'sorted_similarity': sorted_similarity}

        with open(os.path.join(self.cluster_path_root, 'class_sim_{}.pkl'.format(class_folder)), 'wb') as f:
            pickle.dump(all_data, f)

        print("finish process {} in {} seconds".format(process_id, time.time() - start_time))


    @staticmethod
    def iterate_f(funct, filelist, process_id):
        for each in filelist:
            funct(each, process_id)

    @staticmethod
    def load_single(path_to, filename):
        '''Loading one category from a folder'''
        with open(os.path.join(path_to, filename), 'rb') as f:
            fea = pickle.load(f)
        fea_vec = fea['fea']
        img_path_list = fea['path']
        return fea_vec, img_path_list



