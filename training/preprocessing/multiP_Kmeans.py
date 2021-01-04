from utils import multiprocess
import os
import numpy as np
import pickle, random
import shutil
from sklearn import svm
from multiprocessing import Manager
import socket
from utils import fix_imagenet_temp, np_normalize_l2
from sklearn.cluster import KMeans

# exist_list = ['n01978287-4', 'n02137549-4', 'n02835271-4', 'n03796401-4',
# 'n02094433-4', 'n02823750-4','n03733131-4','n04590129-4']
exist_list = []

def remove_generated(whole, existing):
    result = []
    for each in whole:
        if each not in existing:
            result.append(each)
    return result


class KmeansCluster:
    def __init__(self, process_num, K=10, subset=False, test_mode=False, normalize=True, randomize=False,
                 negative_weight_rescale=1.0):
        self.process_num = process_num
        self.K = K
        self.subset = subset
        self.test_mode = test_mode
        self.normalize = normalize
        self.randomize = randomize
        self.negative_weight_rescale = negative_weight_rescale

        if socket.gethostname() == 'deep':
            self.path = '/mnt/md0/2020Spring/VGG_features/train'
            self.cluster_path_root = '/mnt/md0/2020Spring/invariant_imagenet/train_kmeans'
            self.imagenet_path = '/mnt/md0/ImageNet/train'
        elif socket.gethostname() == 'hulk':
            # self.path = '/local/rcs/mcz/2020Spring/VGG_features/train' # THIS is old wrong croped vgg features
            self.path = '/local/rcs/mcz/2020Spring/invariant_imagenet/VGGfeatures/train' # new ones
            self.cluster_path_root = '/local/rcs/mcz/2020Spring/invariant_imagenet/train_kmeans'
            self.imagenet_path = '/local/rcs/mcz/ImageNet-Data'

        elif 'cv' in socket.gethostname():
            self.path = '/proj/vondrick/mcz/ImageNet-Data/VGGfeatures_NoNorm/train' # new ones
            # self.cluster_path_root = '/local/vondrick/cz/cut_img-S3.5-TC5/train_clustered-C{}'.format(self.C) # old
            self.cluster_path_root = '/proj/vondrick/mcz/ImageNet-Data/train_kmeans_{}'.format(self.K)
            # Now I want to split the whole set without removing overlap trains
            self.imagenet_path = '/proj/vondrick/mcz/ImageNet-Data'



        # fix_imagenet_temp(os.path.join(self.imagenet_path, 'train'))
        # fix_imagenet_temp(os.path.join(self.imagenet_path, 'val'))

        os.makedirs(self.cluster_path_root, exist_ok=True)
        class_list = os.listdir(self.path)

        self.class_list = sorted(remove_generated(class_list, exist_list))

    def multi_process_generate(self):

        L = len(self.class_list)
        step = (L // self.process_num) + 1
        args = [(self.run_examplar, self.class_list[i * step:min((i + 1) * step, L)]) for i in
                range(self.process_num)]  # USing COpy
        multiprocess(self.iterate_f, args, self.process_num)

    @staticmethod
    def load_single(path_to, filename):
        '''Loading one category from a folder'''
        with open(os.path.join(path_to, filename), 'rb') as f:
            fea = pickle.load(f)
        fea_vec = fea['fea']
        img_path_list = fea['path']
        return fea_vec, img_path_list

    def run_examplar(self, filename):
        cnt_class_id = 0

        fea_vec, img_path_list = self.load_single(self.path, filename)

        path_name = img_path_list[0]
        path_split = path_name.split('/')
        class_folder = path_split[-3]
        print('class', class_folder)
        # exit(0)

        cluster_path = os.path.join(os.path.join(self.cluster_path_root, class_folder),
                                    str(cnt_class_id))
        os.makedirs(cluster_path, exist_ok=True)

        km = KMeans(n_clusters=self.K, random_state=0).fit(fea_vec)

        print("finish Kmeans")
        # create cluster folders
        for each in range(self.K):
            os.makedirs(os.path.join(os.path.join(self.cluster_path_root, class_folder),
                                    str(each)), exist_ok=True)
            # os.makedirs(os.path.join(self.cluster_path_root, str(each)))

        for cnt in range(km.labels_.shape[0]):
            ddd = img_path_list[cnt]
            pred_K = km.labels_[cnt]
            cluster_path = os.path.join(os.path.join(self.cluster_path_root, class_folder),
                                    str(pred_K))
            file_name = ddd.split('/')
            if 'cv' in socket.gethostname():
                ddd = ddd.replace('/local/vondrick/cz/ImageNet/train', '/proj/vondrick/mcz/ImageNet-Data/train')
            shutil.copy(ddd.replace('/temp', ''), os.path.join(cluster_path, file_name[-1]))
            # Notice: This is server only, i.e., VGG features pickle are generated on the same server as here,
            # If not, more need to be done (replace with the correct path)



    @staticmethod
    def iterate_f(funct, filelist):
        for each in filelist:
            funct(each)



if __name__ == '__main__':
    # EC = ExamplarCluster(process_num=50, C=4, subset=True)
    EC = KmeansCluster(process_num=1, K=4, subset=False, test_mode=True)
    EC.multi_process_generate()



