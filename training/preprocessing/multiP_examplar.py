from utils import multiprocess
import os
import numpy as np
import pickle, random
import shutil
from sklearn import svm
from multiprocessing import Manager
import socket
from utils import fix_imagenet_temp, np_normalize_l2

# exist_list = ['n01978287-4', 'n02137549-4', 'n02835271-4', 'n03796401-4',
# 'n02094433-4', 'n02823750-4','n03733131-4','n04590129-4']
exist_list = []

def remove_generated(whole, existing):
    result = []
    for each in whole:
        if each not in existing:
            result.append(each)
    return result


class ExamplarCluster:
    def __init__(self, process_num, C=3.5, subset=False, test_mode=False, normalize=True, randomize=False,
                 negative_weight_rescale=1.0):
        self.process_num = process_num
        self.C = C
        self.subset = subset
        self.test_mode = test_mode
        self.normalize = normalize
        self.randomize = randomize
        self.negative_weight_rescale = negative_weight_rescale

        if socket.gethostname() == 'deep':
            self.path = '/mnt/md0/2020Spring/VGG_features/train'
            self.cluster_path_root = '/mnt/md0/2020Spring/invariant_imagenet/train_examplar'
            self.imagenet_path = '/mnt/md0/ImageNet/train'
        elif socket.gethostname() == 'hulk':
            # self.path = '/local/rcs/mcz/2020Spring/VGG_features/train' # THIS is old wrong croped vgg features
            self.path = '/local/rcs/mcz/2020Spring/invariant_imagenet/VGGfeatures/train' # new ones
            self.cluster_path_root = '/local/rcs/mcz/2020Spring/invariant_imagenet/train_examplar'
            self.imagenet_path = '/local/rcs/mcz/ImageNet-Data'
            if self.test_mode:
                self.path = '/local/rcs/mcz/2020Spring/invariant_imagenet/VGGfeatures/test'
                self.cluster_path_root = '/local/rcs/mcz/2020Spring/test_view/C+{}'.format(self.C)
                self.imagenet_path = '/local/rcs/mcz/ImageNet-Data/val'

        elif socket.gethostname() == 'cv04':
            self.path = '/local/vondrick/cz/ImageNet/VGGfeatures_NoNorm/train' # new ones
            # self.cluster_path_root = '/local/vondrick/cz/cut_img-S3.5-TC5/train_clustered-C{}'.format(self.C) # old
            self.cluster_path_root = '/local/vondrick/cz/train_whole_clustered-C{}'.format(self.C)
            # Now I want to split the whole set without removing overlap trains
            self.imagenet_path = '/local/vondrick/cz/ImageNet'
            if subset:
                self.path = '/local/vondrick/cz/cut_img-both3.5/VGGfeatures/train'
                self.cluster_path_root = '/local/vondrick/cz/cut_img_S3.5-C{}_traincluster'.format(self.C)
                self.imagenet_path = '/local/vondrick/cz/cut_img-both3.5'


        # fix_imagenet_temp(os.path.join(self.imagenet_path, 'train'))
        # fix_imagenet_temp(os.path.join(self.imagenet_path, 'val'))

        os.makedirs(self.cluster_path_root, exist_ok=True)
        class_list = os.listdir(self.path)

        self.class_list = sorted(remove_generated(class_list, exist_list))
        # print('class list', self.class_list)
        all_feature = []

        # manager = Manager()
        # self.random_fea_dict = manager.dict()
        self.random_fea_dict = {}
        self.fea_dict_name = []

        subsample_rate1 = 200
        subsample_num = 60
        self.subsample_rate2 = 40
        self.topk = 100
        # self.topk = 10

        # TODO: so for each SVM, 300 is the usual negative num for fast and good examplar

        self.relax = 1.05

        # Loading the whole Set
        for jj, each in enumerate(sorted(os.listdir(self.path))):
            with open(os.path.join(self.path, each), 'rb') as f:
                fea = pickle.load(f)

            choice = [i for i in range(fea['fea'].shape[0])]
            if subsample_num < len(choice):
                choice = random.sample(choice, subsample_num)
            temp = fea['fea'][choice, :]

            if self.normalize:
                temp = np_normalize_l2(temp)

            print(jj, "temp", temp.shape)
            # all_feature.append(temp)
            # print(each)
            self.random_fea_dict[each] = temp
            # self.fea_dict_name.append(each)
            # if jj > 150:
            #     break

    def multi_process_generate(self):

        L = len(self.class_list)
        step = (L // self.process_num) + 1
        args = [(self.run_examplar, self.class_list[i * step:min((i + 1) * step, L)], self.random_fea_dict, i) for i in
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

    def run_examplar(self, filename, random_fea_dict, process_id):
        cnt_class_id = 0
        C = self.C

        fea_vec, img_path_list = self.load_single(self.path, filename)

        mean_fea_vec = np.mean(fea_vec, axis=0)

        if self.normalize:
            fea_vec = np_normalize_l2(fea_vec)

        # Randomization the loaded category to be examplared, because we use greedy method
        # if self.randomize:
        #     permutate_index = np.random.permutation([i for i in range(fea_vec.shape[0])])
        #     fea_vec = fea_vec[permutate_index]
        #     img_path_list_new = [img_path_list[permutate_index[jj]] for jj in range(fea_vec.shape[0])]
        #     img_path_list = img_path_list_new

        # with open(os.path.join(self.path, filename), 'rb') as f:
        #     fea = pickle.load(f)

        # list_select = [random.randint(0, len(random_fea_dict.keys())) for _ in range(self.topk)]
        with open('cluster_nneighbor.pkl', 'rb') as f:
            nn = pickle.load(f)
        top_k_index = nn['cluster_topk_index'][filename]

        #retrieve the nearest cluster
        list_select = []
        for ii in range(min(top_k_index.shape[0], self.topk)):
            list_select.append(nn['name_list'][top_k_index[ii]])
        print('select', list_select)

        neg_matrix = []
        for jjj, key in enumerate(random_fea_dict.keys()):
            if key != filename:
                if jjj in list_select or key in list_select:  # The 2nd one is for cluster nearest neighbor version
                    neg_matrix.append(random_fea_dict[key])

        neg_matrix = np.concatenate(neg_matrix)
        # print("negative sample size", neg_matrix.shape[0])

        # Greedily get examplar
        num = fea_vec.shape[0]
        label = np.zeros((num, 1))

        path_name = img_path_list[0]
        path_split = path_name.split('/')
        class_folder = path_split[-3]
        print('class', class_folder)


        while fea_vec.shape[0] != 0:
            # Creating SVM training set
            positive = fea_vec[0]
            positive = np.expand_dims(positive, axis=0)

            neg_len = neg_matrix.shape[0]
            per_ind = np.random.permutation(neg_len)
            keep_num = neg_len // self.subsample_rate2
            # ind_keep = per_ind[:keep_num]

            # retrieve the nearest neighbor in the given subset
            # Do negative sampling here:
            # print('pos shpae', positive.shape, neg_matrix.shape)
            similarity = np.dot(positive, neg_matrix.T)
            # print('sim shape', similarity.shape)
            top_k_close_neg = similarity[0].argsort()[-keep_num:][::-1]
            # print('top k', top_k_close_neg)
            # print(neg_matrix[top_k_close_neg, :].shape)

            all = np.concatenate((positive, neg_matrix[top_k_close_neg, :]), axis=0)
            svm_temp_label = np.zeros((all.shape[0],))
            svm_temp_label[0] = 1

            # Build SVM
            # balanced_dict = {1: 0.5, 0: 0.01}
            Loop = True
            current = 1 / self.negative_weight_rescale
            pred_neg_sum = 0
            lcnt = 0
            add_temp_C = 0

            while Loop:
                lcnt += 1
                if pred_neg_sum > 0:
                    print("current", current)
                    if lcnt <= 3:
                        current = current / self.relax
                    else:
                        print('add C', add_temp_C)

                sample_weights = np.ones(((all.shape[0],))) / all.shape[0] / current
                sample_weights[0] = 1.0
                sample_weights = sample_weights
                clf = svm.SVC(C=C + add_temp_C, kernel='linear')
                # print('start training examplar, split remaining train', filename, 'process id', process_id)
                clf.fit(all, svm_temp_label, sample_weights)

                pred_neg = clf.predict(neg_matrix[top_k_close_neg, :]) ###
                # print('pred neg', np.sum(pred_neg))
                pred_neg_sum = np.sum(pred_neg)

                pred = clf.predict(fea_vec)

                if pred_neg_sum == 0:
                    break
                else:
                    if lcnt > 3:
                        add_temp_C += 0.5
                    if lcnt > 6:
                        pred = np.zeros_like(pred)
                        pred[0] = 1
                        print("WARNING: THERE's still misclassified negative examples, which means SVM is not reliable.")
                        # TODO: here, rather no neighbor for the examplar than mistakenly predict no examplar to be examplar
                        break

            if pred_neg_sum > 0:
                print(pred_neg_sum, 'flase negative exists, pred: number of examples close to that examplar')


            # Loop=True
            # counting = 0
            # list_ratio = [1, 2, 3, 4, 5]
            # current = 1
            # pred_neg_sum=0
            # escape = False
            # while Loop and counting<1:
            #     if pred_neg_sum >0:
            #         print("current", current)
            #         escape = True
            #         # current = (current+list_ratio[counting-1])/2
            #         current = current / 2.0
            #     else:
            #         current = list_ratio[counting]
            #     sample_weights = np.ones(((all.shape[0],))) / all.shape[0] / current #all.shape[0]
            #     sample_weights[0] = 1.0
            #     sample_weights = sample_weights*0.5
            #     clf = svm.SVC(C=C, kernel='linear') #, class_weight=balanced_dict
            #     # Seems need me to write manually
            #
            #     print('start training examplar', filename, 'process id', process_id)
            #     clf.fit(all, svm_temp_label, sample_weights)
            #     pred_neg = clf.predict(neg_matrix)
            #     print('pred neg', np.sum(pred_neg))
            #     pred_neg_sum = np.sum(pred_neg)
            #
            #     pred = clf.predict(fea_vec)
            #     print('pred: number of examples close to that examplar', np.sum(pred))
            #     if pred_neg_sum >0:
            #         continue
            #     if pred_neg_sum and escape:  # If already failed (misclassify neg examples) before, then no need to increase count
            #         break
            #     if np.sum(pred)<2:
            #         counting += 1
            #         continue
            #     else:
            #         break

            # print(clf.decision_function())
            # not_used = label ==0
            # print(not_used.shape, fea_vec.shape)
            # remaining = fea_vec[not_used[:, 0], :]
            # print(remaining.shape)


            # keep_index = []
            # dump_img_index = [img_path_list[0]]
            # keep_img_index = []
            # for iii in range(1, fea_vec.shape[0]):
            #     if pred[iii] == 1:
            #         dump_img_index.append(img_path_list[iii])
            #     else:
            #         keep_index.append(iii)
            #         keep_img_index.append(img_path_list[iii])
            #
            # cluster_path = os.path.join(os.path.join(self.cluster_path_root, class_folder+'-'+str(C)), str(cnt_class_id))
            # os.makedirs(cluster_path, exist_ok=True)

            keep_index = []
            dump_img_index = [img_path_list[0]]
            keep_img_index = []

            for iii in range(1, fea_vec.shape[0]):
                if pred[iii] == 1:
                    dump_img_index.append(img_path_list[iii])
                else:
                    keep_index.append(iii)
                    keep_img_index.append(img_path_list[iii])

            cluster_path = os.path.join(os.path.join(self.cluster_path_root, class_folder),
                                        str(cnt_class_id))
            os.makedirs(cluster_path, exist_ok=True)

            if len(dump_img_index) > 1:
                print(dump_img_index)

            for cnt, ddd in enumerate(dump_img_index):
                file_name = ddd.split('/')
                if cnt == 0:
                    file_pure = file_name[-1].split('.')
                    file_name[-1] = '0a-query_' + file_pure[0] + '.' + file_pure[1]
                if socket.gethostname() == 'cv04' and not self.subset:
                    ddd = ddd.replace('/local/rcs/mcz/ImageNet-Data/train', '/local/vondrick/cz/ImageNet/train')
                shutil.copy(ddd.replace('/temp', ''), os.path.join(cluster_path, file_name[-1]))
                # Notice: This is server only, i.e., VGG features pickle are generated on the same server as here,
                # If not, more need to be done (replace with the correct path)

            fea_vec = fea_vec[keep_index, :]
            img_path_list = keep_img_index

            cnt_class_id += 1



    @staticmethod
    def iterate_f(funct, filelist, fea_dict, process_id):
        for each in filelist:
            funct(each, fea_dict, process_id)



if __name__ == '__main__':
    # EC = ExamplarCluster(process_num=50, C=4, subset=True)
    EC = ExamplarCluster(process_num=1, C=4, subset=False, test_mode=True)
    EC.multi_process_generate()



