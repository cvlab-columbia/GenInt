"""This file removes the example in the training set if the examplar exists in the testset"""
'''I tried C=3, and the training has fewer examples, though well clustered, not separated individually as C=5, but C=5 remove fewer training examples'''
from utils import multiprocess
import os
import numpy as np
import pickle, random
import shutil
from sklearn import svm
from multiprocessing import Manager
import socket

# exist_list = ['n01978287-4', 'n02137549-4', 'n02835271-4', 'n03796401-4',
# 'n02094433-4', 'n02823750-4','n03733131-4','n04590129-4']
exist_list = ['']


def remove_generated(whole, existing):
    result = []
    for each in whole:
        if each not in existing:
            result.append(each)
    return result


class TrainsetRemove:
    def __init__(self, process_num, C=3.5, cluster_training_selected=True):
        self.C = C
        self.cluster_training_selected = cluster_training_selected


        self.process_num = process_num
        if socket.gethostname() == 'deep':
            self.path_train = '/mnt/md0/2020Spring/VGG_features/train'
            self.path_test = '/mnt/md0/2020Spring/VGG_features/test'
            self.target_train = '/mnt/md0/2020Spring/subcut_imagnet/train'
        elif socket.gethostname() == 'hulk':
            self.path_train = '/local/rcs/mcz/2020Spring/invariant_imagenet/VGGfeatures/train'
            self.path_test = '/local/rcs/mcz/2020Spring/invariant_imagenet/VGGfeatures/test'
            self.cluster_path_root = '/local/rcs/mcz/2020Spring/invariant_imagenet/train_examplar'
            self.target_train = '/local/rcs/mcz/2020Spring/cut_imgnet-{}/train'.format(self.C)
            self.target_train_cluster = '/local/rcs/mcz/2020Spring/cut_imgnet-{}/train_clustered'.format(self.C)
            self.removed_train = '/local/rcs/mcz/2020Spring/cut_imgnet-{}/train-test-removed'.format(self.C)
            self.source_imagenet = '/local/rcs/mcz/ImageNet-Data/train'

        os.makedirs(self.cluster_path_root, exist_ok=True)
        os.makedirs(self.target_train, exist_ok=True)

        class_list = os.listdir(self.path_train)

        self.class_list = sorted(class_list)
        # print('class list', self.class_list)
        all_feature = []

        self.random_fea_dict = {}

        subsample_rate1 = 100
        self.subsample_rate2 = 4
        self.topk = 100
        # self.topk = 10

        self.relax = 1.1
        self.c_gap = 0


        # Loading the whole Training Support Set
        for jj, each in enumerate(os.listdir(self.path_train)):
            with open(os.path.join(self.path_train, each), 'rb') as f:
                fea = pickle.load(f)

            temp = fea['fea'][::subsample_rate1, :]
            print(jj, "temp", temp.shape)
            # all_feature.append(temp)
            self.random_fea_dict[each] = temp
            # if jj > 100:
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
        C = self.C

        fea_train, img_path_train = self.load_single(self.path_train, filename)
        fea_test, img_path_test = self.load_single(self.path_test, filename)


        list_select = [random.randint(0, len(random_fea_dict.keys())) for _ in range(self.topk)]
        # list_select = [i for i in range(100)]
        print('select', list_select)

        '''Create Random Negtive'''
        neg_matrix = []
        for jjj, key in enumerate(random_fea_dict.keys()):
            if key != filename:
                if jjj in list_select:
                    neg_matrix.append(random_fea_dict[key])  # May not good  [::self.subsample_rate2]

        neg_matrix = np.concatenate(neg_matrix)

        # Greedily get examplar
        path_name = img_path_train[0]
        path_split = path_name.split('/')
        class_folder = path_split[-3]
        print('class', class_folder)

        # enumerate over all test set:
        # remove the selected training set examples, in order to create view point generation gap.
        # see how many training examples left? Then examplar grouping the training examples again.
        #
        # possible update for examplar: 1. downsample the fea vector for larger cluster
        #

        # while fea_test.shape[0] != 0:
        for cnt in range(fea_test.shape[0]):

            # print("remaining train sample", fea_train.shape[0])
            # Creating SVM training set
            positive = fea_test[cnt]
            positive = np.expand_dims(positive, axis=0)

            # Randomized the negative set (Due to speed, The SVM handles at most 400 examples)
            neg_len = neg_matrix.shape[0]
            per_ind = np.random.permutation(neg_len)
            keep_num = neg_len // self.subsample_rate2
            ind_keep = per_ind[:keep_num]

            all = np.concatenate((positive, neg_matrix[ind_keep, :]), axis=0)
            # print("negative sample size", all.shape[0]-1)
            assert all.shape[0] < 450
            # Negative cannot be large, it will be too slow

            svm_temp_label = np.zeros((all.shape[0],))
            svm_temp_label[0] = 1

            # Build SVM
            # balanced_dict = {1: 0.5, 0: 0.01}
            current = 1
            pred_neg_sum=0
            lcnt=0
            add_temp_C = 0
            while True:
                lcnt += 1
                if pred_neg_sum >0:
                    print("current", current)
                    if lcnt<=3:
                        current = current / self.relax  # Increase the negative example weights if any neg is misclassified

                sample_weights = np.ones(((all.shape[0],))) / all.shape[0] / current #all.shape[0]
                sample_weights[0] = 1.0
                sample_weights = sample_weights*0.5
                clf = svm.SVC(C=C + add_temp_C, kernel='linear') #, class_weight=balanced_dict
                # Seems need me to write manually

                # print('start training examplar', filename, 'process id', process_id)
                clf.fit(all, svm_temp_label, sample_weights)
                pred_neg = clf.predict(neg_matrix[ind_keep, :])
                # print('removing', filename, 'process id', process_id, 'pred neg', np.sum(pred_neg))
                pred_neg_sum = np.sum(pred_neg)

                # debug_pred = clf.predict(all)
                # print('debug sum=', np.sum(debug_pred))

                pred = clf.predict(fea_train)
                # print('removing', filename, 'process id', process_id, 'pred neg', np.sum(pred_neg), 'pred positive', np.sum(pred))
                if pred_neg_sum == 0:
                    break
                else:
                    if lcnt>3:
                        add_temp_C += 0.5
                    if lcnt>5:
                        break

            # We should remove the selected ones instantly after getting them to speed up a lot, and tracking the remaining
            # Training set Size

            if pred_neg_sum >0:
                print('removing with minor error', filename, 'process id', process_id, 'pred neg ', pred_neg_sum,
                      'positive ', np.sum(pred))
            else:
                print('retrive train', np.sum(pred))


            keep_index = []
            dump_img_index = [img_path_test[cnt]] #TODO: ! for visualization, test is included
            keep_img_index = []
            for iii in range(fea_train.shape[0]):
                if pred[iii] == 0:
                    keep_index.append(iii)
                    keep_img_index.append(img_path_train[iii])
                else:
                    dump_img_index.append(img_path_train[iii])

            # Cache the deleted train images
            cluster_path = os.path.join(os.path.join(self.removed_train, class_folder + '-' + str(C)), str(cnt))
            os.makedirs(cluster_path, exist_ok=True)
            for cnt, ddd in enumerate(dump_img_index):
                file_name = ddd.split('/')
                if cnt == 0:
                    file_pure = file_name[-1].split('.')
                    file_name[-1] = '0a-query_' + file_pure[0]+'.' + file_pure[1]
                shutil.copy(ddd.replace('/temp', ''), os.path.join(cluster_path, file_name[-1]))

            fea_train = fea_train[keep_index, :]
            img_path_train = keep_img_index


        # All Remaining Train Data together as Baseline
        for kkk, fname in enumerate(img_path_train):
            file_name = fname.split('/')
            os.makedirs(os.path.join(self.target_train, file_name[-3]), exist_ok=True)
            source = os.path.join(os.path.join(self.source_imagenet, file_name[-3]), file_name[-1])

            target = os.path.join(os.path.join(self.target_train, file_name[-3]), file_name[-1])
            # print(source, 'to', target)
            shutil.copy(source, target)

        # Directly examplar cluster the remaining training set

        if self.cluster_training_selected:
            print("Starting clustering the remain train set")
            print(len(img_path_train), "training examples remained to be classified")
            cnt_class_id = 0

            while fea_train.shape[0] != 0:
                positive = fea_train[0]
                positive = np.expand_dims(positive, axis=0)

                neg_len = neg_matrix.shape[0]
                per_ind = np.random.permutation(neg_len)
                keep_num = neg_len // self.subsample_rate2
                ind_keep = per_ind[:keep_num]

                all = np.concatenate((positive, neg_matrix[ind_keep, :]), axis=0)
                svm_temp_label = np.zeros((all.shape[0],))
                svm_temp_label[0] = 1

                assert all.shape[0] < 400
                # Negative cannot be large, it will be too slow

                Loop = True
                current = 1
                pred_neg_sum = 0
                lcnt = 0
                add_temp_C = 0
                while Loop:
                    lcnt += 1
                    if pred_neg_sum >0:
                        print("current", current)
                        if lcnt<=3:
                            current = current / self.relax
                        else:
                            print('add C', add_temp_C)

                    sample_weights = np.ones(((all.shape[0],))) / all.shape[0] / current
                    sample_weights[0] = 1.0
                    sample_weights = sample_weights * 0.5
                    clf = svm.SVC(C=C + add_temp_C - self.c_gap, kernel='linear')
                    # print('start training examplar, split remaining train', filename, 'process id', process_id)
                    clf.fit(all, svm_temp_label, sample_weights)

                    pred_neg = clf.predict(neg_matrix[ind_keep, :])
                    # print('pred neg', np.sum(pred_neg))
                    pred_neg_sum = np.sum(pred_neg)

                    pred = clf.predict(fea_train)

                    if pred_neg_sum == 0:
                        break
                    else:
                        if lcnt>3:
                            add_temp_C += 0.5
                        if lcnt>6:
                            pred = np.zeros_like(pred)
                            pred[0] = 1
                            # TODO: here, rather no neighbor for the examplar than mistakenly predict wrong examplar to be examplar
                            break

                if pred_neg_sum > 0:
                    print(pred_neg_sum, 'flase negative exists, pred: number of examples close to that examplar', np.sum(pred))

                keep_index = []
                dump_img_index = [img_path_train[0]]
                keep_img_index = []

                for iii in range(1, fea_train.shape[0]):
                    if pred[iii] == 1:
                        dump_img_index.append(img_path_train[iii])
                    else:
                        keep_index.append(iii)
                        keep_img_index.append(img_path_train[iii])

                cluster_path = os.path.join(os.path.join(self.target_train_cluster + '-' + str(C), class_folder), str(cnt_class_id))
                os.makedirs(cluster_path, exist_ok=True)

                if len(dump_img_index)>1:
                    print(dump_img_index)

                for cnt, ddd in enumerate(dump_img_index):
                    file_name = ddd.split('/')
                    if cnt == 0:
                        file_pure = file_name[-1].split('.')
                        file_name[-1] = '0a-query_' + file_pure[0]+'.' + file_pure[1]
                    shutil.copy(ddd.replace('/temp', ''), os.path.join(cluster_path, file_name[-1]))
                    # Notice: This is server only, i.e., VGG features pickle are generated on the same server as here,
                    # If not, more need to be done (replace with the correct path)

                fea_train = fea_train[keep_index, :]
                img_path_train = keep_img_index

                cnt_class_id += 1

            # remove_index = []
            # for iii in range(1, fea_train.shape[0]):
            #     if pred[iii] == 1:
            #         remove_index.append(img_path_train[iii])
                # else:
                #     keep_index.append(iii)
                #     keep_img_index.append(img_path_train[iii])

            # cluster_path = os.path.join(os.path.join(self.cluster_path_root, class_folder+'-'+str(C)), str(cnt_class_id))

            # os.makedirs(cluster_path, exist_ok=True)
            #
            # if len(dump_img_index)>1:
            #     print(dump_img_index)
            # for ddd in dump_img_index:
            #     file_name = ddd.split('/')
            #
            #     if socket.gethostname() == 'hulk':
            #         ddd = ddd.replace('/mnt/md0/ImageNet/train', '/local/rcs/mcz/ImageNet-Data/train')
            #     source = ddd.replace('/temp', '')
            #     shutil.copy(source, os.path.join(cluster_path, file_name[-1]))
            #
            # fea_vec = fea_vec[keep_index, :]
            # img_path_list = keep_img_index



    def iterate_f(self, funct, filelist, fea_dict, process_id):
        for each in filelist:
            funct(each, fea_dict, process_id)



if __name__ == '__main__':
    EC = TrainsetRemove(process_num=48)
    EC.multi_process_generate()



