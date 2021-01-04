'''Maybe need to generate equally num of train for train of rare and rich viewpoints'''
import os
from random import randint
from math import pi
import random
from utils import is_range

import socket


if socket.gethostname() == 'deep':
    root_path = '/mnt/md0/2020Spring/ShapeNet_output'
    source_path = '{}/v3'.format(root_path)
elif socket.gethostname() == 'cv04':
    root_path = '/proj/vondrick/mcz/ShapeNetRender'
    source_path = '{}/v3'.format(root_path)


split_to_path_train = '{}/splited/train'.format(root_path)
split_to_path_test = '{}/splited/test'.format(root_path)
split_to_path_test_diff_obj = '{}/splited/test_diff_obj'.format(root_path)
split_to_path_test_rand_rich = '{}/splited/test_richview'.format(root_path)
split_to_path_test_rand_rare = '{}/splited/test_rareview'.format(root_path)
split_to_path_test_newview = '{}/splited/test_newview'.format(root_path)

category_list = sorted(os.listdir(source_path))

FLAG_test_sub_obj = False  # Whether use obj subfolder

cat_list = os.listdir(source_path)

test_ratio = 0.1
test_whole_object_ratio = 0.2
# Consturct list of copying criteria
test_moving_list = []
test_diff_obj_moving_list = []
test_moving_list_rich = []
test_moving_list_rare = []


train_whole_view_list = []
train_partial_view_list = []
test_newview_list = []

partial_cat_ratio = 0.7  # 70% of the category is with limited viewpoints
partial_cat = random.sample(cat_list, int(len(cat_list) * partial_cat_ratio))

rich_cat = list(set(cat_list) - set(partial_cat))
import pickle
with open("{}/splited/rare_class.pkl", 'wb') as f:
    pickle.dump(partial_cat, f)
with open("{}/splited/rich_class.pkl", 'wb') as f:
    pickle.dump(rich_cat, f)

for each in cat_list:
    cat_path = os.path.join(source_path, each)

    os.makedirs(os.path.join(split_to_path_train, each))
    os.makedirs(os.path.join(split_to_path_test, each))
    os.makedirs(os.path.join(split_to_path_test_diff_obj, each))

    os.makedirs(os.path.join(split_to_path_test_rand_rich, each))
    os.makedirs(os.path.join(split_to_path_test_rand_rare, each))

    os.makedirs(os.path.join(split_to_path_test_newview, each))

    model_list=os.listdir(cat_path)

    model_test_list = random.sample(model_list, int(len(model_list) * test_whole_object_ratio))

    # The held out 3D obj from category
    for obj in model_test_list:
        obj_path = os.path.join(cat_path, obj)
        viewpoint_list = os.listdir(obj_path)

        for ea in viewpoint_list:
            if FLAG_test_sub_obj:
                tp = (os.path.join(obj_path, ea), os.path.join(os.path.join(os.path.join(split_to_path_test_diff_obj, each), obj), ea))
            else:
                tp = (os.path.join(obj_path, ea),
                      os.path.join(os.path.join(split_to_path_test_diff_obj, each), ea))
            test_diff_obj_moving_list.append(tp)

    # The training 3D obj from category
    model_list = list(set(model_list) - set(model_test_list))

    for obj in model_list:
        obj_path = os.path.join(cat_path, obj)
        viewpoint_list = os.listdir(obj_path)
        viewpoint_list.sort()
        total_num = len(viewpoint_list)
        test_ones = random.sample(viewpoint_list[1:], int(total_num * test_ratio))
        # Keep the first 0,0,0 view point in train, we need this as reference

        # Split those test:
        for ii in range(len(test_ones)):
            if FLAG_test_sub_obj:
                tp = (os.path.join(obj_path, test_ones[ii]), os.path.join(os.path.join(os.path.join(split_to_path_test, each), obj), test_ones[ii]))
            else:
                tp = (os.path.join(obj_path, test_ones[ii]),
                      os.path.join(os.path.join(split_to_path_test, each), test_ones[ii]))

            test_moving_list.append(tp)

            if each in partial_cat:  # Rare view point class
                if FLAG_test_sub_obj:
                    tp = (
                    os.path.join(obj_path, test_ones[ii]), os.path.join(os.path.join(os.path.join(split_to_path_test_rand_rare, each), obj), test_ones[ii]))
                    test_moving_list_rich.append(tp)
                else:
                    tp = (
                        os.path.join(obj_path, test_ones[ii]),
                        os.path.join(os.path.join(split_to_path_test_rand_rare, each),
                                     test_ones[ii]))
                    test_moving_list_rich.append(tp)

            else:                    # Rich view point class
                if FLAG_test_sub_obj:
                    tp = (
                        os.path.join(obj_path, test_ones[ii]),
                        os.path.join(os.path.join(os.path.join(split_to_path_test_rand_rich, each), obj), test_ones[ii]))
                else:
                    tp = (
                        os.path.join(obj_path, test_ones[ii]),
                        os.path.join(os.path.join(split_to_path_test_rand_rich, each),
                                     test_ones[ii]))
                test_moving_list_rare.append(tp)


        viewpoint_list = list(set(viewpoint_list) - set(test_ones))

        if each in partial_cat:  # THE new viewpoint class
            for jj in viewpoint_list:
                view_str = jj.split('_')
                c = int(view_str[-6])
                x = int(view_str[-4])
                y = int(view_str[-2])
                z = int(view_str[-1].split('.')[0][1:])

                # if is_range(x, y, z, [0, 90]):
                if c <=2:  # Training category
                    tp = (os.path.join(obj_path, jj), os.path.join(os.path.join(os.path.join(split_to_path_train, each), obj),jj))
                    train_partial_view_list.append(tp)
                else:      # testing category
                    if FLAG_test_sub_obj:
                        tp = (os.path.join(obj_path, jj), os.path.join(os.path.join(os.path.join(split_to_path_test_newview, each), obj),jj))
                    else:
                        tp = (os.path.join(obj_path, jj),
                              os.path.join(os.path.join(split_to_path_test_newview, each), jj))
                    test_newview_list.append(tp)

        else:  # For Training
            for jj in viewpoint_list:
                tp = (os.path.join(obj_path, jj), os.path.join(os.path.join(os.path.join(split_to_path_train, each), obj), jj))
                train_whole_view_list.append(tp)

from utils import move_list
# move_list(test_moving_list)
move_list(test_diff_obj_moving_list)
move_list(test_moving_list_rich)
move_list(test_moving_list_rare)

move_list(train_whole_view_list)
move_list(train_partial_view_list)
move_list(test_newview_list)


