import os
import sys
import glob
import json
import socket
from shutil import copyfile
import tqdm
import shutil

# source_path="/local/rcs/shared/objectnet-1.0/images"
source_path="/proj/vondrick/augustine/objectnet-1.0/images"
target_path="/proj/vondrick/augustine/objectnet-1.0/overlap_category_test"

def create_obj_img(path_json):
    '''
    Copy the objectnet's overlapping categories wtih imagenet to new folder
    :param path_json:
    :return:
    '''
    with open(path_json) as f:
        dict_str_objnet2imagenet = json.load(f)

        cnt=0
        for str_objectnet_category, str_imagenet_categories in dict_str_objnet2imagenet.items():
            key_objectnet_category = str_objectnet_category.lower().replace(" ", "_")
            key_objectnet_category = key_objectnet_category.replace("(", "").replace(")", "").replace("'", "").replace("/", "_")

            key_objectnet_category = key_objectnet_category.replace("___","_")
            key_objectnet_category = key_objectnet_category.replace("__", "_")

            print("copy", cnt, key_objectnet_category)
            src = os.path.join(source_path, key_objectnet_category)
            tar = os.path.join(target_path, key_objectnet_category)
            shutil.copytree(src, tar)
            cnt += 1

def gen_obj_imgnet_index(path_json, imagenet_path="/local/rcs/mcz/ImageNet-Data/train", save=True):

    """generate the sorted index from imagenet and append to the corresponding objectnet class"""
    filelist = sorted(os.listdir(imagenet_path))
    name2int_dict = {}

    imgnet_mapping = getDictImageNetClasses()

    for i, each in enumerate(filelist):
        name2int_dict[imgnet_mapping[each]] = i

    dict_objectnet2imagenet = {}
    with open(path_json) as f:
        dict_str_objnet2imagenet = json.load(f)

        cnt=0
        for str_objectnet_category, str_imagenet_categories in dict_str_objnet2imagenet.items():
            key_objectnet_category = str_objectnet_category.lower().replace(" ", "_")
            key_objectnet_category = key_objectnet_category.replace("(", "").replace(")", "").replace("'", "").replace("/", "_")

            key_objectnet_category = key_objectnet_category.replace("___","_")
            key_objectnet_category = key_objectnet_category.replace("__", "_")

            str_categories = str_imagenet_categories
            str_categories = str_categories.replace(";", ",")
            list_str_cat = str_categories.split(",")
            list_str_cat = [c.strip().replace(" ", "_") for c in list_str_cat]

            label_num_list = []
            for each in list_str_cat:
                try:
                    label_num_list.append(name2int_dict[each.lower()])
                except:
                    # print("not find", each)
                    pass

            if len(label_num_list)==0:
                print(key_objectnet_category, "nothing find!")
            dict_objectnet2imagenet[key_objectnet_category] = label_num_list

    if save:
        with open('obj2imgnet_id.txt', 'w') as outfile:
            json.dump(dict_objectnet2imagenet, outfile)

    return dict_objectnet2imagenet


def getDictImageNetClasses(path_imagenet_classes_name='imagenet_classes_names.txt'):
    '''
    Returns dictionary of classname --> classid. Eg - {n02119789: 'kit_fox'}
    '''

    count = 0
    dict_imagenet_classname2id = {}
    with open(path_imagenet_classes_name) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            if cat_name in dict_imagenet_classname2id.keys():
                print(cat_name)
            dict_imagenet_classname2id[id] = cat_name.lower()
            count += 1
            print(cat_name, id)
            line = f.readline()
    print("Total categories categories", count)
    return dict_imagenet_classname2id

path_objectnet_to_imagenet_json = "objectnet_to_imagenet_1k.json"
"""Create Objectnet testing images overlap with iamgenet"""
create_obj_img(path_objectnet_to_imagenet_json)

# gen_obj_imgnet_index(path_objectnet_to_imagenet_json)




