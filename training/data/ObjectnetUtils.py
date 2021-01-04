'''
RUN FROM THIS FOLDER
NOTE: Save and copyfile may be commented out, uncomment those.

Class to do following things on ObjectNet:
1. Remove border and save in a new folder
2.

TODO:
1. Add argparse -- add an option to print out info(life verbose - which categories overlapping etc.)
2. Make more independent/modular functions
3. Have a single script to setup objectnet and print details. See other dataset scripts like Cityscapes to learn.
4. handle if paths are passed or not, pass in the default ones.
'''

import os
import sys
import glob
import json
import numpy as np
import socket
from shutil import copyfile
import tqdm
import shutil
from PIL import Image, ImageOps


class ObjectnetUtils:
    '''
    Class for ObjectNet
    '''

    def __init__(self, path_objnet, path_imagenet_train):

        self.path_objnet = path_objnet
        self.path_imagenet_train= path_imagenet_train

        self.path_objnet_images = os.path.join(self.path_objnet, 'images')
        # print(self.path_objnet)

    def removeBorder(self, path_source_objectnet_folders, path_destination, list_categories=[]):
        '''
        Removes border for objectnet images and saves at path_destination.
        Only processes list_categories if not empty
        '''
        path_objnet = path_source_objectnet_folders
        
        categories = os.listdir(path_objnet)

        if len(list_categories)>0:
            categories = list_categories
        
        for category in tqdm.tqdm(categories):

            path_category_folder = os.path.join(path_objnet, category)
            list_path_images_in_category = glob.glob(path_category_folder + "/*")

            # Create category destination folder
            path_category_destination_folder = os.path.join(path_destination, category)
            if not os.path.exists(path_category_destination_folder):
                os.makedirs(path_category_destination_folder)

            for path_im in list_path_images_in_category:

                # Open and crop image
                im = Image.open(path_im)
                cropped_border = (2,2,2,2)
                cropped_im = ImageOps.crop(im, cropped_border)

                # Set the destination path and save the image
                im_basename = os.path.basename(path_im)
                path_im_dest = os.path.join(path_category_destination_folder, im_basename)
                #print(path_im_dest, im.size,cropped_im.size)
                cropped_im.save(path_im_dest)

    def getDictImageNetClasses(self, path_imagenet_classes_name):
        '''
        Returns dictionary of classname --> classid. Eg - {'kit_fox': n02119789}
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
                dict_imagenet_classname2id[cat_name] = id
                count += 1
                print(cat_name, id)
                line = f.readline()
        print("Total categories categories", count)
        return dict_imagenet_classname2id

    def getDictImageNetID2Classname(self,path_imagenet_classes_name):
        '''
        Returns dictionary. Eg - {n02119789: 'kit_fox'}
        '''
        count = 0
        dict_imagenet_id2classname = {}
        with open(path_imagenet_classes_name) as f:
            line = f.readline()
            while line:
                split_name = line.strip().split()
                cat_name = split_name[2]
                id = split_name[0]
                if cat_name in dict_imagenet_id2classname.keys():
                    print(cat_name)
                dict_imagenet_id2classname[id] = cat_name.lower()
                count += 1
                print(cat_name, id)
                line = f.readline()
        print("Total categories categories", count)
        return dict_imagenet_id2classname

    def getCategoriesDictFromJson(self,path_json):
        '''
        Returns dict from the values in json file eg - objectnet_to_imagenet_1k.json
        output eg - {"tie": ["bow_tie, bow-tie, bowtie; Windsor_tie"], ...}
        '''

        with open(path_json) as f:
            dict_str_objnet2imagenet = json.load(f)
            dict_objectnet2imagenet = {}

            for str_objectnet_category, str_imagenet_categories in dict_str_objnet2imagenet.items():

                # Prepare the key
                # key_objectnet_category = str_objectnet_category.lower().replace(" ", "_")
                key_objectnet_category = str_objectnet_category

                # Prepare the values (list of imagenet like categories --> separate by ,; and split by " ")
                str_categories = str_imagenet_categories
                str_categories = str_categories.replace(";", ",")
                list_str_cat = str_categories.split(",")
                list_str_cat = [c.strip().replace(" ", "_") for c in list_str_cat]

                # Assign the list of imagenet categories to objectnet category key in dictionary dict_objectnet2imagenet
                dict_objectnet2imagenet[key_objectnet_category] = list_str_cat

        return dict_objectnet2imagenet

    def generateFoldersObjectNetToImagenet(self, dict_objectnet2imagenet,
                                           destination_folder_path,
                                           list_classes_imagenet,
                                           path_imagenet_train,
                                           dict_imagenet_classname2id):
        list_total_overlapping_categories = []
        count = 0
        for objnet_cat, list_imgnet_cat in tqdm.tqdm(dict_objectnet2imagenet.items()):
            # print(objnet_cat)
            # Define the path for objnet category and make the folder
            path_objnet_category = os.path.join(destination_folder_path, objnet_cat)
            if not os.path.exists(path_objnet_category):
                os.makedirs(path_objnet_category)

            # Check if the list of imagenet categories in dict objectnet2imagenet exists in ImageNet, collect in list_overlapping_categories
            set_overlapping_categories = set(list_imgnet_cat)
            list_overlapping_categories = []
            for imgnet_cat in set_overlapping_categories:
                # print(list_classes_imagenet)
                if imgnet_cat in dict_imagenet_classname2id.keys():
                    # print(imgnet_cat)
                    list_overlapping_categories.append(imgnet_cat)
            list_total_overlapping_categories.extend(list_overlapping_categories)
            print("Set of imagenet categories overlapping with {} objectnet category: ".format(objnet_cat), set_overlapping_categories, " and found in imagenet: ",list_overlapping_categories)
            if len(list_overlapping_categories) > 0:
                count += 1
            # Make folders and copy images for each category
            for imgnet_cat in list_overlapping_categories:
                imgnet_class_id = dict_imagenet_classname2id[imgnet_cat]
                imgnet_class_directory_path = os.path.join(path_imagenet_train, imgnet_class_id)
                list_class_image_paths = glob.glob(imgnet_class_directory_path + "/*")

                for im_path in list_class_image_paths:
                    im_basename = os.path.basename(im_path)
                    target_im_path = os.path.join(path_objnet_category, im_basename)
                    # print("Copying im_path to target path", im_path, target_im_path)
                    copyfile(im_path, target_im_path)

        print("Number of overlapping categories found: ", count, len(list_total_overlapping_categories),
              list_total_overlapping_categories)

    def copyObjectnetOverlappingClasses(self, path_json_objectnet_folder2imagenet_classes, path_destination_folder):
        '''
        Copies the imagenet overlapping folders in objectnet to a new directory.
        '''

        path_objectnet_images = self.path_objnet_images

        with open(path_json_objectnet_folder2imagenet_classes) as f:
            dict_objectnet_folder2imagenet_classes= json.load(f)

        list_objectnet_overlapping_folders = dict_objectnet_folder2imagenet_classes.keys()
        print("Number of overlapping classes: ", len(list_objectnet_overlapping_folders))

        # Create destination folder if it does not exist
        if not os.path.exists(path_destination_folder):
            os.makedirs(path_destination_folder)

        count_overlap = 0
        for objectnet_category_folder in tqdm.tqdm(list_objectnet_overlapping_folders):
            src = os.path.join(path_objectnet_images, objectnet_category_folder)
            dst= os.path.join(path_destination_folder, objectnet_category_folder)
            # shutil.copytree(src, dst)
            print("Copying {} to {}".format(src,dst))
            count_overlap += 1

        print("Copied {} folders in {}".format(count_overlap, path_destination_folder))

    def genObjImagenetIndex(self, path_objectnet_foldername_to_imagenet_json,
                            path_imagenet_classname,
                            path_preprocessing,
                            save=True):
        """generate the sorted index from imagenet and append to the corresponding objectnet class"""
        filelist = sorted(os.listdir(self.path_imagenet_train))
        name2int_dict = {}

        imgnet_mapping = self.getDictImageNetID2Classname(path_imagenet_classname)

        for i, each in enumerate(filelist):
            name2int_dict[imgnet_mapping[each]] = i

        dict_objectnet2imagenet = {}
        with open(path_objectnet_foldername_to_imagenet_json) as f:
            dict_str_objnet2imagenet = json.load(f)

            cnt = 0
            for str_objectnet_category, str_imagenet_categories in dict_str_objnet2imagenet.items():

                key_objectnet_category = str_objectnet_category

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

                if len(label_num_list) == 0:
                    print(key_objectnet_category, "nothing find!")
                dict_objectnet2imagenet[key_objectnet_category] = label_num_list

        if save:
            path_out = os.path.join(path_preprocessing, 'obj2imgnet_id.txt')
            with open(path_out, 'w') as outfile:
                json.dump(dict_objectnet2imagenet, outfile)

        return dict_objectnet2imagenet

def parse_args():

    parser = argparse.ArgumentParser(Description='Objectnet util script')
    subparser = parser.add_subparsers()

    border_parser = subparser.add_parser('border')
    border_parser.add_argument('--dst', type=str)

    overlap_parser = subparser.add_parser('overlap')
    overlap_parser.add_argument('--dst', type=str)


if __name__ == "__main__":

    if socket.gethostname() == 'hulk':
        traindir = '/local/rcs/mcz/ImageNet-Data/train'
        valdir = '/local/rcs/mcz/ImageNet-Data/val'
        obj_valdir = '/local/rcs/shared/objectnet-1.0/overlap_category_test'
        imageneta_valdir = ''
        
    elif 'cv' in socket.gethostname():
        path_imagenet_train = '/proj/vondrick/mcz/ImageNet-Data/train'
        path_imagenet_val = '/proj/vondrick/mcz/ImageNet-Data/val'
        path_objectnet = '/proj/vondrick/augustine/objectnet-1.0/'
        path_objectnet_full = '/proj/vondrick/augustine/objectnet-1.0/images'
        path_objectnet_overlap = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test' #
        path_objectnet_overlap_noborder = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_noborder'

    elif socket.gethostname() == 'amogh':
        traindir = '/local/vondrick/cz/cut_img-both3.5/train_clustered-C3.5'
        valdir = '/local/vondrick/cz/ImageNet/val'
        obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test'

    path_preprocessing = "../preprocessing/"
    path_imagenet_classes_name = "../preprocessing/imagenet_classes_names.txt"
    path_objectnet_foldername_to_imagenet_json = "../preprocessing/folder_to_imagenet_1k.json"

    # Make the class for ObjectnetUtils
    objectnet_utils = ObjectnetUtils(path_objectnet, path_imagenet_train)

    ############ GENERATE OBJ2IDX FILE
    objectnet_utils.genObjImagenetIndex(path_objectnet_foldername_to_imagenet_json,
                                        path_imagenet_classes_name,
                                        path_preprocessing,
                                        save=True)

    ############ REMOVE BORDER
    # print(path_objectnet_overlap)
    # # Remove border and save
    # objectnet_utils.removeBorder(path_objectnet_overlap)
    # objectnet_utils.removeBorder(path_objectnet_overlap, path_destination=path_objectnet_overlap_noborder, list_categories=['full_sized_towel'])

    ############ MAKE OVERLAPPING FOLDER FOR OBJECTNET CLASSES WITH IMAGENET IMAGES
    # Get dictionaries :
    # dict_imagenet_classname2id = objectnet_utils.getDictImageNetClasses(path_imagenet_classes_name)
    # dict_objectnet_to_imagenet = oqbjectnet_utils.getCategoriesDictFromJson(path_objectnet_foldername_to_imagenet_json)
    #
    # # list_classes_objectnet = os.listdir(path_objectnet_classes)
    # list_classes_imagenet = os.listdir(path_imagenet_train)
    #
    # # Rename and make new destination folder --> images_by_id
    # if not os.path.exists(destination_folder_path):
    #     os.makedirs(destination_folder_path)
    #
    # # Copy image files for each category.
    # objectnet_utils.generateFoldersObjectNetToImagenet(dict_objectnet_to_imagenet,
    #                                    destination_folder_path,
    #                                    list_classes_imagenet,
    #                                    path_imagenet_train,
    #                                    dict_imagenet_classname2id)

    ############ MAKE OVERLAPPING FOLDER FOR OBJECTNET CLASSES WHICH ARE OVERLAPPING
    # objectnet_utils.copyObjectnetOverlappingClasses(path_objectnet_foldername_to_imagenet_json,
    #                                                path_objectnet_overlap)
