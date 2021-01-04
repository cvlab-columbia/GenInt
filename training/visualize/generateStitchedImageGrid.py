'''
Usage: python script 'images/dot-*.png' 2 3
'''

import os, sys
import glob
import json
import tqdm
import random
import socket
from PIL import Image
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
# for test only
#sys.argv += ['images/dot-*.png', 2, 3]

# get arguments
# pattern = sys.argv[1]
# rows = int(sys.argv[2])
# cols = int(sys.argv[3])

def generateImageGridFromImageList(list_images, rows=10, cols=10, resize=(100,100), shuffle=False):

    num_images = rows*cols
    if num_images > len(list_images):
        print("Not enough images for this grid: ", len(list_images))
        rows = 6
        cols = 6
        num_images = rows*cols

    if shuffle:
        list_images = random.sample(list_images, len(list_images))
    filenames = list_images[:num_images]
    # load images and resize to (100, 100)
    images = [Image.open(name).resize(resize) for name in filenames]

    # create empty image to put thumbnails
    new_image = Image.new('RGB', (cols*100, rows*100))

    # put thumbnails
    i = 0
    for y in range(rows):
        if i >= len(images):
            break
        y *= 100
        for x in range(cols):
            x *= 100
            img = images[i]
            new_image.paste(img, (x, y, x+100, y+100))
            i += 1

    return new_image

def main():
    if socket.gethostname() == 'hulk':
        root_path = ""
    elif 'cv' in socket.gethostname():
        path_imagenet_train = '/proj/vondrick/mcz/ImageNet-Data/train'
        path_imagenet_val = '/proj/vondrick/mcz/ImageNet-Data/val'
        path_objectnet = '/proj/vondrick/augustine/objectnet-1.0/'
        path_objectnet_full = '/proj/vondrick/augustine/objectnet-1.0/images'
        path_objectnet_overlap_noborder = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_noborder'
        # path_results_output = '/proj/vondrick/www/amogh/data/image_grid/'
        path_results_output = '/proj/vondrick/www/amogh/data/image_grid'

    else:
        print("No paths defined")
        exit(0)

    rows = 6
    cols = 6
    resize = (100,100)
    # dataset = 'objectnet'
    dataset = 'imagenet'
    required_categories = ['banana','candle', 'bench']


    # Get overlapping class folders
    if dataset == 'objectnet':
        list_objnet_folders = glob.glob(path_objectnet_overlap_noborder + "/*")
        path_folder_dataset_results = os.path.join(path_results_output, "objectnet")
        if not os.path.exists(path_folder_dataset_results):
            os.makedirs(path_folder_dataset_results)
        for f in tqdm.tqdm(list_objnet_folders):
        
            list_images = glob.glob(f + "/*")
            category = os.path.basename(f)
            # if category in required_categories:
            #     print(list_images)
            #     continue
            # else:
            #     continue
            try:
                stitched_grid_img = generateImageGridFromImageList(list_images,
                                                                rows=rows,
                                                                cols=cols,
                                                                resize=(resize))

                path_img_out = os.path.join(path_folder_dataset_results, "{}.jpg".format(category))
                print("Saving at ", path_img_out)
                stitched_grid_img.save(path_img_out)
            except:
                print("Couldnt save for category", category, f)


    # Generate for Imagenet
    if dataset == 'imagenet':

        # Get the list of images in overlapping categories from imagenet
        dict_folder2imagenetid = getDictObjFolder2ImgnetID()

        list_objnet_folders = glob.glob(path_objectnet_overlap_noborder + "/*")
        categories = os.listdir(path_objectnet_overlap_noborder)
        path_folder_dataset_results = os.path.join(path_results_output, "imagenet")
        if not os.path.exists(path_folder_dataset_results):
            os.makedirs(path_folder_dataset_results)
        for category in tqdm.tqdm(categories):

            list_images = []

            list_imagenet_id = dict_folder2imagenetid[category]
            for id in list_imagenet_id:
                path_imagenet_category_folder = os.path.join(path_imagenet_train, id)
                list_id_images =glob.glob(path_imagenet_category_folder + "/*")
                list_images.extend(list_id_images)
                print(len(list_id_images),id, category)
            random.shuffle(list_images)
            print(len(list_images))
            if category in required_categories:
                print(list_images)
                continue
            else:
                continue
            
            # category = os.path.basename(f)
            try:
                stitched_grid_img = generateImageGridFromImageList(list_images,
                                                                   rows=rows,
                                                                   cols=cols,
                                                                   resize=(resize),
                                                                   shuffle=True)

                path_img_out = os.path.join(path_folder_dataset_results, "{}.jpg".format(category))
                print("Saving at ", path_img_out)
                stitched_grid_img.save(path_img_out)
            except:
                print("Couldnt save for category", category, f)

def getDictObjFolder2ImgnetID():
    labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    id_name = list(idx_to_labels.values())
    sorted_id_name = sorted(id_name,key=lambda x: x[1])
    cats = [id_name[1] for i,id_name in enumerate(sorted_id_name)]

    path_json = "preprocessing/folder_to_imagenet_1k.json"
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

    dict_folder2imagenetid = {}
    lower_cat_imagenet = [cat[1].lower() for cat in sorted_id_name]
    o = 0

    for obj_cat, list_img_cat in dict_objectnet2imagenet.items():

        list_id = []
        # Search for these imagenet categories in dict
        for img_cat in list_img_cat:
            img_cat = img_cat.lower()
            try:
                idx = lower_cat_imagenet.index(img_cat)
                list_id.append(sorted_id_name[idx][0])
            except:
                pass
        if len(list_id) ==0:
            print(obj_cat,list_img_cat)
        o+=len(list_id)
        dict_folder2imagenetid[obj_cat] = list_id

    return dict_folder2imagenetid

if __name__ == "__main__":
    main()


# save it
# new_image.save('output.jpg')
