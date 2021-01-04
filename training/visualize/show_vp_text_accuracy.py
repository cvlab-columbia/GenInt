'''
Usage :
python show_vp_text_accuracy.py <class_name>

Horizontally stacks the images from each viewpoint in class_path, and arranges them vertically (sorting by the accuracy that's given in the folder_name)

'''



import os 
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse

# parser = argparse.ArgumentParser(description='Choose a Category')
# parser.add_argument('name', metavar='name', type=str,
#                    help='name of class')
#
# args = parser.parse_args()

def save_class_image(class_name, data_folder_path, output_folder_path):

    class_path = data_folder_path + class_name
    image_name = output_folder_path + class_name + ".png"

    n_vp = len(listdir(class_path))
    row = min(50, n_vp) # number of view points
    vp_imgs = {}
    h = 128
    print(row)

    # Go to each row(also viewpoint_num), stack the images and save in vp_imgs
    viewpoint_folder_names = os.listdir(class_path)
    sorted_viewpoint_folder_names = sorted(viewpoint_folder_names)
    dict_i_to_viewpoint_folder = {}
    a = 1

    for i in range(row):
        view_point_folder_path = viewpoint_folder_names[i]
        dict_i_to_viewpoint_folder[i] = view_point_folder_path
        # view_point_folder_path = dict_viewpoint_to_viewpoint_folder[viewpoint_folder_names]
        img_path = class_path + '/' + view_point_folder_path
        imgs = [Image.open(img_path + '/' + x).convert('RGB') for x in sorted(listdir(img_path))]
        min_shape = (h,h)
        imgs_stack = np.hstack( (np.asarray( j.resize(min_shape) ) for j in imgs ) )
        vp_imgs[i] = Image.fromarray(imgs_stack)

    # sort vp_images by number of images, currently sorted by
    sorted_vpImg = sorted([(vp_imgs[i].size, i,dict_i_to_viewpoint_folder[i]) for i in vp_imgs], reverse=True) # looks like [(size_vp1,v1_num),(size_vp2,v2_num)] sorted by the size in decreasing order.
    w = vp_imgs[sorted_vpImg[0][1]].size[0] # holds max size.
    
    a = sorted([float(x[2].split('___')[0]) for x in sorted_vpImg])
    # Current sorting by size, just sort sorted_vpImg by accuracy
    sorted_vpImg = sorted(sorted_vpImg, key=lambda x: float(x[2].split('___')[0]))

    # Combine the images
    imgs_comb = Image.new('RGB', (w, h))
    new_im = vp_imgs[sorted_vpImg[0][1]] #image hstack for max
    imgs_comb.paste(new_im, (0,0))

    fnt = ImageFont.truetype('/Pillow/Tests/fonts/FreeMono.ttf', 55)
    d = ImageDraw.Draw(imgs_comb)
    d.text((40, 40), sorted_vpImg[0][2], font=fnt, fill=(255, 0, 0))

    for i in tqdm(range(1, row)):
        im = vp_imgs[sorted_vpImg[i][1]]
        new_im = Image.new('RGB', (w, h))
        new_im.paste(im, (0,0))

        # Draw the text for accuracy

        d = ImageDraw.Draw(new_im)
        d.text((40,40), sorted_vpImg[i][2],font=fnt, fill=(255,0,0))

        imgs_comb = np.vstack((imgs_comb, new_im))
        imgs_comb = Image.fromarray(imgs_comb)

    # Save the generated_image
    imgs_comb.save(image_name)

def saveAll(data_folder_path, output_folder_path):
    list_categories = os.listdir(data_folder_path)
    for category in list_categories:
        save_class_image(category, data_folder_path, output_folder_path)

def stitch_images(folder1, folder2, output_folder):

    categories_images = os.listdir(folder1)
    for category_image in categories_images:

        im1_path = folder1 + category_image
        im2_path = folder2 + category_image

        im1 = Image.open(im1_path).convert('RGB')
        im2 = Image.open(im2_path).convert('RGB')
        try:
            final_im_array = np.hstack((np.asarray(im1),np.asarray(im2)))
            final_im = Image.fromarray(final_im_array)

            final_im_path = output_folder + category_image
            final_im.save(final_im_path)

        except:
            print(category_image, im1.size, im2.size)

if __name__ == "__main__":
    # class_name, data_folder_path, output_folder_path
    c4canonical_remove_folder = '/home/amogh/columbia/research/detection/try/C+4Canonical-remove-overlap/'
    c4canonical_folder = '/home/amogh/columbia/research/detection/try/C+4Canonical/'

    data_folder_path = '/home/amogh/columbia/research/detection/try/C+4Canonical-remove-overlap/'
    output_folder_path = '/home/amogh/columbia/research/detection/try/stitched images/'
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)



    # Saving all images
    # saveAll(data_folder_path, output_folder_path)

    # Saving a particular category
    # category = "n01558993"
    # save_class_image(category, data_folder_path, output_folder_path)


    # Stitch images
    # output_folder1 = '/home/amogh/columbia/research/detection/try/images-C+4canonical/'
    # output_folder2 = '/home/amogh/columbia/research/detection/try/images-C+4canonical-remove-overlap/'
    # stitch_images(output_folder1, output_folder2, output_folder_path)
