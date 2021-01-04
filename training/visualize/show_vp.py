import os 
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Choose a Category')
parser.add_argument('name', metavar='name', type=str,
                   help='name of class')

args = parser.parse_args()

class_name = args.name
class_path = os.getcwd() + "/train_examplar/" + class_name

n_vp = len(listdir(class_path)) - 1
row = min(50, n_vp) # number of view points
vp_imgs = {}
h = 128

print(row)
for i in range(row):
    view_point = str(i)
    img_path = class_path + '/' + view_point
    imgs = [Image.open(img_path + '/' + x).convert('RGB') for x in listdir(img_path)]
    min_shape = (h,h)    
    imgs_stack = np.hstack( (np.asarray( j.resize(min_shape) ) for j in imgs ) )
    vp_imgs[i] = Image.fromarray(imgs_stack)
    
sorted_vpImg = sorted([(vp_imgs[i].size, i) for i in vp_imgs], reverse=True)
w = vp_imgs[sorted_vpImg[0][1]].size[0]

imgs_comb = vp_imgs[sorted_vpImg[0][1]]
for i in tqdm(range(1, row)):
    t = vp_imgs[sorted_vpImg[i][1]]
    new_im = Image.new('RGB', (w, h))
    new_im.paste(t, (0,0))
    imgs_comb = np.vstack((imgs_comb, new_im))
    imgs_comb = Image.fromarray( imgs_comb)
    
imgs_comb.show()