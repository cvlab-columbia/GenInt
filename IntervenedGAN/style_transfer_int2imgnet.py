from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import os
import random

import numpy as np
import os

import PIL
from style_transfer_utils import image_loader, run_style_transfer


Transfer_intGAN=True

# start_cat=800
# end_cat=1000

start_cat=0
end_cat=1000

# desired size of the output image
imsize = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor



import socket

root_cont = '/your/path/to/imageNet'
root_style = '/your/path/of/generative interventions'
save_synthesis = '/your/path/to/save'

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



bs_size=8
for category in os.listdir(root_cont)[start_cat:end_cat]:

    if os.path.isdir(os.path.join(save_synthesis, category)):
        print(category, ' exists')
        continue
    else:
        print('Now running', category)

    os.makedirs(os.path.join(save_synthesis, category), exist_ok=True)
    cat_path = os.path.join(root_cont, category)
    cnt = 0
    for each_img in os.listdir(cat_path):
        cont_tmp = image_loader(os.path.join(cat_path, each_img))

        if cnt == 0:
            cont_list=[]
            cont_list.append(cont_tmp)
            cont_img_list=[]
            cont_img_list.append(each_img)
        else:
            cont_list.append(cont_tmp)
            cont_img_list.append(each_img)

        cnt += 1
        if cnt==bs_size:
            flag=True
            cnt=0
        else:
            flag=False


        if flag:
            content_img = torch.cat(cont_list, dim=0)
            intervene_path_list = os.listdir(root_style)
            selected_path = random.sample(intervene_path_list, bs_size)

            snt = 0

            for ee in selected_path:
                folder_path = os.path.join(root_style, ee)

                img_list = os.listdir(folder_path)
                img_select = random.sample(img_list, 1)[0]

                while True:
                    try:
                        style_tmp = image_loader(os.path.join(folder_path, img_select))
                        break
                    except:
                        img_select = random.sample(img_list, 1)[0]

                if snt == 0:
                    sty_list = []
                    sty_list.append(style_tmp)
                    sty_name_list = []
                    sty_name_list.append(ee+'_'+img_select)
                else:
                    sty_list.append(style_tmp)
                    sty_name_list.append(ee+'_'+img_select)

                snt += 1

            style_img = torch.cat(sty_list, dim=0)

            # print("img size", content_img.size(), style_img.size())

            input_img = content_img.clone()
            optim_steps = random.randint(20, 70)
            print("optimization steps chosen", optim_steps)

            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                        content_img, style_img, input_img, num_steps=optim_steps)
            output = output.detach().cpu().numpy()

            for ind in range(output.shape[0]):
                single_img = output[ind]
                single_img = single_img * 255
                out_array = np.asarray(np.uint8(single_img), dtype=np.uint8)
                # print('out_array', out_array)
                # print('out_array', out_array.shape)
                out_array = np.transpose(out_array, (1,2,0))
                img_out = PIL.Image.fromarray(out_array)
                img_out = img_out.resize((256, 256), PIL.Image.ANTIALIAS)

                save_folder_path = os.path.join(save_synthesis, category)
                os.makedirs(save_folder_path, exist_ok=True)
                current_file_name = os.path.join(save_folder_path,
                                                 "{}_with_{}_step{}.png".format(cont_img_list[ind].split('.')[0],
                                                                       sty_name_list[ind], optim_steps))
                img_out.save(current_file_name, 'png')

            del output

            cont_list = []
            cont_img_list = []


        else:
            continue





