os.environ["CUDA_VISIBLE_DEVICES"]="7"

import argparse
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import shutil
from PIL import Image
from tqdm import trange
from types import SimpleNamespace

trainloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../dataset",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=3), 
                transforms.Resize(32), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]),
        ),
        batch_size=128,
        shuffle=True,
    )

testloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../dataset",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=3), 
                transforms.Resize(32), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]),
        ),
        batch_size=128,
        shuffle=True,
    )

# Directory where dataset for IRM will be saved:
color_mnist_path = '/proj/vondrick/datasets/color_MNIST/irm_mnist/'


for i in range(10):
    os.makedirs(os.path.join(color_mnist_path,"env1",str(i)), exist_ok=True)
    os.makedirs(os.path.join(color_mnist_path,"env2",str(i)), exist_ok=True)
    
e1 = True

for batch_idx, (imgs, labels) in enumerate((trainloader)):
    if e1:
        imgs[labels==0,0,:,:] = torch.ones_like(imgs[labels==0,0,:,:])
        imgs[labels==0,1,:,:] = torch.ones_like(imgs[labels==0,1,:,:])

        imgs[labels==1,1,:,:] = torch.ones_like(imgs[labels==1,1,:,:])
        imgs[labels==1,2,:,:] = torch.ones_like(imgs[labels==1,2,:,:])

        imgs[labels==2,2,:,:] = torch.ones_like(imgs[labels==2,2,:,:])
        imgs[labels==2,0,:,:] = torch.ones_like(imgs[labels==2,0,:,:])

        imgs[labels==3,0,:,:] = torch.ones_like(imgs[labels==3,0,:,:])

        imgs[labels==4,1,:,:] = torch.ones_like(imgs[labels==4,1,:,:])

        imgs[labels==5,2,:,:] = torch.ones_like(imgs[labels==5,2,:,:])

        imgs[labels==6,0,:,:] = torch.zeros_like(imgs[labels==6,0,:,:])
        imgs[labels==6,1,:,:] = torch.zeros_like(imgs[labels==6,1,:,:])

        imgs[labels==7,1,:,:] = torch.zeros_like(imgs[labels==7,1,:,:])
        imgs[labels==7,2,:,:] = torch.zeros_like(imgs[labels==7,2,:,:])

        imgs[labels==8,2,:,:] = torch.zeros_like(imgs[labels==8,2,:,:])
        imgs[labels==8,0,:,:] = torch.zeros_like(imgs[labels==8,0,:,:])            

        imgs[labels==9,1,:,:] = torch.zeros_like(imgs[labels==9,1,:,:])

        e1 = not e1
        
        for digit in range(10):
            for i in range(imgs[labels==digit].size(0)):
                torchvision.utils.save_image(imgs[labels==digit][i, :, :, :], 
                                             color_mnist_path+"/env1/"+str(digit)+'/{}.png'.format(batch_idx*128+i), 
                                             normalize=True)

    else:
        imgs[labels==5,0,:,:] = torch.ones_like(imgs[labels==5,0,:,:])
        imgs[labels==5,1,:,:] = torch.zeros_like(imgs[labels==5,1,:,:])

        imgs[labels==3,1,:,:] = torch.ones_like(imgs[labels==3,1,:,:])
        imgs[labels==3,2,:,:] = torch.zeros_like(imgs[labels==3,2,:,:])

        imgs[labels==4,2,:,:] = torch.ones_like(imgs[labels==4,2,:,:])
        imgs[labels==4,0,:,:] = torch.zeros_like(imgs[labels==4,0,:,:])

        imgs[labels==1,0,:,:] = torch.zeros_like(imgs[labels==1,0,:,:])

        imgs[labels==2,1,:,:] = torch.zeros_like(imgs[labels==2,1,:,:])

        imgs[labels==0,2,:,:] = torch.zeros_like(imgs[labels==0,2,:,:])

        imgs[labels==6,0,:,:] = torch.zeros_like(imgs[labels==6,0,:,:])
        imgs[labels==6,1,:,:] = torch.ones_like(imgs[labels==6,1,:,:])

        imgs[labels==7,1,:,:] = torch.zeros_like(imgs[labels==7,1,:,:])
        imgs[labels==7,2,:,:] = torch.ones_like(imgs[labels==7,2,:,:])

        imgs[labels==8,2,:,:] = torch.zeros_like(imgs[labels==8,2,:,:])
        imgs[labels==8,0,:,:] = torch.ones_like(imgs[labels==8,0,:,:])            

        imgs[labels==9,1,:,:] = torch.ones_like(imgs[labels==9,2,:,:])

        e1 = not e1
        
        for digit in range(10):
            for i in range(imgs[labels==digit].size(0)):
                torchvision.utils.save_image(imgs[labels==digit][i, :, :, :], 
                                             color_mnist_path+"/env2/"+str(digit)+'/{}.png'.format(batch_idx*128+i), 
                                             normalize=True)

# Test path

color_mnist_test_path = '/proj/vondrick/datasets/color_MNIST/irm_mnist_test/'

for i in range(10):
    os.makedirs(color_mnist_test_path+str(i), exist_ok=True)

for batch_idx, (test_imgs, test_labels) in enumerate(testloader):
    channel = np.random.choice(3,2)
    color = np.random.sample([3])>0.5
    for ch in channel:
        if color[ch]:
            test_imgs[:,ch,:,:] = torch.ones_like(test_imgs[:,ch,:,:])            
        else:
            test_imgs[:,ch,:,:] = torch.zeros_like(test_imgs[:,ch,:,:])
                
    for digit in range(10):
        for i in range(test_imgs[test_labels==digit].size(0)):
            torchvision.utils.save_image(test_imgs[test_labels==digit][i, :, :, :], 
                                         color_mnist_test_path+str(digit)+'/{}.png'.format(batch_idx*128+i), 
                                         normalize=True