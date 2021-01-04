import os
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

img_shape = (3, 32, 32)

# Parameters
image_size = 32
label_dim = 10

learning_rate = 0.0002
betas = (0.5, 0.999)
batch_size = 2048
num_epochs = 20

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,1024 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    BATCH_SIZE = 64
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
    
            var_X_batch = Variable(imgs.type(FloatTensor))
            var_y_batch = Variable(labels.type(LongTensor))
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 200 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                            epoch, 
                            batch_idx*len(imgs), 
                            len(train_loader.dataset), 
                            100.*batch_idx / len(train_loader), 
                            loss.item(), 
                            float(correct*100) / float(BATCH_SIZE*(batch_idx+1))
                        )
                     )
                
# def fit_augment(model, train_loader):
#     optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
#     error = nn.CrossEntropyLoss()
#     EPOCHS = 5
#     BATCH_SIZE = 64
#     model.train()
#     for epoch in range(EPOCHS):
#         correct = 0
#         for batch_idx, (data1, data2) in enumerate(train_loader):
            
#             imgs = torch.cat((data1[0],data2[0]))
#             labels = torch.cat((data1[1],data2[1]))            
            
#             var_X_batch = Variable(imgs.type(FloatTensor))
#             var_y_batch = Variable(labels.type(LongTensor))
#             optimizer.zero_grad()
#             output = model(var_X_batch)
#             loss = error(output, var_y_batch)
#             loss.backward()
#             optimizer.step()

#             # Total correct predictions
#             predicted = torch.max(output.data, 1)[1] 
#             correct += (predicted == var_y_batch).sum()
#             #print(correct)
#             if batch_idx % 200 == 0:
#                 print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
#                             epoch, 
#                             batch_idx*len(imgs)/2, 
#                             len(train_loader.dataset), 
#                             100.*batch_idx / len(train_loader), 
#                             loss.item(), 
#                             float(correct*100) / float(BATCH_SIZE*(batch_idx+1))
#                         )
#                      )
                
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
                
def evaluate(model, test_loader):
    correct = 0 
    BATCH_SIZE = 100
    top1 = 0
    top5 = 0
    for batch_idx, (test_imgs, test_labels) in enumerate(test_loader):
        test_imgs = Variable(test_imgs.type(FloatTensor))
        test_labels = Variable(test_labels.type(LongTensor))
        output = model(test_imgs)
#         predicted = torch.max(output,1)[1]
#         correct += (predicted == test_labels).sum()
        
        acc1, acc5 = accuracy(output, test_labels, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        
    print("Test accuracy top1:{:.3f}% ".format( float(top1*100) / (len(test_loader)*BATCH_SIZE)))
    print("Test accuracy top5:{:.3f}% ".format( float(top5*100) / (len(test_loader)*BATCH_SIZE)))


# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return min(len(d) for d in self.datasets)


color_mnist_test = '/proj/vondrick/datasets/color_MNIST/test/'
color_mnist_confound_test = '/proj/vondrick/datasets/color_MNIST/confound_test/'
color_mnist_train = '/proj/vondrick/datasets/color_MNIST/train/'
color_mnist_train_intervened = '/proj/vondrick/datasets/color_MNIST/intervene_train/'


batch_size = 64

composed_transforms = transforms.Compose([
                        transforms.Resize(32), 
                        transforms.ToTensor(), 
                    ])

test_set = datasets.ImageFolder(color_mnist_test, composed_transforms)
confound_test_set = datasets.ImageFolder(confound_test_set, composed_transforms)
color_mnist_train_set = datasets.ImageFolder(color_mnist_train, composed_transforms)
color_mnist_train_intervened_set = datasets.ImageFolder(color_mnist_train_intervened, composed_transforms)


testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
confound_testloader = torch.utils.data.DataLoader(confound_test_set, batch_size=batch_size, shuffle=True)

color_mnist_trainloader = torch.utils.data.DataLoader(
                                                        color_mnist_train_set, 
                                                        batch_size=batch_size, shuffle=True)

intervened_trainloader = torch.utils.data.DataLoader(
                                                        color_mnist_train_intervened_set, 
                                                        batch_size=batch_size, shuffle=True)

print('train baseline')
cnn_baseline = CNN()
cnn_baseline = cnn_baseline.cuda()
fit(cnn_baseline, color_mnist_trainloader)

print('train intervened')
cnn_intervened = CNN()
cnn_intervened = cnn_intervened.cuda()
fit(cnn_intervened, intervened_trainloader)

print("baseline")
evaluate(cnn_baseline, testloader)

print("intervened")
evaluate(cnn_baseline, testloader)