import torch
import torchvision.models as models
import torch.nn as nn


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, save_as_images_predictions)
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable

from utils import getDictImageNetClasses
id2nameDict = getDictImageNetClasses()

ikeys = list(id2nameDict.keys())
num2name={}
ikeys.sort()
for ii, each in enumerate(ikeys):
    num2name[ii] = id2nameDict[each]

print(num2name[770]) # correct: this is running shoe

# print(num2id[770])
# exit(0)
#
# print(id2nameDict)
# print(id2nameDict[770])
# exit(0)

root_path='/home/mcz/2020Spring/BigGAN/images/running_shoe'
# root_path='/home/mcz/2020Spring/BigGAN/images/running_shoe/gen1'
img_file_list = ["3968e6aa1fab4d6.png","55548e1eea15486.png", "83fcccb411f64bf.png"]  #,"f03534e6e1a6464.png"
# img_file_list = ["output_0_0.png","output_1_0.png", "output_2_0.png"]  #,"f03534e6e1a6464.png"
img_file_list_whole = [os.path.join(root_path, each) for each in img_file_list]

from pytorch_pretrained_biggan.utils import load_images, convert_img_to_model_output
img_list = load_images(img_file_list_whole)
img_arr = convert_img_to_model_output(img_list)

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')

model = nn.DataParallel(model)

use_VGG=False


model_predict = models.__dict__["resnet152"](pretrained=True)
model_predict = torch.nn.DataParallel(model_predict)
# VGG.features = torch.nn.DataParallel(VGG.features)
model_predict.cuda()
model_predict.eval()
print("finish loading VGG")


# Prepare a input
truncation = 0.4
# class_vector = one_hot_from_names(['teapot', 'coffeepot', 'car_wheel'], batch_size=3)
# # class_vector = one_hot_from_names(['running_shoe', 'running_shoe', 'running_shoe'], batch_size=3)
# class_vector = one_hot_from_names(['car_wheel', 'car_wheel', 'car_wheel'], batch_size=3)
# class_vector = one_hot_from_names(['pitcher', 'pitcher', 'pitcher'], batch_size=3)
# class_vector = one_hot_from_names(['running_shoe', 'chair', 'car_wheel'], batch_size=3)
# class_vector = one_hot_from_names(['king_penguin', 'running_shoe', 'backpack'], batch_size=3)
# # class_vector = one_hot_from_names(['bench', 'bench', 'bench'], batch_size=3)
# noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3, seed=838383)

## Random Walk

Walk_steps = 5

class_vector1 = one_hot_from_names(['husky'], batch_size=1)
class_vector2 = one_hot_from_names(['banana'], batch_size=1)
class_vector3 = one_hot_from_names(['car_wheel'], batch_size=1)
class_vector4 = one_hot_from_names(['bicycle'], batch_size=1)
class_vector5 = one_hot_from_names(['bench'], batch_size=1)
class_vector = (class_vector1 + class_vector2 + class_vector3 + class_vector4 + class_vector5)/7

print("hot size", class_vector.shape)
exit(0)

noise_vector = truncated_noise_sample(truncation=truncation, batch_size=Walk_steps, seed=3442)

# All in tensors
print("noise_vector", noise_vector)
noise_vector = torch.from_numpy(noise_vector)
# noise_vector =torch.randn(3, 128)
class_vector = torch.from_numpy(class_vector)

# walk_list = [torch.randn_like(noise_vector) * 0.7 for _ in range(Walk_steps)]
walk_point = [noise_vector[ii].unsqueeze(dim=0) for ii in range(Walk_steps)]

interpolate = 10
points_all = []
for ii in range(len(walk_point)-1):
    start = walk_point[ii]
    next = walk_point[ii+1]
    for jj in range(interpolate-1):
        points_all.append(start * (1-jj*1.0/10) + next * (jj*1.0/10))


points_all = torch.cat(points_all, dim=0)

label_list = []
for ii in range(points_all.size(0)):
    label_list.append(class_vector)

class_vector = torch.cat(label_list, dim=0)


norm_of_noise = torch.sum(noise_vector[0]**2)**0.5



def normalize_back(ten):
    print(torch.sum(ten**2, dim=1).size(), norm_of_noise.size())

    tp = torch.sum(ten**2, dim=1)**0.5

    ten = ten / tp.unsqueeze(1) * norm_of_noise
    return ten

# noise_vector_2 = normalize_back(noise_vector_2)
# noise_vector_3 = normalize_back(noise_vector_3)
# noise_vector_4 = normalize_back(noise_vector_4)

target_img = torch.from_numpy(img_arr)

# If you have a GPU, put everything on cuda
# noise_vector = noise_vector.to('cuda')
points_all = points_all.to('cuda')
class_vector = class_vector.to('cuda')

_, gt = class_vector.topk(1, 1, True, True)

target_img=target_img.to('cuda')
model.to('cuda')
model.eval()
model_predict.eval()


# Generate an image
# with torch.no_grad():

output_tmp = model(points_all, class_vector, truncation)

# print('output imag size', output_tmp.size(), output_tmp.max(), output_tmp.min())
# exit(0)

# normalize as standard imagenet input
img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]
output_tmp = torch.clamp((output_tmp + 1)/2, 0, 1)
for channel in range(3):
    output_tmp[:, channel, :, :] = (output_tmp[:, channel, :, :] - img_mean[channel]) / img_std[channel]

softmax_pred = model_predict(output_tmp)
_, pred = softmax_pred.topk(1, 1, True, True)
# print()
pred_np = pred.cpu().numpy()

output_tmp = output_tmp.to('cpu')
# output = target_img.to('cpu') # I checked the img, it is correct.

# Save results as png images
# save_as_images(output_tmp, j=0)
save_as_images_predictions(output_tmp, pred_np, num2name, j=0)


# iter=10
# lr=1e-5
# old_grad=None
# noise_level = 1
# for i in range(iter):
#     noise_level = noise_level - noise_level*1.0/iter*i
#     # lr = lr * 0.5 ** (i*1.0/200)
#     output = model(noise_vector, class_vector, truncation)
#     # print(output.size(), target_img.size())
#
#     loss = torch.sum((output - target_img) ** 2)
#
#     if use_VGG:
#         bsize = output.size(0)
#         # in_cont = torch.cat((output, target_img), dim=0)
#         # _ = VGG(in_cont)
#         # fea = VGG.features
#         gan_fea = VGG.features(output)
#         gt_fea = VGG.features(target_img)
#         # gt_fea = VGG(target_img)
#         loss_vgg = torch.sum((gan_fea-gt_fea)**2)
#
#         print("loss", loss.item(), 'loss vgg', loss_vgg.item())
#         loss = loss+loss_vgg
#
#     model.zero_grad()
#
#     loss.backward()
#     grad = noise_vector.grad
#     # print("grad", grad)
#
#     # if i==0:
#     #     grad = grad
#     # else:
#     #     grad = old_grad * 0.9 + grad*0.1
#     # old_grad = grad
#
#     # print("noise vec", noise_vector)
#     # print("grad", grad)
#     # print("torch.randn_like(noise_vector)", torch.randn_like(noise_vector))
#
#     noise_vector = noise_vector - grad * lr + torch.randn_like(noise_vector) * noise_level #TODO: should be -
#
#     if i%1==0:
#         print("iteration", i, "loss", loss.item())
#         output_tmp = model(noise_vector, class_vector, truncation)
#         output_tmp = output_tmp.to('cpu')
#         # output = target_img.to('cpu') # I checked the img, it is correct.
#
#         # Save results as png images
#         save_as_images(output_tmp, j=i)
#
#     noise_vector = Variable(noise_vector.data, requires_grad=True)
# #
#
#
#
# output = model(noise_vector, class_vector, truncation)
#
#
# # If you have a GPU put back on CPU
# output = output.to('cpu')
# # output = target_img.to('cpu') # I checked the img, it is correct.
#
# # Save results as png images
# save_as_images(output)