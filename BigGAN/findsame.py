import torch
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable

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

use_VGG=False

if use_VGG:
    print("load VGG")
    VGG = models.__dict__["vgg16"](pretrained=True)
    # VGG.features = torch.nn.DataParallel(VGG.features)
    VGG.cuda()
    print("finish loading VGG")


# Prepare a input
truncation = 0.4
class_vector = one_hot_from_names(['teapot', 'coffeepot', 'car_wheel'], batch_size=3)
# class_vector = one_hot_from_names(['running_shoe', 'running_shoe', 'running_shoe'], batch_size=3)
class_vector = one_hot_from_names(['car_wheel', 'car_wheel', 'car_wheel'], batch_size=3)
class_vector = one_hot_from_names(['pitcher', 'pitcher', 'pitcher'], batch_size=3)
class_vector = one_hot_from_names(['running_shoe', 'chair', 'car_wheel'], batch_size=3)
class_vector = one_hot_from_names(['king_penguin', 'running_shoe', 'backpack'], batch_size=3)
# class_vector = one_hot_from_names(['bench', 'bench', 'bench'], batch_size=3)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3, seed=838383)



# All in tensors
print("noise_vector", noise_vector)
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

norm_of_noise = torch.sum(noise_vector[0]**2)**0.5

t1 = torch.randn_like(noise_vector) * 0.2
t2 = torch.randn_like(noise_vector) * 0.2

t3 = (t1+t2)

# print("t size", t.size(), t[0].size())
noise_vector_2 = noise_vector + t1[0].unsqueeze(dim=0)
noise_vector_3 = noise_vector - t2[0].unsqueeze(dim=0)
noise_vector_4 = noise_vector - t3[0].unsqueeze(dim=0)


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
noise_vector = noise_vector.to('cuda')
noise_vector_2 = noise_vector_2.to('cuda')
noise_vector_3 = noise_vector_3.to('cuda')
noise_vector_4 = noise_vector_4.to('cuda')
noise_vector = Variable(noise_vector, requires_grad=True)
class_vector = class_vector.to('cuda')
target_img=target_img.to('cuda')
model.to('cuda')
model.train()

# Generate an image
# with torch.no_grad():

output_tmp = model(noise_vector, class_vector, truncation)
output_tmp = output_tmp.to('cpu')
# output = target_img.to('cpu') # I checked the img, it is correct.

# Save results as png images
save_as_images(output_tmp, j=0)

output_tmp = model(noise_vector_2, class_vector, truncation)
output_tmp = output_tmp.to('cpu')
# output = target_img.to('cpu') # I checked the img, it is correct.

# Save results as png images
save_as_images(output_tmp, j=1)


output_tmp = model(noise_vector_3, class_vector, truncation)
output_tmp = output_tmp.to('cpu')
# output = target_img.to('cpu') # I checked the img, it is correct.

# Save results as png images
save_as_images(output_tmp, j=2)

output_tmp = model(noise_vector_4, class_vector, truncation)
output_tmp = output_tmp.to('cpu')
# output = target_img.to('cpu') # I checked the img, it is correct.

# Save results as png images
save_as_images(output_tmp, j=3)

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