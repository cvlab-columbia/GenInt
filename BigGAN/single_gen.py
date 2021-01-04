import torch
import torchvision.models as models
import torch.nn as nn
import os, random
import numpy as np
from torch.autograd import Variable

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

use_all = False

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, save_as_images_predictions, save_allto_dataset)

# num: the i-th folders after sorting
# id: n01234545
# name: husky
from utils import getDictImageNetClasses, get_imagenet_overlap
id2nameDict = getDictImageNetClasses()
ikeys = list(id2nameDict.keys())
num2id={}
num2name={}
ikeys.sort()
for ii, each in enumerate(ikeys):
    num2name[ii] = id2nameDict[each]
    num2id[ii] = each


# Load Objectnet classes:
overlapping_list, non_overlapping = get_imagenet_overlap()


save_root = '*/rand'


# significant_class=["cars", "ffhq", "husky"]

model = BigGAN.from_pretrained('biggan-deep-256')
model = nn.DataParallel(model)
model.to('cuda')
model.eval()
model_predict = models.__dict__["resnet152"](pretrained=True)
model_predict = torch.nn.DataParallel(model_predict)
# VGG.features = torch.nn.DataParallel(VGG.features)
model_predict.cuda()
model_predict.eval()

# model_predict = models.__dict__["resnet152"](pretrained=True)
# model_predict = torch.nn.DataParallel(model_predict)
# # VGG.features = torch.nn.DataParallel(VGG.features)
# model_predict.cuda()
# model_predict.eval()

# numlist = overlapping_list

start=0
endnum=100
num_list_all = [i for i in range(start, endnum)]

numlist = num_list_all

used_names = []
# Walk_steps = 128*2
Walk_steps = 2000
interpolate = 0
batchsize = 20  # 4GPU needed for batch 10 , 2 GPU for batch 5.
batchsize = 200  # 4GPU needed for batch 10 , 2 GPU for batch 5.
truncation = 0.2

cnt=0
# repeat = 100

for eachnum in numlist:
    cnt += 1
    print("{}-th".format(eachnum))

    idname = num2id[eachnum]

    # outputpath = os.path.join(save_root, "{}_{}".format(idname, num2name[eachnum]))
    outputpath = os.path.join(save_root, "{}".format(idname))
    os.makedirs(outputpath, exist_ok=True)

    class_vector_single = torch.zeros((1, 1000))
    class_vector_single[0, eachnum] = 1
    # class_vector_single = torch.from_numpy(class_vector_single)

    class_vector = None

    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=Walk_steps, seed=993442+cnt)

    batchnum = Walk_steps//batchsize


    count = 0
    count_all = 0

    for bb in range(batchnum):
        # for ee in batchsize:
        walk_point = noise_vector[bb*batchsize:(bb+1)*batchsize, :]
        walk_point = torch.from_numpy(walk_point)

        if interpolate>1:
            points_all = []
            for ii in range(walk_point.shape[0] - 1):
                start = walk_point[ii, :].unsqueeze(dim=0)
                next = walk_point[ii + 1, :].unsqueeze(dim=0)
                # print("start", start.size())
                for jj in range(interpolate):
                    points_all.append(start * (1 - jj * 1.0 / interpolate) + next * (jj * 1.0 / interpolate))

            points_all = torch.cat(points_all, dim=0)
        else:
            points_all = walk_point

        # print('input bsize', points_all.size(0))

        if class_vector is None or class_vector.size(0) != points_all.size(0):
            # print("solo need regenerate class vector")
            label_list = []
            for ii in range(points_all.size(0)):
                label_list.append(class_vector_single)
            class_vector = torch.cat(label_list, dim=0)

        points_all = points_all.to('cuda')
        class_vector = class_vector.to('cuda')

        with torch.no_grad():
            output_tmp = model(points_all, class_vector, truncation)

        gen_imgs = output_tmp.clone()
        gen_imgs = gen_imgs.to('cpu')

        # img_mean = [0.485, 0.456, 0.406]
        # img_std = [0.229, 0.224, 0.225]
        # output_tmp = torch.clamp((output_tmp + 1) / 2, 0, 1)
        # for channel in range(3):
        #     output_tmp[:, channel, :, :] = (output_tmp[:, channel, :, :] - img_mean[channel]) / img_std[channel]

        # with torch.no_grad():
        #     softmax_pred = model_predict(output_tmp)
        #     _, pred = softmax_pred.topk(1, 1, True, True)
        # # print()
        # pred_np = pred.cpu().numpy()
        #
        # correct_mask = np.zeros_like(pred_np)
        #
        # gt_label = np.ones_like(pred_np) * eachnum
        # correct_mask = correct_mask + (gt_label == pred_np)
        pred_np = np.zeros((output_tmp.size(0), 1))

        """Images are saved : {}_{}_{}_{} : id,  original_id, prediction_in_number, prediction name in English"""
        # count, count_all = save_to_dataset(gen_imgs, pred_np, correct_mask, outputpath, num2name, count, count_all)

        count, count_all = save_allto_dataset(gen_imgs, pred_np, outputpath, num2name, count, count_all)






