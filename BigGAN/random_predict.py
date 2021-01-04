import torch
import torchvision.models as models
import torch.nn as nn
import os
from torch.autograd import Variable

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, save_as_images_predictions)

# num: the i-th folders after sorting
# id: n01234545
# name: husky
from utils import getDictImageNetClasses
id2nameDict = getDictImageNetClasses()
ikeys = list(id2nameDict.keys())
num2id={}
num2name={}
ikeys.sort()
for ii, each in enumerate(ikeys):
    num2name[ii] = id2nameDict[each]
    num2id[ii] = each

save_root = ''



model = BigGAN.from_pretrained('biggan-deep-256')
model = nn.DataParallel(model)

# model_predict = models.__dict__["resnet152"](pretrained=True)
# model_predict = torch.nn.DataParallel(model_predict)
# # VGG.features = torch.nn.DataParallel(VGG.features)
# model_predict.cuda()
# model_predict.eval()


for eachid in ikeys:
    os.makedirs(os.path.join(save_root, eachid), exist_ok=True)







