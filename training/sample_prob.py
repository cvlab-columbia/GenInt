# 1. select subset of imgnet category
# 2. select subset of query images
# 3. load support set img
# 4. calculate res18 feature for support, say 500.
# 5. for each query img, retrieve the nearest neighbor in res18 feature, calculate the distance of P(x|x') by cross_entropy
# 6. calculate the P(y|x')
import os, random
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch,json
import numpy as np

glb_feature = []


def hook_fn(model, input, output):
    global glb_feature
    glb_feature = output

def normalize2d(mat):
    return mat / torch.sum(mat ** 2, dim=1, keepdim=True)

print("=> creating model resnet18")
model = models.resnet18(pretrained=True)
model.avgpool.register_forward_hook(hook_fn)
model.eval()

# imgnet_path = '/proj/vondrick/mcz/ImageNet-Data/val'
imgnet_path = '/local/rcs/mcz/ImageNet-Data/val'
obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_noborder'
imgnet_path =obj_valdir

gan_path = '/local/vondrick/cz/GANdata/rand'
# gan_path = '/local/vondrick/cz/GANdata/setting_50_16_sub'
# gan_path = '/local/rcs/mcz/GANgendata/GanSpace/setting_500_20_1_s2_lam_7.0'
# gan_path = '/local/vondrick/cz/GANdata/setting_1000_20_1_s2_lam_9.0'
class_list = os.listdir(imgnet_path)
print(gan_path)

cat_num=10
img_num = 50
dict_size=500
bs=100

class_list.sort()
category2id = {filename: fileintkey for fileintkey, filename in enumerate(class_list)}
id2category = {fileintkey: filename for fileintkey, filename in enumerate(class_list)}
# class_select = random.sample(class_list, cat_num)
# class_select = class_list[:10]
# class_select = class_list[50::100]
class_select = class_list
print(len(class_select))

with open('preprocessing/obj2imgnet_id.txt') as f:
    dict_obj2imagenet_id = json.load(f)

cnt = 0
Pxx = 0
Pxx_KL = 0
Pxy = 0

objectnet=True

for each in class_select:
    img_path_cat = os.path.join(imgnet_path, each)
    img_list = os.listdir(img_path_cat)
    img_list.sort()

    if objectnet:
        target = random.sample(dict_obj2imagenet_id[each], 1)[0]
        each = id2category[target]
    else:

        target = category2id[each]


    # img_select = random.sample(img_list, img_num)
    img_select = img_list  # [:img_num]

    gan_path_cat = os.path.join(gan_path, each)
    if 'rand' in gan_path:
        print('rand')
        gan_list = os.listdir(gan_path_cat)
    else:
        print('intervene')
        gan_list = []
        for root, _, fnames in sorted(os.walk(gan_path_cat, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                gan_list.append(path)
        # print(gan_list)
    gan_select = random.sample(gan_list, dict_size)

    # build dictionary

    from data.dataset import SpecifiedClassLoader

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = SpecifiedClassLoader(
        gan_path_cat,
        gan_select,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=10, pin_memory=True, sampler=None)

    cc=0
    feature = torch.zeros((dict_size, 512))
    feature_softmax = torch.zeros((dict_size, 512))
    # dict_out = torch.zeros((dict_size, 1000))
    for i, (images, path) in enumerate(train_loader):
        images.cuda()
        label = torch.ones((images.size(0))) * target
        label.cuda()

        output = model(images)
        fea = torch.flatten(glb_feature, 1)
        fea_softmax = torch.log_softmax(fea, dim=1)
        batchs = fea.size(0)
        feature[cc:cc + batchs, :] = fea.data.cpu()
        feature_softmax[cc:cc + batchs, :] = fea_softmax.data.cpu()
        # output_logsoftmax = torch.log_softmax(output, dim=1)
        # dict_out[cc:cc + batchs, :] = output_logsoftmax.data.cpu()

            # .data.cpu().numpy()

        cc = cc + batchs

    print("get dictionary", feature.shape)
    feature = normalize2d(feature)


    # get query image
    query_dataset = SpecifiedClassLoader(
        img_path_cat,
        img_list,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=bs, shuffle=False,
        num_workers=10, pin_memory=True, sampler=None)

    for i, (images, path) in enumerate(query_loader):
        images.cuda()
        label = torch.ones((images.size(0))).long() * int(target)
        label.cuda()

        output = model(images)

        output_softmax = torch.softmax(output, dim=1)
        # target_vec = output_softmax[:, target]

        # print(targetloss)

        fea = torch.flatten(glb_feature, 1)
        # fea = fea.data.cpu().numpy()
        fea_softmax = torch.softmax(fea, dim=1)
        fea = normalize2d(fea.data).cpu()

        # score = np.dot(fea, np.transpose(feature, (1, 0)))
        score_mat = torch.mm(fea, feature.t())
        val, ind = torch.topk(score_mat, k=1, dim=1, largest=True, sorted=True)

        KL_dist = torch.mm(fea_softmax.cpu(), feature_softmax.t())
        KLval, KLind = torch.topk(KL_dist, k=1, dim=1, largest=True, sorted=True)
        # val = torch.log(val)

        # val_prod = torch.sum(val)
        val_np = val.data.cpu().numpy()
        ind_np = ind.data.cpu().numpy()

        valKL_np = KLval.data.cpu().numpy()
        indKL_np = KLind.data.cpu().numpy()

        # this_pxyz = 0
        this_pxx = 0
        this_pxx_KL=0
        length = ind_np.shape[0]
        for jj in range(length):
            index = ind_np[jj]
            # print(index, dict_out[index, target])

            # gt = output_softmax[jj]
            # xent = torch.sum(gt * dict_out[index])

            # print(index, xent)

            # this_pxyz += (xent + np.log(np.max([val[jj], 1e-100])))
            this_pxx += np.log(np.max([val[jj], 1e-100]))
            this_pxx_KL += valKL_np[jj]
            cnt += 1

        # print('pxx', this_pxx, this_pxyz)
        Pxx += this_pxx
        Pxx_KL += this_pxx_KL
        # Pxy += this_pxyz
        print('gan path', gan_path)
        print("Average results : Pxx", Pxx/cnt, "Pxy", Pxx_KL/cnt)
        # print(val)
        # exit(0)

print("Pxx", Pxx, "Pxy", Pxy)





