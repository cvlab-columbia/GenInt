import time
import warnings
import numpy
import numpy as np

from utils import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import pickle, socket, json
from learning.resnet import resnet50, resnet18, resnet152, MLPNetWide
from data.KNN_dataloader import ObjectNetClassWiseLoader, ObjectNetLoader, ImagenetALoader

class Args():
    def __init__(self):
        pass



def test(checkpoint_path, load_emb, emb_name, use_mlp, arch='resnet152', save_separate=True,
         exclude_nonoverlap=False, batch_size=300, test_dataset=['objectnet'], CM=False):
    args = Args()
    args.load_emb = load_emb
    args.emb_name = emb_name
    args.resume = checkpoint_path
    args.arch=arch
    args.mlp = use_mlp
    args.gpu=None
    args.save_separate=save_separate
    args.exclude_nonoverlap=exclude_nonoverlap
    args.root_path = "/proj/vondrick/mcz/ModelInvariant/Pretrained/KNN_emb"
    args.batch_size = batch_size
    args.test_dataset = test_dataset
    args.CM=CM

    model = None
    if args.arch == 'resnet50':
        model = resnet50(normalize=False)
    elif args.arch == 'resnet18':
        model = resnet18(normalize=False)
    elif args.arch == 'resnet152':
        model = resnet152(normalize=False)
        if args.mlp:
            MLP = MLPNetWide(2048)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
        # model = torch.nn.DataParallel(model) #cpu
        if args.mlp:
            MLP = torch.nn.DataParallel(MLP).cuda()

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        # args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        if args.mlp:
            MLP.load_state_dict(checkpoint['MLP_state_dict'])
            print("Load MLP too")

        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}', This is an Error".format(args.resume))
        exit(0)

    cudnn.benchmark = True

    traindir = '/proj/vondrick/mcz/ImageNet-Data/train'
    valdir = '/proj/vondrick/mcz/ImageNet-Data/val'
    if socket.gethostname() == 'cv03':
        # use local fast SSD first
        traindir = "/local/vondrick/cz/ImageNet-Data/train"
        traindir_noview = "/local/vondrick/cz/ImageNet-Data/train"
        valdir = "/local/vondrick/cz/ImageNet-Data/val"
    elif socket.gethostname() == 'cv04':
        # traindir = '/local/vondrick/cz/train_whole_clustered-C1'
        # if args.noviewpoint:
        traindir = '/local/vondrick/cz/ImageNet/train'
        traindir_noview = '/local/vondrick/cz/ImageNet/train'
        valdir = '/local/vondrick/cz/ImageNet/val'

    obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_noborder'
    imageneta_valdir = '/proj/vondrick/amogh/imagenet-a/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    from data.test_imagefolder import MyImageFolder, ImageNetTrainOverlapObject


    if args.load_emb:
        if args.save_separate and not args.exclude_nonoverlap:
            foldername = '{}_bl_representation_{}'.format(args.arch, args.emb_name)
            openpath = '{}/{}'.format(args.root_path, foldername)
            embed_list = []
            # output_list = []
            label_list = []
            for each in os.listdir(openpath):
                filename = os.path.join(openpath, each)
                with open(filename,
                          'rb') as f:
                    print("loading saved emb", each)
                    train_all = pickle.load(f)
                    print("finished load emb")
                    embed_list_sub = train_all['embedding']
                    label_list_sub = train_all['label']
                    output_list_sub = train_all['output']
                    embed_list.append(embed_list_sub)
                    label_list.append(label_list_sub)
                    # output_list.append(output_list_sub)
            # embed_list = torch.cat(embed_list, 0)
            # embed_list = np.concatenate(embed_list, 0)
            # output_list = torch.cat(output_list, 0)
            # label_list = np.concatenate(label_list, axis=0)
            output_list = None
            # print("finish concat")
        else:
            openpath = '{}/{}_bl_representation_{}.pkl'.format(args.root_path, args.arch, args.emb_name)
            if args.exclude_nonoverlap:
                openpath = '{}/{}_bl_representation_{}{}.pkl'.format(args.root_path, args.arch, args.emb_name, '_exclude')
            with open(openpath,
                      'rb') as f:
                print("loading saved emb")
                train_all = pickle.load(f)
                print("finished load emb")
                embed_list = train_all['embedding']
                label_list = train_all['label']
                output_list = train_all['output']
                print("load old model successfully; num of data={}".format(embed_list.shape[0]))
    else:
        if args.exclude_nonoverlap:
            train_loader = torch.utils.data.DataLoader(
                ImageNetTrainOverlapObject(traindir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,  # Can be false
                num_workers=10, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                MyImageFolder(traindir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,  # Can be false
                num_workers=10, pin_memory=True)

        print("doing inference on whole data training set")
        embed_list, label_list, output_list = train_embed(train_loader, model, args)

    if args.save_separate and not args.exclude_nonoverlap:
        foldername = '{}_bl_representation_{}'.format(args.arch, args.emb_name)
        openpath = '{}/{}'.format(args.root_path, foldername)
        embed_list = []
        # output_list = []
        label_list = []
        for each in os.listdir(openpath):
            filename = os.path.join(openpath, each)
            with open(filename,
                      'rb') as f:
                print("loading saved emb", each)
                train_all = pickle.load(f)
                print("finished load emb")
                embed_list_sub = train_all['embedding']
                label_list_sub = train_all['label']
                output_list_sub = train_all['output']
                embed_list.append(embed_list_sub)
                label_list.append(label_list_sub)

    test_loader_all = []
    if "imagenet" in args.test_dataset:
        val_loader = torch.utils.data.DataLoader(
            MyImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=256, shuffle=True,
            num_workers=10, pin_memory=True)
        test_loader_all.append(val_loader)

        # if args.LR:
        #     LogisticRegression_test_objectnet_shuffle(val_loader, model, args,
        #                                               [embed_list, label_list, output_list])
    if 'imageneta' in args.test_dataset:
        test_loader = torch.utils.data.DataLoader(
            ImagenetALoader(imageneta_valdir),
            batch_size=100, shuffle=True,
            num_workers=10, pin_memory=True)
        test_loader_all.append(test_loader)

    if 'objectnet' in args.test_dataset:
        composed_transform = transforms.Compose([
            transforms.Compose([
                transforms.Resize(int(256 * 1.4)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ])
        test_loader = torch.utils.data.DataLoader(
            ObjectNetLoader(obj_valdir, composed_transform=composed_transform),
            batch_size=100, shuffle=True,
            num_workers=10, pin_memory=True)
        test_loader_all.append(test_loader)

    LogisticRegression_test_shuffle(test_loader_all, model, args,
                                    [embed_list, label_list, output_list])


def train_embed(train_loader, model, args):
    model.eval()
    if args.mlp:
        args.MLP.eval()

    embed_list = []
    output_list = []
    label_list = []
    embed_list_all = []
    output_list_all = []
    label_list_all = []
    print("total=", len(train_loader))
    split_flag=False
    if args.save_separate and not args.exclude_nonoverlap:
        foldername = '{}_bl_representation_{}'.format(args.arch, args.emb_name)
        openpath = '{}/{}'.format(args.root_path, foldername)
        os.makedirs(openpath, exist_ok=True)
        interval=200
        cnt_fe=0
        split_flag=True



    with torch.no_grad():
        # We need the same category in train be together, thus easier for later one to count the knn
        for i, (images, target, path) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output, feature, norm = model(images)
            # feature, _ = normalize_l2(feature)
            if args.mlp:
                feature = args.MLP(feature)
            embed_list.append(feature.cpu())
            output_list.append(output.cpu())
            label_list.append(target.data.cpu().numpy())

            if i%100==0 or i<5:
                print(i)

            if split_flag and ((i+1) % interval ==0 or i==len(train_loader)-1):
                cnt_fe += 1
                embed_list = torch.cat(embed_list, 0)
                output_list = torch.cat(output_list, 0)
                label_list = np.concatenate(label_list, axis=0)
                train_fea = {'embedding': embed_list.numpy(), 'label': label_list, 'output': output_list}
                filename = os.path.join(openpath, '{}_part_{}.pkl'.format(cnt_fe, args.emb_name))

                with open(filename, 'wb') as f:
                    pickle.dump(train_fea, f, protocol=4)
                    print("Dumping embeddings at: ", args.root_path)

                embed_list_all.append(embed_list)
                output_list_all.append(output_list)
                label_list_all.append(label_list)

                embed_list = []
                output_list = []
                label_list = []

                # break

            # if i==50:
            #     break

    if not split_flag:
        embed_list = torch.cat(embed_list, 0)
        output_list = torch.cat(output_list, 0)
        label_list = np.concatenate(label_list, axis=0)

        train_fea = {'embedding': embed_list.numpy(), 'label': label_list, 'output': output_list}
        # with open('../{}_representation.pkl'.format(args.arch), 'wb') as f:
        os.makedirs(args.root_path, exist_ok=True)

        openpath = '{}/{}_bl_representation_{}.pkl'.format(args.root_path, args.arch, args.emb_name)
        if args.exclude_nonoverlap:
            openpath = '{}/{}_bl_representation_{}{}.pkl'.format(args.root_path, args.arch, args.emb_name, '_exclude')


        with open(openpath, 'wb') as f:
            pickle.dump(train_fea, f, protocol=4)
            print("Dumping embeddings at: ", args.root_path)

        return [embed_list], [label_list], [output_list]
    else:
        return embed_list_all, label_list_all, output_list_all


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).cuda()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def LogisticRegression_test_shuffle(val_loader_list, model, args, train_list):
    model.eval()

    if len(train_list) == 2:
        emb_list, label_list = train_list
    else:
        emb_list, label_list, output_list = train_list

    # try:
    #     emb_list = torch.from_numpy(emb_list)
    #     label_list = torch.from_numpy(label_list)
    #     output_list = torch.from_numpy(output_list)
    # except:
    #     pass

    LR = LogisticRegression(2048, 1000)
    LR.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LR.parameters(), lr=0.01)

    # if args.emb_name == '152bl':
    #     LR_path = '/proj/vondrick/augustine/checkpointBL.pth.tar'
    # else:
    #     LR_path = '/proj/vondrick/augustine/checkpoint.pth.tar'
    #
    # if os.path.exists(LR_path):
    #     checkpoint = torch.load(LR_path)
    #     LR.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(LR_path, checkpoint['epoch']))
    # else:

    b_sz = 320292
    LR.train()
    if not args.save_separate or args.exclude_nonoverlap:
        emb_list = emb_list[0]
        emb_list = torch.from_numpy(emb_list)
        label_list = label_list[0]
        label_list = torch.from_numpy(label_list)
        print("what is len(emb_list)", len(emb_list))
        for epoch in tqdm(range(500), desc='training Logistic Regressor'):
            for i in range(0, len(emb_list), b_sz):
                batch = emb_list[i:i + b_sz]
                batch = batch.cuda()
                label = label_list[i:i + b_sz]
                label = label.cuda()
                out = LR(batch)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
    else:
        # If saved splited
        print('train the splited saving version', len(emb_list))
        for epoch in tqdm(range(500), desc='training Logistic Regressor'):  # 500
            for i in range(0, len(emb_list), b_sz):
                batch = torch.from_numpy(emb_list[i])
                batch = batch.cuda()
                label = torch.from_numpy(label_list[i])
                label = label.cuda()
                out = LR(batch)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

    if args.emb_name == '152bl':
        fname = 'checkpoint.pth.tar'
    else:
        fname = 'checkpointBL.pth.tar'

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': "LogisticRegression",
        'state_dict': LR.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, False, filename=args.emb_name, experiment_backup_folder='/proj/vondrick/mcz/ModelInvariant/Pretrained/LR')



    print("Loading ObjectNet-ImageaNet dictionary")
    with open('preprocessing/obj2imgnet_id.txt') as f:
        dict_obj2imagenet_id = json.load(f)

    test_hidden = True
    print("start evaluation")
    LR.eval()

    if args.CM:
        mapped_pred = []
        mapped_target = []
    args.test_dataset.sort()
    pbar_prefix = 'Testing on ' + str(args.test_dataset)
    for datatype, val_loader in zip(args.test_dataset, val_loader_list):
        top1_cnt = 0
        top5_cnt = 0
        cnt = 0
        with torch.no_grad():
            for sample, example in tqdm(enumerate(val_loader), total=len(val_loader), desc=pbar_prefix):
                if datatype == 'objectnet':
                    images = example['images']
                    target = example['labels']
                    path = example['path']
                elif datatype == 'imagenet':
                    # print("example", example)
                    images = example[0]
                    target = example[1]
                    path = example[2]

                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)

                o, fea, n = model(images)
                maxk = 5
                if datatype == 'objectnet':
                    batch_size = target[0].shape[0]
                else:
                    batch_size = target.shape[0]

                output = LR(fea)

                # if args.CM:
                #     _, p1 = output.topk(1)
                #     # p1 = torch.from_numpy(np.array([[i for i in dict_obj2imagenet_id if j in dict_obj2imagenet_id[i]] for j in p1]))
                #     # t1 = torch.from_numpy(np.array([[i for i in dict_obj2imagenet_id if j in dict_obj2imagenet_id[i]] for j in target[0]]))
                #     for num, j1 in enumerate(p1):
                #         for i1 in dict_obj2imagenet_id:
                #             if j1 in dict_obj2imagenet_id[i1]:
                #                 mapped_pred.append(i1)
                #                 break
                #         if sample * args.batch_size + num == len(mapped_pred):
                #             mapped_pred.append("None")
                #
                #     for j2 in target[0]:
                #         for i2 in dict_obj2imagenet_id:
                #             if j2 in dict_obj2imagenet_id[i2]:
                #                 mapped_target.append(i2)
                #                 break

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

                if "objectnet" == datatype:
                    for k in range(len(target)):
                        correct = pred.eq(target[k].cuda().view(1, -1).expand_as(pred))
                        top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
                        top5_cnt += correct[:5].view(-1).float().sum(0, keepdim=True)
                elif "imagenet" == datatype:
                    correct = pred.eq(target.cuda().view(1, -1).expand_as(pred))
                    top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
                    top5_cnt += correct[:5].view(-1).float().sum(0, keepdim=True)

                cnt += batch_size

            print('{} Overlapping ImgNet'.format(datatype), sample + 1, "/%d  top 1: %.5f, top 5: %.5f" %
                  (len(val_loader), top1_cnt * 100.0 / cnt, top5_cnt * 100.0 / cnt))

            # if args.CM:
            #     conf_mat = confusion_matrix(mapped_target, mapped_pred, list(dict_obj2imagenet_id.keys()) + ["None"])
            #     np.set_printoptions(precision=2)
            #     # Plot non-normalized confusion matrix
            #     plt.figure()
            #     plot_confusion_matrix(conf_mat, classes=list(dict_obj2imagenet_id.keys()),
            #                           title='Ours_Confusion')



if __name__ == '__main__':
    # checkpoint_path=""
    # load_emb=""

    # test(checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/xentfinetune/tune_alllayer_alldata_res152_aug/train_2020-04-21_13:55:13_23ed7ea4_rotation_True/model_best.pth.tar",
    #      load_emb=True,
    #      emb_name="152_tune_xent_aug",
    #      use_mlp=False,
    #      arch='resnet152',
    #      save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet', 'objectnet'])
    # 34.909 58.926   ImageNet: 73.34  91.86


    # test(checkpoint_path="/proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/model_best.pth.tar",
    #      load_emb=True,
    #      emb_name="baseline-152",
    #      use_mlp=False,
    #      arch='resnet152',
    #      save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet', 'objectnet'])
    # # # 32.33014, top 5: 55.39464  ImageNet: top 1: 75.74200, top 5: 93.08400

    # Hulk: 500:
    # test(checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/Hulk/152_augGAN300/epoch92/model_best.pth.tar",
    #      load_emb=True,
    #      emb_name="152-aug-mixHulk-500",
    #      use_mlp=False,
    #      arch='resnet152',
    #      save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet', 'objectnet'])
    "/proj/vondrick/mcz/ModelInvariant/SavedModels/Hulk/152_augGAN300/epoch92/epoch90_checkpoint.pth.tar"

    test(checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/Hulk/152_augGAN500/train_2020-04-30_14:19:07_28113a3a_rotation_True/model_best_96.pth.tar",
         load_emb=False,
         emb_name="152-aug-mixHulk-500-96",
         use_mlp=False,
         arch='resnet152',
         save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet', 'objectnet'])


    # test(checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/MLP_finetune_noview_152_tune32nomlp_exhaust_epoch600/train_2020-03-31_16:45:04_0e3f7ed9_rotation_False/model_best.pth.tar",
    #      load_emb=True,
    #      emb_name="152_tune32_nomlp",
    #      use_mlp=False,
    #      arch='resnet152',
    #      save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet'])
    # imagenet:  top 1: 76.42000, top 5: 92.76400      objectnet: 33.945  55.6853


    # RUNNING
    # test(
    #     checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/MLP_finetune_noview_152_tune32nomlp_exhaust-aug/train_2020-04-18_13:57:07_f0c28ad6_rotation_True/model_best.pth.tar",
    #     load_emb=True,
    #     emb_name="152_tune32_nomlp_aug",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500)
    ## object: 35.878  59.104
    # test(
    #     checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/MLP_finetune_noview_152_tune32nomlp_exhaust-aug/train_2020-04-18_13:57:07_f0c28ad6_rotation_True/model_best.pth.tar",
    #     load_emb=True,
    #     emb_name="152_tune32_nomlp_aug",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet'])
    # # imagenet: top 1: 73.08000, top 5: 91.18400


    # test(
    #     checkpoint_path='/proj/vondrick/mcz/ModelInvariant/SavedModels/152_augGAN100/train_2020-04-28_08:07:47_fdd8ef6f_rotation_True/model_best.pth.tar',
    #     load_emb=False,
    #     emb_name="152_tune32_nomlp_aug_GANmix",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500)
    # Objectnet: epoch 47:  34.386   57.10
    # test(
    #     checkpoint_path='/proj/vondrick/mcz/ModelInvariant/SavedModels/152_augGAN100/train_2020-04-28_08:07:47_fdd8ef6f_rotation_True/model_best.pth.tar',
    #     load_emb=True,
    #     emb_name="152_tune32_nomlp_aug_GANmix",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet'])
    # Objectnet: epoch 47:  34.386   57.10   ImageNet: 71.53000, top 5: 89.91199

    # test(
    #     checkpoint_path='/proj/vondrick/mcz/ModelInvariant/SavedModels/152_augGAN100/train_2020-04-29_00:04:24_2f68abbe_rotation_True/model_best.pth.tar',
    #     load_emb=True,
    #     emb_name="152_tune32_nomlp_aug_GANmix",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet', 'objectnet'])
    # epoch 101:  Imagenet: top 1: 73.78000, top 5: 90.90800   ObjectNet: top 1:  # 36.24960, top 5: 58.07042 has bug9.82588, top 5: 15.75378


    # test(
    #     checkpoint_path='/proj/vondrick/mcz/ModelInvariant/SavedModels/152_augGAN100/train_2020-04-29_00:04:24_2f68abbe_rotation_True/model_best.pth.tar',
    #     load_emb=True,
    #     emb_name="152_tune32_nomlp_aug_GANmix_100",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['imagenet', 'objectnet'])
    # # epoch 117:  Imagenet: top 1: 73.78000, top 5: 90.90800   ObjectNet: top 1:  # 36.24960, top 5: 58.07042 has bug9.82588, top 5: 15.75378



    # !!!
    # Highest one on ImageNet
    # test(
    #     checkpoint_path="/proj/vondrick/mcz/ModelInvariant/SavedModels/MLP_finetune_noview_152_tune32_exhaust_longer/train_2020-03-27_23:02:22_abaf732d_rotation_False/model_best.pth.tar",
    #     load_emb=True,
    #     emb_name="152_tune32_nomlp_highest",
    #     use_mlp=False,
    #     arch='resnet152',
    #     save_separate=True, exclude_nonoverlap=False, batch_size=500, test_dataset=['objectnet'])  #,
    # Imagenet: 75.80600, top 5: 93.07200  ObjectNet: 33.19694, top 5: 55.49155

