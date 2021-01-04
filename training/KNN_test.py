import argparse
import os
import random
import shutil
import time
import warnings
import numpy
import numpy as np
import json
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
import json

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import pickle, socket, json

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_interval', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--emb_name', default='', type=str, metavar='emb_name',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--KNN', default=1000, type=int,
                    help='number of K nearest neighbors')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-d', '--dataset', default='', type=str, metavar='dataset',
                    help='dataset objectnet/imageneta')
# parser.add_argument('--backup_output_dir',          type=str,       default='/local/rcs/mcz/ImageNet-Data/SavedModels',  help='')
parser.add_argument('--rotate',         action='store_true')
parser.add_argument('--visualize',         action='store_true')
parser.add_argument('--exclude_nonoverlap',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')

parser.add_argument('--mlp',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--zoom_in_center',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--CM',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--LR',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--save_separate',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--crop',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--test_imgnet',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')

best_acc1 = 0


if socket.gethostname() == 'hulk':
    root_path = "/local/rcs/mcz"
elif 'cv' in socket.gethostname():
    root_path = "/proj/vondrick/mcz/ModelInvariant/Pretrained/KNN_emb"

print("save emb path", root_path)
def main():
    args = parser.parse_args()

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    from learning.resnet import resnet50, resnet18, resnet152, MLPNetWide
    model=None
    if args.arch == 'resnet50':
        model = resnet50(normalize=False)
    elif args.arch == 'resnet18':
        model = resnet18(normalize=False)
    elif args.arch == 'resnet152':
        model = resnet152(normalize=False)
        if args.mlp:
            MLP = MLPNetWide(2048)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model) #cpu
            if args.mlp:
                MLP = torch.nn.DataParallel(MLP).cuda()


    # optionally resume from a checkpoint
    print("could start resume")
    if args.resume:
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

    if args.mlp:
        args.MLP = MLP

    # Data loading code
    import socket
    if socket.gethostname() == 'hulk':
        # traindir = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train_clustered-3.5'
        # traindir = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train_clustered-C3.5'
        # traindir = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train'
        traindir = '/local/rcs/mcz/ImageNet-Data/train'
        valdir = '/local/rcs/mcz/ImageNet-Data/val'
        imageneta_valdir = ''
        obj_valdir ='/local/rcs/shared/objectnet-1.0/overlap_category_test'

    elif 'cv' in socket.gethostname():
        # traindir = '/local/vondrick/cz/cut_imgnet-3.5/train_clustered-3.5'
        traindir = '/proj/vondrick/mcz/ImageNet-Data/train'
        valdir = '/proj/vondrick/mcz/ImageNet-Data/val'
        obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test'
        obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_noborder'
        # obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_old'
        imageneta_valdir = '/proj/vondrick/amogh/imagenet-a/'
        path_visualization_results = '/proj/vondrick/amogh/results/interpret'

    elif socket.gethostname() == 'amogh':
        traindir = '/local/vondrick/cz/cut_img-both3.5/train_clustered-C3.5'
        valdir = '/local/vondrick/cz/ImageNet/val'
        obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test'
        imageneta_valdir = '/media/amogh/Stuff/Data/natural-adversarial-examples-imageneta/imagenet-a/imagenet-a/'
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # print("Loading Train dataloader")
    # from data.test_imagefolder import MyImageFolder
    # train_loader = torch.utils.data.DataLoader(
    #     MyImageFolder(traindir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,  # Can be false
    #     num_workers=50, pin_memory=True)


    # print("Loading Val dataloader")
    # val_loader = torch.utils.data.DataLoader(
    #     MyImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=1024, shuffle=True,
    #     num_workers=20, pin_memory=True)

    # from data.KNN_dataloader import ObjectNetClassWiseLoader
    # objectnet_loader = torch.utils.data.DataLoader(
    #     ObjectNetClassWiseLoader(obj_valdir),
    #     batch_size=1, shuffle=False,
    #     num_workers=10, pin_memory=True)
    from data.KNN_dataloader import ObjectNetClassWiseLoader, ObjectNetLoader, ImagenetALoader
    # objectnet_loader = torch.utils.data.DataLoader(
    #     ObjectNetClassWiseLoader(obj_valdir),
    #     batch_size=1, shuffle=False,
    #     num_workers=10, pin_memory=True)


    if args.dataset == 'objectnet':
        print('obj data', obj_valdir)
        test_loader = torch.utils.data.DataLoader(
            ObjectNetLoader(obj_valdir),
            batch_size=args.batch_size, shuffle=True,
            num_workers=10, pin_memory=True)
    elif args.dataset == 'imageneta':
        test_loader = torch.utils.data.DataLoader(
            ImagenetALoader(imageneta_valdir),
            batch_size=100, shuffle=True,
            num_workers=10, pin_memory=True)

    # print('args.emb_name', args.emb_name)
    # print('args.emb_name', args.emb_name.replace('#',''))
    # print('args.emb_name', args.emb_name)
    output_list = None
    from data.test_imagefolder import MyImageFolder, ImageNetTrainOverlapObject
    try:
        if args.save_separate and not args.exclude_nonoverlap:
            foldername = '{}_bl_representation_{}'.format(args.arch, args.emb_name)
            openpath = '{}/{}'.format(root_path, foldername)
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
            embed_list = np.concatenate(embed_list, 0)
            # output_list = torch.cat(output_list, 0)
            label_list = np.concatenate(label_list, axis=0)
            output_list = None
            print("finish concat")

        else:
            openpath = '{}/{}_bl_representation_{}.pkl'.format(root_path, args.arch, args.emb_name)
            if args.exclude_nonoverlap:
                openpath = '{}/{}_bl_representation_{}{}.pkl'.format(root_path, args.arch, args.emb_name, '_exclude')
            with open(openpath,
                      'rb') as f:
                print("loading saved emb")
                train_all = pickle.load(f)
                print("finished load emb")
                embed_list = train_all['embedding']
                label_list = train_all['label']
                output_list = train_all['output']
                print("load old model successfully; num of data={}".format(embed_list.shape[0]))
    except:

        if args.exclude_nonoverlap:
            train_loader = torch.utils.data.DataLoader(
                ImageNetTrainOverlapObject(traindir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=True,  # Can be false
                num_workers=30, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                MyImageFolder(traindir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=True,  # Can be false
                num_workers=30, pin_memory=True)

        print("doing inference on whole data training set")
        embed_list, label_list, output_list = train_embed(train_loader, model, args)

        if args.save_separate and not args.exclude_nonoverlap:
            foldername = '{}_bl_representation_{}'.format(args.arch, args.emb_name)
            openpath = '{}/{}'.format(root_path, foldername)
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
            embed_list = np.concatenate(embed_list, 0)
            # output_list = torch.cat(output_list, 0)
            label_list = np.concatenate(label_list, axis=0)
            output_list=None

    if args.test_imgnet:
        # Test ImageNet
        val_loader = torch.utils.data.DataLoader(
            MyImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=1024, shuffle=True,
            num_workers=20, pin_memory=True)
        if args.LR:
            LogisticRegression_test_objectnet_shuffle(val_loader, model, args,
                                                      [embed_list, label_list, output_list])
        else:
            KNN_test(val_loader, model, args, [embed_list, label_list, output_list])
        # return
        # KNN_test_objectnet(objectnet_loader, model, args, [embed_list, label_list, output_list])

    if args.zoom_in_center:
        # TODO: but notice that the imagenet dictionary is not changing.
        # zoom_list = [1, 2, 3, 4, 5, 6]
        zoom_list = [1 + 0.2 * iii for iii in range(0, 10)]
        for each_z in zoom_list:
            print("now zoom in {} Times".format(each_z - 1))
            if args.exclude_nonoverlap:
                print("remove overlapping")

            if args.dataset == 'objectnet':
                composed_transform = transforms.Compose([
                    transforms.Compose([
                        transforms.Resize(int(256 * each_z)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
                ])
                test_loader = torch.utils.data.DataLoader(
                    ObjectNetLoader(obj_valdir, composed_transform=composed_transform),
                    batch_size=200, shuffle=True,
                    num_workers=10, pin_memory=True)

                if args.LR:
                    LogisticRegression_test_objectnet_shuffle(test_loader, model, args,
                                                          [embed_list, label_list, output_list])
                else:
                    KNN_test_objectnet_shuffle(test_loader, model, args, [embed_list, label_list, output_list])
        return

    else:
        if args.dataset == 'objectnet':
            print('obj data', obj_valdir)
            composed_transform = None
            if args.crop:
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
                batch_size=200, shuffle=True,
                num_workers=10, pin_memory=True)
        elif args.dataset == 'imageneta':
            test_loader = torch.utils.data.DataLoader(
                ImagenetALoader(imageneta_valdir),
                batch_size=100, shuffle=True,
                num_workers=10, pin_memory=True)

        if args.LR:
            LogisticRegression_test_objectnet_shuffle(test_loader, model, args,
                                                      [embed_list, label_list, output_list])
        else:

            KNN_test_objectnet_shuffle(test_loader, model, args, [embed_list, label_list, output_list])

from utils import normalize_l2

def train_embed(train_loader, model, args):
    model.eval()
    if args.mlp:
        args.MLP.eval()

    embed_list = []
    output_list = []
    label_list = []
    print("total=", len(train_loader))
    split_flag=False
    if args.save_separate and not args.exclude_nonoverlap:
        foldername = '{}_bl_representation_{}'.format(args.arch, args.emb_name)
        openpath = '{}/{}'.format(root_path, foldername)
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
                    print("Dumping embeddings at: ", root_path)
                embed_list = []
                output_list = []
                label_list = []

            # if i==50:
            #     break

    if not split_flag:
        embed_list = torch.cat(embed_list, 0)
        output_list = torch.cat(output_list, 0)
        label_list = np.concatenate(label_list, axis=0)

        train_fea = {'embedding': embed_list.numpy(), 'label': label_list, 'output': output_list}
        # with open('../{}_representation.pkl'.format(args.arch), 'wb') as f:
        os.makedirs(root_path, exist_ok=True)

        openpath = '{}/{}_bl_representation_{}.pkl'.format(root_path, args.arch, args.emb_name)
        if args.exclude_nonoverlap:
            openpath = '{}/{}_bl_representation_{}{}.pkl'.format(root_path, args.arch, args.emb_name, '_exclude')


        with open(openpath, 'wb') as f:
            pickle.dump(train_fea, f, protocol=4)
            print("Dumping embeddings at: ", root_path)

    return embed_list, label_list, output_list


def KNN_test(val_loader, model, args, train_list):

    model.eval()  # crucial for use when loading a model with bn, if we don't run eval, its very bad
    if args.mlp:
        args.MLP.eval()

    if len(train_list)==2:
        emb_list, label_list = train_list
    else:
        emb_list, label_list, output_list = train_list

    try:
        emb_list = torch.from_numpy(emb_list)
        # output_list = torch.from_numpy(output_list)
    except:
        pass

    cnt=0
    top1_cnt = 0
    top5_cnt = 0

    test_hidden = True # TODO: I believe if this is False, i.e., use output feature, results in random prediction, let's see

    with torch.no_grad():
        for i, (images, target, path) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            output, fea, norm = model(images)
            if args.mlp:
                fea = args.MLP(fea)

            # fea, _ = normalize_l2(fea)
            if test_hidden:
                score_mat = torch.mm(fea.cpu(), emb_list.t())  # test_num * train_num
            else:
                score_mat = torch.mm(output.cpu(), output_list.t())  # test_num * train_num
            # print('topk', TOPK_NN)
            _, pred = torch.topk(score_mat, k=args.KNN, dim=1, largest=True, sorted=True)
            # print("done")
            pred_np = pred.data.cpu().numpy()
            # print("sim", score_mat)
            # print("pred", pred)


            pred_label = []

            for each in range(pred_np.shape[0]):
                label_gt = target[each]
                # print('label gt', label_gt)
                cnt += 1

                index = pred[each]
                retrieved_labels = label_list[index.cpu().numpy()]

                retrieved_labels = np.sort(retrieved_labels)
                # print("retrieved labels", retrieved_labels)

                label_num_dict = {}
                last = None
                num = None

                for kk in range(retrieved_labels.shape[0]):
                    if kk==0:
                        last = retrieved_labels[0]
                        num=1

                    else:
                        if retrieved_labels[kk] == last:
                            num += 1
                        else:
                            label_num_dict[retrieved_labels[kk-1]] = num
                            num = 1
                            last = retrieved_labels[kk]

                    if kk == retrieved_labels.shape[0] - 1:
                        label_num_dict[retrieved_labels[kk]] = num


                # print("label dict num", label_num_dict)
                # Sort dict from most frequent to least frequent
                sort_topk_neighbor = sorted(label_num_dict.items(), key=lambda kv: (-kv[1], kv[0]))

                # print("sort topk neighbor", sort_topk_neighbor)

                try:
                    if sort_topk_neighbor[0][0] == label_gt:
                        top1_cnt += 1
                except:
                    print(sort_topk_neighbor, label_gt, top1_cnt)

                for nnn in range(5):
                    try:
                        if sort_topk_neighbor[nnn][0] == label_gt:
                            top5_cnt += 1
                            break
                    except:
                        # print("top 5 but number less than 5")
                        break

            print(i, "/%d  top 1: %.5f, top 5: %.5f" % (len(val_loader), top1_cnt*100.0/cnt, top5_cnt*100.0/cnt))

    # cnt = 0
    # top1_cnt = 0
    # top5_cnt = 0
    # with torch.no_grad():
    #     for i, (images, target, path) in enumerate(val_loader):
    #         if args.gpu is not None:
    #             images = images.cuda(args.gpu, non_blocking=True)
    #         # target = target.cuda(args.gpu, non_blocking=True)
    #
    #         output, fea, norm = model(images)
    #         # fea, _ = normalize_l2(fea)
    #         if not test_hidden:
    #             score_mat = torch.mm(fea.cpu(), emb_list.t())  # test_num * train_num
    #         else:
    #             score_mat = torch.mm(output.cpu(), output_list.t())  # test_num * train_num
    #         # print('topk', TOPK_NN)
    #         _, pred = torch.topk(score_mat, k=TOPK_NN, dim=1, largest=True, sorted=False)
    #         # print("done")
    #         pred_np = pred.data.cpu().numpy()
    #         # print("sim", score_mat)
    #         # print("pred", pred)
    #
    #
    #         pred_label = []
    #
    #         for each in range(pred_np.shape[0]):
    #             label_gt = target[each]
    #             # print('label gt', label_gt)
    #             cnt += 1
    #
    #             index = pred[each]
    #             retrieved_labels = label_list[index.cpu().numpy()]
    #
    #             retrieved_labels = np.sort(retrieved_labels)
    #             # print("retrieved labels", retrieved_labels)
    #
    #             label_num_dict = {}
    #             last = None
    #             num = None
    #
    #             for kk in range(retrieved_labels.shape[0]):
    #                 if kk==0:
    #                     last = retrieved_labels[0]
    #                     num=1
    #
    #                 else:
    #                     if retrieved_labels[kk] == last:
    #                         num += 1
    #                     else:
    #                         label_num_dict[retrieved_labels[kk-1]] = num
    #                         num = 1
    #                         last = retrieved_labels[kk]
    #
    #                 if kk == retrieved_labels.shape[0] - 1:
    #                     label_num_dict[retrieved_labels[kk]] = num
    #
    #
    #             # print("label dict num", label_num_dict)
    #             # Sort dict from most frequent to least frequent
    #             sort_topk_neighbor = sorted(label_num_dict.items(), key=lambda kv: (-kv[1], kv[0]))
    #
    #             # print("sort topk neighbor", sort_topk_neighbor)
    #
    #             try:
    #                 if sort_topk_neighbor[0][0] == label_gt:
    #                     top1_cnt += 1
    #             except:
    #                 print(sort_topk_neighbor, label_gt, top1_cnt)
    #
    #             for nnn in range(5):
    #                 try:
    #                     if sort_topk_neighbor[nnn][0] == label_gt:
    #                         top5_cnt += 1
    #                         break
    #                 except:
    #                     # print("top 5 but number less than 5")
    #                     break
    #
    #         print(i, "/%d  top 1: %.5f, top 5: %.5f" % (len(val_loader), top1_cnt*100.0/cnt, top5_cnt*100.0/cnt))
def LogisticRegression_test_objectnet_shuffle(val_loader, model, args, train_list):
    model.eval()

    if len(train_list)==2:
        emb_list, label_list = train_list
    else:
        emb_list, label_list, output_list = train_list

    try:
        emb_list = torch.from_numpy(emb_list)
        label_list = torch.from_numpy(label_list)
        output_list = torch.from_numpy(output_list)
    except:
        pass

    LR = LogisticRegression(2048, 1000)
    LR.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LR.parameters(), lr=0.01)

    if args.emb_name == '152bl':
        LR_path = '/proj/vondrick/augustine/checkpointBL.pth.tar'
    else:
        LR_path = '/proj/vondrick/augustine/checkpoint.pth.tar'
    
    if os.path.exists(LR_path):
        checkpoint = torch.load(LR_path)
        LR.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(LR_path, checkpoint['epoch']))
    else:
        b_sz = 320292
        LR.train()
        
        for epoch in tqdm(range(500), desc='training Logistic Regressor'):
            for i in range(0, len(emb_list),b_sz):
                batch = emb_list[i:i+b_sz]
                batch = batch.cuda() 
                label = label_list[i:i+b_sz]
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
            'optimizer' : optimizer.state_dict(),
        }, False, filename=fname, experiment_backup_folder='/proj/vondrick/augustine/')

    cnt=0
    top1_cnt = 0
    top5_cnt = 0

    if args.visualize:
        labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
        with open(labels_path) as json_data:
            idx_to_labels = json.load(json_data)
    print("Loading ObjectNet-ImageaNet dictionary")
    with open('preprocessing/obj2imgnet_id.txt') as f:
        dict_obj2imagenet_id = json.load(f)


    test_hidden=True
    print("start evaluation")
    LR.eval()

    if args.CM:
        mapped_pred = []
        mapped_target = []
    pbar_prefix = 'Testing on '+ str(args.dataset)
    with torch.no_grad():
        for sample, example in tqdm(enumerate(val_loader),total=len(val_loader),desc=pbar_prefix):
            images = example['images']
            target = example['labels']
            path = example['path']
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            
            o, fea, n = model(images)
            maxk = 5
            batch_size = target[0].shape[0]

            output = LR(fea)

            if args.CM:
                _, p1 = output.topk(1)
                # p1 = torch.from_numpy(np.array([[i for i in dict_obj2imagenet_id if j in dict_obj2imagenet_id[i]] for j in p1]))
                # t1 = torch.from_numpy(np.array([[i for i in dict_obj2imagenet_id if j in dict_obj2imagenet_id[i]] for j in target[0]]))
                for num, j1 in enumerate(p1):
                    for i1 in dict_obj2imagenet_id:
                        if j1 in dict_obj2imagenet_id[i1]:
                            mapped_pred.append(i1)
                            break
                    if sample * args.batch_size + num == len(mapped_pred):
                        mapped_pred.append("None")


                for j2 in target[0]:
                    for i2 in dict_obj2imagenet_id:
                        if j2 in dict_obj2imagenet_id[i2]:
                            mapped_target.append(i2)
                            break

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            for k in range(len(target)):
                correct = pred.eq(target[k].cuda().view(1, -1).expand_as(pred))
                top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
                top5_cnt += correct[:5].view(-1).float().sum(0, keepdim=True)
                        
            cnt += batch_size

        print('{} Overlapping ImgNet'.format(args.dataset), sample+1, "/%d  top 1: %.5f, top 5: %.5f" %
              (len(val_loader), top1_cnt*100.0/cnt, top5_cnt*100.0/cnt))
            
        if args.CM:
            conf_mat = confusion_matrix(mapped_target, mapped_pred, list(dict_obj2imagenet_id.keys()) + ["None"])
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(conf_mat, classes=list(dict_obj2imagenet_id.keys()),
                                  title='Ours_Confusion')



def KNN_test_objectnet_shuffle(val_loader, model, args, train_list, path_visualization_results=None):
    # crucial for use when loading a model with bn, if we don't run eval, its very bad
    """This test function, randomly exhaust all the objectnet examples"""
    model.eval()
    if args.mlp:
        args.MLP.cuda()

    if len(train_list)==2:
        emb_list, label_list = train_list
    else:
        emb_list, label_list, output_list = train_list

    try:
        emb_list = torch.from_numpy(emb_list)
        output_list = torch.from_numpy(output_list)
    except:
        pass

    cnt=0
    top1_cnt = 0
    top1_cnt_old = 0
    top5_cnt = 0

    test_hidden=True
    
    if args.visualize:
        labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
        with open(labels_path) as json_data:
            idx_to_labels = json.load(json_data)


    with torch.no_grad():
        for i, example in enumerate(val_loader):
            images = example['images']
            target = example['labels']
            path = example['path']
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)


            # print("target", target)
            # exit(0)

            # print('img', images.size())
            output, fea, norm = model(images)
            if args.mlp:
                fea = args.MLP(fea)

            # fea, _ = normalize_l2(fea)
            # fea = MLP(fea)

            if test_hidden:
                score_mat = torch.mm(fea.cpu(), emb_list.t())  # test_num * train_num
            else:
                score_mat = torch.mm(output.cpu(), output_list.t())  # test_num * train_num
            # print('topk', TOPK_NN)

            _, pred = torch.topk(score_mat, k=args.KNN, dim=1, largest=True, sorted=False)
            # print("done")
            pred_np = pred.data.cpu().numpy()
            # print("sim", score_mat)
            # print("pred", pred)

            # print("Getting scores: ", score, pred)

            # print('target b4', target)
            # The way to organize is that, i-th sublist in the target, contains the i-th label for all test examples
            target_num = len(target)
            target = [each.numpy() for each in target]
            # print('target after', target)

            pred_label = []

            for each in range(pred_np.shape[0]):  # for each test images
                # label_gt_all = target[each]
                label_gt_list = []
                # for mogu in label_gt_all:
                #     label_gt_list.append(mogu.numpy()[0])
                # print('label gt', label_gt)
                for kk in range(target_num):
                    label_gt_list.append(target[kk][each])  # Since target size is: "the multilabel dim, number of test examples"

                cnt += 1

                index = pred[each]
                retrieved_labels = label_list[index.cpu().numpy()]

                retrieved_labels = np.sort(retrieved_labels)
                # print("retrieved labels", len(retrieved_labels))

                label_num_dict = {}
                last = None
                num = None

                for kk in range(retrieved_labels.shape[0]):
                    if kk==0:
                        last = retrieved_labels[0]
                        num=1

                    else:
                        if retrieved_labels[kk] == last:
                            num += 1
                        else:
                            label_num_dict[retrieved_labels[kk-1]] = num
                            num = 1
                            last = retrieved_labels[kk]

                    if kk == retrieved_labels.shape[0] - 1:
                        label_num_dict[retrieved_labels[kk]] = num
                # print(label_num_dict)

                # print("label dict num", label_num_dict)
                # Sort dict from most frequent to least frequent
                sort_topk_neighbor = sorted(label_num_dict.items(), key=lambda kv: (-kv[1], kv[0]))

                # print("sort topk neighbor", sort_topk_neighbor, len(label_gt_list), label_gt_list,)


                # Most frequent label for this test image is in sort_topk_neighbour[0][0], and gt is label_gt_list, so check if our top prediction is among
                if args.visualize:
                    #path_viz = os.path.join(path_visualization_results,'knn' )
                    #path_viz = os.path.join(path_visualization_results,<category_objnet>/knn_prediction/<basename> )
                    

                    path_image = path[each]
                    list_path_split = path_image.split('/')
                    objnet_category = list_path_split[-2]
                    image_basename = list_path_split[-1]
                    path_to_append = objnet_category + '/' + 'knn_output' + '/' + image_basename
                    path_destination_image = os.path.join(path_visualization_results, path_to_append)
                    # print("__path__",path[each])
                    # path_dest_visualization_img = os.path.join(path_viz, 'dfd')
                    
                    true_label = objnet_category
                    predicted_labels_id = [kv[0] for kv in sort_topk_neighbor[:5]]
                    predicted_labels = [idx_to_labels[str(id)][1] for id in predicted_labels_id]
                    predicted_label = objnet_category                
                    
                    temp = images[each].detach().numpy()
                    temp = np.moveaxis(temp, 0, 2)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    temp = np.clip((temp*std)+mean,0,1)
                    f,ax = plt.subplots()
                    ax.imshow(temp)
                    str_title = 'Truth Objnet category: {} \n Predicted labels: {} '.format(true_label,predicted_labels)
                    f.suptitle(str_title, fontsize=12)
                    print("savign at : ", path_destination_image)
                    if i%500 ==0:
                        print(i, 'done') 
                    dirname = os.path.dirname(path_destination_image)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    f.savefig(path_destination_image,bbox_inches='tight')
                    plt.close()

                top1_cnt_old=top1_cnt
                try:
                    for label_gt in label_gt_list:
                        if sort_topk_neighbor[0][0] == label_gt:
                            top1_cnt += 1
                            break
                except:
                    print(sort_topk_neighbor, label_gt_list, top1_cnt)

                if top1_cnt-top1_cnt_old > 1:
                    print(top1_cnt-top1_cnt_old, label_gt_list)

                # Make sure if see one category correct, then stop loop
                flag = True
                for nnn in range(5):
                    try:
                        if flag == False:
                            break
                        for label_gt in label_gt_list:
                            if sort_topk_neighbor[nnn][0] == label_gt:
                                top5_cnt += 1
                                flag = False
                                break
                    except:
                        # print("top 5 but number less than 5")
                        break

            print('{} Overlapping ImgNet'.format(args.dataset), i, "/%d  top 1: %.5f, top 5: %.5f" %
                  (len(val_loader), top1_cnt*100.0/cnt, top5_cnt*100.0/cnt))

def KNN_test_objectnet(val_loader, model, args, train_list):
    # crucial for use when loading a model with bn, if we don't run eval, its very bad
    model.eval()

    if len(train_list)==2:
        emb_list, label_list = train_list
    else:
        emb_list, label_list, output_list = train_list

    try:
        emb_list = torch.from_numpy(emb_list)
        output_list = torch.from_numpy(output_list)
    except:
        pass

    cnt=0
    top1_cnt = 0
    top5_cnt = 0

    test_hidden=True
    with torch.no_grad():
        for i, example in enumerate(val_loader):
            images = example['images']
            target = example['labels']
            path = example['path']
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)


            # print("target", target)
            # exit(0)

            output, fea, norm = model(images[0])

            if test_hidden:
                score_mat = torch.mm(fea.cpu(), emb_list.t())  # test_num * train_num
            else:
                score_mat = torch.mm(output.cpu(), output_list.t())  # test_num * train_num
            # print('topk', TOPK_NN)
            _, pred = torch.topk(score_mat, k=args.KNN, dim=1, largest=True, sorted=False)
            # print("done")
            pred_np = pred.data.cpu().numpy()
            # print("sim", score_mat)
            # print("pred", pred)


            pred_label = []

            for each in range(pred_np.shape[0]):
                label_gt_all = target[each]
                label_gt_list = []
                for mogu in label_gt_all:
                    label_gt_list.append(mogu.numpy()[0])
                # print('label gt', label_gt)
                cnt += 1

                index = pred[each]
                retrieved_labels = label_list[index.cpu().numpy()]

                retrieved_labels = np.sort(retrieved_labels)
                #print("retrieved labels", len(retrieved_labels))

                label_num_dict = {}
                last = None
                num = None

                for kk in range(retrieved_labels.shape[0]):
                    if kk==0:
                        last = retrieved_labels[0]
                        num=1

                    else:
                        if retrieved_labels[kk] == last:
                            num += 1
                        else:
                            label_num_dict[retrieved_labels[kk-1]] = num
                            num = 1
                            last = retrieved_labels[kk]

                    if kk == retrieved_labels.shape[0] - 1:
                        label_num_dict[retrieved_labels[kk]] = num


                # print("label dict num", label_num_dict)
                # Sort dict from most frequent to least frequent
                sort_topk_neighbor = sorted(label_num_dict.items(), key=lambda kv: (-kv[1], kv[0]))

                # print("sort topk neighbor", sort_topk_neighbor)

                try:
                    for label_gt in label_gt_list:
                        if sort_topk_neighbor[0][0] == label_gt:
                            top1_cnt += 1
                            break
                except:
                    print(sort_topk_neighbor, label_gt_list, top1_cnt)

                # Make sure if see one category correct, then stop loop
                flag = True
                for nnn in range(5):
                    try:
                        if flag == False:
                            break
                        for label_gt in label_gt_list:
                            if sort_topk_neighbor[nnn][0] == label_gt:
                                top5_cnt += 1
                                flag = False
                                break
                    except:
                        # print("top 5 but number less than 5")
                        break

            print('ObjectNet Overlapping ImgNet', i, "/%d  top 1: %.5f, top 5: %.5f" %
                  (len(val_loader), top1_cnt*100.0/cnt, top5_cnt*100.0/cnt))


def KNN_test_NCM(val_loader, model, args, train_list):
    # Only retrieve the cluster center
    emb_list, label_list = train_list
    emb_list = torch.from_numpy(emb_list)

    model.eval()  # crucial for use when loading a model with bn, if we don't run eval, its very bad

    cnt=0
    top1_cnt = 0
    top5_cnt = 0

    with torch.no_grad():
        for i, (images, target, path) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            output, fea, norm = model(images)
            fea, _ = normalize_l2(fea)
            score_mat = torch.mm(fea.cpu(), emb_list.t())  # test_num * train_num
            # print('topk', TOPK_NN)
            _, pred = torch.topk(score_mat, k=args.KNN, dim=1, largest=True, sorted=False)
            # print("done")
            pred_np = pred.data.cpu().numpy()
            # print("sim", score_mat)
            # print("pred", pred)


            pred_label = []

            for each in range(pred_np.shape[0]):
                label_gt = target[each]
                # print('label gt', label_gt)
                cnt += 1

                index = pred[each]
                retrieved_labels = label_list[index.cpu().numpy()]

                retrieved_labels = np.sort(retrieved_labels)
                # print("retrieved labels", retrieved_labels)

                label_num_dict = {}
                last=None
                num=None
                for kk in range(retrieved_labels.shape[0]):
                    if kk==0:
                        last = retrieved_labels[0]
                        num=1

                    else:
                        if retrieved_labels[kk] == last:
                            num += 1
                        else:
                            label_num_dict[retrieved_labels[kk-1]] = num
                            num = 1
                            last = retrieved_labels[kk]

                    if kk == retrieved_labels.shape[0] - 1:
                        label_num_dict[retrieved_labels[kk]] = num


                # print("label dict num", label_num_dict)
                # Sort dict from most frequent to least frequent
                sort_topk_neighbor = sorted(label_num_dict.items(), key=lambda kv: (-kv[1], kv[0]))

                # print("sort topk neighbor", sort_topk_neighbor)

                if sort_topk_neighbor[0][0] == label_gt:
                    top1_cnt += 1

                for nnn in range(5):
                    if sort_topk_neighbor[nnn][0] == label_gt:
                        top5_cnt += 1
                        break

            print(i, "/%d  top 1: %.5f, top 5: %.5f" % (len(val_loader), top1_cnt*100.0/cnt, top5_cnt*100.0/cnt))

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).cuda()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def LogisticRegression_test_objectnet_shuffle(val_loader, model, args, train_list):
    model.eval()

    if len(train_list) == 2:
        emb_list, label_list = train_list
    else:
        emb_list, label_list, output_list = train_list

    try:
        emb_list = torch.from_numpy(emb_list)
        label_list = torch.from_numpy(label_list)
        output_list = torch.from_numpy(output_list)
    except:
        pass

    LR = LogisticRegression(2048, 1000)
    LR.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LR.parameters(), lr=0.01)

    if args.emb_name == '152bl':
        LR_path = '/proj/vondrick/augustine/checkpointBL.pth.tar'
    else:
        LR_path = '/proj/vondrick/augustine/checkpoint.pth.tar'

    if os.path.exists(LR_path):
        checkpoint = torch.load(LR_path)
        LR.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(LR_path, checkpoint['epoch']))
    else:
        b_sz = 320292
        LR.train()

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

        if args.emb_name == '152bl':
            fname = 'checkpoint.pth.tar'
        else:
            fname = 'checkpointBL.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "LogisticRegression",
            'state_dict': LR.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename=fname, experiment_backup_folder='/proj/vondrick/augustine/')

    cnt = 0
    top1_cnt = 0
    top5_cnt = 0

    print("Loading ObjectNet-ImageaNet dictionary")
    with open('preprocessing/obj2imgnet_id.txt') as f:
        dict_obj2imagenet_id = json.load(f)

    test_hidden = True
    print("start evaluation")
    LR.eval()

    if args.CM:
        mapped_pred = []
        mapped_target = []
    pbar_prefix = 'Testing on ' + str(args.dataset)
    with torch.no_grad():
        for sample, example in tqdm(enumerate(val_loader), total=len(val_loader), desc=pbar_prefix):
            images = example['images']
            target = example['labels']
            path = example['path']
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            o, fea, n = model(images)
            maxk = 5
            batch_size = target[0].shape[0]

            output = LR(fea)

            if args.CM:
                _, p1 = output.topk(1)
                # p1 = torch.from_numpy(np.array([[i for i in dict_obj2imagenet_id if j in dict_obj2imagenet_id[i]] for j in p1]))
                # t1 = torch.from_numpy(np.array([[i for i in dict_obj2imagenet_id if j in dict_obj2imagenet_id[i]] for j in target[0]]))
                for num, j1 in enumerate(p1):
                    for i1 in dict_obj2imagenet_id:
                        if j1 in dict_obj2imagenet_id[i1]:
                            mapped_pred.append(i1)
                            break
                    if sample * args.batch_size + num == len(mapped_pred):
                        mapped_pred.append("None")

                for j2 in target[0]:
                    for i2 in dict_obj2imagenet_id:
                        if j2 in dict_obj2imagenet_id[i2]:
                            mapped_target.append(i2)
                            break

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            for k in range(len(target)):
                correct = pred.eq(target[k].cuda().view(1, -1).expand_as(pred))
                top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
                top5_cnt += correct[:5].view(-1).float().sum(0, keepdim=True)

            cnt += batch_size

        print('{} Overlapping ImgNet'.format(args.dataset), sample + 1, "/%d  top 1: %.5f, top 5: %.5f" %
              (len(val_loader), top1_cnt * 100.0 / cnt, top5_cnt * 100.0 / cnt))

        if args.CM:
            conf_mat = confusion_matrix(mapped_target, mapped_pred, list(dict_obj2imagenet_id.keys()) + ["None"])
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(conf_mat, classes=list(dict_obj2imagenet_id.keys()),
                                  title='Ours_Confusion')


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    wrong_list = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target, path) in enumerate(val_loader):
            # print(path)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.squeeze()
            mask_correct = pred == target

            for ii in range(mask_correct.size(0)):
                if mask_correct[ii] == 0:
                    wrong_list.append(path[ii])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            # print('wrong', wrong_list)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    print(len(wrong_list))
    import pickle
    resume_name_list = args.resume.split('/')

    with open("{}_wonglist.pkl".format(resume_name_list[-3]), 'wb') as fp:
        pickle.dump(wrong_list, fp)
        print('dump_finish')
    return top1.avg, top5.avg

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).cuda()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs



if __name__ == '__main__':
    main()

