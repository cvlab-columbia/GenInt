import argparse
import os
import random

'''Baseline, Standard Xent Loss trainig '''
import shutil
import time
import warnings

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

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_interval', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=112, type=int,
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
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
parser.add_argument('--backup_output_dir', type=str, default='/local/rcs/mcz/ImageNet-Data/SavedModels', help='')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--optim', type=str, default='SGD')
best_acc1 = 0


class SimplifiedResNet18(nn.Module):
    def __init__(self):
        super(SimplifiedResNet18, self).__init__()
        resnet = models.__dict__['resnet18'](pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=256, out_features=128, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    args = parser.parse_args()

    for k, v in args.__dict__.items():  # Prints arguments and contents of config file
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
    # model = SimplifiedResNet18()
    from learning.resnet import resnet18, resnet50, resnet152
    if args.arch == 'resnet18':
        model = resnet18(oneoutput=True, num_classes=55)
    print(model)

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

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
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
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    import socket
    if socket.gethostname() == 'hulk':
        traindir = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train'
        valdir = '/local/rcs/mcz/ImageNet-Data/val'
    else:
        # elif socket.gethostname() == 'cv04':
        # traindir = '/local/vondrick/cz/ShapeNet/splited/train'
        # test_rich_dir = '/local/vondrick/cz/ShapeNet/splited/test_richview'
        # test_rare_dir = '/local/vondrick/cz/ShapeNet/splited/test_rareview'
        # test_newview = '/local/vondrick/cz/ShapeNet/splited/test_newview'

        # traindir = '/proj/vondrick/mcz/ShapeNetRender/splited_temp/train'
        # test_rich_dir = '/proj/vondrick/mcz/ShapeNetRender/splited_temp/test_richview'
        # test_rare_dir = '/proj/vondrick/mcz/ShapeNetRender/splited_temp/test_rareview'
        # test_newview = '/proj/vondrick/mcz/ShapeNetRender/splited_temp/test_newview'
        # test_diff_obj = '/proj/vondrick/mcz/ShapeNetRender/splited_temp/test_diff_obj'

        traindir = '/proj/vondrick/mcz/ShapeNetRender/splited_128/train'
        test_rich_dir = '/proj/vondrick/mcz/ShapeNetRender/splited_128/test_richview'
        test_rare_dir = '/proj/vondrick/mcz/ShapeNetRender/splited_128/test_rareview'
        test_newview = '/proj/vondrick/mcz/ShapeNetRender/splited_128/test_newview'
        test_diff_obj = '/proj/vondrick/mcz/ShapeNetRender/splited_128/test_diff_obj'

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # ### Stats for ShapeNet128
    normalize = transforms.Normalize(mean=[0.3083855, 0.30563545, 0.3026681],
    std=[0.08926533, 0.08842195, 0.08989957])
    print('Normalize!!!')
    print('train dir', traindir)
    if args.rotate:
        print("do huge rotate")
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomRotation(90, resample=2),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        print("not huge rotate")
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_rich_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_rich_dir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_rare_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_rare_dir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_newview_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_newview, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    val_newobj_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_diff_obj, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('Dataset Loading Done')

    backup_output_dir = args.backup_output_dir
    if not args.evaluate:
        os.makedirs(backup_output_dir, exist_ok=True)

    if os.path.exists(backup_output_dir):
        import uuid
        import datetime
        unique_str = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_name = args.arch + str(args.optim) + "-lr:" + str(args.lr) + "-" + timestamp
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_name)
        print("experiment folder", experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder)  #
        log_dir = os.path.join(experiment_backup_folder, "runs")

        # os.makedirs(log_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        eval_writer = SummaryWriter(log_dir=log_dir + '/validate_runs/')

    if args.evaluate:
        print("Evaluation Mode")
        validate(val_rich_loader, model, criterion, args, 'rich')
        validate(val_rare_loader, model, criterion, args, 'rare')
        validate(val_newview_loader, model, criterion, args, 'new')
        validate(val_newobj_loader, model, criterion, args, 'new')
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        # acc1, acc5 = validate(val_loader, model, criterion, args)
        acc1_rich, acc5_rich = validate(val_rich_loader, model, criterion, args, "rand_rich ")
        acc1_rare, acc5_rare = validate(val_rare_loader, model, criterion, args, "rand_rare ")
        acc1_new, acc5_new = validate(val_newview_loader, model, criterion, args, "newview ")
        acc1_newobj, acc5_newobj = validate(test_diff_obj, model, criterion, args, "newobj ")

        eval_writer.add_scalar('Test/rich top1 acc', acc1_rich, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/rich top5 acc', acc5_rich, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/rare top1 acc', acc1_rare, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/rare top5 acc', acc5_rare, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/newview top1 acc', acc1_new, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/newview top5 acc', acc5_new, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/newview top1 acc', acc1_newobj, epoch)  # * len(train_loader))
        eval_writer.add_scalar('Test/newview top5 acc', acc5_newobj, epoch)  # * len(train_loader))

        # remember best acc@1 and save checkpoint
        is_best = acc1_new > best_acc1
        best_acc1 = max(acc1_new, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='checkpoint.pth.tar', experiment_backup_folder=experiment_backup_folder)


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        if i % 1000 == 1:
            writer.add_scalar('Train/top1 acc', top1.avg, epoch)  # *len(train_loader) + i)
            writer.add_scalar('Train/top5 acc', top5.avg, epoch)  # *len(train_loader) + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, prefix):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix=prefix + 'Test: ')

    # switch to evaluate mode
    model.eval()
    if args.evaluate:
        preds = torch.from_numpy(np.array([])).cuda(args.gpu)
        label = torch.from_numpy(np.array([])).cuda(args.gpu)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(val_loader), leave=False):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            if args.evaluate:
                with torch.no_grad():
                    _, pred = output.topk(1)

                preds = torch.cat([preds.view(-1, 1).int(), pred.view(-1, 1).int()], dim=0)
                label = torch.cat([label.view(-1, 1).int(), target.view(-1, 1).int()], dim=0)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
            # break
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if args.evaluate:
            conf_mat = confusion_matrix(label.view(-1).cpu(), preds.view(-1).cpu(), list(range(0, 55)))
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(conf_mat, classes=list(range(0, 55)),
                                  title='ShapeNet_Baseline-' + prefix)
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
