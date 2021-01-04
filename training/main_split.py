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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
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
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('--optim', default='sgd', type=str, metavar='optimizer',
                    help='optimizer type')
parser.add_argument('-d', '--dataset', default='', type=str, metavar='dataset',
                    help='dataset objectnet/imageneta')
parser.add_argument('--backup_output_dir',          type=str,       default='/local/rcs/mcz/ImageNet-Data/SavedModels',  help='')
parser.add_argument('--rotate',         action='store_true')
parser.add_argument('--exclude_nonoverlap',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--zoom_in_center',         action='store_true', help='only looking at overlapping prediction of obj and imagenet')
parser.add_argument('--gan_bl',         action='store_true', help='')
parser.add_argument('--rand',         action='store_true', help='')
parser.add_argument('--rotate_gan_only',         action='store_true', help='')
parser.add_argument('--concat',         action='store_true', help='')
parser.add_argument('--finetune',         action='store_true', help='')
parser.add_argument('--notloadoptim',         action='store_true', help='')

parser.add_argument('--bs_gan', default=64, type=int,
                    help='gan loader batchsize')
parser.add_argument('--large', default=50, type=int,
                    help='gan loader batchsize')
parser.add_argument('--gan_lambda', default=1, type=float, metavar='M',
                    help='gan_lambda')
parser.add_argument('--ratio', default=1, type=float, metavar='M',
                    help='gan_lambda')
parser.add_argument('--iter', default=-1, type=int,
                    help='number of total epochs to run')

best_acc1 = 0


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
    if args.pretrained or args.finetune:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

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

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)


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
            if args.notloadoptim == True:
                pass
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    import socket

    traindir_img = '/ImageNet-Data/train'
    valdir = 'ImageNet-Data/val'
    obj_valdir = 'overlap_category_test_noborder'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.large == 1000:
        traindir = '/*/GANgendata/GanSpace/setting_1000_20_1_s2_lam_9.0'
    if args.rand:
        traindir = '/*/GANdata/rand'
    if args.evaluate:
        print("start testing")
        from data.KNN_dataloader import ObjectNetClassWiseLoader, ObjectNetLoader, ImagenetALoader



        # if args.dataset == 'objectnet':
        print("data", obj_valdir)
        val_loader = torch.utils.data.DataLoader(
            ObjectNetLoader(obj_valdir),
            batch_size=100, shuffle=True,
            num_workers=10, pin_memory=True)
        print("Objectnet")
        validate_object(val_loader, model, criterion, args)
        validate_object(val_loader, model, criterion, args)
        return

    backup_output_dir = args.backup_output_dir
    os.makedirs(backup_output_dir, exist_ok=True)

    if os.path.exists(backup_output_dir):
        import uuid
        import datetime
        unique_str = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_name = "train_" + timestamp + "_" + unique_str + "_rotation_" + str(args.rotate)
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_name)
        print("experiment folder", experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder)  #
        log_dir = os.path.join(experiment_backup_folder, "runs")

        # os.makedirs(log_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        eval_writer = SummaryWriter(log_dir=log_dir + '/validate_runs/')


    print('train dir', traindir)
    if args.rotate:
        composed_transforms = transforms.Compose([
               # transforms.RandomRotation(90, resample=2, expand=True),
               transforms.RandomRotation((0,360), resample=2, expand=True),
               transforms.RandomResizedCrop(224),
               transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
               transforms.RandomVerticalFlip(),
               transforms.ToTensor(),
               normalize,
           ])
    else:
        composed_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        print("not huge rotate")


    if args.concat:
        assert args.gan_bl==True
        from data.concat_loader import LoaderConcat_split, Loader_Random
        # if args.large == -1:  # use same imagenet
        #     train_dataset_gan = datasets.ImageFolder(
        #         traindir_img, composed_transforms
        #     )
        if args.rotate_gan_only:
            assert args.rotate==False
            print("rotate gan only")
            composed_transforms_ganonly = transforms.Compose([
                # transforms.RandomRotation(90, resample=2, expand=True),
                transforms.RandomRotation((0, 360), resample=2, expand=True),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if args.large > 50:
            print("train dir", traindir)
            train_dataset_gan = Loader_Random(path=traindir,
                composed_transforms=composed_transforms if not args.rotate_gan_only else composed_transforms_ganonly)
        else:
            print("train dir", traindir)
            train_dataset_gan = LoaderConcat_split(path_1=traindir,
                                  path_2=traindir_img, restrict1= None,
                                     composed_transforms=composed_transforms)
        #1008 if args.rand else
        # 1000 is for restricting the used GAN images

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    # ImageNet loader
    train_dataset = datasets.ImageFolder(
        traindir_img, composed_transforms
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_gan = torch.utils.data.DataLoader(
        train_dataset_gan, batch_size=args.bs_gan, shuffle=(train_sampler is None),
        num_workers=args.workers//2, pin_memory=True, sampler=train_sampler)



    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        print("train dir", traindir)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        for k, v in args.__dict__.items():  # Prints arguments and contents of config file
            print(k, ':', v)

        # train for one epoch
        # if args.large <= 50 and args.large>0:
        #     train_dataset_gan.reshuffle()
        train(train_loader, train_loader_gan, model, criterion, optimizer, epoch, args, writer)


        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)

        eval_writer.add_scalar('Test/top1 acc', acc1, epoch * len(train_loader))
        eval_writer.add_scalar('Test/top5 acc', acc5, epoch * len(train_loader))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename='checkpoint{}.pth.tar'.format(epoch), experiment_backup_folder=experiment_backup_folder)


def train(train_loader, train_loader_gan, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses_g = AverageMeter('Loss_g', ':.4e')
    top1_g = AverageMeter('Acc_g@1', ':6.2f')
    top5_g = AverageMeter('Acc_g@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, losses_g, top1_g, top5_g],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, all in enumerate(zip(train_loader, train_loader_gan)):
        if args.iter ==i:
            break
        # print(all)
        e1, e2 =all

        images_i, target_i = e1
        images, target_gan =e2
        # print("img size", images_i.size(), images.size())
        # measure data loading time
        data_time.update(time.time() - end)

        bs=images_i.size(0)

        if args.gpu is not None:
            images_i = images_i.cuda(args.gpu, non_blocking=True)
            images = images.cuda(args.gpu, non_blocking=True)

        target_i = target_i.cuda(args.gpu, non_blocking=True)
        target_gan = target_gan.cuda(args.gpu, non_blocking=True)
        images = torch.cat((images_i, images), dim=0)

        # compute output
        output = model(images)
        out_i = output[:bs]
        out_gan = output[bs:]

        loss_i = criterion(out_i, target_i)
        loss_gan = criterion(out_gan, target_gan)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(out_i, target_i, topk=(1, 5))
        losses.update(loss_i.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        acc1_gan, acc5_gan = accuracy(out_gan, target_gan, topk=(1, 5))
        losses_g.update(loss_gan.item(), images.size(0))
        top1_g.update(acc1_gan[0], images.size(0))
        top5_g.update(acc5_gan[0], images.size(0))

        if i % 1000==1:
            writer.add_scalar('Train/top1 acc', top1.avg, epoch*len(train_loader) + i)
            writer.add_scalar('Train/top5 acc', top5.avg, epoch*len(train_loader) + i)

            writer.add_scalar('Train/top1 gan acc', top1_g.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Train/top5 gan acc', top5_g.avg, epoch * len(train_loader) + i)


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss = loss_i + loss_gan * args.gan_lambda
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        #
        # if i==2500:
        #     break


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

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


import matplotlib.pyplot as plt

def validate_object(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    num_classes = 1000 # change for other dataset

    if args.exclude_nonoverlap:
        # since it is xent loss, we simply manually make those class non overlap prediction score to be very small.
        # This can automatically exclude those classes.
        from preprocessing.obj_img_nonoverlap_id import get_imagenet_overlap
        overlap, non_overlap = get_imagenet_overlap()
        mask_array = torch.zeros((1, 1000))
        for each in overlap:
            mask_array[0, each] = 1
        mask_array = mask_array.cuda()

        # if args.gpu is not None:
        #     mask_array = mask_array.cuda(args.gpu, non_blocking=True)


    with torch.no_grad():
        end = time.time()
        for i, examples in enumerate(val_loader):

            # print(examples)
            images = examples['images']

            # f, axarr = plt.subplots(4, 4)
            # for zz in range(16):
            #     img = images[zz].numpy()
            #     img = np.moveaxis(np.squeeze(img), 0, 2)
            #
            #     axarr[zz//4][zz%4].imshow(img)
            # plt.savefig("mao_viz{}.png".format(i))

            target = examples['labels']
            # if args.gpu is not None:
            images = images.cuda()
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            # loss = criterion(output, target)

            # measure accuracy and record loss

            if args.exclude_nonoverlap:
                output = torch.exp(output) # first make them larger than 0, since there are negative values
                # print(output, 'b4')
                # print('mask', mask_array)
                output = mask_array * output  # by multiply by 0, we can remove those non-relevant categories

            # maxk = 5
            # _, pred = output.topk(maxk, 1, True, True)
            # pred = pred.t() # 5 * num
            # # correct = pred.eq(target[0].cuda().view(1, -1).expand_as(pred))
            # batch_size = output.shape[0]
            #
            # top1cnt=0
            # top5cnt=0
            # for i in range(len(target)):
            #     correct = pred.eq(target[i].cuda().view(1, -1).expand_as(pred)) # 5 * num
            #     top1cnt+= correct[:1].view(-1).float().sum(0, keepdim=True).item()
            #     top5cnt+= correct[:5].view(-1).float().sum(0, keepdim=True).item()
            #
            # # print(top1cnt)
            # acc1 = top1cnt * 100.0 / batch_size
            # acc5 = top5cnt * 100.0 / batch_size

            acc1, acc5 = objectnet_accuracy_B(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate_classwise_object(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, examples in enumerate(val_loader):
            images = examples['images'][0]
            target = examples['labels']
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            # loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = objectnet_accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
