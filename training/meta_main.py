'''The first version of meta training using viewpoints'''
'''hierarchy may solve the problem for too many categories'''
'''e.g. the transformer can have few big category matching in the early layers. We do this by regularization. Saying query and key are less volume in diversity'''
'''k means clustering for example, for reduce -way'''
'''hinge for big cluster label loss'''
import argparse
import os
import random
import shutil
import time
import warnings
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

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


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
parser.add_argument('--lr_interval', default=30, type=int, metavar='I',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-es', '--episode-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-est', '--episode-size-test', default=1000, type=int,
                    metavar='N',
                    help='')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nd', '--norm_decay', default=1e-4, type=float,
                    metavar='ND', help='norm decay for the feature of our metric space', dest='norm_decay')
parser.add_argument('--fplam', '--fea_pair_lambda', default=1, type=float,
                    metavar='FPL', help='lambda for feature mse pairinng loss', dest='fea_pair_lambda')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--optim', default='sgd', type=str, metavar='optimizer',
                    help='optimizer type')
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
parser.add_argument('--backup_output_dir',          type=str,       default='/local/rcs/mcz/ImageNet-Data/SavedModels',  help='')
parser.add_argument('--rotate',         action='store_true')
parser.add_argument('--local_pair',         action='store_true')
parser.add_argument('-sreso', '--split_resolution', default=4, type=int, metavar='S',
                    help='number of 2 power split if needed')
parser.add_argument('--xent_lambda', default=0, type=float,
                    metavar='Lxent', help='initial xent lambda', dest='xent_lambda')
parser.add_argument('--fea_len', default=0, type=int,
                    help='Linear project to next feature for metric')
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
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    print("=> creating model '{}'".format(args.arch))
    if args.fea_len == 0:
        from learning.resnet import resnet18
        model = resnet18()
    else:
        from learning.resnet_flen import resnet18
        model = resnet18(num_fea_len=args.fea_len)

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
    criterion = [nn.CrossEntropyLoss().cuda(args.gpu) for i in range(3)]

    split_sub_criterion = [nn.CrossEntropyLoss().cuda(args.gpu) for i in range(2 ** (args.split_resolution+1) - 1)]

    # TODO:
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
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

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')

    import socket
    if socket.gethostname() == 'hulk':
        # traindir = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train_clustered-3.5'
        traindir = '/local/rcs/mcz/2020Spring/cut_imgnet-3.5/train_clustered-C3.5'
        valdir = '/local/rcs/mcz/ImageNet-Data/val'
    elif socket.gethostname() == 'cv04':
        traindir = '/local/vondrick/cz/cut_imgnet-3.5/train_clustered-3.5'
        valdir = '/local/vondrick/cz/ImageNet/val'
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    if args.rotate:
        pass
        # print("do huge rotate")
        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomRotation(90, resample=2),
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
    else:
        from data.exemplarnet import ExemplarLoader, ExemplarTESTLoader
        train_dataset = ExemplarLoader(train_base_dir=traindir,
                                         length_episode=args.episode_size
                                         )

        test_dataset = ExemplarTESTLoader(train_base_dir=traindir, test_base_dir=valdir, length_episode=args.episode_size_test)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    '''We may start with increasingly batch size, like curriculum'''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) #TODO: num workers ?

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

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer, split_sub_criterion)
        print("finish training, starting testing")
        #
        # # evaluate on validation set

        acc1, acc5 = validate_temp(val_loader, model, criterion, args)
        #
        eval_writer.add_scalar('Test/top1 acc', acc1, epoch * len(train_loader))
        eval_writer.add_scalar('Test/top5 acc', acc5, epoch * len(train_loader))
        #
        # # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        #
        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        # if epoch % 10 == 9:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, False, filename='checkpoint.pth.tar', experiment_backup_folder=experiment_backup_folder)

def train(train_loader, model, criterion, optimizer, epoch, args, writer, split_sub_criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    norm_log = AverageMeter('Feature Norm', ':6.2f')
    k_pairs_log = AverageMeter('k_pairs', ':4e')
    feature_pairs_log = AverageMeter('feature_pairs_mse', ':4e')

    log_list = [batch_time, data_time, losses, top1, top5, norm_log, k_pairs_log, feature_pairs_log]
    if args.xent_lambda > 0:
        xent_losses_log = AverageMeter('Xent Loss', ':.4e')
        xent_top1 = AverageMeter('Xent Acc@1', ':6.2f')
        xent_top5 = AverageMeter('Xent Acc@5', ':6.2f')
        log_list = log_list + [xent_losses_log, xent_top1, xent_top5]

    progress = ProgressMeter(
        len(train_loader),
        log_list,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    cr1, cr2, cr3 = criterion
    # for i, (images, target) in enumerate(train_loader):
    for i, sample in enumerate(train_loader):
        # measure data loading time
        episode_examples = sample["episode_examples"]
        episode_queries = sample["episode_queries"]
        labels = sample["labels"]
        labels = torch.cat(labels, 0)
        # print('episode_examples', episode_examples)
        # print('labels', labels)

        images = torch.cat((episode_examples.squeeze(), episode_queries.squeeze()), 0)
        target = labels
        data_time.update(time.time() - end)

        att_target = torch.from_numpy(np.arange(0, args.episode_size))  # diagonal matrix

        if args.gpu is not None:
            # images = images.cuda(args.gpu, non_blocking=True)
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        att_target = att_target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, feature, norm = model(images)
        # print('feature size', feature_raw.size())

        output_1 = output[:args.episode_size]
        output_2 = output[args.episode_size:]

        # import pdb
        # pdb.set_trace()
        # feature, norm = normalize_l2(feature_raw)

        feature_1 = feature[:args.episode_size]
        feature_2 = feature[args.episode_size:]

        # I can create different length of episode here, can based on the top k to create!
        # also can involve uniformly randomly sampled neg.
        # Global k pairs:

        mat = torch.mm(feature_1, feature_2.t())   # query * support

        # print('shape ', mat.size(), att_target.size())

        # feature_pairing = torch.mean((feature_1 - feature_2) ** 2)

        full_pair_loss = cr1(mat, att_target) # label smoothing trick, should be considered

        def get_local_pair_loss(fea1, fea2, episode_size, split_sub_criterion):
            total_loss = 0
            cnt = 0
            for num_split_power in range(1, args.split_resolution):
                split_num = 2 ** num_split_power
                split_unit_len = episode_size // split_num

                att_target_temp = torch.from_numpy(np.arange(0, split_unit_len))
                att_target_temp = att_target_temp.cuda(args.gpu, non_blocking=True)
                for k in range(split_num):
                    temp_1 = fea1[k*split_unit_len:(k+1)*split_unit_len]
                    temp_2 = fea2[k*split_unit_len:(k+1)*split_unit_len]

                    temp_mat = torch.mm(temp_1, temp_2.t())
                    temp_loss = split_sub_criterion[cnt](temp_mat, att_target_temp)

                    total_loss += temp_loss
            return total_loss


        if args.local_pair:
            local_pair_loss = get_local_pair_loss(feature_1, feature_2, args.episode_size, split_sub_criterion)


        loss_1 = cr2(output_1, target)
        loss_2 = cr3(output_2, target)

        xent_sum_loss = (loss_1 + loss_2) / 2

        decay = torch.mean(norm)
        if args.local_pair:
            loss = full_pair_loss + local_pair_loss + args.norm_decay * decay  # args.fea_pair_lambda * feature_pairing +
        else:
            loss = full_pair_loss + args.norm_decay * decay

        if args.xent_lambda > 0:
            this_lambda = max(args.xent_lambda - args.xent_lambda / 20 * epoch, 0)
            loss = loss + this_lambda * xent_sum_loss

        ##  TODO:
        # print('full_pair_loss', full_pair_loss.item(), 'norm mean', decay.item())

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1, acc5 = accuracy(mat, att_target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        k_pairs_log.update(full_pair_loss.item(), images.size(0))
        # feature_pairs_log.update(feature_pairing.item(), images.size(0))
        norm_log.update(decay.item(), images.size(0))
        if args.xent_lambda > 0:
            xacc1, xacc5 = accuracy(output_1, target, topk=(1, 5))
            xent_top1.update(xacc1[0], images.size(0))
            xent_top5.update(xacc5[0], images.size(0))
            xent_losses_log.update(xent_sum_loss.item(), images.size(0))

        if i % 1000 == 1:
            writer.add_scalar('Train/top1 acc', top1.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Train/top5 acc', top5.avg, epoch * len(train_loader) + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)




def validate_temp(val_loader, model, criterion, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    log_list = [batch_time, losses, top1, top5]
    if args.xent_lambda > 0:
        xent_top1 = AverageMeter('Xent Acc@1', ':6.2f')
        xent_top5 = AverageMeter('Xent Acc@5', ':6.2f')
        log_list = log_list + [xent_top1, xent_top5]

    progress = ProgressMeter(
        len(val_loader),
        log_list,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    cr1, cr2, cr3 = criterion

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            # print('finish loading')
            episode_examples = sample["episode_examples"]
            episode_queries = sample["episode_queries"]  # Unlabeled data can server as additional negative
            labels = sample["labels"]
            target = torch.cat(labels, 0)

            images = torch.cat((episode_examples.squeeze(), episode_queries.squeeze()), 0)
            # target = torch.cat((labels, labels), 0)
            att_target = torch.from_numpy(np.arange(0, args.episode_size_test))  # diagonal matrix

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            att_target = att_target.cuda(args.gpu, non_blocking=True)

            # compute output
            # output = model(images)
            # loss = criterion(output, target)
            output, feature, norm = model(images)

            if args.xent_lambda > 0:
                output_1 = output[:args.episode_size_test]
                xacc1, xacc5 = accuracy(output_1, target, topk=(1, 5))
                xent_top1.update(xacc1[0], images.size(0))
                xent_top5.update(xacc5[0], images.size(0))

            # import pdb
            # pdb.set_trace()
            # feature, norm = normalize_l2(feature_raw)

            feature_1 = feature[:args.episode_size_test]
            feature_2 = feature[args.episode_size_test:]

            mat = torch.mm(feature_1, feature_2.t())

            full_pair_loss = cr1(mat, att_target)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(mat, att_target, topk=(1, 5))
            losses.update(full_pair_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            progress.display(i)

            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #       .format(top1=top1, top5=top5))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

if __name__ == '__main__':
    main()


# TODO: try diff optimizer, like adam, 'expert ones adamW?', write the test episode part, remove detach part for faster convergence, add transformer,
# try diff episode size, do diff episode size in the loss function.






