


import shutil
import torch
import os
from os.path import exists, join, split
import multiprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def img_trans_back(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = img/2 + 0.5

    for i in range(3):
        img[:,i,:,:] = (img[:,i,:,:] - mean[i])/std[i]

    return img

def convert_to_images(obj):
    """ Convert an output tensor from BigGAN in a list of images.
        Params:
            obj: tensor or numpy array of shape (batch_size, channels, height, width)
        Output:
            list of Pillow Images of size (height, width)
    """
    try:
        import PIL
    except ImportError:
        raise ImportError("Please install Pillow to use images: pip install Pillow")

    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()

    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(PIL.Image.fromarray(out_array))
    return img

def np_normalize_l2(center_array):
    center_array = center_array / (np.sum(center_array ** 2, axis=1, keepdims=True) ** 0.5 + 1e-10)
    return center_array

def normalize_l2(feature, epsilon = 1e-7):
    norm2 = torch.sum(feature**2, dim=1, keepdim=True)
    feature = feature / (norm2 + epsilon)
    return feature, norm2.squeeze()

def fix_imagenet_temp(path):
    folder_list = os.listdir(path)
    for each in folder_list:
        cur_path = os.path.join(path, each)
        if os.path.isdir(cur_path):
            sub_folder_list = os.listdir(cur_path)
            if 'temp' in sub_folder_list:
                print("illegal folder need to be fixed", each)
                temp_root_path = os.path.join(cur_path, 'temp')
                temp_list = os.listdir(temp_root_path)
                for mogu in temp_list:
                    shutil.move(os.path.join(temp_root_path, mogu), cur_path)

                if len(os.listdir(temp_root_path)) == 0:
                    os.rmdir(temp_root_path)

def multiprocess(func, args, process_num):
    for i in range(process_num):
        p1 = multiprocessing.Process(target=func, args=args[i])
        p1.start()


def GPU_multiprocess(func, args, process_num):
    from torch.multiprocessing import Process
    torch.multiprocessing.set_start_method('spawn', force=True)
    for i in range(process_num):
        p1 = Process(target=func, args=args[i])
        p1.start()

def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """

    def _ignore_patterns(path, names):
        # print(patterns)
        # print([anme for anme in list(filter(names, patterns[0]))])
        keep = set(name for pattern in patterns
                   for name in filter(names, pattern))
        ignore = set(name for name in names
                     if name not in keep and not os.path.isdir(join(path, name)))
        return ignore

    return _ignore_patterns

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', experiment_backup_folder=None, epoch=None, save=False):
    print("saving checkpoint")
    torch.save(state, os.path.join(experiment_backup_folder, filename))
    if is_best:
        if epoch is None:
            shutil.copyfile(os.path.join(experiment_backup_folder, filename),
                            os.path.join(experiment_backup_folder, 'model_best.pth.tar'))
        else:
            shutil.copyfile(os.path.join(experiment_backup_folder, filename),
                            os.path.join(experiment_backup_folder, 'model_best_{}.pth.tar'.format(epoch)))
    if save:
        shutil.copyfile(os.path.join(experiment_backup_folder, filename),
                        os.path.join(experiment_backup_folder, 'model_{}.pth.tar'.format(epoch)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_interval))
    print('original lr', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_customize(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // args.lr_interval))
    print("list customize lr")
    if epoch>=0 and epoch<args.lr_list[0][0]:
        lr = args.lr_list[0][1]
    elif epoch>=args.lr_list[0][0] and epoch<args.lr_list[1][0]:
        lr = args.lr_list[1][1]
    elif epoch>=args.lr_list[1][0] and epoch<args.lr_list[2][0]:
        lr = args.lr_list[2][1]
    else:
        lr = args.lr_list[3][1]
    print('lr=', lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print('correct size', correct.size())
        # print('correct size', correct[:5].size())
        # print('correct size', correct[:5].reshape(-1).size())
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def objectnet_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    cnt = 0
    top1_cnt = 0
    top5_cnt = 0

    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)

        output = output.data
        _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True) # Change sorted to True

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        pred = pred.cpu().numpy()
        for jj in range(batch_size):
            label_gt_all = target[jj]
            label_gt_list = []

            # print('label gt all', label_gt_list)
            for mogu in label_gt_all:
                label_gt_list.append(mogu.numpy()[0])

            cnt += 1

            pred_index = pred[jj]

            for label_gt in label_gt_list:
                if pred_index[0] == label_gt:
                    top1_cnt += 1
                    break

            flag = True
            for nnn in range(5):  # If each batch is from same category, can make it in matrix to speed up
                if flag == False:
                    break
                for label_gt in label_gt_list:
                    if pred_index[nnn] == label_gt:
                        top5_cnt += 1
                        flag = False
                        break

        return top1_cnt * 100. / cnt, top5_cnt  * 100. / cnt

def objectnet_accuracy_B(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    cnt = 0
    top1_cnt = 0
    top5_cnt = 0

    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)

        output = output.data
        _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)  # Oh forget to sorted it

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        pred = pred.cpu().numpy()

        # target = target # Labels are splited by -1

        target_num = len(target)
        target = [each.numpy() for each in target]

        for jj in range(batch_size):
            # label_gt_all = target[jj]
            label_gt_list = []

            for kk in range(target_num):
                label_gt_list.append(target[kk][jj])

            cnt += 1

            pred_index = pred[jj]

            for label_gt in label_gt_list:
                if pred_index[0] == label_gt:
                    top1_cnt += 1
                    break

            flag = True
            for nnn in range(5):  # If each batch is from same category, can make it in matrix to speed up
                if flag == False:
                    break
                for label_gt in label_gt_list:
                    if pred_index[nnn] == label_gt:
                        top5_cnt += 1
                        flag = False
                        break

        return top1_cnt * 100. / cnt, top5_cnt * 100. / cnt


def set_parameter_not_requires_grad(model_parameters, set_range):
    for cnt, param in enumerate(model_parameters):
        if cnt>=set_range[0] and cnt<set_range[1]:
            param.requires_grad = False

def set_parameter_requires_grad(model_parameters, set_range):
    for cnt, param in enumerate(model_parameters):
        if cnt>=set_range[0] and cnt<set_range[1]:
            param.requires_grad = True


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion_matrix',
                          cmap=plt.cm.Blues,
                          path='./visualize/confusion_matrix/'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.rc('font', size=3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    plt.rc('font', size=1)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.savefig(path+title+'.png',dpi=400)
    plt.savefig(title+'.png',dpi=600)


def getDictImageNetClasses(path_imagenet_classes_name='preprocessing/imagenet_classes_names.txt'):
    '''
    Returns dictionary of classname --> classid. Eg - {n02119789: 'kit_fox'}
    '''

    count = 0
    dict_imagenet_classname2id = {}
    with open(path_imagenet_classes_name) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            if cat_name in dict_imagenet_classname2id.keys():
                print(cat_name)
            dict_imagenet_classname2id[id] = cat_name.lower()
            count += 1
            # print(cat_name, id)
            line = f.readline()
    # print("Total categories categories", count)
    return dict_imagenet_classname2id

def getDictImageNet_ID2Num(path_imagenet_classes_name='preprocessing/imagenet_classes_names.txt'):
    '''
    Returns dictionary of classname --> classid. Eg - {n02119789: 'kit_fox'}
    '''

    count = 0
    dict_imagenet_classname2id = {}
    id_list = []
    with open(path_imagenet_classes_name) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            id_list.append(id)

            # print(cat_name, id)
            line = f.readline()

    id_list = sorted(id_list)
    ans_dict={}
    for jj, each in enumerate(id_list):
        ans_dict[each] = jj
    # print("Total categories categories", count)
    return ans_dict


import torch
import torch.nn as nn
from torch.autograd import Variable
class SoftNLLLoss(nn.NLLLoss):
    """A soft version of negative log likelihood loss with support for label smoothing.

    Effectively equivalent to PyTorch's :class:`torch.nn.NLLLoss`, if `label_smoothing`
    set to zero. While the numerical loss values will be different compared to
    :class:`torch.nn.NLLLoss`, this loss results in the same gradients. This is because
    the implementation uses :class:`torch.nn.KLDivLoss` to support multi-class label
    smoothing.

    Args:
        label_smoothing (float):
            The smoothing parameter :math:`epsilon` for label smoothing. For details on
            label smoothing refer `this paper <https://arxiv.org/abs/1512.00567v1>`__.
        weight (:class:`torch.Tensor`):
            A 1D tensor of size equal to the number of classes. Specifies the manual
            weight rescaling applied to each class. Useful in cases when there is severe
            class imbalance in the training set.
        num_classes (int):
            The number of classes.
        size_average (bool):
            By default, the losses are averaged for each minibatch over observations **as
            well as** over dimensions. However, if ``False`` the losses are instead
            summed. This is a keyword only parameter.
    """

    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', Variable(weight))

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        self.criterion = nn.KLDivLoss(**kwargs)

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)

        if self.weight is not None:
            one_hot.mul_(self.weight)

        return self.criterion(input, one_hot)