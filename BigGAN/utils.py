def getDictImageNetClasses(path_imagenet_classes_name='imagenet_list.txt'):
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
            print(cat_name, id)
            line = f.readline()
    print("Total categories categories", count)
    return dict_imagenet_classname2id

import json

def get_imagenet_overlap():
    with open('obj2imgnet_id.txt') as f:
        dict_obj2imagenet_id = json.load(f)

    overlapping_list= []
    for each in dict_obj2imagenet_id.keys():
        overlapping_list.extend(dict_obj2imagenet_id[each])

    all_list = [i for i in range(1000)]

    non_overlapping = list(set(all_list) - set(overlapping_list))

    print('non-overlapping', len(non_overlapping))

    return overlapping_list, non_overlapping

import torch, os, shutil
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', experiment_backup_folder=None, epoch=None):
    print("saving checkpoint")
    torch.save(state, os.path.join(experiment_backup_folder, filename))
    if is_best:
        if epoch is None:
            shutil.copyfile(os.path.join(experiment_backup_folder, filename),
                            os.path.join(experiment_backup_folder, 'model_best.pth.tar'))
        else:
            shutil.copyfile(os.path.join(experiment_backup_folder, filename),
                            os.path.join(experiment_backup_folder, 'model_best_{}.pth.tar'.format(epoch)))

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
