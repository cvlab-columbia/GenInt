import os,sys
import tqdm
import glob
import argparse
import numpy as np
import socket
import torch
import pickle
from torchvision import datasets
from torch import nn, optim, autograd
from PIL import Image
from collections import defaultdict
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

# Define paths for our datasets
if 'cv' in socket.gethostname():
    # path_color_mnist_test2 = '/proj/vondrick/datasets/color_MNIST/test2/' #Same 
    # path_color_mnist_train = '/proj/vondrick/datasets/color_MNIST/train/'

    path_irm_mnist = '/proj/vondrick/datasets/color_MNIST/irm_mnist/'
    # path_irm_mnist_test = '/proj/vondrick/datasets/color_MNIST/irm_mnist_test/'
    # path_irm_mnist_test = '/proj/vondrick/datasets/color_MNIST/irm_test_same_d/'
    path_irm_mnist_test = '/proj/vondrick/datasets/color_MNIST/test2/' #Same

elif 'hulk' in socket.gethostname():
    # path_color_mnist_test = '/local/rcs/ag4202/color_MNIST/test/'
    # path_color_mnist_test2 = '/local/rcs/ag4202/color_MNIST/test2/' #Same 
    # path_color_mnist_train = '/local/rcs/ag4202/color_MNIST/train/'

    path_irm_mnist = '/local/rcs/ag4202/color_MNIST/irm_mnist/'
    # path_irm_mnist_test = '/local/rcs/ag4202/color_MNIST/irm_test_same_d/'
    # path_irm_mnist_test = '/local/rcs/ag4202/color_MNIST/test2/' #Same
    path_irm_mnist_test = '/local/rcs/ag4202/color_MNIST/irm_mnist_test/'
    path_irm_mnist_test_diff = '/local/rcs/ag4202/color_MNIST/irm_mnist_test/'
    path_irm_mnist_test_same = '/local/rcs/ag4202/color_MNIST/test2/' #Same
     
load = False

path_features_mnist = "./env_list_same.dat"

print("FLAG LOADING: ", load)

def loadIRMMnistTrainEnvironments(path_irm_mnist_folder):

    list_env_folders = os.listdir(path_irm_mnist_folder)

    envs = []

    for env_folder_name in list_env_folders:

        path_env = os.path.join(path_irm_mnist_folder, env_folder_name)

        list_digit_folders = os.listdir(path_env)
        
        list_num_images = []
        list_digit_image_arrays = []
        for digit_fol in tqdm.tqdm(list_digit_folders):

            path_digit_fol = os.path.join(path_env, digit_fol)
            list_path_images = glob.glob(path_digit_fol + '/*')

            list_image_arrays = []
            for path_im in tqdm.tqdm(list_path_images):

                im = Image.open(path_im)
                array_im = np.array(im)

                list_image_arrays.append(array_im)
            
            list_digit_image_arrays.append(list_image_arrays)
            list_num_images.append((len(list_image_arrays)))
        
        num_images_per_digit = min(list_num_images)
        final_list_digits_arrays = [l[:num_images_per_digit] for l in list_digit_image_arrays]
        labels = torch.Tensor([num_images_per_digit*[i] for i in range(1,11)]) 
        tensor_labels = labels.view(-1,1) 
        tensor_labels = tensor_labels.to(torch.long)
        
        array_images = np.concatenate(final_list_digits_arrays)
        array_images_data = np.moveaxis(array_images,-1,-3)
        images = torch.Tensor(array_images_data)

        env = {
                'images' : (images.float()/255.),
                'labels' : tensor_labels
                }

        envs.append(env)
    
    return envs

def loadIRMMnistTestEnvironment(path_irm_mnist_folder):

    list_digit_folders = os.listdir(path_irm_mnist_folder)
    
    list_num_images = []
    list_digit_image_arrays = []
    for digit_fol in tqdm.tqdm(list_digit_folders):

        path_digit_fol = os.path.join(path_irm_mnist_folder, digit_fol)
        list_path_images = glob.glob(path_digit_fol + '/*')

        list_image_arrays = []
        for path_im in tqdm.tqdm(list_path_images):

            im = Image.open(path_im)
            array_im = np.array(im)

            list_image_arrays.append(array_im)
        
        list_digit_image_arrays.append(list_image_arrays)
        list_num_images.append((len(list_image_arrays)))
    
    num_images_per_digit = min(list_num_images)
    final_list_digits_arrays = [l[:num_images_per_digit] for l in list_digit_image_arrays]
    labels = torch.Tensor([num_images_per_digit*[i] for i in range(1,11)]) 
    tensor_labels = labels.view(-1,1) 
    tensor_labels = tensor_labels.to(torch.long)
    
    array_images = np.concatenate(final_list_digits_arrays)
    array_images_data = np.moveaxis(array_images,-1,-3)
    images = torch.Tensor(array_images_data)

    env = {
            'images' : (images.float()/255.),
            'labels' : tensor_labels
            }
    
    return env

if not load:
    print("Computing from train and test", path_irm_mnist, path_irm_mnist_test)
    envs = loadIRMMnistTrainEnvironments(path_irm_mnist)
    test_env = loadIRMMnistTestEnvironment(path_irm_mnist_test)
    envs.append(test_env)
    path_out = path_features_mnist
    with open(path_out, 'wb') as env_file:
        pickle.dump(envs, env_file)
        print("Saved envs at ", path_out)

# Test the loading of features:
else:
    print("Loading")
    path_out = path_features_mnist
    with open(path_out, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        envs = unpickler.load()
    print("Features loaded from", path_out, envs[0]['images'].shape,envs[1]['images'].shape,envs[2]['images'].shape)

# Shuffling the tensor
for j, env in enumerate(envs):
    num_images = len(env['images'])
    perm = torch.randperm(num_images)

    env['images'] = env['images'][perm]
    env['labels'] = env['labels'][perm]

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)


    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
            self.conv3 = nn.Conv2d(32,64, kernel_size=5)
            self.fc1 = nn.Linear(1024, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            # print("forward pass", x.shape)
            x = F.relu(self.conv1(x))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(F.max_pool2d(self.conv3(x),2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.reshape(-1,1024 )
            # x = x.view(-1,1024 )
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x
            # return F.log_softmax(x, dim=1)
 



    mlp = CNN().cuda()

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)
    
    def mnist_loss(logps, y):
        loss = nn.functional.cross_entropy(logps, y.squeeze(1))
        return loss
    
    def mnist_acc(logps, y):
        ps = torch.exp(logps)
        top_p, top_idx = torch.topk(logps,k = 1, dim=1)
        preds = top_idx
        num_correct = torch.sum((y==preds)).to(torch.float)
        num_dp = y.shape[0]
        acc = num_correct/num_dp
        return acc     

    def mnist_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mnist_loss(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)
    
    def pretty_print(*values):
        col_width = 13
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("     ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'][:15000].cuda())
            labels = env['labels'][:15000].cuda()
            env['nll'] = mnist_loss(logits, labels)
            env['acc'] = mnist_acc(logits,labels)
            env['penalty'] = mnist_penalty(logits, labels)

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        
        penalty_weight = (flags.penalty_weight 
                if step >= flags.penalty_anneal_iters else 1.0)

        # PENALTY PART
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
