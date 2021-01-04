import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch, json, numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs
from PIL import Image
from netdissect import proggan, nethook, easydict, zdataset
from netdissect.modelconfig import create_instrumented_model
from estimators import get_estimator
from models import get_instrumented_model
from scipy.cluster.vq import kmeans
import re
import sys
import datetime
import argparse
from tqdm import trange
from tqdm import tqdm
from config import Config
from decomposition import get_random_dirs, get_or_compute, get_max_batch_size, SEED_VISUALIZATION
from utils import pad_frames

from utils import getDictImageNetClasses, get_imagenet_overlap

def img_list_generator(latent, lat_mean, lat_comp, lat_stdev, act_mean, act_comp, act_stdev, scale=1,
                       num_frames=5, make_plots=True, edit_type='latent', english_name=None, allrand=False,
                       args=None):
    """

    :param latent:
    :param lat_mean:
    :param lat_comp:
    :param lat_stdev:
    :param act_mean:
    :param act_comp:
    :param act_stdev:
    :param scale:
    :param num_frames:
    :param make_plots:
    :param edit_type:
    :return:   The image list : tuple (imgname, image_np_array)
    """
    from notebooks.notebook_utils import create_strip_centered

    x_range = np.linspace(-scale, scale, num_frames, dtype=np.float32)

    inst.remove_edits()
    r=0
    curr_row = []

    sigs=None
    if allrand:
        sigma_range = np.linspace(-scale, scale, num_frames)
        selected = random.sample([i for i in range(num_frames)], allrand)
        sigs = sigma_range[selected]

    out_batch = create_strip_centered(inst, mode=edit_type, layer=layer_key, latents=[latent],
                                      x_comp=act_comp[r], z_comp=lat_comp[r], act_stdev=act_stdev[r],
                                      lat_stdev=lat_stdev[r], act_mean=act_mean, lat_mean=lat_mean, sigma=scale,
                                      layer_start=0, layer_end=-1, num_frames=num_frames, allrand=allrand,
                                      sigs=sigs, args=args)[0]  # use [0] since only one latent layer
    for i, img in enumerate(out_batch):
        if allrand:
            curr_row.append(('{}_sigma_{:.2f}_{}.png'.format(i, sigs[i], english_name), img))
        else:
            curr_row.append(('{}_sigma_{:.2f}_{}.png'.format(i, x_range[i], english_name), img))

    return curr_row # The image list : tuple (imgname, image_np_array)

import PIL
def save_img(img_list, save_dir):
    for filename, each in img_list:
        out = each * 256
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img_out = PIL.Image.fromarray(out_array)
        current_file_name = os.path.join(save_dir, filename)
        img_out.save(current_file_name, 'png')
    return

def save_both(both_list, save_dir_img, save_dir_zlabel):
    for filename, data in both_list:
        each, np_noise = data
        out = each * 256
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img_out = PIL.Image.fromarray(out_array)
        current_file_name = os.path.join(save_dir_img, filename)
        img_out.save(current_file_name, 'png')

        np_file_name = os.path.join(save_dir_zlabel, filename.split('.')[0]+".npy")
        np.save(np_file_name, np_noise)
    return



if __name__ == '__main__':
    id2nameDict = getDictImageNetClasses()
    ikeys = list(id2nameDict.keys())
    num2id={}
    num2name={}
    ikeys.sort()
    for ii, each in enumerate(ikeys):
        num2name[ii] = id2nameDict[each]
        num2id[ii] = each


    # Load Objectnet classes:
    overlapping_list, non_overlapping = get_imagenet_overlap()

    global max_batch, sample_shape, feature_shape, inst, args, layer_key, model

    args = Config().from_args()
    args.interval_resolution = 9

    if args.allrand == 0:
        args.allrand=False

    if args.save_simple == 0:
        args.save_simple=False

    t_start = datetime.datetime.now()
    timestamp = lambda: datetime.datetime.now().strftime("%d.%m %H:%M")
    print(f'[{timestamp()}] {args.model}, {args.layer}, {args.estimator}')

    # Ensure reproducibility
    torch.manual_seed(0)  # also sets cuda seeds
    np.random.seed(0)

    has_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if has_gpu else 'cpu')
    layer_key = args.layer
    layer_name = layer_key  # layer_key.lower().split('.')[-1]

    basedir = Path(__file__).parent.resolve()

    # TODO: change * to your local path

    output_dir = '/local/*/*/GANdata/setting_{}_{}_{}_s{}_lam_{}'.format(args.random_image_examplar_num,
                                                                        args.top_k_component_num, args.select_num,
                                                                                         args.allrand, args.sigma)

    # debug=True
    debug=False
    if args.save_noise>0:
        # create data pair :  z noise, img generated
        output_dir = '/*/GANgendata/GanSpace/z-img_{}_{}_{}_s{}_lam_{}'.format(args.random_image_examplar_num,
                                                                        args.top_k_component_num, args.select_num,
                                                                                         args.allrand, args.sigma)
        if debug:
            output_dir = '/*/GANgendata/debug'

        output_dir_img_root = os.path.join(output_dir, 'img')
        output_dir_noise_root = os.path.join(output_dir, 'z')
        os.makedirs(output_dir_img_root, exist_ok=True)
        os.makedirs(output_dir_noise_root, exist_ok=True)
    else:
        if debug:
            output_dir = '/*/GANgendata/debug'
        os.makedirs(output_dir, exist_ok=True)

    inst = get_instrumented_model(args.model, args.output_class, layer_key, device, use_w=args.use_w)
    model = inst.model
    feature_shape = inst.feature_shape[layer_key]
    latent_shape = model.get_latent_shape()
    print('Feature shape:', feature_shape)

    # Layout of activations
    if len(feature_shape) != 4:  # non-spatial
        axis_mask = np.ones(len(feature_shape), dtype=np.int32)
    else:
        axis_mask = np.array([0, 1, 1, 1])  # only batch fixed => whole activation volume used

    # Shape of sample passed to PCA
    sample_shape = feature_shape * axis_mask
    sample_shape[sample_shape == 0] = 1

    # Load or compute components
    dump_name = get_or_compute(args, inst)
    data = np.load(dump_name, allow_pickle=False)  # does not contain object arrays
    X_comp = data['act_comp']
    X_global_mean = data['act_mean']
    X_stdev = data['act_stdev']
    X_var_ratio = data['var_ratio']
    X_stdev_random = data['random_stdevs']
    Z_global_mean = data['lat_mean']
    Z_comp = data['lat_comp']
    Z_stdev = data['lat_stdev']
    n_comp = X_comp.shape[0]
    data.close()

    # Transfer components to device
    tensors = SimpleNamespace(
        X_comp=torch.from_numpy(X_comp).to(device).float(),  # -1, 1, C, H, W
        X_global_mean=torch.from_numpy(X_global_mean).to(device).float(),  # 1, C, H, W
        X_stdev=torch.from_numpy(X_stdev).to(device).float(),
        Z_comp=torch.from_numpy(Z_comp).to(device).float(),
        Z_stdev=torch.from_numpy(Z_stdev).to(device).float(),
        Z_global_mean=torch.from_numpy(Z_global_mean).to(device).float(),
    )
    transformer = get_estimator(args.estimator, n_comp, args.sparsity)
    tr_param_str = transformer.get_param_str()

    print("tensor shape", tensors.Z_comp.size(), tensors.Z_stdev.size())

    # Ensure visualization gets new samples
    torch.manual_seed(SEED_VISUALIZATION)
    np.random.seed(SEED_VISUALIZATION)

    # Measure component sparsity (!= activation sparsity)
    sparsity = np.mean(X_comp == 0)  # percentage of zero values in components
    print(f'Sparsity: {sparsity:.2f}')

    if args.use_w and layer_key in ['style', 'g_mapping'] or args.save_noise>0:
        edit_modes = ['latent'] # activation edit is the same
        print("only latent")
    else:
        edit_modes = ['activation', 'latent']

    num_list_all = [i for i in range(args.start_num, args.end_num)]
    for each in num_list_all:
        # print(each)
        ID = num2id[each]
        english_name = num2name[each]

        if args.save_noise > 0:
            output_dir_img = os.path.join(output_dir_img_root, ID)
            makedirs(output_dir_img, exist_ok=True)
            output_dir_noise = os.path.join(output_dir_noise_root, ID)
            makedirs(output_dir_noise, exist_ok=True)
        else:
            output_class_folder = os.path.join(output_dir, ID)
            makedirs(output_class_folder, exist_ok=True)


        model.set_output_class(each)

        n_random_imgs = args.random_image_examplar_num

        components_all = [i for i in range(args.top_k_component_num)]

        import random

        # print("model truc", model.truncation)
        latents = model.sample_latent(n_samples=n_random_imgs)
        # print(latents.size())
        # print(torch.mean(latents, dim=1))
        # print(torch.std(latents, dim=1))
        # exit()

        for img_idx in trange(n_random_imgs, desc='Random images', ascii=True):
            if args.save_noise >0:
                same_seed_img_dir = os.path.join(output_dir_img, str(img_idx))
                makedirs(same_seed_img_dir, exist_ok=False)
                same_seed_noise_dir = os.path.join(output_dir_noise, str(img_idx))
                makedirs(same_seed_noise_dir, exist_ok=False)
            else:
                same_seed_img_dir = os.path.join(output_class_folder, str(img_idx))
                makedirs(same_seed_img_dir, exist_ok=False)
            if img_idx == 0:
                z = tensors.Z_global_mean
                sigma=args.sigma+2  # we can extend more when on global mean
                num_frames = args.trans_num_frames+2
            else:
                z = latents[img_idx][None, ...]
                sigma = args.sigma
                num_frames = args.trans_num_frames

            components_list = random.sample(components_all, args.select_num)
            for c in components_list:
                if not args.save_simple:
                    img_component_dir = os.path.join(same_seed_img_dir, str("component_{}".format(c)))
                    makedirs(img_component_dir, exist_ok=False)

                if args.allrand:
                    selected_mode = random.sample(edit_modes, 1)
                else:
                    selected_mode = edit_mode
                for edit_mode in selected_mode:  # enumerate over latent or activation
                    if not args.save_simple:
                        img_comp_mode_dir = os.path.join(img_component_dir, edit_mode)
                        makedirs(img_comp_mode_dir, exist_ok=False)

                    frames = img_list_generator(latent=z, lat_mean=tensors.Z_global_mean,
                                                lat_comp=tensors.Z_comp[c:c + 1, :, :],
                                                lat_stdev=tensors.Z_stdev[c:c + 1], act_mean=tensors.X_global_mean,
                                                act_comp=tensors.X_comp[c:c + 1, :, :],
                                                act_stdev=tensors.X_stdev[c:c + 1],
                                                scale=sigma, make_plots=False, edit_type=edit_mode,
                                                num_frames=num_frames, english_name=english_name, allrand=args.allrand,
                                                args=args)
                    # frames are [[],[],[]], each sub list is one layer, where you passsed in in latents list, each element in sublist
                    # is one image, in np, range from 0 to 1, the same as orignal imagenet dataset range

                    #latent, lat_mean, lat_comp, lat_stdev, act_mean, act_comp, act_stdev, scale=1,
                       # num_frames=5, make_plots=True, edit_type='latent'

                    if args.save_simple: # maybe be same name now if has multiple c compoenent, not good idea, can just re-organize the dataset too.
                        if args.save_noise >0:

                            save_both(frames, same_seed_img_dir, same_seed_noise_dir)
                        else:
                            save_img(frames, same_seed_img_dir)
                    else:
                        save_img(frames, img_comp_mode_dir)























