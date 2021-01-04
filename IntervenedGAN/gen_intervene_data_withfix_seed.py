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

import PIL



if __name__ == '__main__':
    rerun=False

    id2nameDict = getDictImageNetClasses()
    ikeys = list(id2nameDict.keys())
    num2id={}
    num2name={}
    ikeys.sort()
    for ii, each in enumerate(ikeys):
        num2name[ii] = id2nameDict[each]
        num2id[ii] = each

    obj_only=False

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

    import socket
    debug = False

    root_path = "/path you want to save to"

    # create data pair :  z noise, img generated
    if obj_only:
        output_dir = '{}/x-z_tr{}-my-objnet'.format(root_path, args.truncation)
    else:
        output_dir = '{}/x-z_tr{}-imgnet-rand{}-int5time'.format(root_path, args.truncation, args.random_num_end)

    print("Info: random z from {} to {} \n    component from {} to {}, \n random offset {}  ".format(
        args.random_num_start, args.random_num_end, args.start_component, args.end_component, args.num_rand_scale))
    print("truncation {}".format(args.truncation))


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
    torch.manual_seed(args.global_seed + args.random_num_start*args.random_num_end)
    np.random.seed(args.global_seed+999+ args.random_num_start*args.random_num_end*3)

    # Measure component sparsity (!= activation sparsity)
    sparsity = np.mean(X_comp == 0)  # percentage of zero values in components
    print(f'Sparsity: {sparsity:.2f}')

    if args.use_w and layer_key in ['style', 'g_mapping'] or args.save_noise>0:
        edit_modes = ['latent'] # activation edit is the same
        print("only latent")
    else:
        edit_modes = ['activation', 'latent']

    mode = 'latent'

    # 1: (3, 3, -2, -4, 'zoom')
    #
    # 13: (2, 7, -10, 'round', 'small'),
    #
    # 2: (2, 3, -3, 'elongate'),
    # 3: (2, 3, -4, 'double, longer', 'bigger'), 4: (2, 5, -3, 'double', 'middle small'),
    # 5: (2, 4, -4, 'zoom out', 'zoom in'),
    # 15: (3, 12, 6, -8, 'crop to rantangle', 'zoom out'),
    # 16: (2, 10, -12, 'backgroudn plain', 'rich background'),
    # 17: (2, 10, -11, 'dark blue background', 'yellowish bk'),

    # TODO: the reason we use PCA, is not to uniformly sample the z, but uniformly sample the hidden feature (transformation),
    # TODO: if we uniformly sample z, the feature space will be not uniformly sampled, since we want to upweight the rare transformation
    # due to the y,

    scale_pair_updown = {
        0: (2, -3, 3, 'orientation'), 1: (3, 2, -2, -4, 'zoom'), 2: (2, 3, -3, 'elongate'),
        3: (2, 3, -4, 'double, longer', 'bigger'), 4: (2, 5, -3, 'double', 'middle small'),
        5: (2, 4, -4, 'zoom out', 'zoom in'),
        6:(2, 5,-5), 7:(2, 5,-5),8:(2, 6,-6),9:(2, 6,-6),10:(2, 7,-7),
        11: (2, 8, -8, 'crop', 'longer'),
        12: (3, 10, 6, -8, 'whitebackground', 'dirtbackground'), 13: (2, 7, -10, 'round', 'small'),
        14: (2, 10, -10, 'background'), 15: (3, 8, 6, -8, 'crop to rantangle', 'zoom out'),
        16: (2, 10, -10, 'backgroudn plain', 'rich background'),
        17: (2, 10, -10, 'dark blue background', 'yellowish bk'), 18: (2, 10, -5, 'whitebottom', ''),
        19: (2, 8, -10, 'unk', 'blackwhite'), 20: (2, 12, -8),
        21: (2, 10, -10, 'peoplebk', 'bk'), 22: (2, 7, -10, '', 'lesscolor'),
        23: (4, 10, 6, -6, -10, 'impressive_pencil_sketch', 'dark_bk'),
        24: (2, 10, -12, 'unk', 'blueyellow bk'),
        25: (2, 10, -7, 'crop vertical', '?'), 26: (2, 8, -10, '?', 'whitening'),
        27: (2, 10, -10, 'heatmap?', 'whitepaint'), 28: (2, 8, -8, '?', 'beachbk'),
        29: (2, 8, -8, 'blackwhite', 'red?'),
        30: (2, 8, -8),
        31: (2, 12, -8, 'red', 'insidegrass'),
        32: (2, 9, -10, 'grass', 'red'),
        33: (2, 10, -10), 34: (2, 10, -10), 35: (2, 10, -10), 36: (2, 12, -12),

        37: (2, 12, -10, 'red', 'blue'), 38: (2, 12, -12), 40: (2, 12, -12), 41: (2, 10, -12),
        42: (2, 12, -10), 43: (2, 12, -12), 44: (2, 8, -12, 'red', 'green'),
        45: (2, 12, -12, '.', 'add text to img '), 46: (2, 12, -12), 47: (2, 12, -12),
        48: (2, 12, -13, '', 'everhting in green water!'),
        49: (2, 12, -12), 50: (2, 12, -12, 'sharp', 'blur'),
        51: (2, 12, 6, -12, 'colorful', 'not color'),
        52: (2, 12, -13, 'bokeh', 'not colorful'), 53: (2, 13, -12, 'sharp color', 'not sharp'),
        55: (2, 15, -12, 'color', 'less color'), 56: (2, 15, -12), 57: (2, 15, -12, 'dawn', ), 58: (2,12, -12),
        59: (2, 12, -10),
        39: (4, 10, -12, 5, -6, 'blackwhite', 'saturation'), 54: (2, 12, -12, 'purple', 'green')}

    # TODO: thinking: If PCA captures the major variation of generator (by random sampling), then the least PCA direction
    # should be the direction that is under-estimated during random sampling, thus larger intervention on those direction
    # can produce out-of-distribution data.

    # THese intervention has poor quality at Objnet.
    # additional_intervention = {}
    # for each in range(60, 80):
    #     if each==69:
    #         additional_intervention[each] = (2, 8, -14)
    #     else:
    #         additional_intervention[each] = (2, 10, -10)
    #
    # scale_pair_updown.update(additional_intervention)

    import random

    center = True
    normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + 1e-8)

    layer_start = 0
    layer_end = -1
    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)

    all_cat = [i for i in range(args.class_start, args.class_end)]
    if obj_only:
        all_cat = overlapping_list

    for each_category in all_cat:
        if os.path.isdir(os.path.join(output_dir, num2id[each_category])):
            print(each_category, ' exists')
            continue
        else:
            print('Now running', num2id[each_category])


        print("category", each_category)
        folder_name = os.path.join(output_dir, '{}'.format(num2id[each_category]))
        makedirs(folder_name, exist_ok=False)

        model.set_output_class(each_category)
        english_name = num2name[each_category]

        for randseed in range(args.random_num_start, args.random_num_end):
            if rerun:
                np_file_name = os.path.join(folder_name, "rand_seed_{}.npy".format(randseed))
                latents = torch.from_numpy(np.load(np_file_name)).cuda()

            else:
                latents = model.sample_latent(n_samples=1, truncation=args.truncation)

                np_file_name = os.path.join(folder_name, "rand_seed_{}.npy".format(randseed))
                np.save(np_file_name, latents.clone().cpu().numpy())

        # for diff_seed in range(args.random_num_start, args.random_num_end):  #######3
        #     latents = model.sample_latent(n_samples=1, truncation=args.truncation) # TODO: maybe small truncation yield better result.

            comp_sort = scale_pair_updown.keys()
            selected = random.sample(comp_sort, 5)  # For each random image, we select random PCA direction for intervention, here 5 for verion 1.
            for each_comp in selected: # We can get all
                # scale = args.sigma
                # sigma_range = np.linspace(-scale, scale, num_resolution)

                c = each_comp
                lat_mean = tensors.Z_global_mean  # TODO: we might need individual class's mean value, if they are very different
                lat_comp = tensors.Z_comp[c:c + 1, :, :]
                lat_stdev = tensors.Z_stdev[c:c + 1]

                act_mean = tensors.X_global_mean   # x means activation
                act_comp = tensors.X_comp[c:c + 1, :, :]
                act_stdev = tensors.X_stdev[c:c + 1]

                # is it ok for r=0? Yes. becauwe it select the component dim, but it need to kepp that vec, not just first element .
                # print('act_comp shape', act_comp.shape, lat_comp.shape)
                r=0
                x_comp = act_comp[r]
                z_comp = lat_comp[r]
                act_stdev = act_stdev[r]
                lat_stdev = lat_stdev[r]

                img_list_len = scale_pair_updown[each_comp][0]

                max_offset = max(scale_pair_updown[each_comp][1:1+img_list_len])
                min_offset = min(scale_pair_updown[each_comp][1:1+img_list_len])

                # # sig_list = np.asarray([scale_pair_updown[each_comp][0], scale_pair_updown[each_comp][1]])
                # for random_idx in range(img_list_len):
                #     # selected = random.sample([i for i in range(num_resolution)], 1)
                #     # offset = sig_list[[random_idx]]
                #     offset = scale_pair_updown[each_comp][random_idx + 1]
                #     if offset>10:
                #         offset = offset / 1.2

                for random_idx in range(1):
                    # offset = random.randrange(min_offset, max_offset, 200)

                    offset = random.random() * (max_offset-min_offset) + min_offset
                    # if each_category in overlapping_list:
                    #     if abs(offset)>8:
                    #         offset = offset/1.2


                    # folder_name = os.path.join(output_dir,
                    #                            "comp_{}_rand_{}_offset_{}".format(each_comp, diff_seed, float(offset)))

                    np_dict_save = {}
                    np_dict_save['noise_seed'] = latents
                    np_dict_save['component'] = each_comp
                    np_dict_save['offset'] = offset

                    # actual input seed?
                    # np_file_name = os.path.join(folder_name, "label.npy")
                    # np.save(np_file_name, np_dict_save)

                    # offset = torch.from_numpy(offset).float().to(x_comp.device)

                    z_single = latents
                    z_batch = z_single.repeat_interleave(1, axis=0)  # TODO:?
                    layer = layer_key

                    zeroing_offset_act = 0
                    zeroing_offset_lat = 0
                    if center:
                        if mode == 'activation':
                            # Center along activation before applying offset
                            inst.retain_layer(layer)
                            _ = inst.model.sample_np(z_single)
                            value = inst.retained_features()[layer].clone()
                            dotp = torch.sum((value - act_mean) * normalize(x_comp), dim=-1, keepdim=True)
                            zeroing_offset_act = normalize(x_comp) * dotp  # offset that sets coordinate to zero
                        else:
                            # Shift latent to lie on mean along given component
                            dotp = torch.sum((z_single - lat_mean) * normalize(z_comp), dim=-1, keepdim=True)
                            zeroing_offset_lat = dotp * normalize(z_comp)

                    with torch.no_grad():
                        z = z_batch

                        if mode in ['latent', 'both']:
                            z = [z] * inst.model.get_max_latents()
                            delta = z_comp * offset * lat_stdev  # .reshape([-1] + [1] * (z_comp.ndim - 1))
                            for i in range(layer_start, layer_end):
                                z[i] = z[i] - zeroing_offset_lat + delta

                            # print('latent z', latents)
                            # print('after', z[0])
                            # import pdb;
                            # pdb.set_trace()

                        if mode in ['activation', 'both']:
                            comp_batch = x_comp.repeat_interleave(1, axis=0)
                            delta = comp_batch * offset  # .reshape([-1] + [1] * (comp_batch.ndim - 1))
                            inst.edit_layer(layer, offset=delta * act_stdev - zeroing_offset_act)

                        img_batch = inst.model.sample_np(z)  # THIS is where data is forwarded
                        if img_batch.ndim == 3:
                            img_batch = np.expand_dims(img_batch, axis=0)

                        single_img = img_batch[0]

                        single_img = single_img * 256
                        out_array = np.asarray(np.uint8(single_img), dtype=np.uint8)
                        img_out = PIL.Image.fromarray(out_array)
                        current_file_name = os.path.join(folder_name,
                                                         "rand_{}_comp_{}_offset_{}{}.png".format(randseed, each_comp, offset, rerun))
                        img_out.save(current_file_name, 'png')


