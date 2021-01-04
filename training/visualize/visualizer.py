'''
Contains script and class to visualise during evaluation.
Scripts to run:
- Invariant/scripts/interpret/interpret_evaluate.sh
- Invariant/scripts/interpret/save_predictions.sh

'''

import os
import sys
import glob
import tqdm
import json
import random
import socket
import shutil
import pickle
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import warnings
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", module="matplotlib")

import logging
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

from PIL import Image
from skimage.transform import resize

import torch
import torch.nn.functional as F


import torchvision
from torchvision import models
from torchvision import transforms
from torchvision import datasets

from captum.attr import IntegratedGradients
from captum.attr import GuidedGradCam 
from captum.attr import LayerGradCam 
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
sys.path.append("./")
sys.path.append("./..")
from learning.resnet import resnet18, resnet50, resnet152, Identity
from data.KNN_dataloader import ObjectNetLoader

torch.manual_seed(0)
np.random.seed(0)

class Visualizer:
    '''
    Class to visualise the predictions of trained models.
    Allows seeing images with labels, showing attributions.
    Current datasets:
    1. Imagenet
    2. Objectnet
    3. ShapeNet
    4. Imagenet A
    5. Custom
    Current models: 
    1. Resnet 152
    '''
    def __init__(self, model=None):
        # self.model = model
        pass


    def savePredictions(self):
        # Predict


        # Save the predictions for corresponding images
        pass

    def getAttributionFigure(self, attribution_algorithm, model, input,transformed_img, top_pred_idx, target_layer=None):

        if attribution_algorithm == "integrated_gradients":
            integrated_gradients = IntegratedGradients(model)
            attributions_ig = integrated_gradients.attribute(input, target=top_pred_idx, n_steps=5)
            default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                             [(0, '#ffffff'),
                                                              (0.25, '#000000'),
                                                              (1, '#000000')], N=256)

            f = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                         np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                         method='heat_map',
                                         cmap=default_cmap,
                                         show_colorbar=True,
                                         sign='positive',
                                         outlier_perc=1)

        elif attribution_algorithm == "gradient_shap":
            gradient_shap = GradientShap(model)
            rand_img_dist = torch.cat([input * 0, input * 1])
            attributions_gs = gradient_shap.attribute(input,
                                                      n_samples=20,
                                                      stdevs=0.0001,
                                                      baselines=rand_img_dist,
                                                      target=top_pred_idx)
            default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                             [(0, '#ffffff'),
                                                              (0.25, '#000000'),
                                                              (1, '#000000')], N=256)
            f = viz.visualize_image_attr_multiple(
                np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                ["original_image", "heat_map"],
                ["all", "absolute_value"],
                cmap=default_cmap,
                show_colorbar=True)

        elif attribution_algorithm == 'guided_gradcam':
            #TODO: get the last conv layer depending on the architecture, not hardcoded like here.
            if not target_layer:
                if isinstance(model, torch.nn.DataParallel):
                    target_layer = list(model.module.children())[-3][1].conv2
                else:
                    target_layer = list(model.children())[-3][1].conv2

            guided_gc = GuidedGradCam(model, target_layer)
            attribution = guided_gc.attribute(input, top_pred_idx)
            assert (attribution.shape == input.shape)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            def normalise_image_array(arr, mean=mean, std=std):
                ''' Reverses normalisation for displaying in plt, and moves axis to later'''
                # print(np.min(arr), np.max(arr), np.mean(arr))
                im = np.moveaxis(arr[0], 0, 2) * std + mean
                return im

            input_array = input.detach().cpu().numpy()
            normalised_input_array = normalise_image_array(input_array)

            gc = attribution.detach().cpu().numpy()
            display_array = normalise_image_array(gc * 500)
            fig,ax = plt.subplots()
            ax.imshow(display_array)
            f = fig,ax
        
        elif attribution_algorithm == 'guided_gradcam':
            #TODO: get the last conv layer depending on the architecture, not hardcoded like here.
            if not target_layer:
                if isinstance(model, torch.nn.DataParallel):
                    target_layer = list(model.module.children())[-3][1].conv2
                else:
                    target_layer = list(model.children())[-3][1].conv2

            guided_gc = GuidedGradCam(model, target_layer)
            attribution = guided_gc.attribute(input, top_pred_idx)
            assert (attribution.shape == input.shape)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            def normalise_image_array(arr, mean=mean, std=std):
                ''' Reverses normalisation for displaying in plt, and moves axis to later'''
                # print(np.min(arr), np.max(arr), np.mean(arr))
                im = np.moveaxis(arr[0], 0, 2) * std + mean
                return im

            input_array = input.detach().cpu().numpy()
            normalised_input_array = normalise_image_array(input_array)

            gc = attribution.detach().cpu().numpy()
            display_array = normalise_image_array(gc * 500)
            fig,ax = plt.subplots()
            ax.imshow(display_array)
            f = fig,ax

        elif attribution_algorithm == 'layer_gradcam':
            #TODO: get the last conv layer depending on the architecture, not hardcoded like here.


            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            def normalise_image_array(arr, mean=mean, std=std):
                ''' Reverses normalisation for displaying in plt, and moves axis to later'''
                # print(np.min(arr), np.max(arr), np.mean(arr))
                im = np.moveaxis(arr[0], 0, 2) * std + mean
                return im

            if not target_layer:
                if isinstance(model, torch.nn.DataParallel):
                    target_layer = list(model.module.children())[-3][1].conv2
                else:
                    target_layer = list(model.children())[-3][1].conv2

            gc = LayerGradCam(model, target_layer)
            attribution = gc.attribute(input, top_pred_idx)
            
            array_image = input.detach().cpu().numpy()
            array_attribution = attribution.detach().cpu().numpy()[0][0]
            array_attribution = resize(array_attribution, (256,256))

            assert (array_attribution.shape == input.shape[-2:])

            input_array = input.detach().cpu().numpy()
            normalised_input_array = normalise_image_array(input_array)

            fig,ax = plt.subplots()
            ax.imshow(normalised_input_array)
            ax.imshow(array_attribution,cmap='jet',interpolation='bilinear', alpha=0.4, 
                    vmin=0.0, vmax=0.5,
                    )
            f = fig,ax


        return f

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def getModel(path_model=None, arch='resnet152', gpu=None, load_pretrained=True):
    # TODO : Flag for pretrained=True would allow downloading and loading model from URL.
    if arch == 'resnet152':
        model = resnet152(finetune_last=False, normalize=False, oneoutput=True)
        model = torch.nn.parallel.DataParallel(model)
        if load_pretrained:
            if os.path.isfile(path_model):
                print("=> loading checkpoint '{}'".format(path_model))
                if gpu is None:
                    loc = 'cpu'
                else:
                    loc = 'cuda:{}', format(gpu)
                checkpoint = torch.load(path_model, map_location=loc)  # TODO map_location for GPU
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path_model, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path_model))

        # self.model = model
        return model

    # elif arch == 'resnet18':
    #     model = resnet18(finetune_last=False, normalize=False, oneoutput=True)
    #     model = torch.nn.parallel.DataParallel(model)
    #     if load_pretrained:
    #         if os.path.isfile(path_model):
    #             print("=> loading checkpoint '{}'".format(path_model))
    #             if gpu is None:
    #                 loc = 'cpu'
    #             else:
    #                 loc = 'cuda:{}',format(gpu)
    #             checkpoint = torch.load(path_model, map_location=loc)  # TODO map_location for GPU
    #             model.load_state_dict(checkpoint['state_dict'])
    #             print("=> loaded checkpoint '{}' (epoch {})"
    #                   .format(path_model, checkpoint['epoch']))
    #         else:
    #             print("=> no checkpoint found at '{}'".format(path_model))
    #     return model

    elif arch == 'resnet152_mlp':
        model = resnet152(finetune_last=False, normalize=False, feature_only=True)
        model = torch.nn.parallel.DataParallel(model)

        MLP = Identity()
        MLP = torch.nn.DataParallel(MLP)

        if load_pretrained:
            if os.path.isfile(path_model):
                print("=> loading checkpoint '{}'".format(path_model))
                if gpu is None:
                    loc = 'cpu'
                else:
                    loc = 'cuda:{}', format(gpu)
                checkpoint = torch.load(path_model, map_location=loc)  # TODO map_location for GPU
                # best_acc1 = checkpoint['best_acc1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path_model, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path_model))

        # self.model = model
        return model, MLP

    else:
        model = models.__dict__[arch]()
        # model = resnet18(finetune_last=False, normalize=False, oneoutput=True)
        model = torch.nn.DataParallel(model)
        #### debug
        # model = models.resnet18(pretrained=True)
        # model = torch.nn.DataParallel(model)
        # return model
        ###
        if load_pretrained:
            if os.path.isfile(path_model):
                print("=> loading checkpoint '{}'".format(path_model))
                if gpu is None:
                    loc = 'cpu'
                else:
                    loc = 'cuda:{}', format(gpu)
                checkpoint = torch.load(path_model, map_location=loc)  # TODO map_location for GPU
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path_model, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path_model))

        # self.model = model
        # model = model.eval()
        return model

def getListPathsFromPredictionDict(dict_predictions1, dict_predictions2, only_correct=False, dict_obj2imagenet_id=None):
    ''' Get the list of paths for which to get attributions'''

    # Remember that predictions are actually the indices of the imagenet categories which are also in objectnet.
    array_paths1 = dict_predictions1['paths']
    array_paths2 = dict_predictions2['paths']
    # print(len(array_paths1))
    # exit(0)
    objnet_predictions1 = dict_predictions1['objnet_predictions']
    objnet_predictions2 = dict_predictions2['objnet_predictions']

    top1_objnet_predictions1 = objnet_predictions1[:,0]
    top1_objnet_predictions2 = objnet_predictions2[:,0]
    
    array_labels = np.array([p.split('/')[-2] for p in array_paths1])
    array_labels_indices = np.array([dict_obj2imagenet_id[label] for label in array_labels])
    if only_correct:
        condition1 = (top1_objnet_predictions1 != top1_objnet_predictions2)
        condition2 = []
        for idx_pred, pred_label_idx in enumerate(array_labels_indices):
            # print(array_paths1[idx_pred], top1_objnet_predictions2[idx_pred],pred_label_idx)
            condition2.append(top1_objnet_predictions2[idx_pred] in pred_label_idx)
        condition2 = np.array(condition2)
        where_predictions_mismatch = np.where(condition1 & condition2)

    else:
        where_predictions_mismatch = np.where((top1_objnet_predictions1 != top1_objnet_predictions2))
    assert(np.all(array_paths1 == array_paths2)) # Check that all paths are same

    list_paths_mismatch = array_paths1[where_predictions_mismatch]

    print("Only Correct: {}, {} out of {} image paths are selected for top 1 predictions".format(only_correct,len(list_paths_mismatch), len(array_paths1)))
    return list_paths_mismatch

def parse_args():
    
    parser = argparse.ArgumentParser(description='Parse arguments for visualisations')
    parser.add_argument('-e', '--eval_mode', metavar='EVAL_MODE', default='resnet18_xent',
                        help='Evaluation mode: renet152_xent/resnet18_xent')
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--path_model', default='', type=str, metavar='PATH_MODEL',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--path_save_predictions', default='', type=str, metavar='PATH_SAVE_PREDICTIONS',
                        help='path where you want to save the predictions/load the predictions from')
    
    
    parser.add_argument('-d', '--dataset', default='', type=str, metavar='dataset',
                        help='dataset objectnet/imageneta/imagenetc/imagenetp')
    parser.add_argument('-n', '--num_images_to_save', default=10, type=int, metavar='N', help='Specify number of images to save')
    parser.add_argument('-b', '--batch_size', default=10, type=int, metavar='BS', help='Specify batch size')
    parser.add_argument('--num_categories', type=int, metavar='N', help='Specify number of categories to save')
    parser.add_argument('-a', '--attribution_algorithms',nargs='+', metavar='N', help='Specify the algorithms to use for interpretability')
    parser.add_argument('-c', '--config_name', metavar='CONFIG',
                        help='Config name to see which model is under use')
    
    parser.add_argument('--shuffle', action='store_true') # Currently only being used in mismatch interpret (eval mode: resnet_xent, dataset: objectnet_custom_path_list)
    parser.add_argument('--save_nolabel', action='store_true') # Currently only being used in mismatch interpret (eval mode: resnet_xent, dataset: objectnet_custom_path_list)

    # ONLY FOR --eval_mode objectnet_custom_path_list
    parser.add_argument('--path_save_images', type=str, metavar='PATH_SAVE_IMAGES',
                        help='path where you want to save the images')
    parser.add_argument('--path_list_images', type=str, metavar='PATH_LIST_IMAGES',
                        help='path where you want to save the images')

    parser.add_argument('--config_name2', metavar='CONFIG-2',help='')
    parser.add_argument('--path_model2', type=str, metavar='PATH_MODEL',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--config_name3', metavar='CONFIG-2',help='')
    parser.add_argument('--path_model3', type=str, metavar='PATH_MODEL',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()
    print(args)

    return args

def main():
    
    args = parse_args()
    
    # Define paths
    if socket.gethostname() == 'hulk':
        traindir = '/local/rcs/mcz/ImageNet-Data/train'
        valdir = '/local/rcs/mcz/ImageNet-Data/val'
        obj_valdir = '/local/rcs/shared/objectnet-1.0/overlap_category_test'
        imageneta_valdir = ''

    elif 'cv' in socket.gethostname():
        path_imagenet_train = '/proj/vondrick/mcz/ImageNet-Data/train'
        path_imagenet_val = '/proj/vondrick/mcz/ImageNet-Data/val'
        path_objectnet = '/proj/vondrick/augustine/objectnet-1.0/'
        path_objectnet_full = '/proj/vondrick/augustine/objectnet-1.0/images'
        path_objectnet_overlap = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test'  #
        path_objectnet_overlap_noborder = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test_noborder'
        path_model_resnet152 = '/proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/model_best.pth.tar'
        #path_interpret_results = '/proj/vondrick/amogh/results/interpret'
        path_interpret_results = '/proj/vondrick/www/amogh/examples/results/interpret'

        path_save_predictions = '/proj/vondrick/www/amogh/examples/results/predictions'


        # path_model_resnet18 = '/proj/vondrick/mcz/ModelInvariant/SavedModels/Neurips2020/res18-bl-imgnet/train_2020-05-07_19:39:38_7364c0a8_rotation_False/model_best.pth.tar'
        # path_model_resnet18 = '/proj/vondrick/mcz/ModelInvariant/SavedModels/Neurips2020/res18-split-sig7-lam0.05-ibs192/train_2020-05-14_14:11:54_a52c34c9_rotation_False/model_best.pth.tar'
        # path_model_resnet18 = '/proj/vondrick/mcz/ModelInvariant/SavedModels/Neurips2020/res18-split-gan-large500-longer-lam0.1/train_2020-05-09_21:46:51_43ac6bbe_rotation_False/model_best.pth.tar'
        # path_model_resnet18 = '/proj/vondrick/mcz/ModelInvariant/SavedModels/Neurips2020/res18-split-gan/train_2020-05-09_12:45:32_b0a3118a_rotation_False/model_best.pth.tar'
        # path_model_resnet18 = '/proj/vondrick/mcz/ModelInvariant/SavedModels/Neurips2020/res18-split-gan-lam0.05-rot/train_2020-05-11_23:00:10_f5447777_rotation_True/model_best.pth.tar'


    elif socket.gethostname() == 'amogh':
        traindir = '/local/vondrick/cz/cut_img-both3.5/train_clustered-C3.5'
        valdir = '/local/vondrick/cz/ImageNet/val'
        obj_valdir = '/proj/vondrick/augustine/objectnet-1.0/overlap_category_test'


    evaluation_mode = args.eval_mode
    dataset = args.dataset
    path_model = args.path_model
    num_images_to_save = args.num_images_to_save
    attribution_algorithms = args.attribution_algorithms
    batch_size = args.batch_size
    num_workers = args.num_workers

    if args.path_save_predictions: # This is set by default but you can override it
        path_save_predictions = args.path_save_predictions
    
    if args.path_save_images:
        path_save_images = args.path_save_images

    save_figures = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)

    # evaluation_mode = 'resnet152_xent'
    # evaluation_mode = 'resnet18_xent'
    # evaluation_mode = 'resnet152_meta_knn'
    # dataset = 'objectnet'
    # path_model = path_model_resnet18
    # num_images_to_save = 20
    # attribution_algorithms = ["integrated_gradients", "gradient_shap"]
    # attribution_algorithms = ["integrated_gradients"]
    # attribution_algorithms = ["gradient_shap"]
    # attribution_algorithms = ["guided_gradcam"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    #Another transform to experiment with zoom
    transform2 = transforms.Compose([
        transforms.Resize(int(256*1.4)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # !wget - P $HOME /.torch / models
    # https: // s3.amazonaws.com / deep - learning - models / image - models / imagenet_class_index.json

    #if dataset == 'objectnet':
    #    test_loader = torch.utils.data.DataLoader(
    #            ObjectNetLoader(path_objectnet_overlap_noborder),
    #            batch_size=1, shuffle=False,
    #            num_workers=10, pin_memory=True)

    # Load Imagenet Info
    labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    # Evaluating with xent loss - Load images from folder, get predictions from the model
    if evaluation_mode == 'resnet152_xent':

        visualizer = Visualizer()

        model = getModel(path_model=path_model,
                         arch='resnet152')  # TODO Allow loading other model architecture
        model = model.eval()
        # print("___here")
        # exit(0)
        ### debug
        # model = models.resnet18(pretrained=True)
        # model = torch.nn.DataParallel(model)
        # model = model.eval()


        # Code for ResNet18. #TODO - put the redundant code at end, shouldnt copy objectnet code for all models.
        if dataset == 'objectnet':
            
            # Get overlapping indices of imagenet with objectnet (Read obj2imgnet.txt)
            # Get the max only from these.
            with open('./preprocessing/obj2imgnet_id.txt') as f:
                dict_obj2imagenet_id = json.load(f)
            list_idx_imgnet_overlap_objnet = [] # Contains the category numbers from imagenet that are also in Objectnet
            for list_indices in dict_obj2imagenet_id.values():
                list_idx_imgnet_overlap_objnet.extend(list_indices)
            list_idx_imgnet_overlap_objnet.sort()

            # Go over all categories of imagenet and process num_images_to_save images for each category
            list_categories_objectnet = os.listdir(path_objectnet_overlap_noborder)
            if args.num_categories:
                list_categories_objectnet = list_categories_objectnet[:args.num_categories]
                print("Saving {} categories".format(len(list_categories_objectnet)))
            else:
                print("Saving {} categories".format(len(list_categories_objectnet)))
            for category_objectnet in tqdm.tqdm(list_categories_objectnet):
                path_category_objectnet = os.path.join(path_objectnet_overlap_noborder, category_objectnet)
                list_path_category_images = glob.glob(path_category_objectnet + '/*')

                for attr_algorithm in attribution_algorithms:
                    if save_figures:
                        # Saving for multiple models would require config name
                        config_name = args.config_name
                        if config_name:
                            # print("Using config name", config_name )
                            path_results_directory = os.path.join(path_interpret_results,
                                                                "{}/{}_{}_{}".format(category_objectnet, config_name, attr_algorithm,
                                                                                    "objnet_predictions"))
                        else: 
                            path_results_directory = os.path.join(path_interpret_results,
                                                                "{}/{}_{}".format(category_objectnet, attr_algorithm,
                                                                                    "objnet_predictions"))
                                                                    
                        # Saving for multiple models would require config name
                        if not os.path.exists(path_results_directory):
                            os.makedirs(path_results_directory)
                            # print("Making directory: ", path_results_directory)
                        # print("Images saved in folder : ", path_results_directory)

                # print("In category : ", category_objectnet)
                #TODO: Change this to be a dataloader. STEPS --> Assure that Captum is able to return attributions for a batch and not just a single image.
                for num_image, path_image in enumerate(list_path_category_images):
                    # print("path : ", path_image)
                    if num_image == num_images_to_save:
                        break
                    img = Image.open(path_image)

                    # transformed_img = transform(img) # For imagenet images
                    transformed_img = transform(img)[:3]

                    input = transform_normalize(transformed_img)
                    input = input.unsqueeze(0)

                    model = model.to(device)
                    input = input.to(device)

                    output = model(input)
                    output = F.softmax(output, dim=1)
                    prediction_score, pred_label_idx = torch.topk(output, 5)
                    pred_label_idx.squeeze_()
                    predicted_labels = [idx_to_labels[str(idx)][1] for idx in pred_label_idx.detach().cpu().numpy()]
                    predicted_scores = prediction_score.squeeze().detach().cpu().numpy()
                    predicted_scores = [round(t, 4) * 100 for t in predicted_scores]
                    predicted_label = predicted_labels[0]
                    # print('Predicted:', predicted_label, '(', prediction_score.squeeze()[0].item(), ')')
                    top_pred_idx = pred_label_idx[0].item()

                    # GET OBJECTNET SPECIFIC LABELS
                    # Get top 10 labels (Among objectnet only)
                    overlap_output = output.squeeze()[list_idx_imgnet_overlap_objnet]
                    overlap_prediction_score, overlap_pred_label_idx = torch.topk(overlap_output, 10)
                    overlap_predicted_labels = [idx_to_labels[str(list_idx_imgnet_overlap_objnet[idx])][1] for idx in
                                                overlap_pred_label_idx.detach().cpu().numpy()]
                    overlap_predicted_scores = overlap_prediction_score.squeeze().detach().cpu().numpy()
                    overlap_predicted_scores = [round(t * 100, 4) for t in overlap_predicted_scores]
                    # print("Overlapping predictions are : ", overlap_predicted_labels, predicted_label, overlap_predicted_scores)

                    # Use visualizer object to get visualisation figure for each algorithm and save.
                    for attr_algorithm in attribution_algorithms:
                        #print((input.requires_grad))
                        #input.requires_grad_(True)
                        f = visualizer.getAttributionFigure(attr_algorithm, model, input, transformed_img, top_pred_idx)

                        # print(type(f))
                        # f[0].suptitle('Predicted labels: {} \n Scores: {}'.format(predicted_labels, predicted_scores), fontsize=12)
                        f[0].suptitle(
                            'True label (objnet): {} \n Predicted labels(objnet only): {} \n Scores: {} \n Predicted labels(include Imgnet): {}'.format(
                                category_objectnet, overlap_predicted_labels, overlap_predicted_scores, predicted_labels), fontsize=12)

                        if save_figures:
                            path_save = os.path.join(path_results_directory, os.path.basename(path_image))
                            # print("saving at", path_save)
                            f[0].savefig(path_save, bbox_inches='tight')
                        plt.close()


        # IF WE WANT TO PROCESS A LIST OF PATHS

        elif dataset == 'objectnet_custom_path_list':

            # Get overlapping indices of imagenet with objectnet (Read obj2imgnet.txt)
            # Get the max only from these.
            with open('./preprocessing/obj2imgnet_id.txt') as f:
                dict_obj2imagenet_id = json.load(f)
            list_idx_imgnet_overlap_objnet = [] # Contains the category numbers from imagenet that are also in Objectnet
            for list_indices in dict_obj2imagenet_id.values():
                list_idx_imgnet_overlap_objnet.extend(list_indices)
            list_idx_imgnet_overlap_objnet.sort()

            # Get the path of predictions based on config_name

            # Get the list of images that we want to get predictions on
            with open(path_predictions1, 'rb') as f:
                dict_predictions1 = pickle.load(f)
            with open(path_predictions2, 'rb') as f:
                dict_predictions2 = pickle.load(f)
            list_paths = getListPathsFromPredictionsDict(dict_predictions1, dict_predictions2, only_correct=True)
            print("Getting predictions for {} paths".format(len(list_paths)))
            #####INCOMPLETE --> see under resnet18

            exit(0) 

        # TO TEST A CUSTOM DATASET (eg - Print a model's predictions with paths)
        else:
            
            # Make a dataloader for the cutom dataset

            path_dataset = '/local/vondrick/cz/GANdata/rand'

            dataset = ImageFolderWithPaths(path_dataset,transform=transforms.Compose([transform,transform_normalize])) # our custom dataset
            dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True 
                                                )

            model = model.to(device)
            images_misclassified = []

            for ii, (inputs, labels, paths) in enumerate(dataloader):
                # print(inputs.shape, labels, paths) # (batch_size, 3, 224,224), tensor([0,0,0,0,0..]), [path1, path2...]

                inputs = inputs.to(device)

                output = model(inputs)
                output = F.softmax(output, dim=1)
                prediction_score, pred_label_idx = torch.topk(output, 5)

                # Get top5 text labels and rounded off scores
                predicted_labels = np.array([[idx_to_labels[str(idx)][1] for idx in im_i_pred_idx] for im_i_pred_idx in pred_label_idx.squeeze().detach().cpu().numpy()])
                predicted_scores = np.round(prediction_score.squeeze().detach().cpu().numpy(),4)
                
                top1_predicted_labels = np.array([l[0] for l in predicted_labels])
                # print('Predicted:', predicted_labels, top1_predicted_labels, predicted_scores)

                
                array_labels = labels.detach().cpu().numpy()
                top1_predicted_idx = pred_label_idx[:,0].detach().cpu().numpy()
                incorrect_idx = np.where(top1_predicted_idx != array_labels)
                
                # print(array_labels, top1_predicted_idx, incorrect_idx)
                if len(incorrect_idx[0])>0:
                    images_misclassified.extend(np.array(paths)[incorrect_idx])
                    print("Incorrect images are {}, labels were {}, predictions  are: {}".format(np.array(paths)[incorrect_idx], 
                                                                                                array_labels[incorrect_idx],
                                                                                                predicted_labels[incorrect_idx]),
                        # array_labels[incorrect_idx],top1_predicted_idx[incorrect_idx]
                        )
                    # print(continue)

                # SAVE INFORMATION
                # dict_image2predictions = {}
                # for 
                # exit(0)

                if ii % 5000 ==0:
                    print(ii, "batches done")
            

    # The meta evaluation needs a different dataloader to generate episodes to visualize
    elif evaluation_mode == 'resnet152_meta_knn':

        # Initialise the dataloader for evaluation

        # Get the batches of episodes and get predictions
        pass

    elif evaluation_mode == 'resnet18_xent':

        visualizer = Visualizer()

        model = getModel(path_model=path_model,
                         arch='resnet18')  # TODO Allow loading other model architecture
        model = model.eval()
        # model = model.to(device)
        # print("___here")
        # exit(0)
        ### debug
        # model = models.resnet18(pretrained=True)
        # model = torch.nn.DataParallel(model)
        # model = model.eval()


        # Code for ResNet18. #TODO - put the redundant code at end, shouldnt copy objectnet code for all models.
        if dataset == 'objectnet':
            
            # Get overlapping indices of imagenet with objectnet (Read obj2imgnet.txt)
            # Get the max only from these.
            with open('./preprocessing/obj2imgnet_id.txt') as f:
                dict_obj2imagenet_id = json.load(f)
            list_idx_imgnet_overlap_objnet = [] # Contains the category numbers in imagenet that are also in Objectnet
            for list_indices in dict_obj2imagenet_id.values():
                list_idx_imgnet_overlap_objnet.extend(list_indices)
            list_idx_imgnet_overlap_objnet.sort()

            # Go over all categories of imagenet and process num_images_to_save images for each category
            list_categories_objectnet = os.listdir(path_objectnet_overlap_noborder)
            if args.num_categories:
                list_categories_objectnet = list_categories_objectnet[:args.num_categories]
                print("Saving {} categories".format(len(list_categories_objectnet)))
            else:
                print("Saving {} categories".format(len(list_categories_objectnet)))
            for category_objectnet in tqdm.tqdm(list_categories_objectnet):
                path_category_objectnet = os.path.join(path_objectnet_overlap_noborder, category_objectnet)
                list_path_category_images = glob.glob(path_category_objectnet + '/*')

                for attr_algorithm in attribution_algorithms:
                    if save_figures:
                        # Saving for multiple models would require config name
                        config_name = args.config_name
                        if config_name:
                            # print("Using config name", config_name )
                            path_results_directory = os.path.join(path_interpret_results,
                                                                "{}/{}_{}_{}".format(category_objectnet, config_name, attr_algorithm,
                                                                                    "objnet_predictions"))
                        else: 
                            path_results_directory = os.path.join(path_interpret_results,
                                                                "{}/{}_{}".format(category_objectnet, attr_algorithm,
                                                                                    "objnet_predictions"))
                                                                    
                        # Saving for multiple models would require config name
                        if not os.path.exists(path_results_directory):
                            os.makedirs(path_results_directory)
                            # print("Making directory: ", path_results_directory)
                        # print("Images saved in folder : ", path_results_directory)

                # print("In category : ", category_objectnet)
                #TODO: Change this to be a dataloader. STEPS --> Assure that Captum is able to return attributions for a batch and not just a single image.
                for num_image, path_image in enumerate(list_path_category_images):
                    # print("path : ", path_image)
                    if num_image == num_images_to_save:
                        break
                    img = Image.open(path_image)

                    # transformed_img = transform(img) # For imagenet images
                    transformed_img = transform(img)[:3]

                    input = transform_normalize(transformed_img)
                    input = input.unsqueeze(0)

                    model = model.to(device)
                    input = input.to(device)

                    output = model(input)
                    output = F.softmax(output, dim=1)
                    prediction_score, pred_label_idx = torch.topk(output, 5)
                    pred_label_idx.squeeze_()
                    predicted_labels = [idx_to_labels[str(idx)][1] for idx in pred_label_idx.detach().cpu().numpy()]
                    predicted_scores = prediction_score.squeeze().detach().cpu().numpy()
                    predicted_scores = [round(t, 4) * 100 for t in predicted_scores]
                    predicted_label = predicted_labels[0]
                    # print('Predicted:', predicted_label, '(', prediction_score.squeeze()[0].item(), ')')
                    top_pred_idx = pred_label_idx[0].item()

                    # GET OBJECTNET SPECIFIC LABELS
                    # Get top 10 labels (Among objectnet only)
                    overlap_output = output.squeeze()[list_idx_imgnet_overlap_objnet]
                    overlap_prediction_score, overlap_pred_label_idx = torch.topk(overlap_output, 10)
                    overlap_predicted_labels = [idx_to_labels[str(list_idx_imgnet_overlap_objnet[idx])][1] for idx in
                                                overlap_pred_label_idx.detach().cpu().numpy()]
                    overlap_predicted_scores = overlap_prediction_score.squeeze().detach().cpu().numpy()
                    overlap_predicted_scores = [round(t * 100, 4) for t in overlap_predicted_scores]
                    # print("Overlapping predictions are : ", overlap_predicted_labels, predicted_label, overlap_predicted_scores)

                    # Use visualizer object to get visualisation figure for each algorithm and save.
                    for attr_algorithm in attribution_algorithms:
                        #print((input.requires_grad))
                        #input.requires_grad_(True)
                        f = visualizer.getAttributionFigure(attr_algorithm, model, input, transformed_img, top_pred_idx)

                        # print(type(f))
                        # f[0].suptitle('Predicted labels: {} \n Scores: {}'.format(predicted_labels, predicted_scores), fontsize=12)
                        f[0].suptitle(
                            'True label (objnet): {} \n Predicted labels(objnet only): {} \n Scores: {} \n Predicted labels(include Imgnet): {}'.format(
                                category_objectnet, overlap_predicted_labels, overlap_predicted_scores, predicted_labels), fontsize=12)

                        if save_figures:
                            path_save = os.path.join(path_results_directory, os.path.basename(path_image))
                            # print("saving at", path_save)
                            f[0].savefig(path_save, bbox_inches='tight')
                        plt.close()
        # IF WE WANT TO PROCESS A LIST OF PATHS
        elif dataset == 'objectnet_custom_path_list':

            # Get overlapping indices of imagenet with objectnet (Read obj2imgnet.txt)
            # Get the max only from these.
            with open('./preprocessing/obj2imgnet_id.txt') as f:
                dict_obj2imagenet_id = json.load(f)
            list_idx_imgnet_overlap_objnet = [] # Contains the category numbers from imagenet that are also in Objectnet
            for list_indices in dict_obj2imagenet_id.values():
                list_idx_imgnet_overlap_objnet.extend(list_indices)
            list_idx_imgnet_overlap_objnet.sort()

            # Get the path of predictions based on config_name
            config_name = args.config_name
            config_name2 = args.config_name2
            path_predictions1 = os.path.join(path_save_predictions, "{}.dat".format(config_name))
            path_predictions2 = os.path.join(path_save_predictions, "{}.dat".format(config_name2))
            only_correct=True


            # Get the list of images that we want to get predictions on
            with open(path_predictions1, 'rb') as f:
                print("Loading from : ", path_predictions1)
                dict_predictions1 = pickle.load(f)
            with open(path_predictions2, 'rb') as f:
                dict_predictions2 = pickle.load(f)
                print("Loading from : ", path_predictions2)
            
            if args.path_list_images:
                with open(args.path_list_images, 'rb') as f:
                    list_paths = pickle.load(f)
            else:
                list_paths = getListPathsFromPredictionDict(dict_predictions1, dict_predictions2, only_correct=only_correct, dict_obj2imagenet_id=dict_obj2imagenet_id)
            

            
            # SAVE THIS MISMATCH PATH LIST
            path_mismatch_list = os.path.join(path_save_images, "paths_mismatch_c1_{}_c2_{}.dat".format(config_name,config_name2))
            with open(path_mismatch_list, 'wb') as f:
                pickle.dump(np.array(path_mismatch_list), f)
                print("Saved list at : ", path_mismatch_list)
            
            # For these mismatches, generate
            path_interpret_mismatch = os.path.join(path_save_images, "interpret_mismatch_c1_{}_c2_{}_allcorrect_{}_shuffle_{}").format(config_name, config_name2,str(only_correct), str(args.shuffle))
            if not os.path.exists(path_interpret_mismatch):
                os.makedirs(path_interpret_mismatch)
                print("Made directory to save images : ", path_interpret_mismatch)
            
            print(list_paths[:5])
            if args.shuffle:
                random.shuffle(list_paths)

            # Load another model
            path_model2 = args.path_model2
            model2 = getModel(path_model=path_model2,
                         arch='resnet18')  # TODO Allow loading other model architecture
            model2 = model2.eval()
            model = model.to(device)
            model2 = model2.to(device)
            dict_models = {config_name: model, config_name2:model2}

            if args.config_name3:
                config_name3 = args.config_name3
                path_model3 = args.path_model3
                model3 = getModel(path_model=path_model3,
                            arch='resnet18')  # TODO Allow loading other model architecture
                model3 = model3.eval()
                model3 = model3.to(device)
                dict_models[config_name3] = model3

            from collections import defaultdict
            dict_prediction_tags = defaultdict(list)
            
            ######################## ADDED TO GET SOME CATEGORIES ##########################
            selected_categories = False
            selected_categories = ['fan', 'bench', 'paper_towel']
            if selected_categories:
                l = []
                filter=False # Only from mismatched predictions
                if not filter:
                    # This is to predict all images from categories
                    list_paths = []
                    for c in selected_categories:
                        list_paths.extend(glob.glob(os.path.join(path_objectnet_overlap_noborder,c,'*')))
                    print(len(list_paths))
                else:
                    for p in list_paths:
                        for c in selected_categories:
                            if c in p:
                                l.append(p)
                    print(len(l))
                    list_paths = l
                # exit(0)
            ################################################################################
            

            # SAVE ATTRIBUTION IMAGES:
            for num_image, path_image in enumerate(list_paths):
                print("path : ", path_image)
                if num_image == num_images_to_save:
                    break
                img = Image.open(path_image)

                # transformed_img = transform(img) # For imagenet images
                transformed_img = transform(img)[:3]

                input = transform_normalize(transformed_img)
                input = input.unsqueeze(0)

                input = input.to(device)

                category_objectnet = path_image.split("/")[-2]
                # print(category_ob jectnet)

                # Save the original image
                path_interpret_mismatch_gt = os.path.join(path_interpret_mismatch, "gt")
                if not os.path.exists(path_interpret_mismatch_gt):
                    os.makedirs(path_interpret_mismatch_gt)
                path_gt_image_save = os.path.join(path_interpret_mismatch_gt, os.path.basename(path_image))
                print("Saving gt image at : ", path_gt_image_save)
                shutil.copy(path_image, path_gt_image_save)

                for conf in list(dict_models.keys()):
                    model = dict_models[conf]
                    output = model(input)
                    output = F.softmax(output, dim=1)
                    prediction_score, pred_label_idx = torch.topk(output, 5)
                    pred_label_idx.squeeze_()
                    predicted_labels = [idx_to_labels[str(idx)][1] for idx in pred_label_idx.detach().cpu().numpy()]
                    predicted_scores = prediction_score.squeeze().detach().cpu().numpy()
                    predicted_scores = [round(t, 4) * 100 for t in predicted_scores]
                    predicted_label = predicted_labels[0]
                    # print('Predicted:', predicted_label, '(', prediction_score.squeeze()[0].item(), ')')
                    top_pred_idx = pred_label_idx[0].item()

                    # GET OBJECTNET SPECIFIC LABELS
                    # Get top 10 labels (Among objectnet only)
                    overlap_output = output.squeeze()[list_idx_imgnet_overlap_objnet]
                    overlap_prediction_score, overlap_pred_label_idx = torch.topk(overlap_output, 10)
                    overlap_predicted_labels = [idx_to_labels[str(list_idx_imgnet_overlap_objnet[idx])][1] for idx in
                                                overlap_pred_label_idx.detach().cpu().numpy()]
                    overlap_predicted_scores = overlap_prediction_score.squeeze().detach().cpu().numpy()
                    overlap_predicted_scores = [round(t * 100, 4) for t in overlap_predicted_scores]
                    # print("Overlapping predictions are : ", overlap_predicted_labels, predicted_label, overlap_predicted_scores)

                    suptitle = 'True label (objnet): {} \n Predicted labels(objnet only): {} \n Scores: {} \n Predicted labels(include Imgnet): {} \n Config: {} \n'.format(
                                category_objectnet, overlap_predicted_labels, overlap_predicted_scores, predicted_labels, conf)
                        # print(suptitle)
                    dict_prediction_tags[os.path.basename(path_image)].append(suptitle)


                    # Use visualizer object to get visualisation figure for each algorithm and save.
                    for attr_algorithm in attribution_algorithms:
                        #print((input.requires_grad))
                        #input.requires_grad_(True)
                        f = visualizer.getAttributionFigure(attr_algorithm, model, input, transformed_img, top_pred_idx)
                  
                        if save_figures:
                            
                            f[1].set_axis_off()
                            if args.save_nolabel:
                                path_interpret_mismatch_labelled = os.path.join(path_interpret_mismatch, "{}_{}_nolabel".format(conf, attr_algorithm))
                                if not os.path.exists(path_interpret_mismatch_labelled):
                                    os.makedirs(path_interpret_mismatch_labelled)
                                path_save = os.path.join(path_interpret_mismatch_labelled, os.path.basename(path_image))
                                print("Saving without label at", path_save)
                                f[0].savefig(path_save, bbox_inches='tight')                                            
                            
                            # Save image with prediction labels
                            f[0].suptitle( suptitle, fontsize=12)
                            path_interpret_mismatch_labelled = os.path.join(path_interpret_mismatch, "{}_{}_labelled".format(conf, attr_algorithm,str(not args.save_nolabel)))
                            if not os.path.exists(path_interpret_mismatch_labelled):
                                os.makedirs(path_interpret_mismatch_labelled)
                            path_save = os.path.join(path_interpret_mismatch_labelled, os.path.basename(path_image))
                            print("Saving with label at", path_save)
                            f[0].savefig(path_save, bbox_inches='tight')
                        plt.close()

                if (num_image%100==0):    
                    # SAVE THE LABELS TOO FOR ANALYSIS
                    path_dict_image2predictions = os.path.join(path_save_images, "dict_image2predictionlabels.json")
                    with open(path_dict_image2predictions, 'w') as f:
                        json.dump(dict_prediction_tags, f)
                        print("Saved label list at : ", path_dict_image2predictions)

            exit(0)  
    
    #TODO: Organise code and put this in the right place
    # This part is to save the predictions as dat file. (see save_predictions.sh)
    elif evaluation_mode == 'resnet18_xent_save_predictions':
        
        # visualizer = Visualizer()z

        model = getModel(path_model=path_model,
                         arch='resnet18')  # TODO Allow loading other model architecture
        model = model.eval()

        if dataset == 'objectnet':

            path_dataset = path_objectnet_overlap_noborder
            config_name = args.config_name
            # path_save_predictions = '/proj/vondrick/www/amogh/examples/results/predictions'

            dataset = ImageFolderWithPaths(path_dataset,transform=transforms.Compose([transform,transform_normalize])) # our custom dataset
            dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True 
                                                )

            # Get overlapping indices of imagenet with objectnet (Read obj2imgnet.txt)
            # Get the max only from these.
            with open('./preprocessing/obj2imgnet_id.txt') as f:
                dict_obj2imagenet_id = json.load(f)
            list_idx_imgnet_overlap_objnet = [] # Contains the category numbers in imagenet that are also in Objectnet
            for list_indices in dict_obj2imagenet_id.values():
                list_idx_imgnet_overlap_objnet.extend(list_indices)
            list_idx_imgnet_overlap_objnet.sort()

            model = model.to(device)

            # INITIALISE DATASTRUCTURE TO STORE THE PREDICTIONS
            list_image_path = []
            list_imgnet_predictions = []
            list_objnet_predictions = []
            

            for ii, (inputs, labels, paths) in enumerate(dataloader):
                # print(inputs.shape, labels, paths) # (batch_size, 3, 224,224), tensor([0,0,0,0,0..]), [path1, path2...]

                inputs = inputs.to(device)

                output = model(inputs)
                output = F.softmax(output, dim=1)
                prediction_score, pred_label_idx = torch.topk(output, 5)

                # Get top5 text labels and rounded off scores
                predicted_labels = np.array([[idx_to_labels[str(idx)][1] for idx in im_i_pred_idx] for im_i_pred_idx in pred_label_idx.squeeze().detach().cpu().numpy()])
                predicted_scores = np.round(prediction_score.squeeze().detach().cpu().numpy(),4)
                
                top1_predicted_labels = np.array([l[0] for l in predicted_labels])
                # print('Predicted:', predicted_labels, top1_predicted_labels, predicted_scores)


                # FOR OBJECTNET SPECIFIC LABELS
                # Get top 10 labels (Among objectnet only), NOTE THAT THE OVERLAP_PRED_LABEL_IDX_RELATIVE IS RELATIVE WRT LIST_IDX_IMGNET_OVERLAP_OBJECTNET
                overlap_output = output.squeeze()[:,list_idx_imgnet_overlap_objnet]
                overlap_prediction_score, overlap_pred_label_idx_relative = torch.topk(overlap_output, 5)

                overlap_pred_label_idx = np.array([[list_idx_imgnet_overlap_objnet[idx] for idx in im_i_pred_idx] for im_i_pred_idx in overlap_pred_label_idx_relative.squeeze().detach().cpu().numpy()])
                overlap_predicted_labels = np.array([[idx_to_labels[str(idx)][1] for idx in im_i_pred_idx] for im_i_pred_idx in overlap_pred_label_idx])

                overlap_predicted_scores = np.round(overlap_prediction_score.squeeze().detach().cpu().numpy(),4)
                overlap_top1_predicted_labels = np.array([l[0] for l in overlap_predicted_labels])
                
                # print("Predictions are : ", predicted_labels, pred_label_idx)
                # print("Overlapping predictions are : ", overlap_predicted_labels, overlap_pred_label_idx)

                # STORE THE PREDICTION RELEVANT INFORMATION
                list_image_path.extend(paths)
                list_imgnet_predictions.append(pred_label_idx.squeeze().detach().cpu().numpy())
                list_objnet_predictions.append(overlap_pred_label_idx) # Already converted to array while taking care of relative indexing between imagenet and objectnet


                # f[0].suptitle(
                #             'True label (objnet): {} \n Predicted labels(objnet only): {} \n Scores: {} \n Predicted labels(include Imgnet): {}'.format(
                #                 category_objectnet, overlap_predicted_labels, overlap_predicted_scores, predicted_labels), fontsize=12)

                # overlap_array_labels = overlap_labels.detach().cpu().numpy()
                # overlap_top1_predicted_idx = overlap_pred_label_idx[:,0].detach().cpu().numpy()
                # incorrect_idx = np.where(overlap_top1_predicted_idx != overlap_array_labels)
                
                # print(array_labels, top1_predicted_idx, incorrect_idx)
                # if len(incorrect_idx[0])>0:
                    # images_misclassified.extend(np.array(paths)[incorrect_idx])
                    # print("Incorrect images are {}, labels were {}, predictions  are: {}".format(np.array(paths)[incorrect_idx], 
                    #                                                                             array_labels[incorrect_idx],
                    #                                                                             predicted_labels[incorrect_idx]),
                        # array_labels[incorrect_idx],top1_predicted_idx[incorrect_idx]
                        # )
                    # print(continue)

                # SAVE INFORMATION
                # dict_image2predictions = {}
                # for 
                # exit(0)

                if ii % 1000 ==0:
                    print(ii, "batches done")
                    array_img_paths = np.array(list_image_path)
                    array_imgnet_predictions = np.concatenate(list_imgnet_predictions)
                    array_objnet_predictions = np.concatenate(list_objnet_predictions)

                    print("INFO: ", array_img_paths.shape, array_imgnet_predictions.shape, array_objnet_predictions.shape)

                    dict_predictions = {"paths": array_img_paths,
                                        "imgnet_predictions": array_imgnet_predictions,
                                        "objnet_predictions": array_objnet_predictions
                                        }

                    path_out = os.path.join(path_save_predictions, "{}.dat".format(config_name))
                    print("Saving at: ", path_out)
                    with open(path_out, 'wb') as features_dict_file:
                        pickle.dump(dict_predictions, features_dict_file)
                        print("Saved features dictionary at ", path_out)
                # break


            # Print the final dictionary
            print("ALL DONE")
            array_img_paths = np.array(list_image_path)
            array_imgnet_predictions = np.concatenate(list_imgnet_predictions)
            array_objnet_predictions = np.concatenate(list_objnet_predictions)

            print("INFO: ", array_img_paths.shape, array_imgnet_predictions.shape, array_objnet_predictions.shape)

            dict_predictions = {"paths": array_img_paths,
                                "imgnet_predictions": array_imgnet_predictions,
                                "objnet_predictions": array_objnet_predictions
                                }

            path_out = os.path.join(path_save_predictions, "{}.dat".format(config_name))
            print("Saving at: ", path_out)
            with open(path_out, 'wb') as features_dict_file:
                pickle.dump(dict_predictions, features_dict_file)
                print("Saved predictions at ", path_out)
               
                    
                    
if __name__ == "__main__":
    main()
    logger.setLevel(old_level)

    # TYPES OF EVALUATIONS POSSIBLE:
    # 1. Model: resnet152 Loss: xent   - Trained using . Tested using
    # 2. Model: resnet152_mlp  Loss:   - Trained using . Tested using
    # 3. Model: resnet152_mlp Loss:
