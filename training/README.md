# Training

First prepare data with the instructions in the other folders.


## Pretrained Models

Res18:

| Type | ImageNet | Stylized | Mixup | AutoAug | GAN Autmentation | GenInt | GenInt with Transfer |
| ------    | ------ | ------ | ------ | ------ | ------ | ------ |-----|
| Standard | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/res18-bl-imgnet/save/checkpoint.pth.tar)      | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/img+stylized_bl_re18/checkpoint.pth.tar)    | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/mixup_bl_res18/model_best.pth.tar) |  [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/autoaug_res18_bl/model_best.pth.tar) | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/img+gan_bl/model_best_89.pth.tar)  | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/img+intGAN_lam005_b64_epoch90/model_best_87.pth.tar) | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/combined_finetune.pth.tar) |
| Additional Aug | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/rot/res18-bl-imgnet-rot/save/checkpoint.pth.tar) | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/rot/img+stylized_bl_res18_rot/model_best_88.pth.tar) | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/rot/mixup/checkpoint.pth.tar)               |  [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/standard/autoaug_res18_bl/model_best.pth.tar) |  [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/rot/img+gan_rot/checkpoint.pth.tar)        | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/rot/img+intGAN_lam005_b64_rot/model_best_88.pth.tar)              | [res18](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res18/rot/img+intGAN_lam005_b64+tune_b256_lam1sty_rot_lr-3/model_best_135.pth.tar) | 



Res152:

| Type | ImageNet | Stylized | Mixup | AutoAug | GAN Autmentation | GenInt | GenInt with Transfer |
| ------    | ------ | ------ | ------ | ------ | ------ | ------ |-----|
| Standard | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/bseline_res152_checkpoint.pth.tar)      | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/stylized_model_checkpoint.pth.tar)    | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/model_best_mixup.pth.tar) |  [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/model_best_autoaug.pth.tar)              | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/model_best_6_obgan_augganonly.pth.tar)  | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/model_best_23_intGAN_b64_lam02_res152.pth.tar) | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/model_best_38_re152combined_lam02_lam02.pth.tar) |
| Additional Aug | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/rot/res152_rot_bl/checkpoint.pth.tar) | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/rot/img+stylized/model_best_29.pth.tar) | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/rot/mixup_152_rot/model_best.pth.tar)               |  [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/standard/model_best_autoaug.pth.tar) |  [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/rot/res152_obgan_rot_model_best.pth.tar)        | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/rot/intGAN_b64_lam02_res152_tune_epo40/model_best_32.pth.tar)              | [res152](https://cv.cs.columbia.edu/mcz/GenInt/CVPR2021/Res152/rot/trans3_b128b128_lam1_b32_lam002_res152_tune_rot/checkpoint.pth.tar) | 



## Res18

### Standard Augmentation
* baseline, `python main_consise.py -a resnet18 -j 30 --prefix_name baseline  your/path/to/ImageNet/train`
* autoaug `python main_consise.py -a resnet18 -j 30 --autoaug --prefix_name baseline  your/path/to/ImageNet/train`
* mixup: `python mixup.py -a resnet18 -j 30 --prefix_name mixup  your/path/to/ImageNet/train`
* stylized: (1) generated stylized imagenet following the instruction in `https://github.com/rgeirhos/Stylized-ImageNet`, then
`CUDA_VISIBLE_DEVICES=1 python main_consise_combine.py -b 256 --prefix_name img+stylized_bl_res18 -a resnet18 -j 50 --Stylized_original`
* GAN Augmentation, run `CUDA_VISIBLE_DEVICES=0,1 python main_consise_combine.py -b 256 --batch_size_2 116 --prefix_name img+gan_rot -a resnet18 -j 50 --rotate --observation_gan`
* GenInt, run `CUDA_VISIBLE_DEVICES=2,3 python main_consise_combine.py -b 256 --batch_size_2 64 --intGAN --prefix_name img+intGAN_lam005_b64 -a resnet18 -j 40 --lambda_1 0.05`
* Transfer the Int Only, run `CUDA_VISIBLE_DEVICES=2,3 python main_consise_combine.py -b 256 --batch_size_2 256 --intGAN --prefix_name img+transfer_3 -a resnet18 -j 40 --lambda_1 1`
* GenInt with Transfer: GenInt with transfer: 
``CUDA_VISIBLE_DEVICES=2,3 python main_consise_combine.py -b 256 --batch_size_2 256 --batch_size_3 64 --intgenGAN 
--transfer_3 --prefix_name img+intGAN_lam002_b64+tune_b256_lam1sty -a resnet18 -j 40 --lambda_1 1 --lambda_2 0.02 
--finetune --saveall --lr 1e-3 --resume SavedModels/img+transfer3_res18/checkpoint.pth.tar 
--epoch 130``

### Additional Data Augmentation:
 
add `--rotate` to the above command.

* GenInt with Transfer: 
``CUDA_VISIBLE_DEVICES=4,5 python main_consise_combine.py -b 256 --batch_size_2 256 --batch_size_3 64 --intgenGAN 
--transfer_3 --prefix_name img+intGAN_lam005_b64+tune_b256_lam1sty_rot_lr-3 --rotate -a resnet18 -j 40 --lambda_1 1 
--lambda_2 0.05 --finetune --saveall --lr 1e-3
 --resume /proj/vondrick/mcz/Causal/SavedModels/img+transfer3_res18_rot/model_best_88.pth.tar --epoch 150``

## Res152

update the architecture from `-a resnet18` to `-a resnet152`

* Mixup:
 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python mix_up.py -b 256 --epoch 30 -a resnet152 -j 50 --prefix_name mixup_res152_tune --lr 1e-3 --resume /proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/checkpoint.pth.tar /local/vondrick/chengzhi/ImageNet-Data/train`

* AutoAug:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_consise.py -b 256 --epoch 30 -a resnet152 -j 50 --prefix_name autoaug_res152_tune --autoaug --lr 1e-3 --resume /proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/checkpoint.pth.tar /local/vondrick/chengzhi/ImageNet-Data/train`

* Stylized-Imagnet:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_consise_combine.py --epoch 30 -b 128 --batch_size_2 128 --prefix_name img+stylized_bl_res152 -a resnet152 -j 50 --Stylized_original --lr 1e-3 --resume /proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/checkpoint.pth.tar`

* GAN Augmentation:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_consise_combine.py -b 224 --epoch 100 --batch_size_2 128 -a resnet152 -j 50 --prefix_name obgan_roganonly_res152_tune --observation_gan --augganonly2 --finetune --saveall --adam --lr 1e-5 --resume /proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/checkpoint.pth.tar`

* GenInt:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_consise_combine.py -b 256 --batch_size_2 64 --lambda_1 0.05  --epoch 8 -a resnet152 -j 50 --prefix_name intGAN_b64_lam005_res152_tune --intGAN --augganonly2 --finetune --saveall --adam --lr 1e-5 --resume /proj/vondrick/mcz/ModelInvariant/Pretrained/resnet152/checkpoint.pth.tar`

* Transfer Int Only:
`CUDA_VISIBLE_DEVICES=2,3 python main_consise_combine.py -b 256 --batch_size_2 64 --intGAN --prefix_name img+transfer_3 -a resnet152 -j 40 --lambda_1 0.2`

* GenInt with Transfer:
``CUDA_VISIBLE_DEVICES= python main_consise_combine.py -b 192 --batch_size_2 64 --batch_size_3 64 --intgenGAN 
--transfer_3 --prefix_name name -a resnet152 -j 40 --lambda_1 0.2 --lambda_2 0.2 
--finetune --saveall --lr 1e-3 --resume SavedModels/img+transfer3_res152/checkpoint.pth.tar 
--epoch 40``


With Augmentation:
We finetune from rotation trained model from scratch:
For baseline: Train on augmented ImageNet. Model Link:
For IntGen: We train GenInt from scratch with lambda=0.2 and batchsize 64 with augmentation. Model　Link:


