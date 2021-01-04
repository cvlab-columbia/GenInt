# Training

First prepare data with the instructions in the other folders.

## Res18

* baseline, `python main_consise.py -a resnet18 -j 30 --prefix_name baseline  your/path/to/ImageNet/train`
* autoaug `python main_consise.py -a resnet18 -j 30 --autoaug --prefix_name baseline  your/path/to/ImageNet/train`
* mixup: `python mixup.py -a resnet18 -j 30 --prefix_name mixup  your/path/to/ImageNet/train`
* stylized: (1) generated stylized imagenet following the instruction in `https://github.com/rgeirhos/Stylized-ImageNet`, then
`CUDA_VISIBLE_DEVICES=1 python main_consise_combine.py -b 256 --prefix_name img+stylized_bl_res18 -a resnet18 -j 50 --Stylized_original`
* observation GAN, run `CUDA_VISIBLE_DEVICES=0,1 python main_consise_combine.py -b 256 --batch_size_2 116 --prefix_name img+gan_rot -a resnet18 -j 50 --rotate --observation_gan`
* interventional GAN GenInt, run `CUDA_VISIBLE_DEVICES=2,3 python main_consise_combine.py -b 256 --batch_size_2 64 --intGAN --prefix_name img+intGAN_lam005_b64 -a resnet18 -j 40 --lambda_1 0.05`
* interventional Transfer, run `CUDA_VISIBLE_DEVICES=2,3 python main_consise_combine.py -b 256 --batch_size_2 256 --intGAN --prefix_name img+transfer_3 -a resnet18 -j 40 --lambda_1 1`
* combined: GenInt with transfer: `CUDA_VISIBLE_DEVICES=7 python main.py -a resnet18 -j 30 --resume /path/to/best/model/above -e -d objectnet ./`


## Res152

update the architecture from `-a resnet18` to `-a resnet152`




