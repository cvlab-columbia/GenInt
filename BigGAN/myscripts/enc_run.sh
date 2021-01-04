cd ..
path='/local/rcs/mcz/ModelInvariant/SavedModels/EncoderBigGAN/'
path='/proj/vondrick/mcz/ModelInvariant/SavedModels/debug'
optim='adam'
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_encoder.py -j 30 --backup_output_dir $path\
 --epochs 90 --lr_interval 30 -b 10 --bs_gan 10\
 --lr 0.0001 --optim $optim -p 20