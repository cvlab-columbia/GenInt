# Observational GAN
path=''
cd ..
CUDA_VISIBLE_DEVICES=0 python main_split.py -a resnet18 /mogu -j 30 --backup_output_dir $path --epochs 90 --lr_interval 30 -b 256\
 --lr 0.1 --gan_bl --concat --gan_lambda 0.05 --rand --rotate

resume=path
optim='sgd'
CUDA_VISIBLE_DEVICES=0 python main_split.py -a resnet18 /mogu -j 30 --backup_output_dir $path --epochs 20 --lr_interval 30 -b 256\
 --lr 0.0001 --gan_bl --concat --gan_lambda 0.2 --large 1000 --resume $resume --optim $optim