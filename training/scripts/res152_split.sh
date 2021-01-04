path=''
optim='sgd'
cd ..
CUDA_VISIBLE_DEVICES=2,3,1 python main_split.py -a resnet152 /mogu -j 40 --backup_output_dir $path --epochs 90 --lr_interval 7 -b 256\
 --lr 0.1 --gan_bl --concat --gan_lambda 0.05 --rand --bs_gan 64 --rotate

#Finetune
CUDA_VISIBLE_DEVICES=2,3,1 python main_split.py -a resnet152 /mogu -j 50 --backup_output_dir $path --epochs 20\
 --lr_interval 10 -b 256\
 --lr 0.0001 --gan_bl --concat --gan_lambda 0.2 --bs_gan 64\
 --optim $optim   --resume $resume_path --rand --iterate 1000
