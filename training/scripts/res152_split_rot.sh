path=''
optim='sgd'
cd ..


#Finetune
resume_path='the saved baseline add aug train model'
CUDA_VISIBLE_DEVICES=2,3,1 python main_split.py -a resnet152 /mogu -j 50 --backup_output_dir $path --epochs 20\
 --lr_interval 10 -b 256\
 --lr 0.0001 --gan_bl --concat --gan_lambda 0.2 --bs_gan 64\
 --optim $optim   --resume $resume_path --rand --iterate 1000