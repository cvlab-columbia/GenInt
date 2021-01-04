path=''
optim='sgd'
cd ..
CUDA_VISIBLE_DEVICES=2,3,1 python main.py -a resnet152 /mogu -j 40 --backup_output_dir $path --epochs 20 --lr_interval 10\
 --lr 0.1 --rotate --pretrained