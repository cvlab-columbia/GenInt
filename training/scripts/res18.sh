path=''
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18 /mogu -j 30 --backup_output_dir $path --epochs 90 --lr_interval 30 -b 256\
 --lr 0.1  --rotate