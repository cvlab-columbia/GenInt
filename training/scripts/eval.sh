resume_path='your saved model path'

CUDA_VISIBLE_DEVICES=1,2,3 python main.py -a resnet152 /mogu  -j 50 -e -b 400\
 --evaluate  -d objectnet  --imgnet256 --resume $resume_path