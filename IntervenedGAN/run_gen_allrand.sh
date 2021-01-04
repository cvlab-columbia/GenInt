CUDA_VISIBLE_DEVICES=1 python gen.py --model=BigGAN-256 --class=husky --layer=generator.gen_z -n=1000000\
 --batch --start_num 0 --end_num 200 --sigma 9.0 --top_k_component_num 20 --trans_num_frames 12\
  --random_image_examplar_num 1000 --select_num 1 --allrand 2


