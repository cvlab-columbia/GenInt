CUDA_VISIBLE_DEVICES=3 python gen.py --model=BigGAN-256 --class=husky --layer=generator.gen_z -n=1000000\
 --batch --start_num 0 --end_num 200 --sigma 8.0 --top_k_component_num 20 --trans_num_frames 8\
  --random_image_examplar_num 500 --select_num 1 --allrand 3 --save_noise 1 --save_simple 1