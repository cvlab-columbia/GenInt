# Intervene GAN with Interpretable GAN Controls
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
![PyTorch 1.3](https://img.shields.io/badge/pytorch-1.3-green.svg)
![teaser](teaser.jpg)
<p align="justify"><b>Figure 1:</b> Sequences of image edits performed using control discovered with our method, applied to three different GANs. The white insets specify the particular edits using notation explained in Section 3.4 ('Layer-wise Edits').</p>



## Setup
See the [setup instructions](SETUP.md).

## Generate Interventional GAN data
* to generated the interventional GAN data, run `CUDA_VISIBLE_DEVICES=0 python gen_intervene_data_withfix_seed.py  --model=BigGAN-256 --class=husky --layer=generator.gen_z -n=1000000\
 --truncation 1.0 --random_num_start 0 --random_num_end 1000 --global_seed 10319 --class_start 0 --class_end 1000`

* to generated transferred intervention, run `CUDA_VISIBLE_DEVICES=0 python style_transfer_int2imgnet.py`, you can specify the starting
and ending category in `start_cat` and `end_cat`.

## Acknowledgements
We would like to thank:

* The authors of the "GANSpace: Discovering Interpretable GAN Controls".
* The authors of the PyTorch implementations of [BigGAN][biggan_pytorch], [StyleGAN][stylegan_pytorch], and [StyleGAN2][stylegan2_pytorch]:<br>Thomas Wolf, Piotr Bialecki, Thomas Viehmann, and Kim Seonghyeon.
* Joel Simon from ArtBreeder for providing us with the landscape model for StyleGAN.<br>(unfortunately we cannot distribute this model)
* David Bau and colleagues for the excellent [GAN Dissection][gandissect] project.
* Justin Pinkney for the [Awesome Pretrained StyleGAN][pretrained_stylegan] collection.
* Tuomas Kynkäänniemi for giving us a helping hand with the experiments.
* The Aalto Science-IT project for providing computational resources for this project.

## License

The code of this repository is released under the [Apache 2.0](LICENSE) license.<br>
The directory `netdissect` is a derivative of the [GAN Dissection][gandissect] project, and is provided under the MIT license.<br>
The directories `models/biggan` and `models/stylegan2` are provided under the MIT license.


[biggan_pytorch]: https://github.com/huggingface/pytorch-pretrained-BigGAN
[stylegan_pytorch]: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
[stylegan2_pytorch]: https://github.com/rosinality/stylegan2-pytorch
[gandissect]: https://github.com/CSAILVision/GANDissect
[pretrained_stylegan]: https://github.com/justinpinkney/awesome-pretrained-stylegan
