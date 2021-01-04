# PyTorch pretrained BigGAN
An op-for-op PyTorch reimplementation of DeepMind's BigGAN model with the pre-trained weights from DeepMind.

## Introduction

This repository contains PyTorch reimplementation of DeepMind's BigGAN that was released with the paper [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://openreview.net/forum?id=B1xsqj09Fm) by Andrew Brock, Jeff Donahue and Karen Simonyan.

This PyTorch implementation of BigGAN is provided with the [pretrained 128x128, 256x256 and 512x512 models by DeepMind](https://tfhub.dev/deepmind/biggan-deep-128/1). We also provide the scripts used to download and convert these models from the TensorFlow Hub models.

This reimplementation was done from the raw computation graph of the Tensorflow version and behave similarly to the TensorFlow version (variance of the output difference of the order of 1e-5).

This implementation currently only contains the generator as the weights of the discriminator were not released (although the structure of the discriminator is very similar to the generator so it could be added pretty easily. Tell me if you want to do a PR on that, I would be happy to help.)

## Installation

This repo was tested on Python 3.6 and PyTorch 1.0.1

PyTorch pretrained BigGAN can be installed from pip as follows:
```bash
pip install pytorch-pretrained-biggan
```

If you simply want to play with the GAN this should be enough.

If you want to use the conversion scripts and the imagenet utilities, additional requirements are needed, in particular TensorFlow and NLTK. To install all the requirements please use the `full_requirements.txt` file:
```bash
git clone https://github.com/huggingface/pytorch-pretrained-BigGAN.git
cd pytorch-pretrained-BigGAN
pip install -r full_requirements.txt
```

## Models

This repository provide direct and simple access to the pretrained "deep" versions of BigGAN for 128, 256 and 512 pixels resolutions as described in the [associated publication](https://openreview.net/forum?id=B1xsqj09Fm).
Here are some details on the models:

- `BigGAN-deep-256`: a 55.9M parameters model generating 256x256 pixels images, the model dump weights 224 MB,

Please refer to Appendix B of the paper for details on the architectures.

All models comprise pre-computed batch norm statistics for 51 truncation values between 0 and 1 (see Appendix C.1 in the paper for details).

# Usage for Generating Data

set up the path in `single_gen.py`.
run `python single_gen.py`

