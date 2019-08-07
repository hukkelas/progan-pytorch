# Progressive Growing of GANs

A pytorch implementation of the popular paper "[Progressive growing of gans for improved quality, stability, and variation](https://arxiv.org/abs/1710.10196)" ([Official tensorflow code](https://github.com/tkarras/progressive_growing_of_gans)).

## Requirements

- Pytorch >= 1.0 
- [Apex AMP](https://github.com/NVIDIA/apex.git)
- packages in [requirements](requirements.txt)

To reproduce our results, we recommend you to use a docker environment defined in the [Dockerfile](docker/Dockerfile)


## Features

- Fully implemented progressive growing of GANs to reproduce the results on the CelebA-HQ dataset.
- Use of WGAN-GP loss
- Easy-to-use [config files](models/default/config.yml) to change hyperparameters for testing
- Supports both CelebA-HQ and MNIST
- High performance data pre-processing pipeline
- DataParallel support to run on multi-gpu (single node) systems
- Mixed precision support with Apex AMP. We recommend to use optimization level [O1](https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use)

## TODO

- WGAN-GP loss + AC-GAN (as presented in the paper) for class conditional datasets
- CIFAR-10 & LSUN datasets

## Reference implementation
- [Official tensorflow code](https://github.com/tkarras/progressive_growing_of_gans)
- [Naveen Benny's Pytorch implementation](https://github.com/nvnbny/progressive_growing_of_gans)
