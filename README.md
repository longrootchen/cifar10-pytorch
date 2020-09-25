# Re-implementation of ConvNets on CIFAR-10 with PyTorch

## Introduction

Here are some re-implementations of Convolutional Networks on CIFAR-10 dataset.

Note that, in the paper by He et al.[1], the training set that consists of 50k training images was divided into 45k/5k train/val split. So I first made stratefied 10-fold split. The implement details are as in [this repository](https://github.com/longrootchen/stratefied-10-fold-cifar10).

Contact email: imdchan@yahoo.com

## Requirements

A single RTX 2080Ti was used.

- Python 3.7+

- PyTorch 1.0+

## Usage

1. Clone this repository

        git clone https://github.com/longrootchen/cifar10-pytorch.git

2. Train a model, taking resnet20 as an example

        python -u train.py --work-dir ./experiments/resnet20
        
3. Evaluate a model, taking resnet20 as an example

        python -u eval.py --arch resnet20 --num-classes 10 --ckpt-path ./experiments/resnet20/checkpoints/last_checkpoint.pth --gpu cuda:0 --df-path ./datasets/test.csv --img-dir ./datasets/test
        
        
## Results

| Error Rate (%)  | original paper | re-implementation |
| ----- | ----- | ----- |
| resnet20 | 8.75 [1] | 8.58 |
| resnet32 | 7.51 [1] | xxxx |
| resnet44 | 7.17 [1] | xxxx |
| resnet56 | 6.97 [1] | xxxx |
| resnet110 | 6.43 [1] | xxxx |
| resnet1202 | 7.93 [1] | xxxx |

## References

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. In CVPR, 2016.
