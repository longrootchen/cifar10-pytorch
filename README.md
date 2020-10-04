# Re-implementation of ConvNets on CIFAR-10 with PyTorch

Contact email: imdchan@yahoo.com

## Introduction

Here are some re-implementations of Convolutional Networks on CIFAR-10 dataset.

Note that the training set that consists of 50k training images was divided into 45k/5k train/val split. So I first made stratefied 10-fold split, resulting in the 'train_folds.csv'.

## Requirements

- A single TITAN RTX (24G memory) is used.

- Python 3.7+

- PyTorch 1.0+

## Usage

1. Clone this repository

        git clone https://github.com/longrootchen/cifar10-pytorch.git

2. Train a model, taking resnet20 as an example

        python -u train.py --work-dir ./experiments/resnet20 --resume ./experiments/resnet20/checkpoints/last_checkpoint.pth

3. Evaluate a model, taking resnet20 as an example

        python -u eval.py --work-dir ./experiments/resnet20 --ckpt-name last_checkpoint.pth --df-path ./datasets/test.csv --img-dir ./datasets/test
        
        
## Results

| Error Rate (%)  | original paper | re-implementation |
| ----- | ----- | ----- |
| ResNet-20 | 8.75 [1] | 8.24 |
| ResNet-32 | 7.51 [1] | 7.38 |
| ResNet-44 | 7.17 [1] | 7.07 |
| ResNet-56 | 6.97 [1] | 7.01 |
| ResNet-110 | 6.43 [1] | 6.63 |
| ResNet-1202 | 7.93 [1] |  |
| ResNeXt-29, 8x64d | 3.65 [2] | 4.33 |
| ResNeXt-29, 16x64d | 3.58 [2] |  |
| DenseNet-100-BC, k=12 | 4.51 [3] |  |
| DenseNet-250-BC, k=24 | 3.62 [3] |  |
| DenseNet-190-BC, k=40 | 3.46 [3] |  |
| SE-ResNet-101 | 5.21 [4] |  |
| SE-ResNet-164 | 4.39 [4] |  |

## References

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. In CVPR, 2016.

[2] Saining Xie, Ross Girshick, Piotr Dollár, Zhouwen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. In CVPR, 2017.

[3] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. Densely Connected Convolutional Networks. In CVPR, 2017.

[4] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. Squeeze-and-Excitation Networks. In CVPR, 2018.
