from torch import nn
import torch.nn.functional as F

__all__ = ['resnext29_8x64d', 'resnext29_16x64d']


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, expansion):
        super(Bottleneck, self).__init__()

        width_ratio = out_channels / (expansion * 64.)
        D = cardinality * int(base_width * width_ratio)

        self.relu = nn.ReLU(inplace=True)

        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'shortcut_conv',
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_conv.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        identity = self.shortcut.forward(x)
        return self.relu(identity + out)


class ResNeXt(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in https://arxiv.org/pdf/1611.05431.pdf"""

    def __init__(self, cardinality, depth, num_classes, base_width, expansion=4):
        """
        Args:
            cardinality (int): number of convolution groups.
            depth (int): number of layers.
            num_classes (int): number of classes
            base_width (int): base number of channels in each group.
            expansion (int = 4): factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 * self.expansion, 256 * self.expansion]

        self.conv1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stage1 = self.block('stage1', self.stages[0], self.stages[1], 1)
        self.stage2 = self.block('stage2', self.stages[1], self.stages[2], 2)
        self.stage3 = self.block('stage3', self.stages[2], self.stages[3], 2)
        self.fc = nn.Linear(self.stages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(
                    name_,
                    Bottleneck(in_channels, out_channels, pool_stride,
                               self.cardinality, self.base_width, self.expansion))
            else:
                block.add_module(
                    name_,
                    Bottleneck(out_channels, out_channels, 1,
                               self.cardinality, self.base_width, self.expansion))
        return block

    def forward(self, x):
        x = self.conv1_3x3.forward(x)
        x = F.relu(self.bn1.forward(x), inplace=True)
        x = self.stage1.forward(x)
        x = self.stage2.forward(x)
        x = self.stage3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


def resnext29_8x64d(num_classes):
    return ResNeXt(
        cardinality=8,
        depth=29,
        num_classes=num_classes,
        base_width=64)


def resnext29_16x64d(num_classes):
    return ResNeXt(
        cardinality=16,
        depth=29,
        num_classes=num_classes,
        base_width=64)
