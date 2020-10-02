from .resnet import *
from .resnext import *


def get_model(config):
    return globals()[config.arch](config.num_classes)
