import os
import yaml
import warnings
from argparse import ArgumentParser

from easydict import EasyDict
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import *
from data import CIFAR10Dataset, cls_to_idx
from utils import Evaluator

warnings.filterwarnings('ignore')


def eval(device, model, test_loader, vis_conf_mat=False, save_conf_mat=False):
    evaluator = Evaluator(10)

    model.eval()
    with tqdm(test_loader) as pbar:
        pbar.set_description('Eval in test set')

        for i, (input_, target) in enumerate(test_loader):
            input_ = torch.tensor(input_, device=device, dtype=torch.float32)
            target = torch.tensor(target, device=device, dtype=torch.long)

            with torch.no_grad():
                output = model(input_)

            true = target.cpu().numpy()
            pred = output.max(dim=1)[1].cpu().numpy()
            evaluator.update_matrix(true, pred)

            pbar.update()

    if vis_conf_mat:
        evaluator.show_matrix(cls_to_idx, save_matrix=save_conf_mat)

    return evaluator.error()


if __name__ == '__main__':
    # for evaluating resnet20 on CIFAR-10 test set:
    # $ python -u eval.py --work-dir ./experiments/resnet20 --ckpt-name last_checkpoint.pth
    #   --df-path ./datasets/test.csv --img-dir ./datasets/test
    parser = ArgumentParser(description='Train ConvNets on CIFAR-100 in PyTorch')
    parser.add_argument('--work-dir', required=True, type=str)
    parser.add_argument('--ckpt-name', required=True, type=str)
    parser.add_argument('--df-path', required=True, type=str)
    parser.add_argument('--img-dir', required=True, type=str)
    args = parser.parse_args()

    # get experiment settings
    with open(os.path.join(args.work_dir, 'config.yaml')) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    cfgs = EasyDict(cfgs)

    # hardware
    device = torch.device(cfgs.gpu if torch.cuda.is_available() else 'cpu')

    # get model
    model = get_model(cfgs)
    ckpt_path = os.path.join(args.work_dir, 'checkpoints', args.ckpt_name)
    ckpt = torch.load(ckpt_path)
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    # get data
    df = pd.read_csv(args.df_path)
    test_set = CIFAR10Dataset(df, args.img_dir, phase='val')
    test_loader = DataLoader(test_set, batch_size=cfgs.batch_size, shuffle=False, num_workers=cfgs.workers)

    # eval in test set
    err = eval(device, model, test_loader, vis_conf_mat=False, save_conf_mat=False)
    print('Error Rate:', err)
