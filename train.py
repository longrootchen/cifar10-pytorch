import os
import yaml
import datetime
import warnings
from collections import OrderedDict
from argparse import ArgumentParser

from easydict import EasyDict
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from data import CIFAR10Dataset
from utils import AverageMeter, Evaluator

warnings.filterwarnings('ignore')


def train(config, device, train_loader, epoch):
    """
    train an epoch

    Args:
        config (EasyDict): configurations for training
        device (torch.device): the GPU or CPU used for training
        train_loader (torch.utils.data.DataLoader): a DataLoader instance object for training set
        epoch (int): current epoch
    Returns:
        err (float): error rate
    """
    losses = AverageMeter()
    evaluator = Evaluator(config.num_classes)

    model.train()
    with tqdm(train_loader) as pbar:
        pbar.set_description('Train Epoch {}'.format(epoch))

        for step, (input_, target) in enumerate(train_loader):
            # move data to device
            input_ = torch.tensor(input_, device=device, dtype=torch.float32)
            target = torch.tensor(target, device=device, dtype=torch.long)

            # forward and compute loss
            output = model(input_)
            loss = criterion(output, target)

            # backward and update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss and show it in the pbar
            losses.update(loss.item(), input_.size(0))
            postfix = OrderedDict({'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}'})
            pbar.set_postfix(ordered_dict=postfix)
            pbar.update()

            # visualization with TensorBoard
            total_iter = (epoch - 1) * len(train_loader) + step + 1
            writer.add_scalar('training_loss', losses.val, total_iter)

            # update confusion matrix
            true = target.cpu().numpy()
            pred = output.max(dim=1)[1].cpu().numpy()
            evaluator.update_matrix(true, pred)

        return evaluator.error()


def validate(config, device, val_loader, epoch):
    """
    validate the model

    Args:
        config (EasyDict): configurations for training
        device (torch.device): the GPU or CPU used for training
        val_loader (torch.utils.data.DataLoader): a DataLoader instance object for validation set
        epoch (int): current epoch
    Returns:
        err: (float) error rate
    """
    losses = AverageMeter()
    evaluator = Evaluator(config.num_classes)

    model.eval()
    with tqdm(val_loader) as pbar:
        pbar.set_description('Valid Epoch {}'.format(epoch))

        for i, (input_, target) in enumerate(val_loader):
            # move data to GPU
            input_ = torch.tensor(input_, device=device, dtype=torch.float32)
            target = torch.tensor(target, device=device, dtype=torch.long)

            with torch.no_grad():
                # compute output and loss
                output = model(input_)
                loss = criterion(output, target)

            # record loss and show it in the pbar
            losses.update(loss.item(), input_.size(0))
            postfix = OrderedDict({'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}'})
            pbar.set_postfix(ordered_dict=postfix)
            pbar.update()

            # update confusion matrix
            true = target.cpu().numpy()
            pred = output.max(dim=1)[1].cpu().numpy()
            evaluator.update_matrix(true, pred)

    return evaluator.error()


def save(err, epoch, path):
    model.eval()
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'err': err,
        'epoch': epoch
    }, path)


def load(path):
    checkpoint = torch.load(path)

    # whether the checkpoint contains other training info
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        err = checkpoint['err']
        epoch = checkpoint['epoch'] + 1
    else:
        model.load_state_dict(checkpoint)

    return err, epoch


def log(config, msg):
    if config.verbose:
        print(msg)

    log_path = os.path.join(config.log_dir, 'log.txt')
    with open(log_path, 'a+') as logger:
        logger.write(f'{msg}\n')


def fit(config, device, train_loader, val_loader, num_epochs, start_epoch=1, best_err=1.1):
    """
    train and evaluate the model

    Args:
        config (EasyDict): configurations for training
        device (torch.device): the GPU or CPU used for training
        train_loader (torch.utils.data.DataLoader): DataLoader instance object for training set
        val_loader (torch.utils.data.DataLoader): DataLoader instance object for validation set
        num_epochs (int): number of epochs for training
        start_epoch (int): start training from some epoch
        best_err (float): current best error rate
    """
    for epoch in range(start_epoch, num_epochs + 1):
        if config.verbose:
            lr = optimizer.param_groups[0]['lr']
            timestamp = datetime.datetime.now().isoformat()
            log(config, '{}\tEpoch: {}\tLR: {}'.format(timestamp, epoch, lr))

        # train an epoch and save the checkpoint and visualize metrics using TensorBoard
        err = train(config, device, train_loader, epoch)
        save(err, epoch, f'{config.save_dir}/last_checkpoint.pth')
        log(config, f'[RESULT]: Train Epoch: {epoch}\t Error Rate: {err:6.4f}')
        writer.add_scalars('error', {'train': err}, epoch)

        # validate the trained model and save the checkpoint if it is the best one
        err = validate(config, device, val_loader, epoch)
        if err < best_err:
            best_err = err
            save(err, epoch, f'{config.save_dir}/best_checkpoint_{str(epoch).zfill(3)}epoch.pth')
        log(config, f'[RESULT]: Valid Epoch: {epoch}\t Error Rate: {err:6.4f}')
        writer.add_scalars('error', {'valid': err}, epoch)

        scheduler.step()


if __name__ == '__main__':
    # for training resnet20 from a checkpoint:
    # $ python -u train.py --work-dir ./experiments/resnet20
    #   --resume ./experiments/resnet20/checkpoints/last_checkpoint.pth
    parser = ArgumentParser(description='Train ConvNets on CIFAR-10 in PyTorch')
    parser.add_argument('--work-dir', required=True, type=str)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # get experiment settings
    with open(os.path.join(args.work_dir, 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    # set paths
    config.save_dir = os.path.join(args.work_dir, config.save_dir)
    config.log_dir = os.path.join(args.work_dir, config.log_dir)
    config.resume = args.resume

    # set device
    device = torch.device(config.gpu if torch.cuda.is_available() else 'cpu')

    # get model
    model = get_model(config)
    model.to(device)

    # get data
    df = pd.read_csv(config.df_path)
    train_df = df[df['fold'] != 1]
    val_df = df[df['fold'] == 1]
    train_set = CIFAR10Dataset(train_df, config.img_dir, phase='train')
    val_set = CIFAR10Dataset(val_df, config.img_dir, phase='val')
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    # get training stuff
    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                    momentum=config.momentum, nesterov=config.nesterov)
    scheduler = MultiStepLR(optimizer, gamma=config.gamma, milestones=config.milestones)
    start_epoch = 1
    best_err = 1.1

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location=device)
            best_err, start_epoch = load(config.resume)
            print("=> loaded checkpoint '{}' (epoch {})".format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    # create directory to checkpoints if necessary
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    # create directory to log file and event files if necessary
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    writer = SummaryWriter(log_dir=config.log_dir)
    log(config, 'Trainer prepared in device: {}'.format(device))

    # train
    fit(config, device, train_loader, val_loader, config.epochs, start_epoch, best_err)
