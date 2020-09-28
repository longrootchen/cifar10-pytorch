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


class Trainer:

    def __init__(self, cfgs, model):
        """
        Args:
            cfgs (class): a class object whose attributes are the hyper-parameters for training
            model (torch.nn.Module): the model to be trained
        """
        self.cfgs = cfgs
        self.model = model
        self.start_epoch = 1
        self.best_err = 1.1

        self.device = torch.device(cfgs.gpu if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay,
                             momentum=cfgs.momentum, nesterov=cfgs.nesterov)
        self.scheduler = MultiStepLR(self.optimizer, gamma=cfgs.gamma, milestones=cfgs.milestones)

        # optionally resume from a checkpoint
        if cfgs.resume:
            if os.path.isfile(cfgs.resume):
                print("=> loading checkpoint '{}'".format(cfgs.resume))
                checkpoint = torch.load(cfgs.resume, map_location=self.device)
                self.load(cfgs.resume)
                print("=> loaded checkpoint '{}' (epoch {})".format(cfgs.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(cfgs.resume))

        # create directory to checkpoints if necessary
        if not os.path.exists(cfgs.save_dir):
            os.makedirs(cfgs.save_dir)
        # create directory to log file and event files if necessary
        if not os.path.exists(cfgs.log_dir):
            os.makedirs(cfgs.log_dir)
        self.writer = SummaryWriter(log_dir=cfgs.log_dir)

        self.log('Trainer prepared in device: {}'.format(self.device))

    def fit(self, train_loader, valid_loader):
        """
        train and evaluate the model

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader instance object for training set
            valid_loader (torch.utils.data.DataLoader): DataLoader instance object for validation set
        """
        for epoch in range(self.start_epoch, self.cfgs.epochs + 1):
            if self.cfgs.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.datetime.now().isoformat()
                self.log('{}\tEpoch: {}\tLR: {}'.format(timestamp, epoch, lr))

            # train an epoch
            err = self.train(epoch, train_loader)

            self.save(epoch, f'{self.cfgs.save_dir}/last_checkpoint.pth')
            self.log(f'[RESULT]: Train Epoch: {epoch}\t Error Rate: {err:6.4f}')
            self.writer.add_scalars('error', {'train': err}, epoch)

            # validate
            err = self.validate(epoch, valid_loader)

            if err < self.best_err:
                self.best_err = err
                self.save(epoch, f'{self.cfgs.save_dir}/best_checkpoint_{str(epoch).zfill(3)}epoch.pth')
            self.log(f'[RESULT]: Valid Epoch: {epoch}\t Error Rate: {err:6.4f}')
            self.writer.add_scalars('error', {'valid': err}, epoch)

            self.scheduler.step()

    def train(self, epoch, train_loader):
        """
        train an epoch

        Args:
            epoch (int): current epoch
            train_loader (torch.utils.data.DataLoader): a DataLoader instance object for training set
        Returns:
            err (float): error rate
        """
        losses = AverageMeter()
        evaluator = Evaluator(self.cfgs.num_classes)

        self.model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description('Train Epoch {}'.format(epoch))

            for step, (input_, target) in enumerate(train_loader):
                # move data to device
                input_ = torch.tensor(input_, device=self.device, dtype=torch.float32)
                target = torch.tensor(target, device=self.device, dtype=torch.long)

                # forward and compute loss
                output = self.model(input_)
                loss = self.criterion(output, target)

                # backward and update params
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record loss and show it in the pbar
                losses.update(loss.item(), input_.size(0))
                postfix = OrderedDict({'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}'})
                pbar.set_postfix(ordered_dict=postfix)
                pbar.update()

                # visualization with TensorBoard
                total_iter = (epoch - 1) * len(train_loader) + step + 1
                self.writer.add_scalar('training_loss', losses.val, total_iter)

                # update confusion matrix
                true = target.cpu().numpy()
                pred = output.max(dim=1)[1].cpu().numpy()
                evaluator.update_matrix(true, pred)

        return evaluator.error()

    def validate(self, epoch, valid_loader):
        """
        validate the model

        Args:
            epoch (int): current epoch
            valid_loader (torch.utils.data.DataLoader): a DataLoader instance object for validation set
        Returns:
            err: (float) error rate
        """
        losses = AverageMeter()
        evaluator = Evaluator(self.cfgs.num_classes)

        self.model.eval()
        with tqdm(valid_loader) as pbar:
            pbar.set_description('Valid Epoch {}'.format(epoch))

            for i, (input_, target) in enumerate(valid_loader):
                # move data to GPU
                input_ = torch.tensor(input_, device=self.device, dtype=torch.float32)
                target = torch.tensor(target, device=self.device, dtype=torch.long)

                with torch.no_grad():
                    # compute output and loss
                    output = self.model(input_)
                    loss = self.criterion(output, target)

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

    def save(self, epoch, path):
        self.model.eval()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_err': self.best_err,
            'epoch': epoch
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        # whether the checkpoint contains other training info
        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.best_err = checkpoint['best_err']
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.model.load_state_dict(checkpoint)

    def log(self, msg):
        if self.cfgs.verbose:
            print(msg)

        log_path = os.path.join(self.cfgs.log_dir, 'log.txt')
        with open(log_path, 'a+') as logger:
            logger.write(f'{msg}\n')


if __name__ == '__main__':
    # for training resnet20: $ python -u train.py --work-dir ./experiments/resnet20
    parser = ArgumentParser(description='Train ConvNets on CIFAR-10 in PyTorch')
    parser.add_argument('--work-dir', required=True, type=str)
    args = parser.parse_args()

    # get experiment settings
    with open(os.path.join(args.work_dir, 'config.yaml')) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    cfgs = EasyDict(cfgs)

    # set paths
    cfgs.save_dir = os.path.join(args.work_dir, cfgs.save_dir)
    cfgs.log_dir = os.path.join(args.work_dir, cfgs.log_dir)
    cfgs.resume = os.path.join(args.work_dir, cfgs.resume)

    # get model
    model = get_model(cfgs)

    # get data
    df = pd.read_csv(cfgs.df_path)
    train_df = df[df['fold'] != 1]
    valid_df = df[df['fold'] == 1]

    train_set = CIFAR10Dataset(train_df, cfgs.img_dir, phase='train')
    valid_set = CIFAR10Dataset(valid_df, cfgs.img_dir, phase='val')

    train_loader = DataLoader(train_set, batch_size=cfgs.batch_size, shuffle=True, num_workers=cfgs.workers)
    valid_loader = DataLoader(valid_set, batch_size=cfgs.batch_size, shuffle=False, num_workers=cfgs.workers)

    # train
    trainer = Trainer(cfgs, model)
    trainer.fit(train_loader, valid_loader)
