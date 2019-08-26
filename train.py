import os
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from bn_lib.nn.modules import patch_replication_callback
from dataset import TrainDataset
from network import EMANet
import settings

logger = settings.logger


def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight
    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], _BatchNorm):
                if m[1].weight is not None:
                    yield m[1].weight
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], _BatchNorm):
                if m[1].bias is not None:
                    yield m[1].bias


def ensure_dir(dir_path):
    if not osp.isdir(dir_path):
        os.makedirs(dir_path)


def poly_lr_scheduler(opt, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    opt.param_groups[0]['lr'] = 1 * new_lr
    opt.param_groups[1]['lr'] = 1 * new_lr
    opt.param_groups[2]['lr'] = 2 * new_lr


class Session:
    def __init__(self, dt_split):
        torch.manual_seed(66)
        torch.cuda.manual_seed_all(66)
        torch.cuda.set_device(settings.DEVICE)

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        self.step = 1
        self.writer = SummaryWriter(osp.join(self.log_dir, 'train.events'))
        dataset = TrainDataset(split=dt_split)
        self.dataloader = DataLoader(
            dataset, batch_size=settings.BATCH_SIZE, pin_memory=True,
            num_workers=settings.NUM_WORKERS, shuffle=True, drop_last=True)

        self.net = EMANet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.opt = SGD(
            params=[
                {
                    'params': get_params(self.net, key='1x'),
                    'lr': 1 * settings.LR,
                    'weight_decay': settings.WEIGHT_DECAY,
                },
                {
                    'params': get_params(self.net, key='1y'),
                    'lr': 1 * settings.LR,
                    'weight_decay': 0,
                },
                {
                    'params': get_params(self.net, key='2x'),
                    'lr': 2 * settings.LR,
                    'weight_decay': 0.0,
                }],
            momentum=settings.LR_MOM)

        self.net = DataParallel(self.net, device_ids=settings.DEVICES)
        patch_replication_callback(self.net)

    def write(self, out):
        for k, v in out.items():
            self.writer.add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            '{}: {:.4g}'.format(k, v) 
            for k, v in out.items()]
        logger.info(' '.join(outputs))

    def save_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        obj = {
            'net': self.net.module.state_dict(),
            'step': self.step,
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path, 
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.error('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])
        self.step = obj['step']

    def train_batch(self, image, label):
        loss, mu = self.net(image, label)

        with torch.no_grad():
            mu = mu.mean(dim=0, keepdim=True)
            momentum = settings.EM_MOM
            self.net.module.emau.mu *= momentum
            self.net.module.emau.mu += mu * (1 - momentum)

        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()


def main(ckp_name='latest.pth'):
    sess = Session(dt_split='trainaug')
    sess.load_checkpoints(ckp_name)

    dt_iter = iter(sess.dataloader)
    sess.net.train()

    while sess.step <= settings.ITER_MAX:
        poly_lr_scheduler(
            opt=sess.opt,
            init_lr=settings.LR,
            iter=sess.step,
            lr_decay_iter=settings.LR_DECAY,
            max_iter=settings.ITER_MAX,
            power=settings.POLY_POWER)

        try:
            image, label = next(dt_iter)
        except StopIteration:
            dt_iter = iter(sess.dataloader)
            image, label = next(dt_iter)

        loss = sess.train_batch(image, label)
        out = {'loss': loss}
        sess.write(out)

        if sess.step % settings.ITER_SAVE == 0:
            sess.save_checkpoints('step_%d.pth' % sess.step)
        if sess.step % (settings.ITER_SAVE // 10) == 0:
            sess.save_checkpoints('latest.pth')
        sess.step += 1

    sess.save_checkpoints('final.pth')


if __name__ == '__main__':
    main()
