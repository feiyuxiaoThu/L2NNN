import os
import random

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchnet as tnt
from tqdm import tqdm


def set_seed(s):
    """ Manually set seed for all random generator.
    To ensure reproducibility, call this before every experiment.
    """
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(s)
    np.random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)


def get_rng_state():
    return {
        'python': random.getstate(),
        'hash': os.environ['PYTHONHASHSEED'],
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
    }


def set_rng_state(s):
    random.setstate(s['python'])
    os.environ['PYTHONHASHSEED'] = s['hash']
    torch.set_rng_state(s['torch'])
    np.random.set_state(s['numpy'])


class Meter(object):
    def __init__(self, num_classes):
        self.loss = tnt.meter.AverageValueMeter()
        self.accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion = tnt.meter.ConfusionMeter(num_classes, normalized=True)

    def add(self, out, labels, loss):
        self.accuracy.add(out.data, labels.data)
        self.loss.add(loss.data.item())
        self.confusion.add(out.data, labels.data)

    def value(self):
        loss = self.loss.value()[0]
        acc = self.accuracy.value()[0]
        confusion = self.confusion.value()
        return acc, loss, confusion

    def reset(self):
        self.loss.reset()
        self.accuracy.reset()
        self.confusion.reset()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, loss_fn, optim, train_loader, meter, device, epoch, prepare_batch, config):
    meter.reset()
    model.train()

    pbar = tqdm(train_loader, leave=False)
    for batch in pbar:
        x, y = prepare_batch(batch, device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # for L2NNN
        if config.get('norm_train', False):
            model.module.normalize()

        meter.add(y_pred, y, loss)
        acc, loss, confusion = meter.value()
        pbar.set_description_str(f'Epoch {epoch:3d}')
        pbar.set_postfix_str(f'Acc:{acc:.4f}%, Loss:{loss:.4f}')
        pbar.update()
    return meter.value()


def test(model, loss_fn, test_loader, meter, device, epoch, prepare_batch, config):
    meter.reset()
    model.eval()

    pbar = tqdm(test_loader, leave=False)
    for batch in pbar:
        x, y = prepare_batch(batch, device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        meter.add(y_pred, y, loss)
    return meter.value()


def save(path, model, epoch, optim, scheduler, acc):
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'acc': acc,
    }
    torch.save(state, path)


def save(path, model, epoch, optim, scheduler, acc):
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'acc': acc,
        'rng': get_rng_state(),
    }
    torch.save(state, path)
