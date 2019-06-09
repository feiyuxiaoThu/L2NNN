import os
import time

import fire
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import nn
import torchnet as tnt
from git import Repo

import datasets
from config import config
import models
from models.layers import MMLDA_Layer
from utils.cleverhans import AccuracyReport, CleverhansAttack
from utils.logger import TensorBoardLogger
from utils.train import Meter, save, train, test, count_params, set_seed, set_rng_state
from utils.regularize import regularize


def main(**kwargs):
    model, optim, scheduler = initialize_or_resume(kwargs)
    print("# parameters =", count_params(model))

    tblogger = TensorBoardLogger(config.save_dir)
    tblogger.add_text('config', str(config._config))

    meter = Meter(config.num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds, test_ds, train_loader, test_loader, classes = datasets.get_data(config)
    if config.model == 'L2NonExpaConvNet':
        def loss_fn(y_, y): return model.module.L2NNN_loss(y_, y)
    else:
        loss_fn = CrossEntropyLoss()

    # construct regularized loss function
    if config.get('reg_coef', 0) != 0:
        print("Use regularization")
        def null_loss(y_, y): return 0
        reg = 0
        model, loss_fn = regularize(model, optim, loss_fn, reg, config.reg_coef)

    # attack section
    if config.get('attack', False):
        print("Use adversarial training/testing")
        attack = CleverhansAttack(model, config)
        alpha = config.get('alpha', 1.0)
        beta = config.get('beta', 0.0)
        if alpha != 1.0 or beta != 0:
            print('Use blind spot attack')

        def prepare_batch(batch, device, non_blocking=False):
            x, y = batch
            adv_x = attack.perturb(x * alpha + beta)
            return adv_x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
    else:
        def prepare_batch(batch, device, non_blocking=False):
            x, y = batch
            return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

    if config.phase == 'train':
        best_acc = 0
        for epoch in range(config.epoch+1, config.max_epochs):
            acc, loss, confusion = train(model, loss_fn, optim, train_loader, meter,
                                         device, epoch, prepare_batch, config)
            print(f'Epoch[{epoch:3d}] Train acc: {acc:.4f}% Loss: {loss:.4f}')
            tblogger.add_scalar('train_acc', acc, epoch)
            tblogger.add_scalar('train_loss', loss, epoch)
            tblogger.add_heatmap('train_confusion', confusion, classes, classes, epoch)
            tblogger.add_model_grad_hist(model, epoch)
            tblogger.add_model_weight_hist(model, epoch)
            scheduler.step(loss)

            acc, loss, confusion = test(model, loss_fn, test_loader, meter, device, epoch, prepare_batch, config)
            print(f'Epoch[{epoch:3d}] Test acc: {acc:.4f}% Loss: {loss:.4f}')
            tblogger.add_scalar('test_acc', acc, epoch)
            tblogger.add_scalar('test_loss', loss, epoch)
            tblogger.add_heatmap('test_confusion', confusion, classes, classes, epoch)
            save(f'{config.save_dir}/ckpt.pt', model, epoch, optim, scheduler, acc)
            if acc > best_acc:
                save(f'{config.save_dir}/best.ckpt.pt', model, epoch, optim, scheduler, acc)
    elif config.phase == 'test':
        epoch = config.epoch
        acc, loss, confusion = test(model, loss_fn, test_loader, meter, device, epoch, prepare_batch, config)
        print(f'Epoch[{epoch:3d}] Test acc: {acc:.4f}% Loss: {loss:.4f}')

        if hasattr(model.module, 'c_gap'):
            c_gap_meter = tnt.meter.AverageValueMeter()
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                gap = model.module.c_gap(x, y)
                c_gap_meter.add(gap.mean().item())
            print(f'Average confidence gap: {c_gap_meter.value()[0]}')

    tblogger.close()


def initialize_or_resume(kwargs):
    if 'config' in kwargs:
        return initialize(kwargs)
    elif 'resume' in kwargs:
        return resume(kwargs)
    else:
        raise Exception('Must specify config or resume path!')


def initialize(kwargs):
    # find config file in ./configs dir
    # nonlocal config
    config.load(f'configs/{kwargs["config"]}').update(**kwargs)
    config.commit_id = get_commit_id(kwargs.get('check_repo', False))
    timestamp = time.strftime('%Y%m%d:%H:%M:%S', time.localtime())
    config.save_dir = 'runs/' + timestamp + "-" + '-'.join(f'{k}:{v}' for k, v in kwargs.items())
    os.makedirs(config.save_dir, exist_ok=True)
    config.save(f'{config.save_dir}/final_config.yaml')
    config.epoch = 0  # start training from scratch

    # Fix random seed is important! Stochastic random seed only brings unproducible results and makes bugs hide deeper.
    if config.get('deterministic', True):
        set_seed(0)

    model = getattr(models, config.model)(config)
    if config.get('MMLDA', False):
        print('Use MMLDA model')
        mmlda_layer = MMLDA_Layer(device='cuda', C=config.mean, sigma=config.var,
                                  p=config.num_classes, L=config.num_classes)
        model = torch.nn.Sequential(model, mmlda_layer)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model.cuda())
    optim = getattr(torch.optim, config.optim)(
        model.parameters(), **config.optim_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=2)
    print('Model and optimizer initialized')

    return model, optim, scheduler


def resume(kwargs):
    # find config file in resume dir
    # nonlocal config
    config.load(f'{kwargs["resume"]}/final_config.yaml').update(**kwargs)
    config.save_dir = kwargs['resume']

    commit_id = get_commit_id(kwargs.get('check_repo', False))
    # if commit_id != config.commit_id:
    #     raise Exception(f'Must checkout to commit {config.commit_id} to resume!')

    print(f'Resuming from {config.save_dir}...')
    model = getattr(models, config.model)(config)
    if config.get('MMLDA', False):
        print('Use MMLDA model')
        mmlda_layer = MMLDA_Layer(device='cuda', C=config.mean, sigma=config.var,
                                  p=config.num_classes, L=config.num_classes)
        model = torch.nn.Sequential(model, mmlda_layer)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model.cuda())
    optim = getattr(torch.optim, config.optim)(
        model.parameters(), **config.optim_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=2)
    ckpt_file = {
        'best': f'{config.save_dir}/best.ckpt.pt',
        'latest': f'{config.save_dir}/ckpt.pt',
    }[config.get('resume_opt', 'best')]
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model'])
    config.epoch = ckpt['epoch']
    optim.load_state_dict(ckpt['optim'])
    scheduler.load_state_dict(ckpt['scheduler'])
    set_rng_state(ckpt['rng'])
    print(f'Resumed from {config.save_dir} at epoch {config.epoch}')

    return model, optim, scheduler


def get_commit_id(check_repo=False):
    repo = Repo('.')
    if check_repo and repo.is_dirty():
        raise Exception('Please commit all your changes first.')
    return repo.head.commit.hexsha


if __name__ == "__main__":
    fire.Fire(main)
