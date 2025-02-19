import os
import random
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import lpips
import numpy as np
import pyiqa
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler
import PIL.Image as PImage
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance

from models import VAR, VQVAE, build_vae_var
from utils import dist_utils
from utils.data import build_dataset
from utils.dataset import UnlabeledImageItem, UnlabeledImageFolder, FFHQ, CelebAHQ
from utils.dataset.my_transforms import denormalize_pm1_into_01, img01_into_img255
from utils.dataset.options import DataOptions
from utils.common import seed_everything


class Timer:
    def __init__(self, dec=1):
        self.dec = dec
        self.rec = {}
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def record(self, desc):
        rec_time = time.time()
        self.rec[desc] = rec_time - self.start_time
        self.start_time = rec_time
        return self.rec[desc]*self.dec

    def record_desc(self, desc):
        return f'{desc}: {self.record(desc)}'

    def __str__(self):
        desc = ''
        for k, v in self.rec.items():
            desc += f'{k}: {v*self.dec}\n'
        return desc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        if hasattr(val, 'shape'):
            if len(val.shape) > 0 and val.shape[0] > 1:
                val = val.mean(dim=0).squeeze(0)

        if hasattr(val, 'item'):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'[{self.name}] score_avg: {self.avg:.9f}'


def create_model_var(args):
    vqvae, var = build_vae_var(
        args=None,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        device=dist_utils.get_device(), patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    )
    vqvae.eval()
    var.eval()

    vae_loaded = False
    if args.var_ckpt_path != '':
        var_ckpt = torch.load(args.var_ckpt_path, map_location='cpu')
        if 'trainer' in var_ckpt.keys():
            vqvae.load_state_dict(var_ckpt['trainer']['vae_local'])
            var.load_state_dict(var_ckpt['trainer']['var_local'], strict=True, compat=False)
            vae_loaded = True
        else:
            # vae_local.load_state_dict(var_ckpt['trainer']['vae_local'])
            missing_keys, unexpected_keys = var.load_state_dict(var_ckpt, strict=False, compat=True)
            print('missing_keys: ', [k for k in missing_keys])
            print('unexpected_keys: ', [k for k in unexpected_keys])
    if not vae_loaded and args.vae_ckpt_path != '':
        vqvae.load_state_dict(args.vae_ckpt_path)

    return var


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=False, default='celeba_hq_blind')
    parser.add_argument('--split', type=str, required=False, default='test', choices=DataOptions.get_splits())
    parser.add_argument('--vae_ckpt_path', type=str, required=False, default='')
    parser.add_argument('--var_ckpt_path', type=str, required=False, default='')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--log_freq', type=int, required=False, default=1)
    parser.add_argument('--nums', type=int, required=False, default=3000)
    parser.add_argument('--seed', type=int, required=False, default=random.randint(1, 10000))

    args = parser.parse_args()
    pprint(args)

    seed_everything(args.seed, benchmark=True)

    data_opt = DataOptions.get_options(args.split)
    pprint(data_opt)
    data_params = {'opt': data_opt}
    dataset = build_dataset(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        params=data_params,
        split=args.split,
    )

    num_samples = min(len(dataset), args.nums)
    data_loader = DataLoader(
        dataset, num_workers=0, pin_memory=True,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset, num_samples=num_samples),
        drop_last=False,
    )

    model = create_model_var(args)

    timer = Timer(dec=1)
    metrics_info = {}
    for m in [
        'fid', 'lpips',
        # 'cos_dist', 'landmark_dist',
        'niqe', 'psnr',
        'ssim', 'musiq',
        'time_cost_avg'
    ]:
        metrics_info[m] = AverageMeter(m)

    def log_metrics(process):
        if (process + 1) % args.log_freq != 0:
            return
        print(f'======== {process} =======')
        for me in metrics_info.values():
            print(me)

    fid = FrechetInceptionDistance()
    # fid = pyiqa.create_metric('fid',  device=dist_utils.get_device())
    lpips = pyiqa.create_metric('lpips-vgg', device=dist_utils.get_device())
    niqe_metric = pyiqa.create_metric('niqe', device=dist_utils.get_device())
    psnr_metric = pyiqa.create_metric('psnr', device=dist_utils.get_device())
    ssim_metric = pyiqa.create_metric('ssim', device=dist_utils.get_device())
    musiq_metric = pyiqa.create_metric('musiq', device=dist_utils.get_device())
    for i, data in enumerate(data_loader):
        data: dict
        lq, hq = data['lq'], data['gt']
        lq = lq.to(dist_utils.get_device(), non_blocking=True)
        hq = hq.to(dist_utils.get_device(), non_blocking=True)

        timer.record(f'{i}_start')
        pred = model.autoregressive_infer_cfg(lq)
        time_cost = timer.record(f'{i}_end')

        hq = denormalize_pm1_into_01(hq)
        pred = denormalize_pm1_into_01(pred)

        # record time cost
        metrics_info['time_cost_avg'].update(time_cost / args.batch_size)

        # calculate fid
        # todo 使用 pyiqa, 运行过程中实时保存 pred, fid 跑完就删, hq 从 dataset 原处读
        fid.update(img01_into_img255(hq), True)
        fid.update(img01_into_img255(pred), False)
        if i > 0 or args.batch_size > 1:
            fid_score = fid.compute()
            metrics_info['fid'].update(fid_score, time_cost)
            # print(f'{i}: {fid_score}, {time_cost}')

        # calculate lpips
        lpips_score = lpips(pred, hq)
        metrics_info['lpips'].update(lpips_score)

        # calculate niqe
        niqe_score = niqe_metric(pred)
        metrics_info['niqe'].update(niqe_score)

        # calculate psnr
        psnr_score = psnr_metric(pred, hq)
        metrics_info['psnr'].update(psnr_score)

        # calculate ssim
        ssim_score = ssim_metric(pred, hq)
        metrics_info['ssim'].update(ssim_score)

        # calculate musiq
        musiq_score = musiq_metric(pred, hq)
        metrics_info['musiq'].update(musiq_score)

        log_metrics(i)


