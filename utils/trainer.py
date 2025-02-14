import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import PIL.Image as PImage

from utils import dist_utils
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.dataset.my_transforms import denormalize_pm1_into_01
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
            self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
            vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
            var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')

        self.first_l = self.var_wo_ddp.first_l
        self.L = self.var_wo_ddp.L - self.var_wo_ddp.first_l + 1
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        # self.loss_weight_first = torch.ones(1, self.first_l, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    @torch.no_grad()
    def test_ep(self, ld_test: DataLoader, ep: int, out_dir: Path):
        print(f'[testing ...] @ep{ep} test_ds={len(ld_test)}')

        self.var_wo_ddp.eval()
        vis_root = out_dir / f'ep{ep:04d}'
        vis_root.mkdir(parents=True, exist_ok=True)
        vis_max_count = 16
        vis_count = 0
        lq_list = []
        hq_list = []
        lq_res_list = []
        hq_res_list = []
        for i, data in enumerate(ld_test):
            data: dict
            lq, hq = data['lq'], data['gt']
            lq = lq.to(dist_utils.get_device(), non_blocking=True)
            hq = hq.to(dist_utils.get_device(), non_blocking=True)
            lq_list.append(lq)
            hq_list.append(hq)

            lq_res = self.var_wo_ddp.autoregressive_infer_cfg(lq)
            hq_res = self.var_wo_ddp.autoregressive_infer_cfg(hq)
            lq_res_list.append(lq_res)
            hq_res_list.append(hq_res)
            vis_count += 1
            if i == len(ld_test) - 1 or vis_count % vis_max_count == 0:
                # save img
                mid = vis_count // 2
                lq_left, lq_right = lq_list[:mid], lq_list[-mid:]
                hq_left, hq_right = hq_list[:mid], hq_list[-mid:]
                lq_res_list_left, lq_res_list_right = lq_res_list[:mid], lq_res_list[-mid:]
                hq_res_list_left, hq_res_list_right = hq_res_list[:mid], hq_res_list[-mid:]

                res = [
                    torch.cat(hq_left), torch.cat(lq_left), torch.cat(hq_res_list_left), torch.cat(lq_res_list_left),
                    torch.cat(hq_right), torch.cat(lq_right), torch.cat(hq_res_list_right),
                    torch.cat(lq_res_list_right),
                ]

                res_img = torch.stack(res, dim=1)
                res_img = torch.reshape(res_img, (-1, 3, 256, 256))
                img = denormalize_pm1_into_01(res_img)
                chw = torchvision.utils.make_grid(img, nrow=len(res), padding=0, pad_value=1.0)
                chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
                chw = PImage.fromarray(chw.astype(np.uint8))
                filename = f'{i:06d}.png'
                filepath = os.path.join(vis_root, filename)
                chw.save(filepath)
                print(f'[testing ...] @ep{ep} saved {filepath}, images: {mid*2}')

                if (len(ld_test) - i - 1) // vis_max_count == 0:
                    break
                vis_count = 0

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()

        for i, data in enumerate(ld_val):
            data: dict
            lq, hq = data['lq'], data['gt']
            B, V = lq.shape[0], self.vae_local.vocab_size
            lq = lq.to(dist_utils.get_device(), non_blocking=True)
            hq = hq.to(dist_utils.get_device(), non_blocking=True)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(hq)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)

            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(lq, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.reshape(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                    gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (
                        100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist_utils.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        lq: FTen, hq: FTen, prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1  # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1  # max prog, as if no prog

        # forward
        B, V = lq.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(hq)
        x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        # gt_BL_first = torch.repeat_interleave(gt_idx_Bl[0], self.first_l, dim=1)

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(lq, x_BLCv_wo_first_l)
            # logits_BLV = logits[:, -self.L:]
            # logits_BLV_first = logits[:, :self.first_l]

            loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.view(-1)).view(B, -1)
            # loss_first = self.train_loss(logits_BLV_first.reshape(-1, V), gt_BL_first.view(-1)).view(B, -1)

            loss = loss.mul(self.loss_weight).sum(dim=-1).mean()
            # loss_first = loss_first.mul(self.loss_weight_first).sum(dim=-1).mean()
            # loss = loss + loss_first

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        # pred_BL_first = logits_BLV_first.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.reshape(-1, V), gt_BL.view(-1)).item()
            # Lfirst = self.val_loss(logits_BLV_first.data.reshape(-1, V),
            #                       gt_BL_first.reshape(-1)).item()
            Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                gt_BL[:, -self.last_l:].reshape(-1)).item()

            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            # acc_first = (pred_BL_first == gt_BL_first).float().mean().item() * 100
            acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100

            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist_utils.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist_utils.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp,
                             step=g_it)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums': self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it': self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        if len(state) == 1:
            ks = {'var_wo_ddp'}
        else:
            ks = ('var_wo_ddp', 'vae_local', 'var_opt')
        for k in ks:
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                if k == 'var_wo_ddp':
                    ret = m.load_state_dict(state[k], strict=False, compat=True)
                else:
                    ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        if config is not None:
            self.prog_it = config.get('prog_it', 0)
            self.last_prog_si = config.get('last_prog_si', -1)
            self.first_prog = config.get('first_prog', True)
            if config is not None:
                for k, v in self.get_config().items():
                    if config.get(k, None) != v:
                        err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                        if strict:
                            raise AttributeError(err)
                        else:
                            print(err)
