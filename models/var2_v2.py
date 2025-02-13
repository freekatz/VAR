import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple, Union, Mapping, Any, Dict
import sys, os

from einops import repeat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
import torch.nn as nn
import numpy as np
import torchvision

from utils import dist_utils
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn, AdaLNCrossAttn, FeatureFusionBlock
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class VAR2(nn.Module):
    def __init__(
            self, vae_local: VQVAE,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate

        self.patch_nums: Tuple[int] = patch_nums
        self.control_l = self.patch_nums[-1] ** 2
        self.first_l = self.patch_nums[0] ** 2
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist_utils.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        self.control_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = self.V
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l+self.control_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        control_pe = torch.empty(1, self.control_l, self.C)
        nn.init.trunc_normal_(control_pe, mean=0, std=init_std)
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC.insert(0, control_pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L+self.control_l, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm, add_cond=True,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        mask_items = [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        mask_items.insert(0, torch.full((self.control_l,), 0))
        d: torch.Tensor = torch.cat(mask_items).view(1, self.L+self.control_l, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L+self.control_l, self.L+self.control_l)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self, lq: torch.Tensor,
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng
        B = lq.shape[0]

        f = self.vae_proxy[0].img_to_f(lq)
        control = f.reshape(B, self.Cvae, self.control_l).permute(0, 2, 1)
        control_tokens = self.control_embed(control)

        cond = self.vae_quant_proxy[0].f_to_idxBl_or_fhat(f, to_fhat=False, stop_si=0)[0]
        with torch.cuda.amp.autocast(enabled=False):
            sos = cond_BD = self.class_emb(cond)

        pos_start = self.pos_start.expand(B, self.first_l+self.control_l, -1)
        lvl_pos = self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC
        next_token_map = torch.cat((control_tokens, sos), dim=1)
        next_token_map = next_token_map + pos_start + lvl_pos[:, :self.first_l+self.control_l]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        cur_L = self.control_l
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            if si == 0:
                logits_BlV = logits_BlV[:, self.control_l:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                lvl_p = lvl_pos[:,cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_p

        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat)  # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, lq: torch.FloatTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        B = lq.shape[0]
        f = self.vae_proxy[0].img_to_f(lq)
        control = f.reshape(B, self.Cvae, self.control_l).permute(0, 2, 1)
        control_tokens = self.control_embed(control)

        cond = self.vae_quant_proxy[0].f_to_idxBl_or_fhat(f, to_fhat=False, stop_si=0)[0]
        with torch.cuda.amp.autocast(enabled=False):
            pos_start = self.pos_start.expand(B, self.first_l+self.control_l, -1)
            lvl_pos = self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC

            sos = cond_BD = self.class_emb(cond)
            sos = torch.cat((control_tokens, sos), dim=1) + pos_start

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += lvl_pos

        attn_bias = self.attn_bias_for_masking
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        x_BLC = x_BLC[:, self.control_l:]
        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

    def load_state_dict(self, state_dict: Dict[str, Any],
                        strict: bool = True, assign: bool = False, compat=False):
        compat = True
        for key in list(state_dict.keys()):
            if key.find('control_embed') != -1:
                compat = False
                strict = True
                break

        if compat:
            strict = False
        if compat:
            # compat encoder and decoder
            new_state_dict = OrderedDict()
            for key in list(state_dict.keys()):
                if key.find('pos_1LC') != -1:
                    pass
                elif key.find('class_emb') != -1:
                    pass
                elif key.find('pos_start') != -1:
                    pass
                elif key.find('lvl_1L') != -1:
                    pass
                elif key.find('attn_bias_for_masking') != -1:
                    pass
                else:
                    new_state_dict[key] = state_dict.pop(key)
        else:
            new_state_dict = state_dict
        return super().load_state_dict(state_dict=new_state_dict, strict=strict, assign=assign)


if __name__ == '__main__':
    import argparse
    import glob
    import math

    import PIL.Image as PImage
    import torch

    from utils import dist_utils
    from utils.dataset.options import DataOptions

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--out', type=str, default='res')
    parser.add_argument('--opts', type=str, default='{}')
    args = parser.parse_args()

    device = 'cpu'
    seed = args.seed
    import random

    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            print(f'[in seed_everything] {self.deterministic=}', flush=True)
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            seed = self.seed + dist_utils.get_rank() * 16384
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.same_seed_for_all_ranks = seed


    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)


    def denormalize_pm1_into_01(x):  # normalize x from [-1, 1] to [0, 1] by (x + 1)/2
        return x.add(1) / 2

    import sys
    from pathlib import Path
    import os
    from pprint import pprint

    root = Path(os.path.dirname(__file__)).parent

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    # VQVAE args
    V = 4096
    Cvae = 32
    ch = 160
    share_quant_resi = 4
    depth = 16
    shared_aln = False
    attn_l2_norm = True
    flash_if_available = True
    fused_if_available = True
    init_adaln = 0.5
    init_adaln_gamma = 1e-5
    init_head = 0.02
    init_std = -1
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums)

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24
    var_wo_ddp = VAR2(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    )
    vae_local.eval()
    var_wo_ddp.eval()
    if args.ckpt != '':
        var_ckpt = torch.load(args.ckpt, map_location='cpu')
        if 'trainer' in var_ckpt.keys():
            vae_local.load_state_dict(var_ckpt['trainer']['vae_local'])
            var_wo_ddp.load_state_dict(var_ckpt['trainer']['var_wo_ddp'], strict=True, compat=False)
        else:
            # vae_local.load_state_dict(var_ckpt['trainer']['vae_local'])
            missing_keys, unexpected_keys = var_wo_ddp.load_state_dict(var_ckpt, strict=False, compat=True)
            print('missing_keys: ', [k for k in missing_keys])
            print('unexpected_keys: ', [k for k in unexpected_keys])

    from utils.dataset.ffhq_blind import FFHQBlind
    import torchvision
    import numpy as np

    # validate
    opt = DataOptions.val_options()
    pprint(opt)

    import json
    arg_opts = json.loads(args.opts)
    for k, v in arg_opts.items():
        print(f'Option updates: {k} {opt[k]} -> {k} {v}')
        opt[k] = v

    import torch.utils.data as data
    import PIL.Image as PImage

    def inference(i, ds: data.Dataset):
        lq_in_list = []
        hq_in_list = []
        for idx in range(len(ds)):
            if idx > 20:
                break
            res = ds[idx]
            lq, hq = res['lq'], res['gt']
            lq_in_list.append(lq)
            hq_in_list.append(hq)
        lq = torch.stack(lq_in_list, dim=0)
        hq = torch.stack(hq_in_list, dim=0)
        print(i, lq.shape, hq.shape)

        lq_res = var_wo_ddp.autoregressive_infer_cfg(lq)
        hq_res = var_wo_ddp.autoregressive_infer_cfg(hq)

        res = [hq, lq, hq_res, lq_res]
        res_img = torch.stack(res, dim=1)
        res_img = torch.reshape(res_img, (-1, 3, 256, 256))
        img = denormalize_pm1_into_01(res_img)
        chw = torchvision.utils.make_grid(img, nrow=len(res), padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        filename = f'dataset{i}-{args.out}-mode{args.mode}.png'
        chw.save(os.path.join(root, filename))
        print(f'Saved {filename}...')

    def inference2(i, ds: data.Dataset):
        lq_in_list = []
        hq_in_list = []
        for idx in range(len(ds)):
            if idx > 20:
                break
            res = ds[idx]
            lq, hq = res['lq'], res['gt']
            lq_in_list.append(lq)
            hq_in_list.append(hq)
        lq = torch.stack(lq_in_list, dim=0)
        hq = torch.stack(hq_in_list, dim=0)
        print(i, lq.shape, hq.shape)

        idx_hq = var_wo_ddp.vae_proxy[0].img_to_idxBl(hq)
        x_BLCv_wo_first_l = var_wo_ddp.vae_quant_proxy[0].idxBl_to_var_input(idx_hq)
        logits = var_wo_ddp(lq, x_BLCv_wo_first_l)
        pred = torch.argmax(logits, dim=-1)
        pred = list(torch.split(pred, [ph * pw for (ph, pw) in var_wo_ddp.vae_proxy[0].patch_hws], dim=1))
        lq_res = var_wo_ddp.vae_proxy[0].idxBl_to_img(pred, same_shape=True)[-1]

        logits = var_wo_ddp(hq, x_BLCv_wo_first_l)
        pred = torch.argmax(logits, dim=-1)
        pred = list(torch.split(pred, [ph * pw for (ph, pw) in var_wo_ddp.vae_proxy[0].patch_hws], dim=1))
        hq_res = var_wo_ddp.vae_proxy[0].idxBl_to_img(pred, same_shape=True)[-1]

        res = [hq, lq, hq_res, lq_res]
        res_img = torch.stack(res, dim=1)
        res_img = torch.reshape(res_img, (-1, 3, 256, 256))
        img = denormalize_pm1_into_01(res_img)
        chw = torchvision.utils.make_grid(img, nrow=len(res), padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        filename = f'dataset{i}-{args.out}-mode{args.mode}.png'
        chw.save(os.path.join(root, filename))
        print(f'Saved {filename}...')

    for i in range(4):
        ds = FFHQBlind(root=f'{args.data}{i+1}', split='train', opt=opt)
        if args.mode == 0:
            inference(i + 1, ds)
        elif args.mode == 1:
            inference2(i + 1, ds)
