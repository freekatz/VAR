import math
from functools import partial
from typing import Optional, Tuple, Union
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
import torch.nn as nn
import numpy as np
import torchvision

from utils import dist_utils
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn, AdaLNCrossAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from utils.my_dataset import FFHQBlind


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
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
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

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        # self.num_classes = num_classes
        # self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
        #                                device=dist_utils.get_device())
        self.cond_embed = nn.Embedding(self.V, self.C)
        nn.init.trunc_normal_(self.cond_embed.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
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
            AdaLNCrossAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
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
        d: torch.Tensor = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L,
                                                                                                              1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
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
            self, lq: torch.FloatTensor,
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng
        B = lq.shape[0]

        idx_lq = self.vae_proxy[0].img_to_idxBl(lq)
        cond, x_BLCv_lq = self.vae_quant_proxy[0].idxBl_to_var2_input(idx_lq)
        x_BLC_lq = self.word_embed(x_BLCv_lq.float())
        x_BLC_lq = torch.split(x_BLC_lq, [ph * pw for (ph, pw) in self.vae_proxy[0].patch_hws], dim=1)

        sos = cond_BD = self.cond_embed(cond.squeeze(-1))
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        pos_start = self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map_hq = sos.unsqueeze(1).expand(B, self.first_l, -1) + pos_start
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            next_token_map_lq = x_BLC_lq[si]

            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn * pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map_hq
            y = next_token_map_lq
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, y=y, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(y, cond_BD)

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map_hq = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map_hq = next_token_map_hq.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map_hq = self.word_embed(next_token_map_hq) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, lq: torch.FloatTensor, x_BLCv_wo_first_l_hq: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = lq.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            idx_lq = self.vae_proxy[0].img_to_idxBl(lq)
            cond, x_BLCv_lq = self.vae_quant_proxy[0].idxBl_to_var2_input(idx_lq)
            sos = cond_BD = self.cond_embed(cond.squeeze(-1))
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC_lq = sos
                x_BLC_hq = sos
            else:
                x_BLC_lq = self.word_embed(x_BLCv_lq.float())
                x_BLC_hq = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l_hq.float())), dim=1)
            # x_BLC_lq += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC
            x_BLC_hq += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC_hq.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC_lq = x_BLC_lq.to(dtype=main_type)
        x_BLC_hq = x_BLC_hq.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNCrossAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC_hq = b(x=x_BLC_hq, y=x_BLC_lq, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        # x_BLC torch.Size([1, 680, 1024])
        x_BLC_hq = self.get_logits(x_BLC_hq.float(), cond_BD)
        return x_BLC_hq  # logits BLV, V is vocab_size

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


if __name__ == '__main__':
    import sys

    device = 'cpu'

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

    data_root =  sys.argv[1]
    arg_idx = 2
    if len(sys.argv) > arg_idx:
        var_ckpt = sys.argv[arg_idx]
        var_ckpt = torch.load(var_ckpt, map_location='cpu')
        vae_local.load_state_dict(var_ckpt['trainer']['vae_local'])

        print(var_ckpt['trainer'].keys())
        var_wo_ddp.load_state_dict(var_ckpt['trainer']['var_wo_ddp'], strict=False)
        arg_idx+=1

    import PIL.Image as PImage
    from torchvision.transforms import InterpolationMode, transforms
    import torch

    from utils.my_transforms import BlindTransform, NormTransform, denormalize_pm1_into_01

    opt = {
        'blur_kernel_size': 41,
        'kernel_list': ['iso', 'aniso'],
        'kernel_prob': [0.5, 0.5],
        'blur_sigma': [1, 15],
        'downsample_range': [4, 30],
        'noise_range': [0, 20],
        'jpeg_range': [30, 80],
        # 'color_jitter_prob': 0.3,
        # 'color_jitter_shift': 20,
        # 'color_jitter_pt_prob': 0.3,
        # 'gray_prob': 0.01,
    }
    final_reso = 256
    mid_reso = 1.125
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_lq_aug = [
        transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
        BlindTransform(opt),
        NormTransform()
    ]
    train_hq_aug = [
        transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        NormTransform()
    ]

    train_lq_transform = transforms.Compose(train_lq_aug)
    train_hq_transform = transforms.Compose(train_hq_aug)

    params = {'lq_transform': train_lq_transform, 'hq_transform': train_hq_transform, 'use_hflip': True}
    ds = FFHQBlind(root=data_root, split='train', **params)
    data_idx = 0
    if len(sys.argv) > arg_idx:
        data_idx = int(sys.argv[arg_idx])
        arg_idx+=1
    lq, hq = ds[data_idx]
    print(lq.shape, hq.shape)

    from pathlib import Path
    out_path = Path('../out')
    out_path.mkdir(parents=True, exist_ok=True)

    img_lq = transforms.ToPILImage()(denormalize_pm1_into_01(lq))
    img_hq = transforms.ToPILImage()(denormalize_pm1_into_01(hq))
    img_lq.save('../out/lq.png')
    img_hq.save('../out/hq.png')

    swap = False
    if len(sys.argv) > arg_idx:
        swap = True
        arg_idx += 1
    inference = False
    if len(sys.argv) > arg_idx:
        inference = True
        arg_idx += 1

    x_BLCv_wo_first_l = None
    if inference:
        img = [var_wo_ddp.autoregressive_infer_cfg(lq.unsqueeze(0))]
    else:
        idx_hq = var_wo_ddp.vae_proxy[0].img_to_idxBl(hq.unsqueeze(0))
        idx_lq = var_wo_ddp.vae_proxy[0].img_to_idxBl(lq.unsqueeze(0))
        x_BLCv_wo_first_l = var_wo_ddp.vae_quant_proxy[0].idxBl_to_var_input(idx_hq)
        logits = var_wo_ddp(lq.unsqueeze(0), x_BLCv_wo_first_l)
        pred = torch.argmax(logits, dim=-1)
        pred = list(torch.split(pred, [ph * pw for (ph, pw) in var_wo_ddp.vae_proxy[0].patch_hws], dim=1))
        img = var_wo_ddp.vae_proxy[0].idxBl_to_img(pred, same_shape=True)
    img.insert(0, lq.unsqueeze(0))
    img.insert(0, hq.unsqueeze(0))
    img = torch.cat(img, dim=0)
    img = denormalize_pm1_into_01(img)
    chw = torchvision.utils.make_grid(img, nrow=3, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    chw.save(f'../out/{data_idx}-lq-res.png')

    if swap:
        lq = hq
        if inference:
            img = [var_wo_ddp.autoregressive_infer_cfg(lq.unsqueeze(0))]
        else:
            logits = var_wo_ddp(lq.unsqueeze(0), x_BLCv_wo_first_l)
            pred = torch.argmax(logits, dim=-1)
            pred = list(torch.split(pred, [ph * pw for (ph, pw) in var_wo_ddp.vae_proxy[0].patch_hws], dim=1))
            img = var_wo_ddp.vae_proxy[0].idxBl_to_img(pred, same_shape=True)
        img.insert(0, lq.unsqueeze(0))
        img.insert(0, hq.unsqueeze(0))
        img = torch.cat(img, dim=0)
        img = denormalize_pm1_into_01(img)
        chw = torchvision.utils.make_grid(img, nrow=3, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        chw.save(f'../out/{data_idx}-hq-res.png')
