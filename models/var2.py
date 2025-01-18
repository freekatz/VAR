import math
from functools import partial
from typing import Optional, Tuple, Union
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


from utils import dist_utils
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from utils.my_dataset import FFHQBlind


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR2(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist_utils.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 3. absolute position embedding
        init_std = math.sqrt(1 / self.C / 3)
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(h.float()).float()
    
    def forward(self, lq: torch.FloatTensor) -> torch.Tensor:  # returns logits_BLV
        bg, ed = (0, self.L)
        B = lq.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            lq_idx  = self.vae_proxy[0].img_to_idxBl(lq)
            x_lq_BLC = self.vae_quant_proxy[0].idxBl_to_var2_input(lq_idx)
            x_BLC = self.word_embed(x_lq_BLC.float())
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=None, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float())
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
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
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


if __name__ == '__main__':
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # var_ckpt = r'/Users/katz/Projects/var/test/ar-ckpt-best.pth'
    var_ckpt = sys.argv[1]
    var_ckpt = torch.load(var_ckpt, map_location='cpu')

    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums)
    vae_local.load_state_dict(var_ckpt['trainer']['vae_local'])

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24
    var_wo_ddp = VAR2(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    )


    print(var_ckpt['trainer'].keys())
    var_wo_ddp.load_state_dict(var_ckpt['trainer']['var_wo_ddp'])

    import PIL.Image as PImage
    from torchvision.transforms import InterpolationMode, transforms
    import torch

    from utils.my_transforms import BlindTransform, NormTransform

    def tensor_to_img(img_tensor: torch.Tensor) -> PImage.Image:
        B, C, H, W = img_tensor.shape
        assert int(math.sqrt(B)) * int(math.sqrt(B)) == B
        b = int(math.sqrt(B))
        img_tensor = torch.permute(img_tensor, (1, 0, 2, 3))
        img_tensor = torch.reshape(img_tensor, (C, b, b * H, W))
        img_tensor = torch.permute(img_tensor, (0, 2, 1, 3))
        img_tensor = torch.reshape(img_tensor, (C, b * H, b * W))
        img = transforms.ToPILImage()(img_tensor)
        return img

    opt = {
        'blur_kernel_size': 41,
        'kernel_list': ['iso', 'aniso'],
        'kernel_prob': [0.5, 0.5],
        'blur_sigma': [1, 15],
        'downsample_range': [4, 30],
        'noise_range': [0, 1],
        'jpeg_range': [30, 80],
        'use_hflip': True,
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0]
    }
    final_reso = 256
    train_lq_aug = [
            transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
            BlindTransform(opt),
            # NormTransform(opt),
        ]
    train_hq_aug = [
            transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            # NormTransform(opt)
        ]

    train_lq_transform = transforms.Compose(train_lq_aug)
    train_hq_transform = transforms.Compose(train_hq_aug)

    root = sys.argv[2]
    transform_dict = {'lq_transform': train_lq_transform, 'hq_transform': train_hq_transform}
    ds = FFHQBlind(root=root, split='val', **transform_dict)
    idx = 0
    if len(sys.argv) > 3:
        idx = int(sys.argv[3])
    lq, hq = ds[idx]
    print(lq.shape, hq.shape)

    img_lq = transforms.ToPILImage()(lq)
    img_hq = transforms.ToPILImage()(hq)
    img_lq.save('../out/lq.png')
    img_hq.save('../out/hq.png')

    logits = var_wo_ddp(lq.unsqueeze(0))
    pred = logits.argmax(dim=-1)
    pred = list(torch.split(pred, [ph*pw for (ph, pw) in var_wo_ddp.vae_proxy[0].patch_hws], dim=1))
    img = var_wo_ddp.vae_proxy[0].idxBl_to_img(pred, same_shape=True)
    for i, im in enumerate(img):
        pim = transforms.ToPILImage()(im.squeeze(0))
        pim.save(f'../out/lq-{i}.png')

    swap = False
    if len(sys.argv) > 4:
        swap = True
    if swap:
        lq = hq
        logits = var_wo_ddp(lq.unsqueeze(0))
        pred = logits.argmax(dim=-1)
        pred = list(torch.split(pred, [ph*pw for (ph, pw) in var_wo_ddp.vae_proxy[0].patch_hws], dim=1))
        img = var_wo_ddp.vae_proxy[0].idxBl_to_img(pred, same_shape=True)
        for i, im in enumerate(img):
            pim = transforms.ToPILImage()(im.squeeze(0))
            pim.save(f'../out/lq-{i}-swaped.png')
