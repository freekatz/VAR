import gc
import os
import random
import shutil
import sys
import time
import warnings
from functools import partial
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import dist_utils
from utils import arg_util, misc
from utils.data import build_data_loader, build_dataset
from utils.dataset.options import DataOptions
from utils.misc import maybe_resume, maybe_pretrain
from utils.lr_control import lr_wd_annealing
from models import VAR, VQVAE, build_vae_var
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params


def build_tensorboard_logger(args: arg_util.Args):
    tensor_board: misc.TensorboardLogger
    with_tensor_board = dist_utils.is_master()
    if with_tensor_board:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tensor_board = misc.DistLogger(
            misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'),
            verbose=True)
        tensor_board.flush()
    else:
        # noinspection PyTypeChecker
        tensor_board = misc.DistLogger(None, verbose=False)
    dist_utils.barrier()
    return tensor_board


def build_model(args):
    vae_local, var_local = build_vae_var(
        args=args,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        device=dist_utils.get_device(), patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    vae_local.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=True)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_local: VAR = args.compile_model(var_local, args.tfast)
    var_ddp: DDP = (DDP if dist_utils.initialized() else NullDDP)(var_local, device_ids=[dist_utils.get_local_rank()],
                                                              find_unused_parameters=False, broadcast_buffers=False)

    print(f'[INIT] VAR model = {var_local}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
    ('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder),
    ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_local),)]) + '\n\n')
    return vae_local, var_local, var_ddp


def build_optimizer(args: arg_util.Args, var_local):
    names, paras, para_groups = filter_params(var_local, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_class = {
        'adam': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_params = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_class}, opt_params={opt_params}\n')

    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_class(params=para_groups, **opt_params), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    return var_optim


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = maybe_resume(args)
    if len(trainer_state) == 0:
        trainer_state = maybe_pretrain(args)
    # create tensorboard logger
    tensor_board = build_tensorboard_logger(args)
    
    # log args
    print(f'global batch size={args.glb_batch_size}, local batch size={args.batch_size}')
    print(f'initial args:\n{str(args)}')

    # build data
    print(f'[build PT data] ...\n')
    train_opt = DataOptions.train_options()
    ld_train = build_data_loader(args, start_ep, start_it, dataset=None,
                                 dataset_params={'opt': train_opt}, split='train')
    val_opt = DataOptions.val_options()
    ld_val = build_data_loader(args, start_ep, start_it, dataset=None,
                               dataset_params={'opt': val_opt}, split='val')

    if len(args.data_path_test) > 0:
        test_opt = DataOptions.test_options()
        test_ds = build_dataset(args.dataset_name_test, args.data_path_test,
                                {'opt': test_opt}, split='test')
        ld_test = build_data_loader(args, start_ep, start_it, dataset=test_ds,
                                    dataset_params=None, split='test')
    else:
        ld_test = None

    [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}')

    # build models
    vae_local, var_local, var_ddp = build_model(args)

    # build optimizer
    var_optim = build_optimizer(args, var_local)
    
    # build trainer
    from utils.trainer import VARTrainer
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_local=var_local, var_ddp=var_ddp,
        var_opt=var_optim, label_smooth=args.ls,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_local, var_ddp, var_optim

    dist_utils.barrier()
    return (
        tensor_board, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val, ld_test
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tensor_board, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val, ld_test
    ) = build_everything(args)
    
    # train
    # Lmean is loss about all scalers logits (680), Ltail is loss about last scaler logits (256)
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tensor_board.set_step(ep * iters_train)

        print(f'[train ...]@ep{ep}')
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tensor_board, ld_train, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        is_val_and_also_saving = (ep + 1) % args.val_freq == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            print(f'[val ...]@ep{ep}')
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist_utils.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_bak = os.path.join(args.local_out_dir_path, 'ar-ckpt-last-bak.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                if os.path.exists(local_out_ckpt):
                    shutil.move(local_out_ckpt, local_out_ckpt_bak)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                    print(f'     [saving ckpt](*) finished! best @ {local_out_ckpt}', flush=True, clean=True)
                print(f'     [saving ckpt](*) finished! last @ {local_out_ckpt}', flush=True, clean=True)
            dist_utils.barrier()

        is_test = (ep + 1) % args.test_freq == 0 or (ep + 1) == args.ep and ld_test is not None
        if is_test and dist_utils.is_local_master():
            print(f'[test ...]@ep{ep}')
            out_dir = Path(args.visual_out_dir_path)
            trainer.test_ep(ld_test, ep, out_dir)
        dist_utils.barrier()

        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tensor_board.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tensor_board.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tensor_board.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tensor_board.flush(); tensor_board.close()
    dist_utils.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tensor_board: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    step_cnt = 0
    me = misc.MetricLogger(delimiter='|')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))  # learn rate
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))  # grad clips
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train

    for it, data in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()

        lq, hq = data['lq'], data['gt']
        lq = lq.to(args.device, non_blocking=True)
        hq = hq.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping,
            metric_lg=me, tensor_board=tensor_board,
            lq=lq, hq=hq
        )
        
        me.update(tlr=max_tlr)
        tensor_board.set_step(step=g_it)
        tensor_board.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tensor_board.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tensor_board.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tensor_board.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tensor_board.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tensor_board.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tensor_board.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist_utils.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
