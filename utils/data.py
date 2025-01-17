import os.path as osp
from typing import List, Dict

import PIL.Image as PImage
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import InterpolationMode, transforms

from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils import dist_utils
from utils import arg_util
from utils.my_dataset import UnlabeledDatasetFolder, FFHQ, FFHQBlind
from utils.my_transforms import BlindTransform, NormTransform


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_transforms(args: arg_util.Args):
    dataset_name = args.dataset_name
    final_reso = args.data_load_reso
    mid_reso = args.mid_reso
    hflip = args.hflip
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    if dataset_name in ['imagenet', 'ffhq']:
        train_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
        if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())

        val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
        train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
        train_transform, val_transform = {'transform': train_aug}, {'transform': val_aug}
    elif dataset_name in ['ffhq_blind']:
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
        train_lq_aug = [
            transforms.Resize(final_reso, interpolation=InterpolationMode.LANCZOS),
            BlindTransform(opt),
            NormTransform(opt),
        ]
        train_hq_aug = [
            transforms.Resize(final_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            NormTransform(opt)
        ]
        val_lq_aug = [
            transforms.Resize(final_reso, interpolation=InterpolationMode.LANCZOS),
            BlindTransform(opt),
            NormTransform(opt),
        ]
        val_hq_aug = [
            transforms.Resize(final_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            NormTransform(opt)
        ]
        train_lq_transform, val_lq_transform = transforms.Compose(train_lq_aug), transforms.Compose(val_lq_aug)
        train_hq_transform, val_hq_transform = transforms.Compose(train_hq_aug), transforms.Compose(val_hq_aug)
        train_transform = {'lq_transform': train_lq_transform, 'hq_transform': train_hq_transform}
        val_transform = {'lq_transform': val_lq_transform, 'hq_transform': val_hq_transform}
    else:
        raise NotImplementedError
    for key, train_aug in train_transform.items():
        print_aug(train_aug, '[train]')
    for key, val_aug in val_transform.items():
        print_aug(val_aug, '[val]')
    return train_transform, val_transform


def build_dataset(
    dataset_name: str, data_path: str, transform_dict: Dict[str, transforms.Compose], split='train'
):
    print('Building dataset: dataset_name={}, split={}'.format(dataset_name, split))
    if dataset_name == 'imagenet':
        dataset = UnlabeledDatasetFolder(root=data_path, split=split, **transform_dict)
    elif dataset_name == 'ffhq':
        dataset = FFHQ(root=data_path, split=split, **transform_dict)
    elif dataset_name == 'ffhq_blind':
        dataset = FFHQBlind(root=data_path, split=split, **transform_dict)
    else:
        raise NotImplementedError
    print(f'[Dataset] length={len(dataset)}')
    return dataset


def build_data_loader(args, start_ep, start_it, dataset=None, transform_list=None, split='train'):
    if dataset is None:
        dataset = build_dataset(args.dataset_name, args.data_path, transform_list, split=split)
    if split == 'train':
        data_loader = DataLoader(
            dataset=dataset, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(),  # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset), glb_batch_size=args.bs,
                same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist_utils.get_rank(), world_size=dist_utils.get_world_size(),
                start_ep=start_ep, start_it=start_it,
            ),
        )
    elif split == 'val':
        data_loader = DataLoader(
            dataset, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size*1.5),
            sampler=EvalDistributedSampler(dataset, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank()),
            shuffle=False, drop_last=False,
        )
    else:
        raise NotImplementedError
    return data_loader


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
