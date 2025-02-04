import os.path as osp
from typing import List, Dict

import PIL.Image as PImage
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import InterpolationMode, transforms

from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils import dist_utils
from utils import arg_util
from utils.dataset.ffhq import FFHQ
from utils.dataset.ffhq_blind import FFHQBlind
from utils.dataset.ffhq_blind_uneven import FFHQBlindUneven
from utils.dataset.img_folder import UnlabeledDatasetFolder


def build_dataset(
    dataset_name: str, data_path: str, params: dict, split='train'
):
    print('Building dataset: dataset_name={}, split={}'.format(dataset_name, split))
    if dataset_name == 'imagenet':
        dataset = UnlabeledDatasetFolder(root=data_path, split=split, **params)
    elif dataset_name == 'ffhq':
        dataset = FFHQ(root=data_path, split=split, **params)
    elif dataset_name == 'ffhq_blind':
        if split == 'train':
            dataset = FFHQBlindUneven(root=data_path, split=split, **params)
        else:
            dataset = FFHQBlind(root=data_path, split=split, **params)
    else:
        raise NotImplementedError
    print(f'Dataset {dataset.__class__} size: {len(dataset)}')
    return dataset


def build_data_loader(args, start_ep, start_it, dataset=None, dataset_params=None, split='train'):
    if dataset is None:
        dataset = build_dataset(args.dataset_name, args.data_path, dataset_params, split=split)
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
