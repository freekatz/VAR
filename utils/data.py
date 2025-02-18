import PIL.Image as PImage
from torch.utils.data import DataLoader

from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils import dist_utils
from utils.dataset import UnlabeledImageItem
from utils.dataset.blind import BlindDataset
from utils.dataset.celeba_hq import CelebAHQ
from utils.dataset.ffhq import FFHQ
from utils.dataset.img_folder import UnlabeledImageFolder


def build_dataset(
    dataset_name: str, data_path: str, params: dict=None, split='train'
):
    if params is None:
        params = {}
    print('Building dataset: dataset_name={}, split={}'.format(dataset_name, split))
    if dataset_name in ['ffhq']:
        dataset = FFHQ(root=data_path, split=split, **params)
    elif dataset_name in ['ffhq_blind']:
        base_dataset = FFHQ(root=data_path, split=split, **params)
        dataset = BlindDataset(base_dataset=base_dataset, **params)
    elif dataset_name in ['celeba_hq']:
        dataset = CelebAHQ(root=data_path, split=split, **params)
    elif dataset_name in ['celeba_hq_blind']:
        base_dataset = CelebAHQ(root=data_path, split=split, **params)
        dataset = BlindDataset(base_dataset=base_dataset, **params)
    elif dataset_name in ['image_folder_blind']:
        base_dataset = UnlabeledImageFolder(root=data_path, split=split, **params)
        dataset = BlindDataset(base_dataset=base_dataset, **params)
    elif dataset_name in ['image_item_blind']:
        base_dataset = UnlabeledImageItem(root=data_path, split=split, **params)
        dataset = BlindDataset(base_dataset=base_dataset, **params)
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
    elif split == 'test':
        data_loader = DataLoader(
            dataset, num_workers=0, pin_memory=True,
            batch_size=1,
            shuffle=False, drop_last=False,
        )
    else:
        raise NotImplementedError
    return data_loader


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img
