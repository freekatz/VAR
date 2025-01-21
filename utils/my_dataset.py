import os

import PIL.Image as PImage
import random
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms

from utils.my_transforms import NormTransform, BlindTransform, print_transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class UnlabeledDatasetFolder(DatasetFolder):
    def __init__(self, root, transform=None, split='train'):
        self.root = os.path.join(root, split)
        super().__init__(root=self.root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class FFHQ(Dataset):
    def __init__(self, root, transform=None, split='train'):
        super().__init__()
        self.root = root
        split_file = os.path.join(self.root, f'ffhq_{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [os.path.join(self.root, line.strip()) for line in file.readlines() if line.find('.png') != -1]
        assert(len(self.samples) > 0)

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample


class FFHQBlind(Dataset):
    def __init__(self, root, lq_transform, hq_transform, split='train', **kwargs):
        super().__init__()
        self.root = root
        self.split = split
        split_file = os.path.join(self.root, f'ffhq_{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [os.path.join(self.root, line.strip()) for line in file.readlines() if line.find('.png') != -1]
        assert(len(self.samples) > 0)

        self.identify_ratio = kwargs.get('identify_ratio', 0.)

        use_hflip = kwargs.get('use_hflip', False)
        base_transform = []
        if use_hflip:
            base_transform.insert(0, transforms.RandomHorizontalFlip(0.5))
        self.base_transform = transforms.Compose(base_transform)
        for aug in base_transform:
            print_transforms(aug, f'[{split}-base]')

        self.lq_transform = lq_transform
        self.hq_transform = hq_transform
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        hq = self.loader(path)
        hq = self.base_transform(hq)
        hq_res = self.hq_transform(hq)
        if self.split == 'val' or random.random() > (1-self.identify_ratio) :
            return hq_res, hq_res
        lq_res = self.lq_transform(hq)
        return lq_res, hq_res


if __name__ == '__main__':
    import math
    import torch
    import cv2
    from torchvision.transforms import InterpolationMode

    data = '../tmp'
    opt = {
        'blur_kernel_size': 41,
        'kernel_list': ['iso', 'aniso'],
        'kernel_prob': [0.5, 0.5],
        'blur_sigma': [1, 15],
        'downsample_range': [4, 30],
        'noise_range': [0, 1],
        'jpeg_range': [30, 80],
    }
    train_lq_aug = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
            BlindTransform(opt),
            NormTransform(),
        ]
    )
    train_hq_aug = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            NormTransform(),
        ]
    )
    train_transform = {'lq_transform': train_lq_aug, 'hq_transform': train_hq_aug, 'use_hflip': True}
    ds = FFHQBlind(root=data, split='train', **train_transform)
    lq, hq = ds[3]
    print(lq.size())
    print(hq.size())
    print(lq.min(), lq.max(), lq.mean(), lq.std())
    print(hq.min(), hq.max(), hq.mean(), hq.std())

    from torchvision.transforms import InterpolationMode

    img_lq = transforms.ToPILImage()(lq)
    img_hq = transforms.ToPILImage()(hq)
    img_lq.save('../lq.png')
    img_hq.save('../hq.png')
