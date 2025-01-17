import os

import PIL.Image as PImage
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torchvision.transforms.v2 import ToTensor

from utils.my_transforms import NormTransform, BlindTransform


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
    def __init__(self, root, lq_transform, hq_transform, split='train'):
        super().__init__()
        self.root = root
        split_file = os.path.join(self.root, f'ffhq_{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [os.path.join(self.root, line.strip()) for line in file.readlines() if line.find('.png') != -1]
        assert(len(self.samples) > 0)
        print(f'Dataset size: {len(self.samples)}')

        self.lq_transform = lq_transform
        self.hq_transform = hq_transform
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        hq = self.loader(path)
        lq = self.lq_transform(hq)
        hq = self.hq_transform(hq)
        return lq, hq


if __name__ == '__main__':
    import math
    import torch
    import cv2
    from torchvision.transforms import InterpolationMode, transforms

    data = '../tmp'
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
    train_lq_aug = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
            BlindTransform(opt),
            NormTransform(opt),
        ]
    )
    train_hq_aug = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            NormTransform(opt),
        ]
    )
    train_transform = {'lq_transform': train_lq_aug, 'hq_transform': train_hq_aug}
    ds = FFHQBlind(root=data, split='train', **train_transform)
    lq, hq = ds[3]
    print(lq.size())
    print(hq.size())
    print(lq.min(), lq.max(), lq.mean(), lq.std())
    print(hq.min(), hq.max(), hq.mean(), hq.std())

    import PIL.Image as PImage
    from torchvision.transforms import InterpolationMode, transforms

    img_lq = transforms.ToPILImage()(lq)
    img_hq = transforms.ToPILImage()(hq)
    img_lq.save('../lq.png')
    img_hq.save('../hq.png')
