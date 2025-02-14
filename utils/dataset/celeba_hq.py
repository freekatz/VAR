import glob
import os
import random

import torch.utils.data as data

from utils.dataset.my_transforms import pil_loader


class CelebAHQ(data.Dataset):
    def __init__(self, root, split='train', **kwargs):
        super().__init__()
        self.root = root
        splits = ['train', 'val']
        classes = ['male', 'female']
        all_samples = []
        for s in splits:
            for c in classes:
                sub_samples = glob.glob(os.path.join(self.root, s, c, '*.jpg'))
                all_samples.extend(sub_samples)
        random.shuffle(all_samples)
        print(f'CelebA HQ total samples: {len(all_samples)}')

        # load dataset
        split_file = os.path.join(self.root, f'{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [all_samples[int(line.strip())] for line in file.readlines()]
        assert (len(self.samples) > 0)
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # load gt image
        path = self.samples[index]
        img_gt = self.loader(path)
        return img_gt
