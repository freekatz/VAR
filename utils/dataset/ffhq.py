import glob
import os

import torch.utils.data as data

from utils.dataset.my_transforms import pil_loader


class FFHQ(data.Dataset):
    def __init__(self, root, split='train', **kwargs):
        super().__init__()
        self.root = root

        load_by_name = kwargs.get('load_by_name', False)
        if not load_by_name:
            all_samples = glob.glob(os.path.join(self.root, '*.png'))
            print(f'FFHQ total samples: {len(all_samples)}')

        # load dataset
        split_file = os.path.join(self.root, f'{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [all_samples[int(line.strip())] if not load_by_name else os.path.join(self.root, line.strip()) for line in file.readlines()]
        assert (len(self.samples) > 0)
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # load gt image
        path = self.samples[index]
        img_gt = self.loader(path)
        return img_gt
