import glob
import os

import torch.utils.data as data

from utils.dataset.my_transforms import pil_loader


class UnlabeledImageFolder(data.Dataset):
    def __init__(self, root, split='train', by_name=False, **kwargs):
        super().__init__()
        self.root = root

        # load dataset
        split_file = os.path.join(self.root, f'{split}.txt')

        if by_name:
            with open(split_file, 'r') as file:
                self.samples = [os.path.join(root, line.strip()) for line in file.readlines()]
        else:
            all_samples = glob.glob(os.path.join(self.root, '*.*g'))
            print(f'Folder total samples: {len(all_samples)}')
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
