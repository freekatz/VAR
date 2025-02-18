import glob
import os

import torch.utils.data as data

from utils.dataset.my_transforms import pil_loader


class UnlabeledImageItem(data.Dataset):
    def __init__(self, root, **kwargs):
        super().__init__()
        # load dataset

        self.samples = [root]
        assert (len(self.samples) > 0)
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # load gt image
        path = self.samples[index]
        img_gt = self.loader(path)
        return img_gt
