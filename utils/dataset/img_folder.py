import os

from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

from utils.dataset.my_transforms import pil_loader


class UnlabeledDatasetFolder(DatasetFolder):
    def __init__(self, root, transform=None, split='train', **kwargs):
        self.root = os.path.join(root, split)
        super().__init__(root=self.root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample