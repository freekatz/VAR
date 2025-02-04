import os

from basicsr import img2tensor
from basicsr.data.transforms import augment
import cv2
import numpy as np
from torch.utils import data as data

from utils.dataset.my_transforms import normalize_01_into_pm1, cv2_loader


class FFHQ(data.Dataset):
    def __init__(self, root, opt, split='train', **kwargs):
        super().__init__()
        self.root = root
        self.opt = opt

        # load dataset
        split_file = os.path.join(self.root, f'ffhq_{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [os.path.join(self.root, line.strip()) for line in file.readlines() if
                            line.find('.png') != -1]
        assert (len(self.samples) > 0)
        self.loader = cv2_loader

        # read options
        self.gray_prob = opt.get('gray_prob', 0.)

        self.exposure_prob = opt.get('exposure_prob', 0.)
        self.exposure_range = opt['exposure_range']

        self.shift_prob = opt.get('shift_prob', 0.)
        self.shift_unit = opt.get('shift_unit', 32)
        self.shift_max_num = opt.get('shift_max_num', 3)

        if self.gray_prob is not None:
            print(f'Use random gray. Prob: {self.gray_prob}')

        if self.exposure_prob is not None:
            print(f'Use random exposure. Prob: {self.exposure_prob}')
            print(f'Use random exposure. Range: [{self.exposure_range[0]}, {self.exposure_range[1]}]')

        if self.shift_prob is not None:
            print(f'Use random shift. Prob: {self.shift_prob}')
            print(f'Use random shift. uint: {self.shift_unit}')
            print(f'Use random shift. max_num: {self.shift_max_num}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # load gt image
        path = self.samples[index]
        img_gt = self.loader(path)

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        h, w, _ = img_gt.shape

        if (self.exposure_prob is not None) and (np.random.uniform() < self.exposure_prob):
                exp_scale = np.random.uniform(self.exposure_range[0], self.exposure_range[1])
                img_gt *= exp_scale

        if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob ):
            shift_vertical_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_horisontal_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_v = self.shift_unit * shift_vertical_num
            shift_h = self.shift_unit * shift_horisontal_num
            img_gt_pad = np.pad(img_gt, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (0,0)), 
                                mode='symmetric')
            img_gt = img_gt_pad[shift_v:shift_v + h, shift_h: shift_h + w,:]

        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        img_gt = normalize_01_into_pm1(img_gt)
        return {'gt', img_gt}
