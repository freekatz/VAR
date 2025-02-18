import math
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import InterpolationMode, transforms
import torchvision.transforms.v2.functional as vision_fn
from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor

from utils.dataset.my_transforms import random_add_jpg_compression, normalize_01_into_pm1, \
    random_add_gaussian_noise, denormalize_pm1_into_01
from utils.dataset.options import DataOptions


class BlindDataset(data.Dataset):
    def __init__(self, base_dataset: data.Dataset, opt: DataOptions, **kwargs):
        super().__init__()
        self.base = base_dataset
        self.opt = opt

        self.mid_size = opt['mid_size']
        self.out_size = opt['out_size']
        self.random_crop_ratio = opt['random_crop_ratio']
        self.identify_ratio = opt['identify_ratio']
        self.random_crop = transforms.RandomCrop((self.out_size, self.out_size))

        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)

        if self.crop_components:
            self.components_list = torch.load(opt.get('component_path'))

        # degradations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = [round(q) for q in opt['jpeg_range']]

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')

        print(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'sigma: [{", ".join(map(str, self.blur_sigma))}]')
        print(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        print(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        print(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            print(f'Use random color jitter. Prob: {self.color_jitter_prob}, '
                        f'shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            print(f'Use random gray. Prob: {self.gray_prob}')

        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = vision_fn.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = vision_fn.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = vision_fn.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = vision_fn.adjust_hue(img, hue_factor)
        return img

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __len__(self):
        return len(self.base)

    def generate_lq(self, img_gt):
        h, w, _ = img_gt.shape
        # blur
        cur_kernel_size = random.choice(self.blur_kernel_size)
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = img_lq.astype(np.float32)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        # BGR to RGB, HWC to CHW, numpy to tensor
        (img_gt, img_lq) = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)
        return img_gt, img_lq

    def __getitem__(self, index):
        # load gt image
        img_gt = self.base[index]

        # resize and random crop
        if self.mid_size > self.out_size and random.random() < self.random_crop_ratio:
            img_gt = vision_fn.resize(img_gt, self.mid_size, interpolation=InterpolationMode.LANCZOS)
            img_gt = self.random_crop(img_gt)
        else:
            img_gt = vision_fn.resize(img_gt, [self.out_size, self.out_size], interpolation=InterpolationMode.LANCZOS)

        # random horizontal flip
        img_gt = cv2.cvtColor(np.asarray(img_gt), cv2.COLOR_RGB2BGR)
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)

        if self.crop_components:
            locations = self.get_component_coordinates(index, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations
        # generate lq
        if self.identify_ratio > 0 and random.random() > (1 - self.identify_ratio):
            img_lq, img_gt = img2tensor([img_gt, img_gt], bgr2rgb=True, float32=True)
        else:
            img_gt, img_lq = self.generate_lq(img_gt)
        # round and clip
        img_lq = torch.clip(img_lq, 0, 255) / 255.
        img_gt = torch.clip(img_gt, 0, 255) / 255.
        # normalize
        img_lq = normalize_01_into_pm1(img_lq)
        img_gt = normalize_01_into_pm1(img_gt)

        return_dict = {
                'lq': img_lq,
                'gt': img_gt,
            }
        if self.crop_components:
            return_dict['loc_left_eye'] = loc_left_eye
            return_dict['loc_right_eye'] = loc_right_eye
            return_dict['loc_mouth'] = loc_mouth

        return return_dict


if __name__ == '__main__':
    import math
    import torch
    import cv2

    from utils.dataset.ffhq import FFHQ
    from utils.dataset.celeba_hq import CelebAHQ

    data = '../../tmp/dataset1'
    # validate
    from pprint import pprint
    opt = DataOptions.val_options()
    pprint(opt)

    base_ds = FFHQ(root=data, split='train', load_by_name=True)
    ds = BlindDataset(base_dataset=base_ds, opt=opt)
    res = ds[-1]
    lq, hq = res['lq'], res['gt']
    print(lq.size())
    print(hq.size())
    print(lq.min(), lq.max(), lq.mean(), lq.std())
    print(hq.min(), hq.max(), hq.mean(), hq.std())

    import torchvision.transforms.functional as vision_fn
    from torchvision import transforms

    lq = denormalize_pm1_into_01(lq)
    hq = denormalize_pm1_into_01(hq)
    img_lq = vision_fn.to_pil_image(lq)
    img_hq = vision_fn.to_pil_image(hq)
    img_lq.save('../../lq.png')
    img_hq.save('../../hq.png')
