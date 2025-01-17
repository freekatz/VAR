import math
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast,
                                        adjust_hue, adjust_saturation, normalize)

from utils.img_util import img2tensor
from utils import gaussian_kernels


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


class BlindTransform(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.component_path = opt.get('component_path', None)
        self.latent_gt_path = opt.get('latent_gt_path', None)

        # perform corrupt
        self.use_corrupt = opt.get('use_corrupt', True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_motion_kernel:
            self.motion_kernel_prob = opt.get('motion_kernel_prob', 0.001)
            motion_kernel_path = opt.get('motion_kernel_path', 'basicsr/data/motion-blur-kernels-32.pth')
            self.motion_kernels = torch.load(motion_kernel_path)

        if self.use_corrupt:
            # degradation configurations
            self.blur_kernel_size = opt['blur_kernel_size']
            self.blur_sigma = opt['blur_sigma']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']
            self.downsample_range = opt['downsample_range']
            self.noise_range = opt['noise_range']
            self.jpeg_range = opt['jpeg_range']
            # print
            print(
                f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
            print(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
            print(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            print(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob', None)
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob', None)
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        if self.color_jitter_prob is not None:
            print(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')

        # to gray
        self.gray_prob = opt.get('gray_prob', 0.0)
        if self.gray_prob is not None:
            print(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def forward(self, img):
        # pil to cv2
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        # random horizontal flip
        img_gt = augment(img, hflip=self.opt['use_hflip'], rotation=False, return_status=False)
        h, w, _ = img_gt.shape

        # generate in image
        img_in = img_gt
        if self.use_corrupt:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0, 31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_in = cv2.filter2D(img_in, -1, k)

            # gaussian blur
            kernel = gaussian_kernels.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None)
            img_in = cv2.filter2D(img_in, -1, kernel)

            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_in = cv2.resize(img_in, (int(w // scale), int(w // scale)),
                                interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range is not None:
                noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
                img_in = img_in + noise
                img_in = np.clip(img_in, 0, 255)

            # jpeg
            if self.jpeg_range is not None:
                jpeg_p = np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_p)]
                _, encimg = cv2.imencode('.jpg', img_in, encode_param)
                img_in = np.float32(cv2.imdecode(encimg, 1))

            # resize to in_size
            img_in = cv2.resize(img_in, (h, w), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_in = self.color_jitter(img_in, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            img_in = np.tile(img_in[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in = img2tensor(img_in, bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_in = self.color_jitter_pt(img_in, brightness, contrast, saturation, hue)

        img_in = torch.clip(img_in, 0, 255) / 255.

        return img_in

class NormTransform(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

    def forward(self, img):
        normalize(img, self.mean, self.std, inplace=True)
        return img
