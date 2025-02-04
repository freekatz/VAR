import math
import random

import cv2
import numpy as np
import PIL.Image as PImage
import torch


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def cv2_loader(path):
    img: np.ndarray = cv2.imread(path, cv2.COLOR_RGB2BGR)
    return img


def random_add_gaussian_noise(img, sigma_range=(0, 1.0)):
    noise_sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = np.float32(np.random.randn(*(img.shape))) * noise_sigma / 255.
    img = img + noise
    return np.clip(img, 0, 255)


def random_add_jpg_compression(img, quality_range=(90, 100)):
    quality = np.random.randint(quality_range[0], quality_range[1])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return np.float32(cv2.imdecode(encimg, 1))


def print_transforms(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class NormTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return normalize_01_into_pm1(img)


def denormalize_pm1_into_01(x):  # denormalize x from [-1, 1] to [0, 1] by (x + 1)/2
    return x.add(1)/2


class DenormTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return denormalize_pm1_into_01(img)


from torchvision.ops import roi_align
import torch

def get_roi_regions(gt, output, loc_left_eyes, loc_right_eyes, loc_mouths,
                    face_ratio=1, eye_out_size=80, mouth_out_size=120):
    # hard code
    eye_out_size *= face_ratio
    mouth_out_size *= face_ratio

    eye_out_size = int(eye_out_size)
    mouth_out_size = int(mouth_out_size)

    rois_eyes = []
    rois_mouths = []
    for b in range(loc_left_eyes.size(0)):  # loop for batch size
        # left eye and right eye
        img_inds = loc_left_eyes.new_full((2, 1), b)
        bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
        rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
        rois_eyes.append(rois)
        # mouse
        img_inds = loc_left_eyes.new_full((1, 1), b)
        rois = torch.cat([img_inds, loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        rois_mouths.append(rois)

    rois_eyes = torch.cat(rois_eyes, 0)
    rois_mouths = torch.cat(rois_mouths, 0)

    # real images
    all_eyes = roi_align(gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes_gt = all_eyes[0::2, :, :, :]
    right_eyes_gt = all_eyes[1::2, :, :, :]
    mouths_gt = roi_align(gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
    # output
    all_eyes = roi_align(output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes = all_eyes[0::2, :, :, :]
    right_eyes = all_eyes[1::2, :, :, :]
    mouths = roi_align(output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

    return {'left_eyes_gt': left_eyes_gt, 'right_eyes_gt': right_eyes_gt, 'mouths_gt': mouths_gt,
            'left_eyes': left_eyes, 'right_eyes': right_eyes, 'mouths': mouths}

