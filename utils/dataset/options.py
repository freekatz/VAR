
class DataOptions(dict):
    def __init__(self, args=None, opt=None):
        if opt is None:
            if args is not None:
                opt = DataOptions.train_options()
            else:
                opt = DataOptions.val_options()
        super().__init__(opt)

    @classmethod
    def get_splits(cls):
        return ['train', 'val', 'test']

    @classmethod
    def get_options(cls, split='train'):
        if split == 'train':
            return DataOptions.train_options()
        elif split == 'val':
            return DataOptions.val_options()
        elif split == 'test':
            return DataOptions.test_options()
        else:
            raise NotImplementedError

    @classmethod
    def train_options(cls, **kwargs):
        opt = {
            'out_size': 256,
            'mid_size': 288,  # train
            'random_crop_ratio': 0.95,  # train
            'identify_ratio': 0.0005,
            'blur_kernel_size': [41],
            'kernel_list': ['iso', 'aniso'],
            'kernel_prob': [0.5, 0.5],
            'blur_sigma': [0.1, 10],
            'downsample_range': [0.8, 8],
            'noise_range': [0, 20],
            'jpeg_range': [60, 100],
            'use_hflip': True,
            'color_jitter_prob': None,
            'color_jitter_shift': 20,
            'color_jitter_pt_prob': None,
            'gray_prob': 0.008,  # train
            'gt_gray': True,
            'crop_components': False,
            'component_path': '',
            'eye_enlarge_ratio': 1.4,
        }
        return DataOptions(opt=opt)

    @classmethod
    def val_options(cls, **kwargs):
        opt = {
            'out_size': 256,
            'mid_size': 288,
            'random_crop_ratio': 0.,
            'identify_ratio': 0.,
            'blur_kernel_size': [41],
            'kernel_list': ['iso', 'aniso'],
            'kernel_prob': [0.5, 0.5],
            'blur_sigma': [0.1, 10],
            'downsample_range': [0.8, 8],
            'noise_range': [0, 20],
            'jpeg_range': [60, 100],
            'use_hflip': True,
            'color_jitter_prob': None,
            'color_jitter_shift': 20,
            'color_jitter_pt_prob': None,
            'gray_prob': None,
            'gt_gray': True,
            'crop_components': False,
            'component_path': '',
            'eye_enlarge_ratio': 1.4,
        }
        return DataOptions(opt=opt)

    @classmethod
    def test_options(cls, **kwargs):
        opt = {
            'out_size': 256,
            'mid_size': 288,
            'random_crop_ratio': 0.,
            'identify_ratio': 0.,
            'blur_kernel_size': [41],
            'kernel_list': ['iso', 'aniso'],
            'kernel_prob': [0.5, 0.5],
            'blur_sigma': [0.1, 10],
            'downsample_range': [0.8, 8],
            'noise_range': [0, 20],
            'jpeg_range': [60, 100],
            'use_hflip': True,
            'color_jitter_prob': None,
            'color_jitter_shift': 20,
            'color_jitter_pt_prob': None,
            'gray_prob': None,
            'gt_gray': True,
            'crop_components': False,
            'component_path': '',
            'eye_enlarge_ratio': 1.4,
        }
        return DataOptions(opt=opt)