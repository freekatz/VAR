
class DataOptions(dict):
    def __init__(self, args=None, opt=None):
        if opt is None:
            if args is not None:
                opt = DataOptions.train_options()
            else:
                opt = DataOptions.val_options()
        super().__init__(opt)

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
            'exposure_prob': None,
            'exposure_range': [0.7, 1.1],
            'shift_prob': 0.2,  # train
            'shift_unit': 1,
            'shift_max_num': 32,
            'uneven_prob': 0.1,  # train
            'hazy_prob': 0.008,  # train
            'hazy_alpha': [0.75, 0.95],
            'crop_components': False,
            'component_path': '',
            'eye_enlarge_ratio': 1.4,
        }
        return DataOptions(opt=opt)

    @classmethod
    def val_options(cls, **kwargs):
        opt = {
            'out_size': 256,
            'mid_size': -1,
            'random_crop_ratio': 1.,
            'identify_ratio': 0.,
            'blur_kernel_size': [13, 19, 23, 29, 31, 37, 41],
            'kernel_list': ['iso', 'aniso'],
            'kernel_prob': [0.5, 0.5],
            'blur_sigma': [0.1, 15],
            'downsample_range': [0.8, 30],
            'noise_range': [0, 20],
            'jpeg_range': [30, 100],
            'use_hflip': True,
            'color_jitter_prob': None,
            'color_jitter_shift': 20,
            'color_jitter_pt_prob': None,
            'gray_prob': None,
            'gt_gray': True,
            'exposure_prob': None,
            'exposure_range': [0.7, 1.1],
            'shift_prob': None,
            'shift_unit': 1,
            'shift_max_num': 32,
            'uneven_prob': None,
            'hazy_prob': None,
            'hazy_alpha': [0.75, 0.95],
            'crop_components': False,
            'component_path': '',
            'eye_enlarge_ratio': 1.4,
        }
        return DataOptions(opt=opt)

    @classmethod
    def test_options(cls, **kwargs):
        opt = {
            'out_size': 256,
            'mid_size': -1,
            'random_crop_ratio': 1.,
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
            'exposure_prob': None,
            'exposure_range': [0.7, 1.1],
            'shift_prob': None,
            'shift_unit': 1,
            'shift_max_num': 32,
            'uneven_prob': None,
            'hazy_prob': None,
            'hazy_alpha': [0.75, 0.95],
            'crop_components': False,
            'component_path': '',
            'eye_enlarge_ratio': 1.4,
        }
        return DataOptions(opt=opt)