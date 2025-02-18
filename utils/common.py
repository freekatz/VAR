import os
import random

import numpy as np
import torch

from utils import dist_utils


def seed_everything(seed, benchmark=False):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = benchmark
    if seed is None:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True
        seed = seed * dist_utils.get_world_size() + dist_utils.get_rank()
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return seed