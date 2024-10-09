import numpy as np
import torch
import random


def use_fix_random_seed():
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

