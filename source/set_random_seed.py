import torch
import numpy as np
import random


# control reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
# TODO : does not seem to work???????!
# https://github.com/pytorch/pytorch/issues/7068

def set_random_seed(custom_seed, benchmark=False, deterministic=False):
    np.random.seed(custom_seed)
    torch.manual_seed(custom_seed)
    torch.cuda.manual_seed(custom_seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(custom_seed)  # Python random module.
    # CuDNN
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
