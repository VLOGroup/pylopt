import random
import numpy as np
import torch

def seed_random_number_generators(seed_val: int) -> None:
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
