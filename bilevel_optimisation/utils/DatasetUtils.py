from typing import List
import torch
from torchvision.transforms import v2

def collate_function(batch: List[torch.Tensor], crop_size: int = 64) -> torch.Tensor:
    if crop_size > 0:
        return torch.cat([v2.RandomCrop(size=crop_size)(item) for item in batch], dim=0)
    else:
        return torch.cat([item for item in batch], dim=0)