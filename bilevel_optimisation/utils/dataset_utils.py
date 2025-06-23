from typing import List
import torch
from torchvision.transforms import v2
import logging

def collate_function(batch: List[torch.Tensor], crop_size: int = 64) -> torch.Tensor:
    if crop_size > 0:
        return torch.cat([v2.RandomCrop(size=crop_size)(item) for item in batch], dim=0)
    else:
        min_width = min([u.shape[-1] for u in batch])
        max_width = max([u.shape[-1] for u in batch])
        min_height = min([u.shape[-2] for u in batch])
        max_height = max([u.shape[-2] for u in batch])

        if (min_width < max_width) or (min_height < max_height):
            logging.info('[COLLATE] Batch contains images of different shapes: apply padding.')
            batch_padded = []
            for i in range(0, len(batch)):
                u = batch[i]
                diff_horizontal = max_height - u.shape[-1]
                diff_vertical = max_width - u.shape[-2]

                u_padded = torch.nn.functional.pad(batch[i], (0, diff_horizontal, 0, diff_vertical),
                                                   mode='constant', value=1)
                batch_padded.append(u_padded)
            ret_val = torch.cat([item for item in batch_padded], dim=0)
        else:
            ret_val = torch.cat([item for item in batch], dim=0)

        return ret_val