import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LowerProblemResult:
    solution: torch.Tensor
    num_iterations: int
    loss: torch.Tensor
    message: str=''
    meta: Optional[Dict[str, Any]]=None
