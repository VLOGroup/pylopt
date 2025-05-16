from dataclasses import dataclass
import torch
from typing import Callable, Optional, Dict, Any

@dataclass
class ParamSpec:
    value: torch.Tensor
    trainable: bool
    projection: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    parameters: Optional[Dict[str, Any]] = None