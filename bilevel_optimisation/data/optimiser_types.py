import torch
from typing import Any, Optional, Dict
from dataclasses import dataclass

@dataclass
class OptimiserResult:
    solution: Any
    num_iterations: int
    loss: torch.Tensor
    meta: Optional[Dict[str, Any]] = None