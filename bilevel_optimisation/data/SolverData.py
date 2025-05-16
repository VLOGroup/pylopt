import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class SolverStats:
    num_iterations: int
    iterate_list: List[torch.Tensor]

@dataclass
class LinearSolverStats(SolverStats):
    residual_norm_list: List[float]

@dataclass
class SolverResult:
    solution: Optional[torch.Tensor]
    info: Dict[str, Any]
    stats: SolverStats