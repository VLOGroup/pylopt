from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
import torch

from bilevel_optimisation.optimiser.StoppingCriterion import StoppingCriterion

@dataclass
class OptimiserSpec:
    optimiser_class: type[torch.optim.Optimizer]
    optimiser_params: Dict[str, Any]
    stopping_class: type[StoppingCriterion]
    stopping_params: Dict[str, Any]
    prox_map_factory: Optional[Callable] = None