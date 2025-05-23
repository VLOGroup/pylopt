from typing import Dict, Any, Mapping
import torch

from bilevel_optimisation.potential.Potential import Potential

class StudentT(Potential):
    """
    Class implementing student-t potential - does not have any trainable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + x ** 2)

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 + x ** 2)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        if '_arch' in state_dict.keys():
            state_dict.pop('_arch')
        super().load_state_dict(state_dict, *args, **kwargs)