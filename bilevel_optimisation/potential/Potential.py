from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Mapping


class Potential(ABC, torch.nn.Module):
    """
    Class which is used as base class for potentials. Subclassing requires the implementation of
    the methods forward(...) and forward_negative_log(...).
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        return state

    @abstractmethod
    def initialisation_dict(self)  -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        pass