from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Mapping, NamedTuple


class Potential(ABC, torch.nn.Module):
    """
    Class which is used as base class for FoE-potentials. By design, there is one potential function
    per filter.
    Subclassing requires the implementation of the method forward_negative_log(...).
    """
    def __init__(self, num_marginals: int):
        """
        Initialisation of an object of class Potential.

        :param num_marginals: Number of marginal potentials required for the FoE-model. By design
            there is one marginal potential per filter.
        """
        super().__init__()
        self.num_marginals = num_marginals

    def get_num_marginals(self) -> int:
        return self.num_marginals

    @abstractmethod
    def get_parameters(self):
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
    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
        pass