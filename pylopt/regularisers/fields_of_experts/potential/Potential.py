from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Self
from confuse import Configuration

class Potential(ABC, torch.nn.Module):
    """
    Class which is used as base class for FoE-potentials. By design, there is one
    potential function per filter.
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

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        return state

    @abstractmethod
    def freeze(self) -> None:
        pass

    @abstractmethod
    def unfreeze(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, path_to_model: str, device: torch.device=torch.device('cpu')):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Configuration) -> Self:
        pass

    @abstractmethod
    def initialisation_dict(self)  -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parameters(self) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        pass
