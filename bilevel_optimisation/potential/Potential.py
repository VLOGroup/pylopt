from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Mapping, NamedTuple, Callable, Type, Self
from confuse import Configuration

class Potential(ABC, torch.nn.Module):
    """
    Class which is used as base class for FoE-potentials. By design, there is one
    potential function per filter. Subclassing requires the implementation of the
    method forward_negative_log(...).
    """

    registry = {}

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

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
        result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
        return result

    @classmethod
    def register_subclass(cls, sub_class_id: str) -> Callable:
        def decorator(subclass: Type[cls.__name__]) -> Type[cls.__name__]:
            cls.registry[sub_class_id] = subclass
            return subclass
        return decorator

    @classmethod
    def from_config(cls, num_marginals: int, config: Configuration, **kwargs) -> Self:
        potential_type = list(config['potential'].get().keys())[-1]
        subclass = cls.registry.get(potential_type, None)
        if subclass is None:
            raise
        return subclass(num_marginals=num_marginals, config=config, **kwargs)

    @abstractmethod
    def initialisation_dict(self)  -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parameters(self) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
        pass
