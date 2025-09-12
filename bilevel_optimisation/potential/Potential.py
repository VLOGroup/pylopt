from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Mapping, NamedTuple, Callable, Type, Self, Optional
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
            raise ValueError('Unable to load potential – unknown class type.')
        return subclass(num_marginals=num_marginals, config=config, **kwargs)

    @abstractmethod
    def initialisation_dict(self)  -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_parameters(self) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_negative_log(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        pass

            # @abstractmethod
            # def first_derivative(self, x: torch.Tensor, grad_outputs: Optional[torch.Tensor]=None) -> torch.Tensor:
            #     """
            #     This function is intended to implement analytical derivatives whenever possible and suitable,
            #     as is the case with splines.
            #
            #     :param x: PyTorch tensor representing the point at which derivative shall be computed.
            #     :param grad_outputs: PyTorch tensor representing the vector the derivative is applied to.
            #     :return: PyTorch tensor corresponding to the derivative of the potential function at x, applied
            #         to grad_outputs.
            #     """
            #     pass
            #
            # @abstractmethod
            # def second_derivative(self, x: torch.Tensor, grad_outputs: Optional[torch.Tensor]=None,
            #                       mixed: bool=True) -> torch.Tensor:
            #     """
            #     As for the function first_derivative().
            #
            #     :param x: PyTorch tensor corresponding to the point at which derivative shall be computed.
            #     :param grad_outputs: PyTorch tensor of the vector at which the derivative shall be appliled.
            #     :param mixed: Flag indicating if mixed derivative needs to be computed. By default, the derivative
            #         is computed w.r.t. the state variable x.
            #     :return: Derivative of potential function at x, applied to grad_outputs in terms of a PyTorch tensor.
            #     """
            #     pass


    @abstractmethod
    def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
        pass
