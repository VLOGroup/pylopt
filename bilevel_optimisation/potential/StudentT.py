from typing import Dict, Any, Mapping, NamedTuple
import torch

from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.potential.Potential import Potential
from bilevel_optimisation.projection.ParameterProjections import non_negative_projection


class StudentT(Potential):
    """
    Class implementing student-t potential for the usage in context of FoE models.
    """

    def __init__(self, num_potentials: int, weights_spec: ParamSpec):
        super().__init__(num_potentials=num_potentials)
        self.weights = torch.nn.Parameter(weights_spec.value, requires_grad=weights_spec.trainable)
        setattr(self.weights, 'proj', non_negative_projection)

    def get_parameters(self) -> torch.Tensor:
        return self.weights.data

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bfhw,f->bfhw', torch.log(1.0 + x ** 2), self.weights)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        return state

    def initialisation_dict(self) -> Dict[str, Any]:
        return {'type': self.__class__.__name__, 'num_potentials': self.num_potentials}

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
        result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
        return result