import torch
from typing import Any, Mapping, Dict, NamedTuple

from bilevel_optimisation.data.Constants import EPSILON
from bilevel_optimisation.potential.Potential import Potential

class CubicSplinePotential(Potential):

    def __init__(self, num_marginals: int, num_nodes: int, box_lower: float, box_upper: float) -> None:
        super().__init__(num_marginals)

        self._num_nodes = num_nodes
        self._box_lower = box_lower
        self._box_upper = box_upper


    def get_parameters(self):
        pass

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass



# class LinearSplinePotential(Potential):
#
#     def __init__(self, num_nodes: int, box_lower: float, box_upper: float,
#                  param_spec) -> None:
#         num_potentials = param_spec.value.shape[0]
#         super().__init__(num_potentials=num_potentials)
#

#         self._node_dist = (box_upper - box_lower) / (num_nodes - 1)
#         nodes = torch.linspace(box_lower, box_upper, self._num_nodes)
#         self.nodes = torch.nn.Parameter(nodes, requires_grad=False)
#         self.nodal_values = torch.nn.Parameter(param_spec.value, requires_grad=param_spec.trainable)
#         setattr(self.nodal_values, 'proj', non_negative_projection)
#
#     def get_parameters(self) -> torch.Tensor:
#         return self.nodal_values.data
#
#     def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.einsum('bfhw->', -torch.log(self.forward(x)))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         idx = torch.searchsorted(self.nodes, x, right=True) - 1
#         idx = torch.clamp(idx, 0, self._num_nodes - 2)
#
#         x_0 = self.nodes[idx]
#
#         batch_size, _, height, width = x.shape
#         y = self.nodal_values.reshape(1, self.get_num_potentials(), 1, 1,
#                                       self._num_nodes).expand(batch_size, -1, height, width, -1)
#         y_0 = torch.gather(y, 4, idx.unsqueeze(-1)).squeeze()
#         y_1 = torch.gather(y, 4, (idx + 1).unsqueeze(-1)).squeeze()
#         slope = (y_1 - y_0) / self._node_dist
#
#         return torch.clamp(y_0 + slope * (x - x_0), min=EPSILON)
#
#     def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
#         state = super().state_dict(*args, kwargs)
#         return state
#
#     def initialisation_dict(self) -> Dict[str, Any]:
#         return {'type': self.__class__.__name__, 'num_potentials': self.num_potentials,
#                 'num_nodes': self._num_nodes, 'box_lower': self._box_lower, 'box_upper': self._box_upper}
#
#     def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
#         result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
#         return result

