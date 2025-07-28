import torch
from typing import Any, Mapping, Dict, NamedTuple
from confuse import Configuration
import os

from bilevel_optimisation.data.Constants import EPSILON
from bilevel_optimisation.potential.Potential import Potential

def sub_function_0(x: torch.Tensor) -> torch.Tensor:
    z = x + 0.5
    return (11 + z * (12 + z * (-6 + z * (-12 + 6 * z)))) / 24

def sub_function_1(x: torch.Tensor) -> torch.Tensor:
    z = 1.5 - x
    return (1 + z * (4 + z * (6 + z * (4 - 4 * z)))) / 24

def sub_function_2(x: torch.Tensor) -> torch.Tensor:
    z = (2.5 - x) * (2.5 - x)
    return (z * z) / 24

def sub_function_3(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)

def base_spline_func(x: torch.Tensor) -> torch.Tensor:
    x_abs = torch.abs(x)
    ret_val = sub_function_0(x_abs)
    sub_functions = [sub_function_1, sub_function_2, sub_function_3]
    thresholds = [0.5, 1.5, 2.5]

    for th, func in zip(thresholds, sub_functions):
        ret_val = torch.where(x_abs >= th, func(x_abs), ret_val)

    return ret_val

class QuarticSpline(Potential):

    def __init__(self, num_marginals: int, config: Configuration) -> None:
        super().__init__(num_marginals)

        initialisation_mode = config['potential']['spline']['initialisation']['mode'].get()
        multiplier = config['potential']['spline']['initialisation']['multiplier'].get()
        trainable = config['potential']['spline']['trainable'].get()

        model_path = config['potential']['spline']['initialisation']['file_path'].get()
        if not model_path:
            self.num_nodes = config['potential']['spline']['num_nodes'].get()
            self.box_lower = config['potential']['spline']['box_lower'].get()
            self.box_upper = config['potential']['spline']['box_upper'].get()

            if initialisation_mode == 'rand':
                weights = torch.log(multiplier * torch.rand(num_marginals, self.num_nodes))
            elif initialisation_mode == 'uniform':
                weights = torch.log(multiplier * torch.ones(num_marginals, self.num_nodes))
            else:
                raise ValueError('Unknown initialisation method')
            self.weight_tensor = torch.nn.Parameter(data=weights, requires_grad=trainable)
        else:
            dummy_data = torch.ones(num_marginals, 1)
            self.weight_tensor = torch.nn.Parameter(data=dummy_data, requires_grad=trainable)
            self._load_from_file(model_path)

            with torch.no_grad():
                self.weight_tensor.add_(torch.log(torch.tensor(multiplier)))

        self.nodes = torch.nn.Parameter(torch.linspace(self.box_lower, self.box_upper,
                                                       self.num_nodes).reshape(1, 1, -1, 1, 1), requires_grad=False)
        self.scale = (self.box_upper - self.box_lower) / (self.num_nodes - 1)

    def get_parameters(self) -> torch.Tensor:
        return self.weight_tensor.data

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:



        x_centered = x.unsqueeze(dim=2) - self.nodes
        x_scaled = x_centered / self.scale

        import torch.profiler
        with torch.profiler.profile() as prof:
            y = base_spline_func(x_scaled)
        print(prof.key_averages().table())

        y = base_spline_func(x_scaled)
        y = torch.einsum('bfnhw, fn->bfhw', y, torch.exp(self.weight_tensor))
        y = torch.log(y + EPSILON)
        return torch.sum(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, kwargs)
        return state

    def initialisation_dict(self) -> Dict[str, Any]:
        return {'num_marginals': self.num_marginals, 'num_nodes': self.num_nodes, 'bow_lower': self.box_lower,
                'box_upper': self.box_upper}

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
        result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
        return result

    def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
        potential_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = potential_data['initialisation_dict']
        self.num_marginals = initialisation_dict['num_marginals']
        self.num_nodes = initialisation_dict['num_nodes']
        self.box_lower = initialisation_dict['box_lower']
        self.box_upper = initialisation_dict['box_upper']

        state_dict = potential_data['state_dict']
        self.load_state_dict(state_dict)

    def save(self, path_to_model_dir: str, model_name: str='spline') -> str:
        path_to_model = os.path.join(path_to_model_dir, '{:s}.pt'.format(os.path.splitext(model_name)[0]))
        potential_dict = {'initialisation_dict': self.initialisation_dict(),
                          'state_dict': self.state_dict()}

        torch.save(potential_dict, path_to_model)
        return path_to_model


