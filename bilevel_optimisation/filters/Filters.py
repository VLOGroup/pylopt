import torch
from typing import Dict, Any, Mapping, NamedTuple

from bilevel_optimisation.data.ParamSpec import ParamSpec

def zero_mean_projection(x: torch.Tensor) -> torch.Tensor:
    return x - torch.mean(x, dim=(-2, -1), keepdim=True)

class ImageFilter(torch.nn.Module):
    def __init__(self, filter_spec: ParamSpec) -> None:
        super().__init__()

        self.filter_tensor = torch.nn.Parameter(data=filter_spec.value, requires_grad=filter_spec.trainable)
        setattr(self.filter_tensor, 'proj', zero_mean_projection)

        self.filter_dim = self.filter_tensor.shape[-1]             # filters are assumed to be quadratic!
        self.padding_mode = filter_spec.parameters['padding_mode']
        if 'padding' in filter_spec.parameters.keys():
            self.padding = filter_spec.parameters['padding']
        else:
            self.padding = self.filter_dim // 2

    def get_filter_tensor(self) -> torch.Tensor:
        return self.filter_tensor.data

    def get_num_filters(self) -> int:
        return self.filter_tensor.shape[0]

    def forward(self, x: torch.Tensor):
        x_padded = torch.nn.functional.pad(x, (self.padding, ) * 4, self.padding_mode)
        return torch.nn.functional.conv2d(x_padded, self.filter_tensor)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        return state

    def initialisation_dict(self)  -> Dict[str, Any]:
        return {'filter_dim': self.filter_dim, 'padding_mode': self.padding_mode}

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
        result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
        return result
