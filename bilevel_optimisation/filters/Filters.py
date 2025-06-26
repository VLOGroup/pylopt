import torch
from typing import Dict, Any, Mapping, NamedTuple
from confuse import Configuration
import numpy as np
from scipy.fftpack import idct

def zero_mean_projection(x: torch.Tensor) -> torch.Tensor:
    return x - torch.mean(x, dim=(-2, -1), keepdim=True)

class ImageFilter(torch.nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()

        self.filter_dim = config['image_filter']['filter_dim'].get()
        self.padding = self.filter_dim // 2
        self.padding_mode = config['image_filter']['padding_mode'].get()

        initialisation_mode = config['image_filter']['initialisation'].get()
        multiplier = config['image_filter']['multiplier'].get()
        trainable = config['image_filter']['trainable'].get()

        model_path = config['image_filter']['file_path'].get()
        if not model_path:
            if initialisation_mode == 'dct':
                can_basis = np.reshape(np.eye(self.filter_dim ** 2, dtype=np.float64),
                                       (self.filter_dim ** 2, self.filter_dim, self.filter_dim))
                dct_basis = idct(idct(can_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
                dct_basis = dct_basis[1:].reshape(-1, 1, self.filter_dim, self.filter_dim)
                filter_data = torch.tensor(dct_basis)
            elif initialisation_mode == 'randn':
                filter_data = torch.randn(self.filter_dim ** 2 - 1, 1, self.filter_dim, self.filter_dim)
            else:
                filter_data = 2 * torch.rand(self.filter_dim ** 2 - 1, 1, self.filter_dim, self.filter_dim) - 1
            self.filter_tensor = torch.nn.Parameter(data=filter_data, requires_grad=trainable)
        else:
            dummy_data = torch.ones(self.filter_dim ** 2 - 1, 1, self.filter_dim, self.filter_dim)
            self.filter_tensor = torch.nn.Parameter(data=dummy_data, requires_grad=trainable)
            self._load_from_file(model_path)
        with torch.no_grad():
            self.filter_tensor.mul_(multiplier)
        if not hasattr(self.filter_tensor, 'proj'):
            setattr(self.filter_tensor, 'proj', zero_mean_projection)

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

    def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
        filter_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = filter_data['initialisation_dict']
        filter_dim = initialisation_dict['filter_dim']
        padding_mode = initialisation_dict['padding_mode']
        self.filter_dim = filter_dim
        self.padding_mode = padding_mode

        state_dict = filter_data['state_dict']
        self.load_state_dict(state_dict)


    def save(self, path_to_model_dir: str, model_name: str):
        pass