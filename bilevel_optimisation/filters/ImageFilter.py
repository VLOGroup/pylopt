import os.path

import torch
from typing import Dict, Any, Mapping, NamedTuple
from confuse import Configuration
import numpy as np
from scipy.fftpack import idct

def zero_mean_projection(x: torch.Tensor) -> torch.Tensor:
    return x - torch.mean(x, dim=(-2, -1), keepdim=True)

def orthogonal_projection_procrustes(x: torch.Tensor, eps: float=1e-7, max_num_iterations: int=5,
                                     rel_tol: float=1e-5) -> torch.Tensor:
    """
    Function which applies the orthogonal procrustes scheme to orthogonalise a set/batch of filters

    :param x: Filter tensor of shape [batch_size, 1, filter_dim, filter_dim]
    :param eps:
    :param max_num_iterations: Maximal number of iterations
    :param rel_tol: Tolerance used to stop iteration as soon the norm of subsequent iterates is less than rel_tol.
    :return: Orthogonalised set of filters in terms of a tensor of the same shape as the input tensor.
    """
    x_flattened = [x[i, 0, :, :].flatten() for i in range(0, x.shape[0])]
    x_stacked = torch.stack(x_flattened, dim=1)
    m, n = x_stacked.shape
    diag = torch.diag(torch.ones(n, dtype=x.dtype, device=x.device))

    v_old = torch.zeros_like(x_stacked)
    for k in range(0, max_num_iterations):
        z = torch.matmul(x_stacked, diag)
        q, s, r_h = torch.linalg.svd(z, full_matrices=False)
        v = torch.matmul(q, r_h)

        tmp = torch.matmul(v.transpose(dim0=0, dim1=1), x_stacked)
        diag_elements = torch.diag(tmp)
        diag_elements = torch.clamp(diag_elements, min=eps)
        diag = torch.diag(diag_elements)
        if torch.linalg.norm(v - v_old) < rel_tol:
            break
        v_old = v.clone()

    x_orthogonal = [torch.unflatten(torch.matmul(v, diag)[:, j], dim=0,
                                    sizes=x.shape[-2:]) for j in range(0, v.shape[1])]
    return torch.stack(x_orthogonal, dim=0).unsqueeze(dim=1)

class ImageFilter(torch.nn.Module):
    """
    Class modelling quadratic image filters by means of a PyTorch module.

    """
    def __init__(self, config: Configuration) -> None:
        """
        Initialisation of class ImageFilter.

        :param config: Configuration object. See bilevel_optimisation/config_data for sample configuration files.
        """
        super().__init__()

        self.filter_dim = config['image_filter']['filter_dim'].get()
        self.padding = self.filter_dim // 2
        self.padding_mode = config['image_filter']['padding_mode'].get()

        initialisation_mode = config['image_filter']['initialisation']['mode'].get()
        multiplier = config['image_filter']['initialisation']['multiplier'].get()
        normalise = config['image_filter']['initialisation']['normalise'].get()
        trainable = config['image_filter']['trainable'].get()
        enforce_orthogonality = config['image_filter']['enforce_orthogonality'].get()

        model_path = config['image_filter']['initialisation']['file_path'].get()
        if not model_path:
            if initialisation_mode == 'dct':
                can_basis = np.reshape(np.eye(self.filter_dim ** 2, dtype=np.float64),
                                       (self.filter_dim ** 2, self.filter_dim, self.filter_dim))
                dct_basis = idct(idct(can_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
                dct_basis = dct_basis[1:].reshape(-1, 1, self.filter_dim, self.filter_dim)
                filter_data = torch.tensor(dct_basis)
            elif initialisation_mode == 'randn':
                filter_data = torch.randn(self.filter_dim ** 2 - 1, 1, self.filter_dim, self.filter_dim)
            elif initialisation_mode == 'rand':
                filter_data = 2 * torch.rand(self.filter_dim ** 2 - 1, 1, self.filter_dim, self.filter_dim) - 1
            else:
                raise ValueError('Unknown initialisation method.')
            self.filter_tensor = torch.nn.Parameter(data=filter_data, requires_grad=trainable)
        else:
            dummy_data = torch.ones(self.filter_dim ** 2 - 1, 1, self.filter_dim, self.filter_dim)
            self.filter_tensor = torch.nn.Parameter(data=dummy_data, requires_grad=trainable)
            self._load_from_file(model_path)

        with torch.no_grad():
            if normalise:
                self.filter_tensor.divide_(torch.linalg.norm(self.filter_tensor, dim=(-2, -1)).reshape(-1, 1, 1, 1))
            self.filter_tensor.mul_(multiplier)
            self.filter_tensor.copy_(zero_mean_projection(self.filter_tensor))

        # define projections
        if not hasattr(self.filter_tensor, 'zero_mean_projection'):
            setattr(self.filter_tensor, 'zero_mean_projection', zero_mean_projection)
        if enforce_orthogonality:
            max_num_iterations = 5
            def orthogonal_projection(x: torch.Tensor):
                return orthogonal_projection_procrustes(x, max_num_iterations=max_num_iterations)
            setattr(self.filter_tensor, 'orthogonal_projection', orthogonal_projection)

    def get_filter_tensor(self) -> torch.Tensor:
        return self.filter_tensor.data

    def get_num_filters(self) -> int:
        return self.filter_tensor.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def save(self, path_to_model_dir: str, model_name: str='filters') -> str:
        path_to_model = os.path.join(path_to_model_dir, '{:s}.pt'.format(os.path.splitext(model_name)[0]))
        potential_dict = {'initialisation_dict': self.initialisation_dict(),
                          'state_dict': self.state_dict()}

        torch.save(potential_dict, path_to_model)
        return path_to_model
