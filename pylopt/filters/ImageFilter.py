import os
import torch
from typing import Dict, Any, Optional, Self
from confuse import Configuration
import numpy as np
from scipy.fftpack import idct

ORTHOGONAL_PROJECTION_NAME = 'orthogonal_projection'
ZERO_MEAN_PROJECTION_NAME = 'zero_mean_projection'

def zero_mean_projection(x: torch.Tensor) -> torch.Tensor:
    return x - torch.mean(x, dim=(-2, -1), keepdim=True)

def orthogonal_projection_procrustes(x: torch.Tensor, eps: float=1e-7, max_num_iterations: int=5,
                                     rel_tol: float=1e-5) -> torch.Tensor:
    """
    Function which applies the orthogonal procrustes scheme to orthogonalise a set of filters

    :param x: Filter tensor of shape [batch_size, 1, filter_dim, filter_dim]
    :param eps: Clipping parameter to ensure positivity of diagonal elements
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

    NOTE
    ----
        > Instance of this class can be created by means of constructor, or
            by means of the class methods
                * from_file()
                * from_config()
            The latter requires as parameter a configuration object from the
            Python package confuse.
    """
    def __init__(
            self, 
            filter_dim: int=7, 
            padding: int=3, 
            padding_mode: str='reflect', 
            initialisation_mode: str='rand', 
            multiplier: float=1.0, 
            trainable: bool=True, 
            orthogonality_options: Optional[Dict[str, int]]=None,
            normalise: bool=False
    ) -> None:

        """
        Initialisation of class ImageFilter.

        TODO
        ----
            > Introduce backward-hooks for proper handling of projections. Currently these
                are applied 'manually' in the optimisation routines.

        :param filter_dim: Filter dimension - 7 by default.
        :param padding: Integer giving the padding which is used when applying filters; 3 by default.
        :param padding_mode: String indicating the padding mode; 'reflect' by default.
        :param initialisation_mode: String indicating how filters are initialised: 'rand', 'randn', 'dct'. The latter
            corresponds to the initialisation of the filters by means of the discrete cosine transform. By default
            'rand' is used.
        :param multiplier: Float used as scaling factor in the initialisation process; default value equals 1
        :param trainable: Flag indicating if filter is trainable or not. True by default.
        :param orthogonality_options: Dictionary with keys 'enable', 'max_num_iterations'. If orthonalisation
            is enabled, the procrustes algorithm is used to orthogonalise filters at initialisation stage and
            after each update step. For this purpose a callable attribute titled ORTHOGONAL_PROJECTION_NAME
            is created.
            By default orthogonalisation is not applied. Default number of procrustes iterations equals to 5.
        :param normalise: Flag indicating if filters are normalised (i.e. each filter has norm 1)
            within the initialisation procedure. By default normalisation is not applied.
        """
        super().__init__()

        self.filter_dim = filter_dim
        self.padding = padding
        self.padding_mode = padding_mode
        self.orthogonality_options = orthogonality_options if orthogonality_options is not None else {}

        # initialisation with constructor parameters
        filter_data = self._init_filter_tensor(filter_dim, initialisation_mode, 
                                               normalise, self.orthogonality_options, multiplier)
        self.filter_tensor = torch.nn.Parameter(data=filter_data, requires_grad=trainable)
        self._bind_projections()

    def _bind_projections(self) -> None:
        """
        Binding of callable attributes to self.filter_tensor. Orthogonal procrustes projection is
        not bound by default.
        """
        if not hasattr(self.filter_tensor, ZERO_MEAN_PROJECTION_NAME):
            setattr(self.filter_tensor, ZERO_MEAN_PROJECTION_NAME, zero_mean_projection)

        orthogonalise = self.orthogonality_options.get('enable', False)
        if orthogonalise:
            max_num_procrustes_iteations = self.orthogonality_options.get('max_num_iterations', 5)
            def orthogonal_projection(x: torch.Tensor):
                return orthogonal_projection_procrustes(x, max_num_iterations=max_num_procrustes_iteations)

            setattr(self.filter_tensor, ORTHOGONAL_PROJECTION_NAME, orthogonal_projection)

    @staticmethod
    def _init_filter_tensor(
            filter_dim: int, 
            initialisation_mode: str,
            normalise: bool,
            orthogonality_options: Dict[str, Any],
            multiplier: float
    ) -> torch.Tensor:
        """
        Utiliy method for the initialisation of the parameter filter_tensor.

        :param filter_dim:
        :param initialisation_mode:
        :param normalise:
        :param orthogonality_options:
        :param multiplier:
        :return: PyTorch tensor of shape [num_filters, 1, filter_dim, filter_dim]
        """
        if initialisation_mode == 'dct':
            can_basis = np.reshape(np.eye(filter_dim ** 2, dtype=np.float64),
                                    (filter_dim ** 2, filter_dim, filter_dim))
            dct_basis = idct(idct(can_basis, axis=1, norm='ortho'), axis=2, norm='ortho')
            dct_basis = dct_basis[1:].reshape(-1, 1, filter_dim, filter_dim)
            filter_data = torch.tensor(dct_basis)
        elif initialisation_mode == 'randn':
            filter_data = torch.randn(filter_dim ** 2 - 1, 1, filter_dim, filter_dim)
        elif initialisation_mode == 'rand':
            filter_data = 2 * torch.rand(filter_dim ** 2 - 1, 1, filter_dim, filter_dim) - 1
        else:
            raise ValueError('Unknown initialisation method.')

        orthogonalise = orthogonality_options.get('enable', False)
        if orthogonalise:
            max_num_procrustes_iteations = orthogonality_options.get('max_num_iterations', 5)
            orthogonal_projection_procrustes(filter_data, max_num_iterations=max_num_procrustes_iteations)
        if normalise:
            filter_data.divide_(torch.linalg.norm(filter_data, dim=(-2, -1)).reshape(-1, 1, 1, 1))
        filter_data = zero_mean_projection(filter_data)
        filter_data.mul_(multiplier)

        return filter_data

    @classmethod
    def from_config(cls, config: Configuration) -> Self:
        """
        Class method for initialisation from config.

        :param config: Configuration object from Python package confuse.
        :return: Instance of class ImageFilter
        """
        filter_dim = config['image_filter']['filter_dim'].get()
        padding = config['image_filter']['padding'].get()
        padding_mode = config['image_filter']['padding_mode'].get()

        initialisation_mode = config['image_filter']['initialisation_mode'].get()
        multiplier = config['image_filter']['multiplier'].get()
        normalise = config['image_filter']['normalise'].get()
        trainable = config['image_filter']['trainable'].get()

        orthogonality_options = {
            'enable': config['image_filter']['orthogonality']['enable'].get(),
            'max_num_iterations': config['image_filter']['orthogonality']['max_num_iterations'].get()
        }

        return cls(filter_dim, padding, padding_mode, initialisation_mode, multiplier, trainable,
                   orthogonality_options, normalise)

    @classmethod
    def from_file(cls, path_to_model: str, device: torch.device=torch.device('cpu')) -> Self:
        """
        Class method for initialisation from file.

        :param path_to_model: String representing the path to the model file.
        :param device: Location where loaded model should be placed; by default model is placed on cpu.
        :return: Instance of class ImageFilter
        """
        filter_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = filter_data['initialisation_dict']
        state_dict = filter_data['state_dict']

        filter_dim = initialisation_dict.get('filter_dim', 7)
        padding = initialisation_dict.get('padding', 3)
        padding_mode = initialisation_dict.get('padding_mode', 'reflect')
        orthogonality_options = initialisation_dict.get('orthogonality_options', {})

        image_filter = cls(filter_dim=filter_dim,
                           padding=padding,
                           padding_mode=padding_mode,
                           orthogonality_options=orthogonality_options)
        image_filter.load_state_dict(state_dict, strict=True)
        return image_filter

    def get_filter_tensor(self) -> torch.Tensor:
        return self.filter_tensor.data

    def get_num_filters(self) -> int:
        return self.filter_tensor.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = torch.nn.functional.pad(x, (self.padding, self.padding,
                                               self.padding, self.padding), self.padding_mode)
        return torch.nn.functional.conv2d(x_padded, self.filter_tensor)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        return state

    def initialisation_dict(self)  -> Dict[str, Any]:
        return {'filter_dim': self.filter_dim,
                'padding': self.padding,
                'padding_mode': self.padding_mode,
                'orthogonality_options': self.orthogonality_options}

    def save(self, path_to_model_dir: str, model_name: str='filters') -> str:
        path_to_model = os.path.join(path_to_model_dir, '{:s}.pt'.format(os.path.splitext(model_name)[0]))
        filter_data_dict = {'initialisation_dict': self.initialisation_dict(),
                            'state_dict': self.state_dict()}

        torch.save(filter_data_dict, path_to_model)
        return path_to_model
