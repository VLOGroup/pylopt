import torch
from typing import Any, Dict, Self, Optional
from confuse import Configuration

from . import Potential
from quartic_bspline_extension.functions import QuarticBSplineFunction

class QuarticBSpline(Potential):

    def __init__(self, 
                 num_marginals: int,
                 box_lower: float=-1.0,
                 box_upper: float=1.0,
                 num_centers: int=87,
                 init_options: Optional[Dict[str, Any]]=None,
                 trainable: bool=True) -> None:
        """
        Constructor of class QuarticBSpline integrating the C++/CUDA extension
        'quartic_bspline_extension'.

        :param num_marginals: Number of potentials - must coincide with number of filters
        :param box_lower: Lower bound of spline domain, defaults to -1.0
        :param box_upper: Upper bound of spline domain, defaults to 1.0
        :param num_centers: _description_, defaults to 87
        :param init_options: _description_, defaults to None
        :param trainable: _description_, defaults to True
        """
        super().__init__(num_marginals)

        default_init = {
            'mode': 'student_t', 
            'multiplier': 1.0, 
        }
        self.init_options = {**default_init, **(init_options or {})}

        self.box_lower = box_lower
        self.box_upper = box_upper
        self.num_centers = num_centers
        self.scale = (self.box_upper - self.box_lower) / (self.num_centers - 1)
        self.register_buffer('centers', torch.linspace(self.box_lower, self.box_upper, self.num_centers))

        weight_data = self._init_weight_tensor(num_marginals, self.centers, self.init_options)
        self.weight_tensor = torch.nn.Parameter(data=weight_data, requires_grad=trainable)

    @staticmethod
    def _init_weight_tensor(num_marginals: int, centers: torch.Tensor, init_options: Dict[str, Any]) -> torch.Tensor:
        multiplier = init_options['multiplier']
        if init_options['mode'] == 'rand':
            weight_data = torch.log(multiplier * torch.rand(num_marginals, len(centers)))
        elif init_options['mode'] == 'uniform':
            weight_data = torch.log(multiplier * torch.ones(num_marginals, len(centers)))
        elif init_options['mode'] == 'student_t':
            weight_data = torch.log(multiplier * torch.log(1 + torch.stack([centers 
                                                                for _ in range(0, num_marginals)], dim=0) ** 2))
        else:
            raise ValueError('Unknown initialisation method')
        return weight_data
        
    def initialisation_dict(self) -> Dict[str, Any]:
        return {'num_marginals': self.num_marginals, 
                'box_lower': self.box_lower,
                'box_upper': self.box_upper,
                'num_centers': self.num_centers}

    def get_parameters(self) -> torch.Tensor:
        return self.weight_tensor.data

    def freeze(self) -> None:
        self.weight_tensor.requires_grad = False
    
    def unfreeze(self) -> None:
        self.weight_tensor.requires_grad = True

    def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        y, _ = QuarticBSplineFunction.apply(2 * torch.arctan(x) / torch.pi, torch.exp(self.weight_tensor), self.centers, self.scale)
        if reduce:
            return torch.sum(y)
        else:
            return y

    @classmethod
    def from_file(cls, path_to_model: str, device: torch.device=torch.device('cpu')) -> Self:
        potential_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = potential_data['initialisation_dict']
        state_dict = potential_data['state_dict']

        num_marginals = initialisation_dict.get('num_marginals', 48)
        box_lower = initialisation_dict.get('box_lower', -1.0)
        box_upper = initialisation_dict.get('box_upper', 1.0)
        num_centers = initialisation_dict.get('num_centers', 87)
        potential = cls(num_marginals=num_marginals, box_lower=box_lower, box_upper=box_upper, num_centers=num_centers)
        potential.load_state_dict(state_dict, strict=True)
        return potential

    @classmethod
    def from_config(cls, config: Configuration) -> Self:
        num_marginals = config['potential']['quartic_bspline']['num_marginals'].get()
        box_lower = config['potential']['quartic_bspline']['box_lower'].get()
        box_upper = config['potential']['quartic_bspline']['box_upper'].get()
        num_centers = config['potential']['quartic_bspline']['num_centers'].get()
        initialisation_mode = config['potential']['quartic_bspline']['initialisation']['mode'].get()
        multiplier = config['potential']['quartic_bspline']['initialisation']['multiplier'].get()
        trainable = config['potential']['quartic_bspline']['trainable'].get()
        
        init_options = {'mode': initialisation_mode, 'multiplier': multiplier}

        return cls(num_marginals=num_marginals, 
                   box_lower=box_lower, 
                   box_upper=box_upper, 
                   num_centers=num_centers,
                   init_options=init_options,
                   trainable=trainable)

    def save(self, path_to_model_dir: str, model_name: str='quartic_spline') -> str:
        potential_dict = {'initialisation_dict': self.initialisation_dict(),
                          'state_dict': self.state_dict()}
        return Potential.save(potential_dict, path_to_model_dir, model_name)

