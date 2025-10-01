import os.path
from typing import Dict, Any, Self, Optional
import torch
from confuse import Configuration

from . import Potential

class StudentT(Potential):
    """
    Class implementing student-t potential for the usage in context of FoE models.
    """
    def __init__(
            self,
            num_marginals: int,
            init_options: Optional[Dict[str, Any]]=None,
            trainable: bool=True
    ) -> None:
        """
        Initialisation of class StudentT

        NOTE
        ----
            > It is not checked whether number of filters and number of potentials coincide.

        :param num_marginals: Number of potentials - per design it must coincide
            with the number of filters.
        :param init_options: Dictionary of initialisation options. Keys:
            - 'mode': String indicating how potentials weights are initialised: 'rand', 'uniform'
            - 'multiplier': Float used as initial scaling factor
            Defaults to None
        :param trainable: Boolean flag indicating if potentials are trainable; defaults to True
        """
        super().__init__(num_marginals)

        default_init = {
            'mode': 'uniform', 
            'multiplier': 1.0
        }
        self.init_options = {**default_init, **(init_options or {})}

        weight_data = self._init_weight_tensor(num_marginals, self.init_options)
        self.weight_tensor = torch.nn.Parameter(data=weight_data, requires_grad=trainable)

    @staticmethod
    def _init_weight_tensor(num_marginals: int, init_options: Dict[str, Any]) -> torch.Tensor:
        multiplier = init_options['multiplier']

        if init_options['mode'] == 'rand':
            weight_data = torch.log(multiplier * torch.rand(num_marginals))
        elif init_options['mode'] == 'uniform':
            weight_data = torch.log(multiplier * torch.ones(num_marginals))
        else:
            raise ValueError('Unknown initialisation method')
        return weight_data

    def get_parameters(self) -> torch.Tensor:
        return self.weight_tensor.data

    def freeze(self) -> None:
        self.weight_tensor.requires_grad = False
    
    def unfreeze(self) -> None:
        self.weight_tensor.requires_grad = True

    def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        if reduce:
            return torch.einsum('bfhw,f->', torch.log(1.0 + x ** 2), torch.exp(self.weight_tensor))
        else:
            return torch.einsum('bfhw,f->bfhw', torch.log(1.0 + x ** 2), torch.exp(self.weight_tensor))

    def initialisation_dict(self) -> Dict[str, Any]:
        return {'num_marginals': self.num_marginals}

    @classmethod
    def from_file(cls, path_to_model: str, device: torch.device=torch.device('cpu')) -> Self:
        potential_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = potential_data['initialisation_dict']
        state_dict = potential_data['state_dict']

        num_marginals = initialisation_dict.get('num_marginals', 48)
        potential = cls(num_marginals=num_marginals)
        potential.load_state_dict(state_dict, strict=True)
        return potential

    @classmethod
    def from_config(cls, config: Configuration) -> Self:
        num_marginals = config['potential']['student_t']['num_marginals'].get()
        initialisation_mode = config['potential']['student_t']['initialisation']['mode'].get()
        multiplier = config['potential']['student_t']['initialisation']['multiplier'].get()
        trainable = config['potential']['student_t']['trainable'].get()

        init_options = {'mode': initialisation_mode, 'multiplier': multiplier}
        return cls(num_marginals, init_options, trainable)

    def save(self, path_to_model_dir: str, model_name: str='student_t') -> str:
        potential_dict = {'initialisation_dict': self.initialisation_dict(),
                          'state_dict': self.state_dict()}
        return Potential.save(potential_dict, path_to_model_dir, model_name)