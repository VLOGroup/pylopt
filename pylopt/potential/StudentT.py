import os.path
from typing import Dict, Any, Self
import torch
from confuse import Configuration

from pylopt.potential.Potential import Potential

class StudentT(Potential):
    """
    Class implementing student-t potential for the usage in context of FoE models.    """

    def __init__(
            self,
            num_marginals: int=48,
            initialisation_mode: str='uniform',
            multiplier: float=1.0,
            trainable: bool=True
    ) -> None:
        super().__init__(num_marginals)

        weight_data = self._init_weight_tensor(num_marginals, initialisation_mode, multiplier)
        self.weight_tensor = torch.nn.Parameter(data=weight_data, requires_grad=trainable)

    @staticmethod
    def _init_weight_tensor(num_marginals: int, initialisation_mode: str, multiplier: float) -> torch.Tensor:
        if initialisation_mode == 'rand':
            weight_data = torch.log(multiplier * torch.rand(num_marginals))
        elif initialisation_mode == 'uniform':
            weight_data = torch.log(multiplier * torch.ones(num_marginals))
        else:
            raise ValueError('Unknown initialisation method')
        return weight_data

    def get_parameters(self) -> torch.Tensor:
        return self.weight_tensor.data

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

        return cls(num_marginals, initialisation_mode, multiplier, trainable)

    def save(self, path_to_model_dir: str, model_name: str='student_t') -> str:
        path_to_model = os.path.join(path_to_model_dir, '{:s}.pt'.format(os.path.splitext(model_name)[0]))
        potential_dict = {'initialisation_dict': self.initialisation_dict(),
                          'state_dict': self.state_dict()}

        torch.save(potential_dict, path_to_model)
        return path_to_model