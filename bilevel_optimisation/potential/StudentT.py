import os.path
from typing import Dict, Any, Optional
import torch
from confuse import Configuration

from bilevel_optimisation.potential.Potential import Potential

@Potential.register_subclass('student_t')
class StudentT(Potential):
    """
    Class implementing student-t potential for the usage in context of FoE models.
    """

    def __init__(self, num_marginals: int, config: Configuration) -> None:
        super().__init__(num_marginals)

        initialisation_mode = config['potential']['student_t']['initialisation']['mode'].get()
        multiplier = config['potential']['student_t']['initialisation']['multiplier'].get()
        trainable = config['potential']['student_t']['trainable'].get()

        model_path = config['potential']['student_t']['initialisation']['file_path'].get()
        if not model_path:
            if initialisation_mode == 'rand':
                weights = torch.log(multiplier * torch.rand(num_marginals))
            elif initialisation_mode == 'uniform':
                weights = torch.log(multiplier * torch.ones(num_marginals))
            else:
                raise ValueError('Unknown initialisation method')
            self.weight_tensor = torch.nn.Parameter(data=weights, requires_grad=trainable)
        else:
            dummy_data = torch.ones(num_marginals)
            self.weight_tensor = torch.nn.Parameter(data=dummy_data, requires_grad=trainable)
            self._load_from_file(model_path)
            with torch.no_grad():
                self.weight_tensor.add_(torch.log(torch.tensor(multiplier)))

    def get_parameters(self) -> torch.Tensor:
        return self.weight_tensor.data

    def forward_negative_log(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        if reduce:
            return torch.einsum('bfhw,f->', torch.log(1.0 + x ** 2), torch.exp(self.weight_tensor))
        else:
            return torch.einsum('bfhw,f->bfhw', torch.log(1.0 + x ** 2), torch.exp(self.weight_tensor))

    def initialisation_dict(self) -> Dict[str, Any]:
        return {'num_marginals': self.num_marginals}

    def _load_from_file(self, path_to_model: str, device: torch.device=torch.device('cpu')) -> None:
        potential_data = torch.load(path_to_model, map_location=device)

        initialisation_dict = potential_data['initialisation_dict']
        self.num_marginals = initialisation_dict['num_marginals']

        state_dict = potential_data['state_dict']
        self.load_state_dict(state_dict)

    def save(self, path_to_model_dir: str, model_name: str='student_t') -> str:
        path_to_model = os.path.join(path_to_model_dir, '{:s}.pt'.format(os.path.splitext(model_name)[0]))
        potential_dict = {'initialisation_dict': self.initialisation_dict(),
                          'state_dict': self.state_dict()}

        torch.save(potential_dict, path_to_model)
        return path_to_model