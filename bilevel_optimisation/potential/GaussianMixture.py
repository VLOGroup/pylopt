from typing import Dict, Any, Mapping, NamedTuple
import torch

from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.potential.Potential import Potential
from bilevel_optimisation.gaussian_mixture_model.GaussianMixtureModel import GaussianMixtureModel

class GaussianMixture(Potential):
    """
    Class, inheriting from Potential, modelling a Gaussian mixture potential for a FoE model. By design
    there is a mixture per filter and every mixture consists of fixed centers and variances. Only the
    weights of each mixture are learnable. Note that by definition, the weights of every Gaussian mixture need to
    be non-negative and sum to one.
    """
    def __init__(self, num_components: int, box_lower: float, box_upper: float,
                 log_weights_spec: ParamSpec, num_potentials: int):
        super().__init__(num_potentials=num_potentials)

        self.register_buffer('num_components', torch.tensor(num_components, dtype=torch.uint16))
        self.register_buffer('box_lower', torch.tensor(box_lower, dtype=torch.float32))
        self.register_buffer('box_upper', torch.tensor(box_upper, dtype=torch.float32))

        centers = torch.linspace(start=self.box_lower, end=self.box_upper, steps=self.num_components)
        self.centers = torch.nn.Parameter(centers, requires_grad=False)

        std_dev = 2 * (self.box_upper - self.box_lower) / (self.num_components.to(dtype=torch.float32) - 1)
        self.variance = torch.nn.Parameter(torch.Tensor([std_dev ** 2]),
                                           requires_grad=False).to(device=self.centers.device)

        self.log_weights = torch.nn.Parameter(log_weights_spec.value, requires_grad=log_weights_spec.trainable)

        self.scale_param = torch.nn.Parameter(torch.rand(self.get_num_potentials()) / 100,
                                              requires_grad=log_weights_spec.trainable)
        setattr(self.scale_param, 'proj', lambda z: torch.clamp(z, min=0.00001))

        self.gaussian_multiplier = torch.nn.Parameter(0.5 * torch.log(2 * torch.pi * self.variance),
                                                      requires_grad=False)

    def get_parameters(self) -> torch.Tensor:
        return self.log_weights.data

    def get_number_of_mixtures(self) -> int:
        return self.get_num_potentials()

    def get_single_mixture(self, j: int) -> GaussianMixtureModel:
        weights = torch.nn.functional.softmax(self.log_weights, dim=1)[j, :]
        gmm = GaussianMixtureModel(weights=weights, centers=self.centers.data, variance=self.variance.data,
                                   box_lower=self.box_lower, box_upper=self.box_upper)
        return gmm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        diff_sq = (self.scale_param.reshape(1, -1, 1, 1, 1) * x.unsqueeze(dim=2) -
                   self.centers.reshape(1, 1, -1, 1, 1)) ** 2

        # diff_sq = (x.unsqueeze(dim=2) - self.centers.reshape(1, 1, -1, 1, 1)) ** 2

        log_weights = torch.nn.functional.log_softmax(self.log_weights, dim=1).reshape(1, self.get_num_potentials(),
                                                                                       self.num_components, 1, 1)
        log_probs = -0.5 * diff_sq / self.variance + log_weights - self.gaussian_multiplier
        neg_log_filter_gmm = -torch.logsumexp(log_probs, dim=2)

        return neg_log_filter_gmm

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, kwargs)
        return state

    def initialisation_dict(self) -> Dict[str, Any]:
        return {'type': self.__class__.__name__, 'num_potentials': self.num_potentials,
                'num_components': self.num_components, 'box_lower': self.box_lower, 'box_upper': self.box_upper}

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> NamedTuple:
        result = torch.nn.Module.load_state_dict(self, state_dict, strict=True)
        return result
