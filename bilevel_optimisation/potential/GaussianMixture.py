from typing import Dict, Any, Mapping
import torch

from bilevel_optimisation.potential.Potential import Potential
from bilevel_optimisation.gaussian_mixture_model.GaussianMixtureModel import GaussianMixtureModel

class GaussianMixture(Potential):
    """
    Class, inheriting from Potential, modelling a Gaussian mixture potential for a FoE model. By design
    there is a mixture per filter and every mixture consists of fixed centers and variances. Only the
    weights of each mixture are learnable. Note that by definition, the weights of every Gaussian mixture need to
    be non-negative and sum to one.

    """
    def __init__(self, **kwargs):
        """
        Initialisation of a Gaussian mixture. The following keyword arguments are required for the initialisation
        of an object of class GaussianMixture
            > num_components: Number of Gaussian's per mixture
            > box_lower: Lower bound of domain of Gaussian mixture
            > box_upper: Upper bound of domain of Gaussian mixture
            > weights_spec: Weights specification for Gaussian mixtures
            > num_filters: Number of filters used for the FoE model

        :param kwargs: Keyword arguments
        """
        super().__init__(**kwargs)

        self._num_components = kwargs.get('num_components')
        self._box_lower = kwargs.get('box_lower')
        self._box_upper = kwargs.get('box_upper')
        weights_spec = kwargs.get('weights_spec')
        self._num_filters = kwargs.get('num_filters')

        self._gmm_list = torch.nn.ModuleList([GaussianMixtureModel(self._num_components, self._box_lower,
                                                                   self._box_upper, weights_spec)
                                              for _ in range(0, self._num_filters)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([gmm(x[i : i + 1, :, :, :]) for i, gmm in enumerate(self._gmm_list)], dim=0)

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([gmm.forward_negative_log(x[i: i + 1, :, :, :])
                          for i, gmm in enumerate(self._gmm_list)])

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = self._gmm_list[0].state_dict(*args, **kwargs)
        state['_arch'] = {'num_components': self._num_components,
                          'box_lower': self._box_lower,
                          'box_upper': self._box_upper,
                          'num_filters': self._num_filters}
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        if '_arch' in state_dict.keys():
            state_dict.pop('_arch')
        for gmm in self._gmm_list:
            gmm.load_state_dict(state_dict, *args, **kwargs)

