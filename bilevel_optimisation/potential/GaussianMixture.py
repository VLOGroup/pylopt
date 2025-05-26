from typing import Dict, Any, Mapping
import torch
import logsumexpv2 as lse

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
                 log_weights_spec: ParamSpec, num_filters: int):
        super().__init__()

        self._num_components = num_components
        self._box_lower = box_lower
        self._box_upper = box_upper
        self._num_filters = num_filters

        centers = torch.linspace(start=self._box_lower, end=self._box_upper,
                                 steps=self._num_components)
        self.centers = torch.nn.Parameter(centers, requires_grad=False)



        std_dev = 2 * (self._box_upper - self._box_lower) / (self._num_components - 1)
        self.variance = torch.nn.Parameter(torch.Tensor([std_dev ** 2]),
                                           requires_grad=False).to(device=self.centers.device)

        self.log_weights = torch.nn.Parameter(log_weights_spec.value, requires_grad=log_weights_spec.trainable)
        self.scale_param = torch.nn.Parameter(torch.rand(self._num_filters) / 100, requires_grad=log_weights_spec.trainable)
        setattr(self.scale_param, 'proj', lambda z: torch.clamp(z, min=0.00001))
                    # setattr(self.weights, 'proj', weights_spec.projection)
                    #
                    # # if weights_spec.projection is not None:
                    # #
        self._gaussian_multiplier = torch.nn.Parameter(0.5 * torch.log(2 * torch.pi * self.variance),
                                                       requires_grad=False)

        # ### terms for cuda accelerated computation
        self.centers_ = torch.nn.Parameter(centers.unsqueeze(dim=0).repeat(num_filters, 1), requires_grad=False)
        self.std_dev_ = torch.nn.Parameter(std_dev * torch.ones(num_filters, num_components), requires_grad=False)


    def get_number_of_mixtures(self) -> int:
        return self.log_weights.shape[0]

    def get_single_mixture(self, j: int) -> GaussianMixtureModel:
        weights = torch.nn.functional.softmax(self.log_weights, dim=1)[j, :]
        gmm = GaussianMixtureModel(weights=weights, centers=self.centers.data, variance=self.variance.data,
                                   box_lower=self._box_lower, box_upper=self._box_upper)
        return gmm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:

        # weights = torch.nn.functional.softmax(self.log_weights, dim=1)
        # neg_log_filter_gmm, _ = lse.pot_act(self.scale_param.reshape(1, -1, 1, 1) * x, weights, self.centers_, self.std_dev_)
        #   there is an issue with gradient computation!
        #   more than that: it is slower than conventional implementation. how is that possible?


        # conventional implementation
        diff_sq = (self.scale_param.reshape(1, -1, 1, 1, 1) * x.unsqueeze(dim=2) - self.centers.reshape(1, 1, -1, 1, 1)) ** 2
        log_weights = torch.nn.functional.log_softmax(self.log_weights, dim=1).reshape(1, self._num_filters,
                                                                                       self._num_components, 1, 1)
        log_probs = -0.5 * diff_sq / self.variance + log_weights - self._gaussian_multiplier
        neg_log_filter_gmm = -torch.logsumexp(log_probs, dim=2)

        return neg_log_filter_gmm


    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        pass
        # state = self._gmm_list[0].state_dict(*args, **kwargs)
        # state['_arch'] = {'num_components': self._num_components,
        #                   'box_lower': self._box_lower,
        #                   'box_upper': self._box_upper,
        #                   'num_filters': self._num_filters}
        # return state

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        pass
        # if '_arch' in state_dict.keys():
        #     state_dict.pop('_arch')
        # for gmm in self._gmm_list:
        #     gmm.load_state_dict(state_dict, *args, **kwargs)

    # def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
    #     state = self._gmm_list[0].state_dict(*args, **kwargs)
    #     state['_arch'] = {'num_components': self._num_components,
    #                       'box_lower': self._box_lower,
    #                       'box_upper': self._box_upper,
    #                       'num_filters': self._num_filters}
    #     return state
    #
    # def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
    #     if '_arch' in state_dict.keys():
    #         state_dict.pop('_arch')
    #     for gmm in self._gmm_list:
    #         gmm.load_state_dict(state_dict, *args, **kwargs)
