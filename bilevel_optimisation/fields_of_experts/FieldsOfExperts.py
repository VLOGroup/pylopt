import torch
import logging

from bilevel_optimisation.data.ParamSpec import ParamSpec
from bilevel_optimisation.potential.Potential import Potential

class FieldsOfExperts(torch.nn.Module):
    """
    Class representing fields of experts (short: FoE) model which is used as regulariser. It consists of mainly
    of the components
        > filters
        > filter weights
        > potential
    In the most general case all of these components are trainable. By assumption, filters shall have zero mean,
    filter weights shall be positive and the potential is assumed to be proportional to a probability density. Thus,
    training these components requires to apply appropriate projections. The projection maps of the filters and
    filter weights can be specified by means of filters_spec and filter_weights_spec respectively and will be set
    as attributes of the corresponding member variables. This approach enables applying these operations by subclassing
    of any torch optimiser. Parameter dependent potentials need to be designed accordingely - see for example
    GaussianMixtureModel.
    """
    def __init__(self, potential: Potential, filters_spec: ParamSpec, filter_weights_spec: ParamSpec) -> None:
        """
        Initialisation of an FoE-model.

        :param potential: Object of class Potential
        :param filters_spec: Object of data class ParamSpec which is used to initialise the filters
        :param filter_weights_spec: Object of data class ParamSpec used to initialise the filter weights.
        """
        super().__init__()

        self.potential = potential

        self.filters = torch.nn.Parameter(filters_spec.value, requires_grad=filters_spec.trainable)
        setattr(self.filters, 'param_name', 'filters')
        if filters_spec.projection is not None:
            setattr(self.filters, 'proj', filters_spec.projection)

        self._padding_mode = filters_spec.parameters['padding_mode']
        self._padding = self.filters.shape[-1] // 2

        self.filter_weights = torch.nn.Parameter(filter_weights_spec.value,
                                                 requires_grad=filter_weights_spec.trainable)
        setattr(self.filter_weights, 'param_name', 'filter_weights')
        if filter_weights_spec.projection is not None:
            setattr(self.filter_weights, 'proj', filter_weights_spec.projection)

    def get_filters(self) -> torch.Tensor:
        return self.filters.detach().clone()

    def get_filter_weights(self) -> torch.Tensor:
        return self.filter_weights.detach().clone()

    def get_potential(self) -> Potential:
        return self.potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logging.debug('[FoE] evaluate regulariser')
        x_padded = torch.nn.functional.pad(x, (self._padding, self._padding, self._padding, self._padding),
                                           self._padding_mode)
        convolutions = torch.nn.functional.conv2d(x_padded, self.filters)
        neg_log_potential = self.potential.forward_negative_log(convolutions)
        return torch.einsum('bfhw,f->', neg_log_potential, self.filter_weights)








