import logging

from bilevel_optimisation.filters.Filters import ImageFilter
from bilevel_optimisation.potential.Potential import Potential

import torch

class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()

        self.delta_time = self.start.elapsed_time(self.end)

class FieldsOfExperts(torch.nn.Module):
    """
    Class representing fields of experts (short: FoE) model which is used as regulariser. It consists of mainly
    of the components
        > filters
        > potential
    In the most general case all of these components are trainable. By assumption, filters shall have zero mean,
    filter weights shall be positive and the potential is assumed to be proportional to a probability density. Thus,
    training these components requires to apply appropriate projections. The projection maps of the filters  can be
    specified by means of filters_spec and will be set as attributes of the corresponding member variables.
    """
    def __init__(self, potential: Potential, image_filter: ImageFilter) -> None:
        """
        Initialisation of an FoE-model.

        :param potential: Object of class Potential
        :param image_filter: Object of class ImageFilter
        """
        super().__init__()

        self.potential = potential
        self.image_filter = image_filter

    def get_image_filter(self) -> ImageFilter:
        return self.image_filter

    def get_potential(self) -> Potential:
        return self.potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logging.debug('[FoE] evaluate regulariser')
        x_conv = self.image_filter(x)
        return torch.einsum('bfhw->', self.potential.forward_negative_log(x_conv))

