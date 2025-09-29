import torch

from .ImageFilter import ImageFilter
from .potential import Potential

class FieldsOfExperts(torch.nn.Module):
    """
    Class representing fields of experts (short: FoE) model which is used as regulariser. It consists
    of the components
        > filters
        > potential
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

    def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        x_conv = self.image_filter(x)
        return self.potential.forward(x_conv, reduce=reduce)

