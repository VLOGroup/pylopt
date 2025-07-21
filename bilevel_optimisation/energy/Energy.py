import torch
from abc import ABC

from bilevel_optimisation.fields_of_experts import FieldsOfExperts
from bilevel_optimisation.measurement_model import MeasurementModel

class Energy(torch.nn.Module, ABC):
    """
    Class, inherited from torch.nn.Module, which is used as base class of the inner bilevel
    energy. Each class, inheriting from InnerEnergy must implement the methods sample() and
    argmin().
    """
    def __init__(self, measurement_model: MeasurementModel,
                 regulariser: FieldsOfExperts, lam: float) -> None:
        """
        Initialisation of an object of class InnerEnergy

        :param measurement_model: Object of class MeasurementModel
        :param regulariser: Fields of experts regulariser
        :param lam: Regularisation parameter
        """
        super().__init__()
        self.measurement_model = measurement_model
        self.regulariser = regulariser
        self.lam = lam

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Returns sum of data fidelty and scaled regularisation term

        :param u: Tensor at which data fidelty and regulariser are evaluated
        :return: Tensor representing the energy at the input x
        """
        return self.measurement_model(u) + self.lam * self.regulariser(u)
