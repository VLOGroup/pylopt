import torch
from abc import ABC
from typing import Dict, Any

from bilevel_optimisation.fields_of_experts import FieldsOfExperts, compile_regulariser
from bilevel_optimisation.measurement_model import MeasurementModel

class Energy(torch.nn.Module, ABC):
    """
    Class, inheriting from torch.nn.Module which represents the energy function assumed to be the sum of
    a data fidelty term and a regularisation term:

        (1 / (2 * sigma**2)) * | u - u_{noisy}| ** 2 + lam * regulariser(u)

    """
    def __init__(self, measurement_model: MeasurementModel, regulariser: FieldsOfExperts, lam: float) -> None:
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

    def compile(self, **compile_options: Dict[str, Any]) -> None:
        """
        Compilation of a regulariser using torch.compile().

        NOTE
        ----
            > Per default the most aggressive compilation options are used.
            > [IMPORTANT] When model shall be applied to different image shapes, then use the option dynamic=True.
                By default, this option is applied. Additionally, note that several warm up calls are performed
                using tensors of shape measurement_model.get_noisy_observation().shape.

        :param compile_options: Optional compilation options
        :return:
        """
        self.regulariser = compile_regulariser(self.regulariser,
                                               self.measurement_model.get_noisy_observation(), **compile_options)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Returns sum of data fidelty and (scaled) regularisation term

        :param u: Tensor at which data fidelty and regulariser are evaluated
        :return: Tensor representing the energy at the input x
        """
        return self.measurement_model(u) + self.lam * self.regulariser(u)
