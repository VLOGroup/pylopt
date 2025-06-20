import torch
from functools import cached_property
import logging

class MeasurementModel(torch.nn.Module):
    """
    A class, inherited from torch.nn.Module modeling the noisy measurement/observation of data.
    Observations are assumed to be a sum of a forward operator applied to the clean data and noise sampled from
    the multivariate normal distribution with zero mean and covariance matrix (noise_level ** 2) * I.
    """
    def __init__(self, u_clean: torch.Tensor, operator: torch.nn.Module, noise_level: float) -> None:
        """
        Initialisation of an object of class MeasurementModel. The clean data tensor u_clean is
        stored as a non-trainable parameter.

        :param u_clean: Tensor of shape [batch_size, channels, height, width]
        :param operator: Module representing the forward operator (e.g. torch.nn.Identity() for image denoising)
        :param noise_level: Standard deviation of additive gaussian noise
        """
        super().__init__()
        self.u_clean = torch.nn.Parameter(u_clean, requires_grad=False)
        self._operator = operator
        self.noise_level = noise_level

    def obs_clean(self) -> torch.nn.Parameter:
        """
        Returns the clean data tensor to the caller

        :return: torch.Tensor
        """
        return self.u_clean

    @cached_property
    def obs_noisy(self) -> torch.nn.Parameter:
        """
        Returns the noisy observation by applying the forward operator to the clean data tensor and
        adding random gaussian white noise.

        :return: torch.nn.Parameter
        """
        obs_clean = self._operator(self.u_clean)
        return torch.nn.Parameter(obs_clean + self.noise_level * torch.randn_like(obs_clean), requires_grad=False)

    def _data_fidelity(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the data fidelty, i.e. the scaled squared l2-norm of
        the difference between the input tensor and a noisy observation. The
        squared l2-norm is scaled by 0.5 / self._noise_level ** 2

        :param u: Tensor of the same shape as clean or noisy data tensor.
        :return: Scaled squared l2-norm in terms of a torch.Tensor.
        """

        return 0.5 * torch.sum((u - self.obs_noisy) ** 2) / self.noise_level ** 2

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the data fidelty of the input tensor u by calling self._data_fidelty(u)

        :param u: Tensor of the same shape as clean or noisy data tensor.
        :return: Data fidelty of u
        """
        return self._data_fidelity(u)

