import torch
from typing import Tuple

class GaussianMixtureModel(torch.nn.Module):
    """
    Class modelling a single Gaussian mixture on the interval [box_lower, box_upper].

    Notes
    -----
        > Intended usage: Visualisation only.
    """

    def __init__(self, weights: torch.Tensor, centers: torch.Tensor, variance: torch.Tensor,
                 box_lower: float, box_upper: float):
        super().__init__()

        self._centers = centers
        self._variance = variance

        self._box_lower = box_lower
        self._box_upper = box_upper

        self._weights = weights
        if torch.abs(torch.sum(self._weights) - 1.0) > 1e-5:
            raise ValueError('Weights are not properly normalised')

    def get_box(self) -> Tuple[float, float]:
        return self._box_lower, self._box_upper

    def get_number_of_components(self) -> int:
        return len(self._centers)

    def forward_single_component(self, x: torch.Tensor, j: int) -> torch.Tensor:
        arg = (-0.5 / self._variance) * (x.unsqueeze(dim=-1) - self._centers[j]) ** 2
        return torch.exp(arg) / torch.sqrt(2 * torch.pi * self._variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff_sq = (x.unsqueeze(dim=1) - self._centers.unsqueeze(dim=0)) ** 2
        gaussian_multiplier = torch.sqrt(2 * torch.pi * self._variance)
        return torch.sum(torch.exp(-0.5 * diff_sq / self._variance) / gaussian_multiplier, dim=1)

    def forward_negative_log(self, x: torch.Tensor) -> torch.Tensor:
        diff_sq = (x.unsqueeze(dim=1) - self._centers.unsqueeze(dim=0)) ** 2
        gaussian_multiplier = 0.5 * torch.log(2 * torch.pi * self._variance)
        log_probs = -0.5 * diff_sq / self._variance + torch.log(self._weights) - gaussian_multiplier
        neg_log_filter_gmm = -torch.logsumexp(log_probs, dim=1)
        return neg_log_filter_gmm
