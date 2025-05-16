import torch

from bilevel_optimisation.data.ParamSpec import ParamSpec

class GaussianMixtureModel(torch.nn.Module):

    """
    Class modelling a single Gaussian mixture.
    """

    def __init__(self, num_components: int, box_lower: float, box_upper: float,
                 weights_spec: ParamSpec):
        super().__init__()

        self._num_components = num_components
        # self._box_lower = torch.nn.Parameter(torch.tensor(box_lower), requires_grad=False)
        # self._box_upper = torch.nn.Parameter(torch.tensor(box_upper), requires_grad=False)
        self._box_lower = box_lower
        self._box_upper = box_upper

        centers = torch.linspace(start=self._box_lower, end=self._box_upper,
                                 steps=self._num_components)
        self.centers = torch.nn.Parameter(centers, requires_grad=False)
        std_dev = 2 * (self._box_upper - self._box_lower) / (self._num_components - 1)
        self.variance = torch.nn.Parameter(torch.tensor(std_dev ** 2), requires_grad=False)

        self.weights = torch.nn.Parameter(weights_spec.value, requires_grad=weights_spec.trainable)
        setattr(self.weights, 'param_name', 'gmm_weights')
        if weights_spec.projection is not None:
            setattr(self.weights, 'proj', weights_spec.projection)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = (-0.5 / self.variance) * (x.unsqueeze(dim=-1) - self.centers.unsqueeze(dim=0)) ** 2
        mixtures = torch.exp(args) / torch.sqrt(2 * torch.pi * self.variance)
        return torch.einsum('...k,...k->...', mixtures, self.weights)

    def forward_negative_log(self, x):
        args = (-0.5 / self.variance) * (x.unsqueeze(dim=-1) - self.centers.unsqueeze(dim=0)) ** 2
        return -torch.logsumexp(args + torch.log(self.weights[None, None, None, :])
                               - torch.log(torch.sqrt(2 * torch.pi * self.variance)), dim=-1)