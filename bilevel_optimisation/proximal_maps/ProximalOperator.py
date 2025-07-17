import torch
from abc import abstractmethod

class ProximalOperator(torch.nn.Module):
    def __init__(self, noise_level: float) -> None:
        super().__init__()
        self.noise_level = torch.nn.Parameter(torch.tensor(noise_level), requires_grad=False)

    @abstractmethod
    def forward(self, u: torch.Tensor, tau: float, u_noisy: torch.Tensor) -> torch.Tensor:
        pass

class DenoisingProx(ProximalOperator):
    def __init__(self, noise_level: float) -> None:
        super().__init__(noise_level)

    def forward(self, u: torch.Tensor, tau: float, u_noisy: torch.Tensor) -> torch.Tensor:
        kappa = (tau / self.noise_level ** 2)
        return (kappa * u_noisy + u) / (1 + kappa)