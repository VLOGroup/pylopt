from abc import abstractmethod
import torch

class BaseLoss(torch.nn.Module):
    """
    Base class to model outer losses. Any outer loss is assumed to be independent of
    the parameters of the regulariser; its value is computed the clean data batch and
    a noisy/denoised observation. The clean data batch is treated as non-differentiable
    parameter of an object of class BaseLoss.
    """
    def __init__(self, u_clean: torch.Tensor):
        """
        Initialisation of an object of class BaseLoss.

        :param u_clean: Clean data batch in terms of a PyTorch tensor.
        """
        super().__init__()
        self.u_clean = torch.nn.Parameter(u_clean, requires_grad=False)

    @abstractmethod
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        pass

class L2Loss(BaseLoss):
    """
    Class, inherited from BaseLoss, implementing the standard l2 loss.
    """
    def __init__(self, u_clean: torch.Tensor) -> None:
        super().__init__(u_clean)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Function which computes the l2-loss of the input and the clean
        data batch.

        :param u: PyTorch tensor of the same shape as self.u_clean
        :return: L2 loss
        """
        return 0.5 * torch.sum((u - self.u_clean) ** 2)
