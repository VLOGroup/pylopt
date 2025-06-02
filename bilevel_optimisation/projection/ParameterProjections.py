import torch

from bilevel_optimisation.data.Constants import EPSILON

def non_negative_projection(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=EPSILON)

def zero_mean_projection(x: torch.Tensor) -> torch.Tensor:
    return x - torch.mean(x, dim=(-2, -1), keepdim=True)

def unit_simplex_projection(x: torch.Tensor) -> torch.Tensor:
    """
    See

      @article{wang2013projection,
        title={Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application},
        author={Wang, Weiran and Carreira-Perpin{\'a}n, Miguel A},
        journal={arXiv preprint arXiv:1309.1541},
        year={2013}
      }

    :param x:
    :return:
    """

    x_sorted, indices = torch.sort(x, descending=True)
    k = max([l for l in range(1, len(x_sorted) + 1)
         if (torch.sum(x_sorted[0 : l]) - 1) / l < x_sorted[l - 1]])
    tau = (torch.sum(x_sorted[0 : k]) - 1) / k
    x_proj = torch.zeros_like(x)
    x_proj[indices] = torch.maximum(x_sorted - tau, torch.zeros_like(x_sorted))
    return x_proj
