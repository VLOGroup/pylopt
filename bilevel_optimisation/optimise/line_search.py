import torch
from typing import List

def compute_quadratic_approximation(param_group_new: List[torch.nn.Parameter],
                                    param_group: List[torch.Tensor],
                                    grad_group_list: List[torch.Tensor], loss: torch.Tensor,
                                    lip_const: float) -> torch.Tensor:
    """
    Function which computes the quadratic approximation of the loss function of the optimisation
    function. It is used when searching via backtracking for a step size which guarantees
    sufficient descent.

    :param param_group_new: List of new parameter candidates of current group
    :param param_group: List of current parameters of current group
    :param grad_group_list: List of gradient of current parameters of current group
    :param loss: Value of loss function at current parameters
    :param lip_const: Current guess of the gradients Lipschitz constant
    :return: Quadratic approximation of loss.
    """
    quadr_approx = loss.clone()
    for p_new, p, grad_p in zip(param_group_new, param_group, grad_group_list):
        quadr_approx += (torch.sum(grad_p * (p_new.data - p.data)) +
                         0.5 * lip_const * torch.sum((p_new.data - p.data) ** 2))
    return quadr_approx