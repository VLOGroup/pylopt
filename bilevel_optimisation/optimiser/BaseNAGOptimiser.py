from typing import Union, Iterable, Dict, Any, List
import torch
import numpy as np

class BaseNAGOptimiser(torch.optim.Optimizer):
    """
    Base NAG optimiser class which is used to implement Nesterov's Accelerated Gradient
    method in the torch.optim.Optimizer environment.
    """

    def __init__(self,  params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 alpha: float = None, beta: float = None, lip_const: float = 10000000,
                 max_num_bt_iterations: int = 10) -> None:
        """
        Initialisation of class BaseNAGOptimiser

        :param params: Parameters to be optimised
        :param alpha: Constant step size - if not provided backtracking is performed to
            find step size
        :param beta: Momentum parameter - if not provided Nesterov's optimal momentum
            parameter is used.
        :param lip_const: Default value for Lipschitz constant which is used
            in the backtracking method to determine the step size. If a constant
            step sizes is applied, the parameter is not required.
        :param max_num_bt_iterations: Maximal number of backtracking iterations - again,
            for constant step sizes this is not required.
        """
        super().__init__(params, defaults={'alpha': alpha, 'beta': beta, 'lip_const': lip_const, 'theta': 0.0})

        self._max_num_bt_iterations = max_num_bt_iterations

    @staticmethod
    def _apply_prox_map(p: torch.nn.Parameter, step_size: float) -> torch.nn.Parameter:
        """
        Function which applies prox map to parameter provided that the parameter has an
        attribute called 'prox'. By assumption this attribute is a callable taking the
        value of the parameter and the step size to be applied as arguments.

        :param p:
        :param step_size:
        :return: Parameter whose value is updated.
        """
        if hasattr(p, 'prox'):
            p.data.copy_(p.prox(p.data, step_size))
        return p

    @staticmethod
    def _apply_projection(p: torch.nn.Parameter) -> torch.nn.Parameter:
        """
        This function, applies a projection to the value of the given parameter. This is
        implemented similarly as for _apply_prox_map, i.e. the projection map has to be provided as callable
        via attribute 'proj' of the parameter.

        :param p:
        :return: Parameter whose value is updated.
        """
        if hasattr(p, 'proj'):
            p.data.copy_(p.proj(p.data))
        return p

    @staticmethod
    def _momentum_param(param_group: Dict[str, Any]) -> float:
        if param_group['beta']:
            return param_group['beta']
        else:
            theta = param_group['theta']

            theta_new = 0.5 * (1 + np.sqrt(1 + 4 * (theta ** 2)))
            beta = (theta - 1) / theta_new
            param_group['theta'] = theta_new

            return beta

    @staticmethod
    def _compute_quadratic_approximation(param_group_new: List[torch.nn.Parameter],
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