from typing import Union, Iterable, Dict, Any, List
import torch
import numpy as np



class BaseNAGOptimiser:
    """
    Base NAG optimiser class which is used to implement Nesterov's Accelerated Gradient method.
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
        self.param_groups = list(params)
        self.defaults = {'alpha': alpha, 'beta': beta, 'lip_const': lip_const, 'theta': 0.0}
        self._max_num_bt_iterations = max_num_bt_iterations
        self.state = {}

    @staticmethod
    def compute_momentum_param(param_group: Dict[str, Any]) -> float:
        if param_group['beta']:
            return param_group['beta']
        else:
            theta = param_group['theta']

            theta_new = 0.5 * (1 + np.sqrt(1 + 4 * (theta ** 2)))
            beta = (theta - 1) / theta_new
            param_group['theta'] = theta_new

            return beta





