from typing import Union, Iterable, Dict, Any, Callable, List
import torch

from bilevel_optimisation.optimise.BaseNAGOptimiser import BaseNAGOptimiser
from bilevel_optimisation.optimise.line_search import compute_quadratic_approximation

class NAGOptimiser(BaseNAGOptimiser):
    """
    Class implementing Nesterov's accelerated gradient method. Note that this implementation
    treats all the parameters belonging to a parameter group uniformly:
        > For all parameters the same step size is applied
        > For backtracking, descent condition is checked for closure as function of all
            parameters.

    To perform Nesterov updates using backtracking line search the loss function of the problem
    needs to be provided in terms of a callable. This allows to find step sizes providing
    sufficient descent.

    Notes
    -----
        > The implementation allows to perform Nesterov accelerated proximal gradient descent. If parameters
            do have a callable attribute 'prox', this map is applied. The function must take a torch tensor and
            a float (the step size) as input; the result is returned as torch tensor.
    """
    def __init__(self, params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 alpha: float = None, beta: float = None, lip_const: float = 10000000,
                 max_num_bt_iterations: int = 10) -> None:
        """
        Initialisation of class NAGOptimiser

        :param params:
        :param alpha:
        :param beta:
        :param lip_const:
        :param max_num_bt_iterations:
        """
        super().__init__(params, alpha, beta, lip_const, max_num_bt_iterations)

    def _make_intermediate_step(self, param_group: Dict[str, Any]) -> None:
        beta = self.compute_momentum_param(param_group)
        for p in [p_ for p_ in param_group['params'] if p_.requires_grad]:
            state = self.state[p]
            if 'history' not in state:
                state['history'] = p.data.clone()

            momentum = p.data - state['history']
            state['history'] = p.data.clone()
            p.data.add_(beta * momentum)

    def _apply_gradient_step(self, param_group_flat: List[torch.nn.Parameter],
                             grad_group_list: List[torch.Tensor], step_size: float):
        for p, grad_p in zip(param_group_flat, grad_group_list):
            p.data.sub_(step_size * grad_p)
            p = self._apply_prox_map(p, step_size)
            p = self._apply_projection(p)

    def _perform_backtracking(self, param_group_flat: List[torch.nn.Parameter],
                               grad_group_list: List[torch.Tensor],
                               lip_const: float, closure: Callable) -> float:
        loss = closure()
        params_orig = [p.clone() for p in param_group_flat]

        for k in range(0, self._max_num_bt_iterations):
            step_size = 1 / lip_const
            self._apply_gradient_step(param_group_flat, grad_group_list, step_size)

            loss_new = closure()
            quadratic_approx = compute_quadratic_approximation(param_group_flat, params_orig,
                                                               grad_group_list, loss, lip_const)
            if loss_new <= quadratic_approx:
                lip_const *= 0.9
                break
            else:
                lip_const *= 2.0
                for p, p_orig in zip(param_group_flat, params_orig):
                    p.data.copy_(p_orig)

        return lip_const

    @torch.no_grad()
    def step(self, func: Callable, grad_func: Callable) -> torch.Tensor:
        """
        Function which performs the update step. The main difference to step(...) is in the closure function.
        This implementation requires the closure to be the loss function only - gradients are computed within the
        implementation of step_() using torch.autograd.grad(). Thus, gradients do not need to be cleared,

        :param closure: Callable, representing the loss function required for this optimisation procedure.
        :return: Loss after update step is performed
        """

        for group in self.param_groups:
            self._make_intermediate_step(group)

            # with torch.enable_grad():
            #     loss = closure()
            # group_params_trainable = [p for p in group['params'] if p.requires_grad]
            # grad_list = list(torch.autograd.grad(outputs=loss, inputs=group_params_trainable))
            group_params_trainable = [p for p in group['params'] if p.requires_grad]
            grad_list = grad_func(group_params_trainable)


            if group['alpha']:
                self._apply_gradient_step(group_params_trainable, grad_list, group['alpha'])
            else:
                group['lip_const'] = self._perform_backtracking(group_params_trainable, grad_list,
                                                                 group['lip_const'], closure)
        return func(group_params_trainable)
