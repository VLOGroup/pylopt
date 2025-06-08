from typing import Union, Iterable, Dict, Any, Callable, List
import torch
import numpy as np
import logging

from bilevel_optimisation.data.Constants import MAX_NUM_UNROLLING_ITER

class BaseNAGOptimiser(torch.optim.Optimizer):
    """
    Base NAG optimiser class which is used to implement Nesterov's Accelerated Gradient
    method in the torch.optim.Optimizer environment.
    """

    def __init__(self,  params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 alpha: float = None, beta: float = None, lip_const_default: float = 10000000,
                 max_num_bt_iterations: int = 10) -> None:
        """
        Initialisation of class BaseNAGOptimiser

        :param params: Parameters to be optimised
        :param alpha: Constant step size - if not provided backtracking is performed to
            find step size
        :param beta: Momentum parameter - if not provided Nesterov's optimal momentum
            parameter is used.
        :param lip_const_default: Default value for Lipschitz constant which is used
            in the backtracking method to determine the step size. If a constant
            step sizes is applied, the parameter is not required.
        :param max_num_bt_iterations: Maximal number of backtracking iterations - again,
            for constant step sizes this is not required.
        """
        self._alpha = alpha
        self._beta = beta

        self._lip_const_default = lip_const_default
        self._max_num_bt_iterations = max_num_bt_iterations

        super().__init__(params, {})

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
            logging.debug('[NAG] apply prox map')
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
            logging.debug('[NAG] apply projection')
            p.data.copy_(p.proj(p.data))
        return p

class NAGOptimiser(BaseNAGOptimiser):
    """
    Class implementing Nesterov's accelerated gradient method. Note that this implementation
    treats all the parameters belonging to a parameter group uniformly:
        > For all parameters the same step size is applied
        > For backtracking, descent condition is checked for closure as function of all
            parameters.

    To perform Nesterov updates using backtracking line search the loss function of the problem
    needs to be provided in terms of a callable. This allows to find step sizes providing
    sufficient descent. The loss function is assumed to be provided as closure function - see [1]

    References
    ----------
    [1] https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html

    Notes
    ----------
        > The implementation allows to perform Nesterov accelerated proximal gradient descent. If parameters
            do have a callable attribute 'prox', this map is applied. The function must take a torch tensor and
            a float (the step size) as input; the result is returned as torch tensor.
    """
    def __init__(self, params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 alpha: float = None, beta: float = None, lip_const_default: float = 10000000,
                 max_num_bt_iterations: int = 10) -> None:
        """
        Initialisation of class NAGOptimiser

        :param params:
        :param alpha:
        :param beta:
        :param lip_const_default:
        :param max_num_bt_iterations:
        """
        super().__init__(params, alpha, beta, lip_const_default, max_num_bt_iterations)
        self._theta = 0.0
        self._lip_const = lip_const_default

    def _line_search_constant(self, params: List[torch.nn.Parameter],
                              closure: Callable) -> List[torch.nn.Parameter]:
        """
        Function which applies constant step size. If provided, prox map and projection map are applied

        :param params: List of trainable parameters
        :param closure: Loss function which does not need to be provided when applying constant
            step sizes
        :return: Altered parameter list
        """
        # NOTE
        #   > this call is required to accumulate gradients
        _ = closure()
        grad_list = [p.grad for p in params]
        for p, grad_p in zip(params, grad_list):
            p.data.sub_(self._alpha * grad_p)
            p = self._apply_prox_map(p, self._alpha)
            p = self._apply_projection(p)
        return params

    def _compute_quadratic_approximation(self, params_new: List[torch.Tensor], loss: torch.Tensor,
                                         params: List[torch.Tensor],
                                         grads: List[torch.Tensor]) -> torch.Tensor:
        """
        Function which computes the quadratic approximation of the loss function of the optimisation
        function. It is used when searching via backtracking for a step size which guarantees
        sufficient descent.

        :param params_new: List of new parameter candidates
        :param loss: Value of loss function at current parameters
        :param params: List of current parameters
        :param grads: List of gradient of current parameters
        :return: Quadratic approximation of loss.
        """
        quadr_approx = loss.clone()
        for p_new, p, grad_p in zip(params_new, params, grads):
            quadr_approx += (torch.sum(grad_p * (p_new.data - p.data)) +
                                 0.5 * self._lip_const * torch.sum((p_new.data - p.data) ** 2))
        return quadr_approx

    def _line_search_backtracking(self, params: List[torch.nn.Parameter],
                                  closure: Callable) -> List[torch.nn.Parameter]:
        """
        Function which performs line search via backtracking. Backtracking is performed such that
        sufficient descent (descent lemma) is obtained.

        :param params: List of trainable parameters
        :param closure: Closure function, corresponding to the loss function of the problem
        :return: Altered list of parameters
        """
        loss = closure()
        params_orig = [p.data.clone() for p in params]
        grads_orig = [p.grad.clone() for p in params]

        for k in range(0, self._max_num_bt_iterations):
            step_size = 1 / self._lip_const

            for p, grad_p in zip(params, grads_orig):
                p.data.sub_(step_size * grad_p)
                p = self._apply_prox_map(p, step_size)
                p = self._apply_projection(p)

            quadratic_approx = self._compute_quadratic_approximation(params, loss, params_orig, grads_orig)
            loss_new = closure()
            if loss_new <= quadratic_approx:
                self._lip_const *= 0.9
                break
            else:
                self._lip_const *= 2.0

                for p, p_orig in zip(params, params_orig):
                    p.data.copy_(p_orig)

        return params

    @torch.no_grad()
    def step(self, closure: Callable) -> torch.Tensor:
        """
        Function which performs update step.

        ------------------------

        # TODO
        #   > current implementation is alternating w.r.t to param_groups.
        #   > this should be considered in AlternatingNAGOptimiser - but not here!
        #   > Since currently in all the scripts of this project parameter groups were NOT used,
        #       it doesn't make a difference. Nevertheless: FIX ME!!!

        ------------------------

        :param closure: Loss function which is required for this optimisation procedure.
        :return: Loss after update step is performed
        """
        if not self._beta:
            theta_new = 0.5 * (1 + np.sqrt(1 + 4 * (self._theta ** 2)))
            beta = (self._theta - 1) / theta_new
            self._theta = theta_new
        else:
            beta = self._beta

        for group in self.param_groups:
            trainable_params = [p for p in group['params'] if p.requires_grad]
            for p in trainable_params:
                state = self.state[p]

                if 'history' not in state:
                    state['history'] = p.data.clone()

                momentum = p.data - state['history']
                state['history'] = p.data.clone()
                p.data.add_(beta * momentum)

            if self._alpha:
                trainable_params = self._line_search_constant(trainable_params, closure)
            else:
                trainable_params = self._line_search_backtracking(trainable_params, closure)

        return closure()

    def step_unroll(self, loss_func: Callable, num_iterations: int) -> torch.Tensor:

        params_unrolling = [[p.detach().clone().requires_grad_(True) for p in group['params'] if p.requires_grad]
                        for group in self.param_groups]

        for k in range(0, min(num_iterations, MAX_NUM_UNROLLING_ITER)):
            if not self._beta:
                theta_new = 0.5 * (1 + np.sqrt(1 + 4 * (self._theta ** 2)))
                beta = (self._theta - 1) / theta_new
                self._theta = theta_new
            else:
                beta = self._beta

            for group_idx, group in enumerate(params_unrolling):
                group_params_inter = []
                for p in group:
                    state = self.state[p]

                    if 'history' not in state:
                        state['history'] = p.data.clone()

                    momentum = p.data - state['history']
                    state['history'] = p.data.clone()
                    p = p + beta * momentum
                    group_params_inter.append(p)
                params_unrolling[group_idx] = group_params_inter

            params_unrolling_flat = [p for group in params_unrolling for p in group]
            y = loss_func(*params_unrolling_flat)
            grad_list = list(torch.autograd.grad(outputs=y, inputs=params_unrolling_flat, create_graph=True))

            # TODO
            #   > backtracking line search!

            idx = 0
            for group_idx, group in enumerate(self.param_groups):
                group_params = params_unrolling_flat[group_idx]
                group_size = len(group['params'])
                group_grads = grad_list[idx: idx + group_size]

                step_size = 1e-4
                tmp_param_list = []
                for p, grad in zip(group_params, group_grads):
                    p = p - step_size * grad
                    if hasattr(p, 'prox'):
                        p = p.prox(p.data, step_size)

                    if hasattr(p, 'proj'):
                        p = p.prox(p.data)

                    tmp_param_list.append(p)
                params_unrolling[group_idx] = tmp_param_list
                idx += group_size

        with torch.no_grad():
            for group, unrolled_group in zip(self.param_groups, params_unrolling):
                for p, new_p in zip(group['params'], unrolled_group):
                    p.copy_(new_p)

        return y