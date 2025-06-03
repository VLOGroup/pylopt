from typing import Union, Iterable, Dict, Any, Callable
import torch
import numpy as np
import logging

from bilevel_optimisation.optimiser.NAGOptimiser import BaseNAGOptimiser

class AlternatingNAGOptimiser(BaseNAGOptimiser):
    """
    Subclass of BaseNAGOptimiser implementing alternating NAG optimisation, i.e. the NAG step
    is applied per parameter independently of all the parameters:
        > Backtracking is applied parameter-wise according to the order parameters are provided during
            initialisation of the optimiser
        > If parameters have attributes 'alpha' and/or 'beta', they will be used to perform update step (step size,
            and momentum respectively). If not, the values specified during initialisation are used as fallback.
    """
    def __init__(self,  params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 alpha: float = None, beta: float = None, lip_const_default: float = 2**10,
                 max_num_bt_iterations: int = 10) -> None:
        """
        Initialisation of class AlternatingNAGOptimiser

        :param params:
        :param alpha:
        :param beta:
        :param lip_const_default:
        :param max_num_bt_iterations:
        """
        super().__init__(params, alpha, beta, lip_const_default, max_num_bt_iterations)

    def _line_search_backtracking(self, p: torch.nn.Parameter, closure: Callable) -> torch.nn.Parameter:
        loss = closure()
        grad = p.grad.clone()
        state = self.state[p]
        p_data_orig = p.data.clone()

        if 'lip_const' in state:
            lip_const = state['lip_const']
        else:
            state['lip_const'] = self._lip_const_default
            lip_const = self._lip_const_default

        for k in range(0, self._max_num_bt_iterations):
            step_size = 1 / lip_const
            p.data.sub_(step_size * grad)
            p = self._apply_prox_map(p, step_size)
            p = self._apply_projection(p)

            loss_new = closure()
            quadratic_approx = (loss + torch.sum(grad * (p.data - p_data_orig)) +
                                0.5 * lip_const * torch.sum((p.data - p_data_orig) ** 2))
            if loss_new <= quadratic_approx:
                lip_const *= 0.9
                break
            else:
                lip_const *= 2.0
                p.data = p_data_orig.clone()

        self.state[p]['lip_const'] = lip_const
        return p

    def _line_search_constant(self, p: torch.nn.Parameter, closure: Callable) -> torch.nn.Parameter:
        _ = closure()
        grad = p.grad.clone()
        if hasattr(p, 'alpha'):
            p.data.sub_(p.alpha * grad)
            p = self._apply_prox_map(p, p.alpha)
            p = self._apply_projection(p)
        else:
            p.data.sub_(self._alpha * grad)
            p = self._apply_prox_map(p, self._alpha)
            p = self._apply_projection(p)
        return p

    def _get_momentum_parameter(self, p: torch.nn.Parameter) -> float:
        state = self.state[p]
        if self._beta:
            beta = self._beta
        elif hasattr(p, 'beta'):
            beta = p.beta
        else:
            if 'theta' not in state:
                state['theta'] = 0.0
            theta_new = 0.5 * (1 + np.sqrt(1 + 4 * (state['theta'] ** 2)))
            beta = (state['theta'] - 1) / theta_new
            state['theta'] = theta_new
        return beta

    @torch.no_grad()
    def step(self, closure: Callable) -> float:
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue

                state = self.state[p]
                if 'history' not in state:
                    state['history'] = p.data.clone()

                beta = self._get_momentum_parameter(p)
                momentum = p.data - state['history']
                state['history'] = p.data.clone()
                p.data.add_(beta * momentum)

                if self._alpha:
                    p = self._line_search_constant(p, closure)
                else:
                    p = self._line_search_backtracking(p, closure)

        return closure()

