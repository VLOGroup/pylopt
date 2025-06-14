import torch
from typing import Union, Iterable, Dict, Any, List, Callable, Tuple

from bilevel_optimisation.data.Constants import MAX_NUM_UNROLLING_ITER
from bilevel_optimisation.optimiser.BaseNAGOptimiser import BaseNAGOptimiser

class UnrollingNAGOptimiser(BaseNAGOptimiser):

    def __init__(self, params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 alpha: float = None, beta: float = None, lip_const: float = 10000000,
                 max_num_bt_iterations: int = 10) -> None:
        super().__init__(params, alpha, beta, lip_const, max_num_bt_iterations)

    @torch.no_grad()
    def step(self, closure: Callable) -> torch.Tensor:
        raise NotImplementedError('Function step() is not implemented for unrolling-type optimisers')

    def _make_intermediate_step(self, param_group_flat: List[torch.nn.Parameter], beta: float) -> List[torch.nn.Parameter]:
        group_params_inter = []
        for p in param_group_flat:
            state = self.state[p]
            if 'history' not in state:
                state['history'] = p.data.clone()

            momentum = p.data - state['history']
            state['history'] = p.data.clone()

            p = p + beta * momentum
            group_params_inter.append(p)
        return group_params_inter

    @staticmethod
    def _compute_gradients(params_grouped: List[List[torch.nn.Parameter]], loss_func: Callable) -> List[torch.Tensor]:
        params_grouped_flat = [p for group in params_grouped for p in group]
        with torch.enable_grad():
            loss = loss_func(*params_grouped_flat)
        grad_list = list(torch.autograd.grad(outputs=loss, inputs=params_grouped_flat, create_graph=True))
        return grad_list

    def _apply_gradient_step(self, param_group_flat: List[torch.nn.Parameter],
                              grad_group_list: List[torch.Tensor], step_size: float):
        updated_param_list = []
        for p, grad in zip(param_group_flat, grad_group_list):
            p = p - step_size * grad
            self._apply_prox_map(p, step_size)
            self._apply_projection(p)

            updated_param_list.append(p)
        return updated_param_list

    @torch.no_grad()
    def _perform_backtracking(self, params_grouped: List[List[torch.nn.Parameter]],
                              grad_group_list: List[torch.Tensor], group_idx: int,
                              loss_func: Callable) -> List[torch.Tensor]:
        params_grouped_ = [[p.detach().clone() for p in group] for group in params_grouped]
        params_grouped_orig_flat = [p for group in params_grouped for p in group]
        loss = loss_func(*params_grouped_orig_flat)

        for k in range(0, self._max_num_bt_iterations):
            step_size = 1 / self.param_groups[group_idx]['lip_const']

            params_grouped_[group_idx] = self._apply_gradient_step(params_grouped_[group_idx],
                                                                   grad_group_list, step_size)
            params_grouped_flat = [p for group in params_grouped_ for p in group]

            loss_new = loss_func(*params_grouped_flat)
            quadratic_approx = self._compute_quadratic_approximation(params_grouped_flat, params_grouped_orig_flat,
                                                                     grad_group_list, loss,
                                                                     self.param_groups[group_idx]['lip_const'])

            if loss_new <= quadratic_approx:
                self.param_groups[group_idx]['lip_const'] *= 0.9
                break
            else:
                self.param_groups[group_idx]['lip_const'] *= 2.0
                params_grouped_ = [[p.detach().clone() for p in group] for group in params_grouped]

        params_grouped[group_idx] = self._apply_gradient_step(params_grouped[group_idx],
                                                              grad_group_list, step_size)

        return params_grouped[group_idx]

    @torch.enable_grad()
    def step_unroll(self, loss_func: Callable, num_iterations: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        params_grouped = [[p.detach().clone().requires_grad_(True) for p in group['params'] if p.requires_grad]
                           for group in self.param_groups]

        dtype = [p for group in self.param_groups for p in group['params'] if p.requires_grad][0].dtype
        loss = torch.tensor(-1.0, dtype=dtype)
        for k in range(0, min(num_iterations, MAX_NUM_UNROLLING_ITER)):

            idx = 0
            for group_idx, group in enumerate(self.param_groups):
                group_size = len(params_grouped[group_idx])

                beta = self._momentum_param(group)

                params_grouped[group_idx] = self._make_intermediate_step(params_grouped[group_idx], beta)
                grad_list = self._compute_gradients(params_grouped, loss_func)
                grad_group_list = grad_list[idx: idx + group_size]

                if group['alpha']:
                    params_grouped[group_idx] = self._apply_gradient_step(params_grouped[group_idx],
                                                                          grad_group_list, group['alpha'])
                else:
                    params_grouped[group_idx] = self._perform_backtracking(params_grouped, grad_group_list, group_idx,
                                                                           loss_func)
                idx += group_size
        return loss, [p for group in params_grouped for p in group]
