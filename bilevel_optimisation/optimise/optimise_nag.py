import torch
from typing import List, Dict, Any, Callable, Optional
import math

from bilevel_optimisation.data import OptimiserResult
from bilevel_optimisation.data.Constants import EPSILON, MAX_NUM_UNROLLING_ITER

DEFAULTS = {'alpha': None, 'beta': None, 'theta': 0.0, 'lip_const': 1e5,
            'max_num_backtracking_iterations': 10, 'proj': None, 'prox': None}

def compute_momentum_parameter(param_group: Dict[str, Any]) -> float:
    if param_group['beta']:
        beta = param_group['beta']
    else:
        theta = param_group['theta']

        theta_new = 0.5 * (1 + math.sqrt(1 + 4 * (theta ** 2)))
        beta = (theta - 1) / theta_new
        param_group['theta'] = theta_new

    return beta

def harmonise_param_groups_nag(param_groups: List[Dict[str, Any]], break_graph: bool=False) -> List[Dict[str, Any]]:
    param_groups_merged = []
    for group in param_groups:
        group_ = {'history': [p.detach().clone() for p in group['params']]}
        if break_graph:
            group_['params'] = [p.detach().clone().requires_grad_(True) for p in group['params']]
        else:
            group_['params'] = [p for p in group['params']]

        for key, value in DEFAULTS.items():
            group_[key] = group.get(key, value)
        param_groups_merged.append(group_)
    return param_groups_merged

def make_intermediate_step(param_group: Dict[str, Any], in_place: bool) -> None:
    beta = compute_momentum_parameter(param_group)
    if in_place:
        for p, p_old in zip([p_ for p_ in param_group['params']], [p_ for p_ in param_group['history']]):
            momentum = p.data - p_old.data
            p_old.data.copy_(p.data)
            p.data.add_(beta * momentum)
    else:
        params_new = []
        history_new = []
        for p, p_old in zip([p_ for p_ in param_group['params']], [p_ for p_ in param_group['history']]):
            history_new.append(p.detach().clone())

            momentum = p - p_old
            params_new.append(p + beta * momentum)
        param_group['params'] = params_new
        param_group['history'] = history_new

def make_gradient_step(param_group: Dict[str, Any], param_group_grads: List[torch.Tensor],
                       step_size: float, in_place: bool) -> None:
    if in_place:
        for p, grad_p in zip(param_group['params'], param_group_grads):
            p.data.sub_(step_size * grad_p)
            if param_group['prox']:
                p.data.copy_(param_group['prox'](p.data, step_size))
            if param_group['proj']:
                p.data.copy_(param_group['proj'](p.data))
    else:
        params_new = []
        for p, grad_p in zip(param_group['params'], param_group_grads):
            p = p - step_size * grad_p
            if param_group['prox']:
                p = param_group['prox'](p, step_size)
            if param_group['proj']:
                p = param_group['proj'](p)
            params_new.append(p)
        param_group['params'] = params_new

def flatten_groups(param_groups: List[Dict[str, Any]]) -> List[torch.Tensor]:
    return [p for group in param_groups for p in group['params']]

def compute_quadratic_approximation(param_group_new: Dict[str, Any],
                                    param_group: Dict[str, Any],
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
    for p_new, p, grad_p in zip(param_group_new['params'], param_group['params'], grad_group_list):
        quadr_approx += (torch.sum(grad_p * (p_new.data - p.data)) +
                         0.5 * lip_const * torch.sum((p_new.data - p.data) ** 2))
    return quadr_approx

def copy_param_groups_partial(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    param_groups_ = []
    for group in param_groups:
        group_ = {'params': [p.detach().clone() for p in group['params']],
                  'prox': group['prox'],
                  'proj': group['proj']}
        param_groups_.append(group_)
    return param_groups_

def apply_backtracking(func: Callable, param_groups: List[Dict[str, Any]], group_idx: int,
                       group_grads: List[torch.Tensor], in_place: bool) -> None:
    param_groups_orig = copy_param_groups_partial(param_groups)
    params_flat_orig = flatten_groups(param_groups_orig)
    loss = func(*params_flat_orig)

    for k in range(0, param_groups[group_idx]['max_num_backtracking_iterations']):
        lip_const = param_groups[group_idx]['lip_const']
        step_size = 1 / lip_const
        make_gradient_step(param_groups[group_idx], group_grads, step_size, in_place)

        params_flat_ = flatten_groups(param_groups)
        loss_new = func(*params_flat_)

        quadratic_approx = compute_quadratic_approximation(param_groups[group_idx], param_groups_orig[group_idx],
                                                           group_grads, loss, lip_const)

        if loss_new <= quadratic_approx:
            param_groups[group_idx]['lip_const'] *= 0.9
            break
        else:
            param_groups[group_idx]['lip_const'] *= 2.0
            param_groups[group_idx]['params'] = [p_orig.detach().clone().requires_grad_(True)
                                                 for p_orig in param_groups_orig[group_idx]['params']]

def compute_relative_error(param_groups: List[Dict[str, Any]]) -> torch.Tensor:
    error = 0.0
    nrm = 0.0
    for group in param_groups:
        for p, p_old in zip(group['params'], group['history']):
            error += torch.sum((p - p_old) ** 2)
            nrm += torch.sum(p ** 2)

    return torch.sqrt(error) / (torch.sqrt(nrm) + EPSILON)

def step_nag(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]], in_place: bool=True) -> torch.Tensor:
    loss = func(*flatten_groups(param_groups))
    grad_idx = 0
    for group_idx, group in enumerate(param_groups):
        make_intermediate_step(group, in_place)

        params_flat = flatten_groups(param_groups)

        # with torch.enable_grad():
        #     f = func(*params_flat)
        # grads = torch.autograd.grad(outputs=f, inputs=params_flat)

        grads = grad_func(*params_flat)
        group_grads = grads[grad_idx: grad_idx + len(group['params'])]

        if group['alpha']:
            make_gradient_step(group, group_grads, group['alpha'], in_place)
        else:
            apply_backtracking(func, param_groups, group_idx, group_grads, in_place)
        grad_idx += len(group['params'])

    return loss


@torch.no_grad()
def optimise_nag(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]], max_num_iterations: int=1000,
                 rel_tol: Optional[float]=None, **unknown_options) -> OptimiserResult:
    num_iterations = max_num_iterations
    param_groups_ = harmonise_param_groups_nag(param_groups)

    for k in range(0, max_num_iterations):
        step_nag(func, grad_func, param_groups_)

        if rel_tol:
            rel_error = compute_relative_error(param_groups_)
            if rel_error <= rel_tol:
                num_iterations = k + 1
                break

    result = OptimiserResult(solution=param_groups_, num_iterations=num_iterations,
                             loss=func(*flatten_groups(param_groups_)))



    return result

def optimise_nag_unrolling(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]],
                           max_num_iterations: int=30, rel_tol: Optional[float]=None,
                           **unknown_options) -> OptimiserResult:
    num_iterations = max_num_iterations
    param_groups_ = harmonise_param_groups_nag(param_groups, break_graph=True)

    for k in range(0, max_num_iterations):
        for group_idx, group in enumerate(param_groups_):
            make_intermediate_step(param_groups_[group_idx], in_place=False)

        param_groups_flat = flatten_groups(param_groups_)
        grads = grad_func(*param_groups_flat)

        grad_idx = 0
        for group_idx, group in enumerate(param_groups_):
            group_grads = list(grads[grad_idx: grad_idx + len(group)])
            if group['alpha']:
                make_gradient_step(group, group_grads, group['alpha'], in_place=False)
            else:
                apply_backtracking(func, param_groups_, group_idx, group_grads, in_place=False)

            grad_idx += len(group['params'])

        if rel_tol:
            rel_error = compute_relative_error(param_groups_)
            if rel_error <= rel_tol:
                num_iterations = k + 1
                break

    return OptimiserResult(solution=param_groups_, num_iterations=num_iterations,
                             loss=func(*flatten_groups(param_groups_)))




# def optimise_nag_unrolling(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]],
#                            max_num_iterations: int=50, num_unrolling_iterations: int=10,
#                            rel_tol: float=None, **unknown_options) -> OptimiserResult:
#     num_iterations = max_num_iterations
#     param_groups_ = harmonise_param_groups_nag(param_groups)
#
#     for k in range(0, max_num_iterations):
#         # step_nag_unrolling(func, grad_func, param_groups_, num_unrolling_iterations)
#         for l in range(0, min(num_unrolling_iterations, MAX_NUM_UNROLLING_ITER)):
#             grad_idx = 0
#             for group_idx, group in enumerate(param_groups_):
#                 make_intermediate_step(group, in_place=False)
#
#                 params_flat = flatten_groups(param_groups_)
#                 grads = grad_func(*params_flat)
#                 group_grads = grads[grad_idx: grad_idx + len(group['params'])]
#                 if group['alpha']:
#                     make_gradient_step(group, group_grads, group['alpha'], in_place=False)
#                 else:
#                     apply_backtracking(func, param_groups_, group_idx, group_grads, in_place=False)
#                 grad_idx += len(group['params'])
#
#         if rel_tol:
#             rel_error = compute_relative_error(param_groups_)
#             if rel_error <= rel_tol:
#                 num_iterations = k + 1
#                 break
#
#     result = OptimiserResult(solution=param_groups_, num_iterations=num_iterations,
#                              loss=func(*flatten_groups(param_groups_)))
#
#     return result