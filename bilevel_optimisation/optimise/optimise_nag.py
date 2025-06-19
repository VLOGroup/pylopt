import torch
from typing import List, Dict, Any, Callable
import math
import logging

from bilevel_optimisation.data import OptimiserResult
from bilevel_optimisation.data.Constants import EPSILON

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

def harmonise_groups(param_groups):
    param_groups_merged = []
    for group in param_groups:
        group_ = {'params': group['params'], 'history': [p.data.clone() for p in group['params']]}
        for key, value in DEFAULTS.items():
            group_[key] = group.get(key, value)
        param_groups_merged.append(group_)
    return param_groups_merged

def make_intermediate_step(param_group: Dict[str, Any]) -> None:
    beta = compute_momentum_parameter(param_group)
    for p, p_old in zip([p_ for p_ in param_group['params']], [p_ for p_ in param_group['history']]):
        momentum = p.data - p_old.data
        p_old.data.copy_(p.data)
        p.data.add_(beta * momentum)

def make_gradient_step(param_group: Dict[str, Any], param_group_grads: List[torch.Tensor], step_size) -> None:
    for p, grad_p in zip(param_group['params'], param_group_grads):
        p.data.sub_(step_size * grad_p)
        if param_group['prox']:
            p.data.copy_(param_group['prox'](p.data, step_size))
        if param_group['proj']:
            p.data.copy_(param_group['proj'](p.data))

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

def copy_param_groups_partial(param_groups):
    param_groups_ = []
    for group in param_groups:
        group_ = {'params': [p.data.clone() for p in group['params']],
                  'prox': group['prox'],
                  'proj': group['proj']}
        param_groups_.append(group_)
    return param_groups_

def apply_backtracking(func: Callable, param_groups: List[Dict[str, Any]], group_idx: int,
                       group_grads: List[torch.Tensor]) -> None:
    params_flat = flatten_groups(param_groups)
    loss = func(*params_flat)

    param_groups_orig = copy_param_groups_partial(param_groups)

    for k in range(0, param_groups[group_idx]['max_num_backtracking_iterations']):
        lip_const = param_groups[group_idx]['lip_const']
        step_size = 1 / lip_const
        make_gradient_step(param_groups[group_idx], group_grads, step_size)

        params_flat_ = flatten_groups(param_groups)
        loss_new = func(*params_flat_)

        quadratic_approx = compute_quadratic_approximation(param_groups[group_idx], param_groups_orig[group_idx],
                                                           group_grads, loss, lip_const)

        if loss_new <= quadratic_approx:
            param_groups[group_idx]['lip_const'] *= 0.9
            break
        else:
            param_groups[group_idx]['lip_const'] *= 2.0
            for p, p_orig in zip(param_groups[group_idx]['params'], param_groups_orig[group_idx]['params']):
                p.data.copy_(p_orig)

def compute_relative_error(param_groups: List[Dict[str, Any]]):
    error = 0.0
    nrm = 0.0
    for group in param_groups:
        for p, p_old in zip(group['params'], group['history']):
            error += torch.sum((p - p_old) ** 2)
            nrm += torch.sum(p ** 2)

    return torch.sqrt(error) / (torch.sqrt(nrm) + EPSILON)


@torch.no_grad()
def optimise_nag(func, grad_func, param_groups: List[Dict[str, Any]], max_num_iterations: int,
                 rel_tol: float = None) -> OptimiserResult:
    num_iterations = -1
    param_groups_ = harmonise_groups(param_groups)

    for k in range(0, max_num_iterations):
        grad_idx = 0
        for group_idx, group in enumerate(param_groups_):
            make_intermediate_step(group)

            params_flat = flatten_groups(param_groups_)
            grads = grad_func(*params_flat)
            group_grads = grads[grad_idx: grad_idx + len(group['params'])]
            if group['alpha']:
                make_gradient_step(group, group_grads, group['alpha'])
            else:
                apply_backtracking(func, param_groups_, group_idx, group_grads)
            grad_idx += len(group['params'])

        if rel_tol:
            rel_error = compute_relative_error(param_groups_)
            if rel_error <= rel_tol:
                num_iterations = k + 1
                break

    result = OptimiserResult(solution=param_groups_, num_iterations=num_iterations,
                             loss=func(*flatten_groups(param_groups_)))

    return result