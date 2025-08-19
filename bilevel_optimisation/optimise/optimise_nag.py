import torch
from typing import List, Dict, Any, Callable, Optional, Tuple
import math

from bilevel_optimisation.data import OptimiserResult

MAX_NUM_ITERATIONS_DEFAULT = 1000
LIP_CONST_KEY = 'lip_const'
DEFAULTS_GROUP = {'alpha': 1e-4, 'beta': 0.71, 'theta': 0.0, LIP_CONST_KEY: 1e5, 'max_num_backtracking_iterations': 10}

def harmonise_param_groups_nag(param_groups: List[Dict[str, Any]], break_graph: bool=False) -> List[Dict[str, Any]]:
    param_groups_merged = []
    for group in param_groups:
        group_ = {'history': [p.detach().clone() for p in group['params']],
                  'name': group.get('name', '')}
        if break_graph:
            group_['params'] = []
            for p in group['params']:
                p_ = p.detach().clone().requires_grad_(True)
                if hasattr(p, 'prox'):
                    setattr(p_, 'prox', p.prox)
                if hasattr(p, 'proj'):
                    setattr(p_, 'proj', p.proj)
                group_['params'].append(p_)
        else:
            group_['params'] = [p for p in group['params']]

        group_['alpha'] = group['alpha']
        group_['beta'] = group['beta']
        if all(item is not None for item in group_['beta']):
            pass
        elif all(item is None for item in group_['beta']):
            group_['theta'] = DEFAULTS_GROUP['theta']
        else:
            group_['beta'] = [item if item is not None else DEFAULTS_GROUP['beta'] for item in group['beta']]

        if all(item is not None for item in group_['alpha']):
            pass
        elif all(item is None for item in group_['alpha']):
            group_[LIP_CONST_KEY] = [item if item is not None else DEFAULTS_GROUP[LIP_CONST_KEY]
                                   for item in group[LIP_CONST_KEY]]
        else:
            group_['alpha'] = [item if item is not None else DEFAULTS_GROUP['alpha'] for item in group['alpha']]

        group_['max_num_backtracking_iterations'] = group.get('max_num_backtracking_iterations',
                                                              DEFAULTS_GROUP['max_num_backtracking_iterations'])
        param_groups_merged.append(group_)
    return param_groups_merged

def flatten_groups(param_groups: List[Dict[str, Any]]) -> List[torch.Tensor]:
    return [p for group in param_groups for p in group['params']]

def compute_relative_error(param_groups: List[Dict[str, Any]], eps: float=1e-7) -> torch.Tensor:
    error = 0.0
    n = 0
    for group in param_groups:
        for p, p_old in zip(group['params'], group['history']):
            n += p.shape[0]
            error += torch.sum(torch.sqrt(torch.sum((p - p_old) ** 2, dim=(-2, -1)))
                               / torch.sqrt(torch.sum(p_old ** 2, dim=(-2, -1)) + eps))
    return error / n

def make_intermediate_step(group: Dict[str, Any], in_place: bool=True) -> Dict[str, Any]:
    """
    Function which computes the intermediate step of the Nesterov scheme.

    :param group: Parameter group to be updated
    :param in_place: Flag indicating if update shall be updated in place. For the unrolling scheme in place
        updates cannot be used, since computational graph would break.
    :return: Group of updated parameters.
    """
    param = group.get('params')[0]
    param_old = group.get('history')[0]

    if all(item is not None for item in group['beta']):
        beta = torch.tensor(group['beta'], dtype=param.dtype, device=param.device)
    else:
        theta = group['theta']
        theta_new = 0.5 * (1 + math.sqrt(1 + 4 * (theta ** 2)))
        beta = torch.tensor((theta - 1) / theta_new, dtype=param.dtype, device=param.device)
        group['theta'] = theta_new

    # make intermediate step
    momentum = beta.reshape(-1, *[1] * (param.dim() - 1)) * (param - param_old)
    if in_place:
        param_old.copy_(param.data)
        param.add_(momentum)
        return group
    else:
        param_old = param.detach().clone()
        param_new = param + momentum
        group_new = {**group, 'params': [param_new], 'history': [param_old]}
        return group_new

def make_gradient_step(param: torch.nn.Parameter, grads: torch.Tensor, alpha: torch.Tensor,
                       in_place: bool=True) -> torch.Tensor:
    """
    Function which performs a gradient step on param.

    NOTE
    ----
    > In case no in place update is made, projections will not be applied. This is because the non in-place update
        is required only for the unrolling scheme of the lower level problem, where no projections are needed.

    :param param: PyTorch parameter to be updated.
    :param grads: Gradient of objective function w.r.t. param
    :param alpha: Step size
    :param in_place: Flag indicating if update is performed in place. For updates within the unrolling scheme,
        updates can't be made in place.
    :return:
    """
    alpha_ = alpha.reshape(-1, *[1] * (param.dim() - 1))
    if in_place:
        param.sub_(alpha_ * grads)
        if hasattr(param, 'prox'):
            param.data.copy_(param.prox(param.data, alpha_))
        if hasattr(param, 'orthogonal_projection'):
            param.data.copy_(param.orthogonal_projection(param))
        if hasattr(param, 'zero_mean_projection'):
            param.data.copy_(param.zero_mean_projection(param))
    else:
        param = param - alpha_ * grads
        if hasattr(param, 'prox'):
            param = param.prox(param.data, alpha_)

    return param

def backtracking_line_search(param: torch.nn.Parameter, grads: torch.Tensor, closure: Callable,
                             lip_const: torch.Tensor, rho_1: float, rho_2: float,
                             max_num_iterations: int=10) -> torch.Tensor:
    """
    Function which performs backtracking line search for the non-unrolling update to find a step size giving
    sufficient descent.

    :param param: Parameter to be updated.
    :param grads: Gradient of objective function w.r.t. param
    :param closure: Closure function, which evaluates the loss function at the given parameters.
    :param lip_const: Current set of Lipschitz constants.
    :param rho_1: Decrease factor for Lipschitz constant
    :param rho_2: Increase factor for Lipschitz constant
    :param max_num_iterations: Maximal number of backtracking iterations.
    :return: Updated set of Lipschitz constants.
    """
    param_orig = param.data.clone()
    loss = closure()

    for k in range(0, max(1, max_num_iterations)):
        _ = make_gradient_step(param, grads, 1 / lip_const)

        loss_new = closure()
        quadr_approx = (loss + torch.sum(grads * (param - param_orig), dim=[-3, -2, -1])
                        + 0.5 * lip_const * torch.sum((param - param_orig) ** 2, dim=[-3, -2, -1]))
        sufficient_descent_met = loss_new <= quadr_approx

        param.data.copy_(param_orig)
        if sufficient_descent_met.all():
            break
        else:
            lip_const = torch.where(sufficient_descent_met, lip_const, lip_const * rho_2)

    _ = make_gradient_step(param, grads, 1 / lip_const)
    return torch.where(sufficient_descent_met, lip_const * rho_1, lip_const)

def step_nag(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]],
                   rho_1: float=0.9, rho_2: float=2.0) -> torch.Tensor:
    """
    Function which performs a single NAG or NAPG step.

    NOTE
    ----
    > Implementation assumes that parameters are given in terms of parameter groups, i.e. a list
        of dictionaries, where each dictionary represents a parameter group. Every parameter group
        must have the following key, value pairs:
            - 'params': List of PyTorch parameters
            - 'alpha': List of floats, or list of Nones indicating the line search strategy w.r.t. to
                batch-optimisation or per-sample optimisation. If a list of floats is specified, these values are
                used as constant step sizes; otherwise backtracking line search is applied.
            - 'beta': List of floats, or list of Nones representing the momentum parameters per optimisation
                variable (batch or samples). If it contains a None, Nesterov's momentum schedule
                is applied for all optimisation variables.
            - 'lip_const': List of floats, or list of Nones specifying the Lipschitz constants used in the
                backtracking line search scheme. If one the provided value is None, the default value is used.
        Projection operators, and the proximal operator are assumed to be given on parameter level in terms
        of one of the attributes 'zero_mean_projection', 'orthogonal_projection', 'prox'.
    > Optimisation procedure is alternating w.r.t. parameter groups. This means, that
        for each group, sequentially, gradients are computed and applied.
    > This implementation is feasible for solving the lower level and the upper level problem.

    :param func: Function representing the objective function
    :param grad_func: Function representing the gradient of func
    :param param_groups: Groups of parameters to be optimised.
    :param rho_1: Decrease factor for Lipschitz constant
    :param rho_2: Increase factor for Lipschitz constant
    :return: Loss at new iterate
    """
    for idx, group in enumerate(param_groups):
        _ = make_intermediate_step(group)

        params_flat = flatten_groups(param_groups)
        params_flat_ = [p.requires_grad_(True) for p in params_flat]

        # NOTE
        #   > Cloning the return value of grad_func is necessary to prevent
        #       the following error when applying torch.compile():
        #
        #           [...]
        #           RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
        #           by a subsequent run.
        #           [...]
        #           To prevent overwriting, clone the tensor outside of torch.compile() or call
        #           torch.compiler.cudagraph_mark_step_begin() before each model invocation.
        #
        grads = grad_func(*params_flat_)[idx].clone()

        param = group.get('params')[0]
        if all(item is not None for item in group['alpha']):
            # use constant step size(s)
            alpha = torch.tensor(group['alpha'], dtype=param.dtype, device=param.device)
            _ = make_gradient_step(param, grads, alpha)
        else:
            # apply backtracking line search
            def closure():
                return func(*flatten_groups(param_groups))

            lip_const = torch.tensor(group[LIP_CONST_KEY], dtype=param.dtype, device=param.device)
            lip_const = backtracking_line_search(param, grads, closure, lip_const, rho_1, rho_2,
                                                 group['max_num_backtracking_iterations'])
            group[LIP_CONST_KEY] = lip_const.tolist()

    return torch.sum(func(*flatten_groups(param_groups)))

@torch.no_grad()
def optimise_nag(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]],
                 max_num_iterations: int=MAX_NUM_ITERATIONS_DEFAULT,
                 rel_tol: Optional[float]=None, **unknown_options) -> OptimiserResult:
    """
    Implementation of Nesterov's accelerated gradient method.

    :param func: Callable representing the loss function
    :param grad_func: Callable representing the gradient of func
    :param param_groups: List of dictionaries, where each dictionary represents a parameter group. See step_nag_lower().
    :param max_num_iterations: Maximal number of iterations to be performed.
    :param rel_tol: Tolerance, which if provided, is used as early stopping criterion.
    :param unknown_options:
    :return: Instance of class OptimiserResult containing the computed the solution, the final loss, the
        number of iterations which were performed and a status message.
    """
    num_iterations = max_num_iterations
    param_groups_ = harmonise_param_groups_nag(param_groups, break_graph=True)

    loss = -1 * torch.ones(1)
    for k in range(0, max_num_iterations):
        loss = step_nag(func, grad_func, param_groups_)

        if rel_tol:
            rel_error = compute_relative_error(param_groups_)

            if rel_error <= rel_tol:
                num_iterations = k + 1
                break

    result = OptimiserResult(solution=param_groups_, num_iterations=num_iterations,
                             loss=loss)
    return result

@torch.no_grad()
def backtracking_line_search_unrolling(param_groups: List[Dict[str, Any]], group_idx: int, grads: torch.Tensor,
                                       closure: Callable, rho_1: float=0.9, rho_2: float=2.0,
                                       max_num_iterations: int=10) -> Tuple[torch.Tensor, torch.Tensor]:
    group = param_groups[group_idx]
    param = group.get('params')[0].clone()
    param_orig = group['params'][0].clone()

    loss = closure(param_groups)

    lip_const = torch.tensor(group[LIP_CONST_KEY], dtype=param.dtype, device=param.device)
    for k in range(0, max(1, max_num_iterations)):
        param_new = make_gradient_step(param, grads, 1 / lip_const, in_place=False)

        group_new = {**group, 'params': [param_new]}
        param_groups[group_idx] = group_new
        loss_new = closure(param_groups)

        quadr_approx = (loss + torch.sum(grads * (param_new - param_orig))
                        + 0.5 * lip_const * torch.sum((param_new - param_orig) ** 2))
        sufficient_descent_met = loss_new <= quadr_approx

        group_orig = {**group, 'params': [param_orig]}
        param_groups[group_idx] = group_orig
        if sufficient_descent_met.all():
            break
        else:
            lip_const = torch.where(sufficient_descent_met, lip_const, lip_const * rho_2)

    lip_const_new = torch.where(sufficient_descent_met, lip_const * rho_1, lip_const)

    return lip_const, lip_const_new

def optimise_nag_unrolling(func: Callable, grad_func: Callable, param_groups: List[Dict[str, Any]],
                           max_num_iterations: int=30, rel_tol: Optional[float]=None,
                           **unknown_options) -> OptimiserResult:
    """
    Implementation of the unrolling scheme for solving the lower problem using NAG or NAPG.

    :param func: Callable representing the objective function
    :param grad_func: Callable representing the gradient function of grad
    :param param_groups: List of dictionaries, where each dictionary represents a parameter group. See step_nag_lower().
    :param max_num_iterations: Maximal number of iterations to be performed. Per default only 30 iterations are
        performed.
    :param rel_tol: Tolerance, which if provided, is used for early stopping.
    :param unknown_options:
    :return: Instance of class OptimiserResult containing the computed the solution, the final loss, the
        number of iterations which were performed and a status message.
    """
    num_iterations = max_num_iterations
    param_groups_ = harmonise_param_groups_nag(param_groups, break_graph=False)

    param_groups_current = param_groups_
    for k in range(0, max_num_iterations):
        param_groups_new = []
        for idx, group in enumerate(param_groups_current):
            param_groups_current[idx] = make_intermediate_step(group, in_place=False)

            params_flat = flatten_groups(param_groups_current)
            params_flat_ = [p.requires_grad_(True) for p in params_flat]
            grads = grad_func(*params_flat_)[idx]

            param = group.get('params')[0]
            if all(item is not None for item in group['alpha']):
                alpha = torch.tensor(group['alpha'], dtype=param.dtype, device=param.device)
                param_new = make_gradient_step(param, grads, alpha, in_place=False)
                group_new = {**group, 'params': [param_new]}
            else:
                def closure(_param_groups: List[Dict[str, Any]]) -> torch.Tensor:
                    return func(*flatten_groups(_param_groups))

                lip_const, lip_const_new = backtracking_line_search_unrolling(param_groups_current, idx, grads,
                                                                              closure,
                                                                              max_num_iterations=
                                                                              group['max_num_backtracking_iterations'])
                param_new = make_gradient_step(param, grads, 1 / lip_const, in_place=False)
                group_new = {**group, 'params': [param_new], LIP_CONST_KEY: lip_const_new.tolist()}

            param_groups_new.append(group_new)
        param_groups_current = param_groups_new

        if rel_tol:
            rel_error = compute_relative_error(param_groups_current)

            if rel_error <= rel_tol:
                num_iterations = k + 1
                break

    return OptimiserResult(solution=param_groups_current, num_iterations=num_iterations,
                             loss=func(*flatten_groups(param_groups_current)))

