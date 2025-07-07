import torch
from typing import Optional, List, Dict, Any, Tuple, Callable

from bilevel_optimisation.data import LowerProblemResult, OptimiserResult
from bilevel_optimisation.energy import Energy
from bilevel_optimisation.optimise import optimise_nag, optimise_adam, optimise_nag_unrolling

def assemble_param_groups_base(u: torch.Tensor, batch_optimisation: bool) -> List[Dict[str, Any]]:
    param_groups = []
    if not batch_optimisation and u.shape[0] > 1:

        # TODO: vectorise me !!!

        for item in torch.split(u, split_size_or_sections=1, dim=0):
            group = {'params': [torch.nn.Parameter(item, requires_grad=True)]}
            param_groups.append(group)
    else:
        group = {'params': [torch.nn.Parameter(u, requires_grad=True)]}
        param_groups.append(group)
    return param_groups

def add_group_options(param_groups: List[Dict[str, Any]], group_options: Dict[str, List[Any]]) -> None:
    for group_idx, group in enumerate(param_groups):
        for key, values in group_options.items():
            if values[group_idx]:
                group[key] = values[group_idx]

def make_prox_map(u: torch.Tensor, noise_level: float) -> Callable:
    def prox_map(x: torch.Tensor, tau: float) -> torch.Tensor:
        return ((tau / noise_level ** 2) * u + x) / (1 + (tau / noise_level ** 2))
    return prox_map

def add_prox(param_groups: List[Dict[str, Any]], u: torch.Tensor, energy: Energy,
             batch_optimisation: bool) -> None:
    u_ = u.detach().clone()
    if batch_optimisation:
        prox_map = make_prox_map(u_, energy.measurement_model.noise_level)

        for group in param_groups:
            for p in group['params']:
                setattr(p, 'prox', prox_map)
    else:
        for idx, group in enumerate(param_groups):
            v = u_[idx: idx + 1, :, :, :].detach().clone()
            for p in group['params']:
                prox_map = make_prox_map(v, energy.measurement_model.noise_level)
                setattr(p, 'prox', prox_map)

def assemble_param_groups_nag(u: torch.Tensor, alpha: List[Optional[float]]=None,
                              beta: List[Optional[float]]=None,
                              lip_const: List[float]=None,
                              batch_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(u, batch_optimisation)

    alpha = [None for _ in range(u.shape[0])] if not alpha else alpha
    beta = [None for _ in range(u.shape[0])] if not beta else beta
    lip_const = [None for _ in range(u.shape[0])] if not lip_const else lip_const
    add_group_options(param_groups, {'alpha': alpha, 'beta': beta, 'lip_const': lip_const})

    return param_groups

def assemble_param_groups_adam(u: torch.Tensor, lr: List[Optional[float]]=None,
                               betas: List[Optional[Tuple[float, float]]]=None,
                               weight_decay: List[Optional[float]]=None,
                               batch_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(u, batch_optimisation)
    lr = [None for _ in range(0, len(param_groups))] if not lr else lr
    betas = [None for _ in range(0, len(param_groups))] if not betas else betas
    weight_decay = [None for _ in range(0, len(param_groups))] if not weight_decay else weight_decay
    add_group_options(param_groups, {'lr': lr, 'betas': betas, 'weight_decay': weight_decay})

    return param_groups

def parse_result(result: OptimiserResult, max_num_iterations: int, **unknown_options) -> LowerProblemResult:
    solution_tensor = build_solution_tensor_from_param_groups(result.solution)
    lower_prob_result = LowerProblemResult(solution=solution_tensor, num_iterations=result.num_iterations,
                                           loss=result.loss,
                                           message='Converged' if result.num_iterations < max_num_iterations
                                           else 'Max. number of iterations reached')
    return lower_prob_result

def build_solution_tensor_from_param_groups(param_groups: List[Dict[str, Any]]):
    solution_list = [group['params'][-1] for group in param_groups]
    return torch.cat(solution_list, dim=0)

def build_objective_func(energy: Energy, batch_optim: bool, use_prox: bool) -> Callable:
    if batch_optim and not use_prox:
        def func(x: torch.Tensor) -> torch.Tensor:
            return energy(x)
    elif batch_optim and use_prox:
        def func(x: torch.Tensor) -> torch.Tensor:
            return energy.lam * energy.regulariser(x)
    elif not batch_optim and not use_prox:
        def func(*x: torch.Tensor) -> torch.Tensor:
            return energy(torch.cat(x, dim=0))
    else:
        def func(*x: torch.Tensor) -> torch.Tensor:
            return energy.lam * energy.regulariser(torch.cat(x, dim=0))
    return func

def build_gradient_func(func: Callable, batch_optim: bool) -> Callable:
    if batch_optim:
        def grad_func(x: torch.Tensor) -> List[torch.Tensor]:
            with torch.enable_grad():
                x_ = x.detach().clone().requires_grad_(True)
                loss = func(x_)
            return list(torch.autograd.grad(outputs=loss, inputs=[x_]))
    else:
        def grad_func(*x: torch.Tensor):
            with torch.enable_grad():
                x = [x_.detach().clone().requires_grad_(True) for x_ in x]
                loss = func(*x)
            return list(torch.autograd.grad(outputs=loss, inputs=x, create_graph=True))

    return grad_func

def solve_lower(energy: Energy, method: str,
                options: Dict[str, Any], noisy_obs: Optional[torch.Tensor]=None) -> LowerProblemResult:
    noisy_obs = noisy_obs if noisy_obs is not None else energy.measurement_model.get_noisy_observation()
    u_ = noisy_obs.detach().clone().requires_grad_(True)

    batch_optim = options.get('batch_optimisation', True)

    # TODO:
    #   > introduce some sanity check w.r.t. parameters!

    if method == 'nag':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim=batch_optim)
        param_groups = assemble_param_groups_nag(u_, **options)

        nag_result = optimise_nag(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'napg':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=True)
        grad_func = build_gradient_func(func, batch_optim=batch_optim)
        param_groups = assemble_param_groups_nag(u_, **options)
        add_prox(param_groups, u_, energy, batch_optim)

        nag_result = optimise_nag(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'adam':
        param_groups = assemble_param_groups_adam(u_, **options)
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)

        adam_result = optimise_adam(func, param_groups, **options)
        lower_prob_result = parse_result(adam_result, **options)
    elif method == 'nag_unrolling':
        # Only for testing purposes ...
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim)
        param_groups = assemble_param_groups_nag(u_, **options)

        nag_result = optimise_nag_unrolling(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)

    elif method == 'napg_unrolling':
        # Only for testing purposes ...
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=True)
        grad_func = build_gradient_func(func, batch_optim)
        param_groups = assemble_param_groups_nag(u_, **options)
        add_prox(param_groups, u_, energy, batch_optim)

        nag_result = optimise_nag_unrolling(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'your_custom_method':
        # Custom method for solving the lower problem or by map
        # estimation of by mmse estimation goes here. Use the following
        # structure
        #
        #       solution = solution_method(...)
        #       lower_prob_result = LowerProblemResult(...)
        pass
    else:
        raise NotImplementedError

    return lower_prob_result


