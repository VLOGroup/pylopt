import torch
from typing import Optional, List, Dict, Any, Tuple, Callable

from bilevel_optimisation.data import LowerProblemResult, OptimiserResult
from bilevel_optimisation.energy import Energy
from bilevel_optimisation.measurement_model import MeasurementModel
from bilevel_optimisation.optimise import optimise_nag, optimise_adam, optimise_nag_unrolling

def add_group_options(param_groups: List[Dict[str, Any]], group_options: Dict[str, List[Any]]) -> None:
    for key, values in group_options.items():
        param_groups[0][key] = values

def make_prox_map(prox_operator: torch.nn.Module, u: torch.Tensor) -> Callable:
    def prox_map(x: torch.Tensor, tau: float) -> torch.Tensor:
        return prox_operator(x, tau, u)
    return prox_map

def add_prox(prox_operator: Optional[torch.nn.Module], param_groups: List[Dict[str, Any]], u: torch.Tensor) -> None:
    u_ = u.detach().clone()
    prox_map = make_prox_map(prox_operator, u_)
    for group in param_groups:
        for p in group['params']:
            setattr(p, 'prox', prox_map)

def assemble_param_groups_nag(u: torch.Tensor, alpha: List[Optional[float]]=None,
                              beta: List[Optional[float]]=None, lip_const: List[float]=None,
                              **unknown_options) -> List[Dict[str, Any]]:
    u_ = u.detach().clone()
    param_groups = [{'params': [torch.nn.Parameter(u_, requires_grad=True)]}]

    alpha = [None for _ in range(u.shape[0])] if not alpha else alpha
    beta = [None for _ in range(u.shape[0])] if not beta else beta
    lip_const = [None for _ in range(u.shape[0])] if not lip_const else lip_const

    add_group_options(param_groups, {'alpha': alpha, 'beta': beta, 'lip_const': lip_const})
    return param_groups

def assemble_param_groups_adam(u: torch.Tensor, lr: List[Optional[float]]=None,
                               betas: List[Optional[Tuple[float, float]]]=None,
                               weight_decay: List[Optional[float]]=None,
                               batch_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    u_ = u.detach().clone()
    param_groups = []
    if batch_optimisation:
        group = {'params': [torch.nn.Parameter(u_, requires_grad=True)]}
        param_groups.append(group)
    else:
        for i in range(0, u_.shape[0]):
            group = {'params': [torch.nn.Parameter(u_[i: i + 1, :, :, :].detach().clone(), requires_grad=True)]}
            param_groups.append(group)

    lr = [None for _ in range(0, len(param_groups))] if not lr else lr
    betas = [(None, None) for _ in range(0, len(param_groups))] if not betas else betas
    weight_decay = [None for _ in range(0, len(param_groups))] if not weight_decay else weight_decay

    for group, lr_, betas_, weight_decay_ in zip(param_groups, lr, betas, weight_decay):
        group['lr'] = lr_
        group['betas'] = betas_
        group['weight_decay'] = weight_decay_

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
        operator = energy.measurement_model.operator
        noise_level = energy.measurement_model.noise_level

        regulariser = energy.regulariser
        lam = energy.lam

        per_sample_energy_models = []
        u_clean = energy.measurement_model.u_clean
        for i in range(u_clean.shape[0]):
            sample_measurement_model = MeasurementModel(u_clean[i:i + 1, :, :, :], operator=operator,
                                                        noise_level=noise_level)
            sample_energy = Energy(sample_measurement_model, regulariser, lam)
            per_sample_energy_models.append(sample_energy)

        # RECALL
        #   > torch.vmap cannot be used for a list of different maps.
        def func(*x: torch.Tensor) -> torch.Tensor:
            return torch.stack([sample_energy_model(sample.unsqueeze(dim=0)) for sample_energy_model, sample in
                                zip(per_sample_energy_models, torch.cat(x, dim=0))])

        return func
    else:
        regulariser = energy.regulariser
        lam = energy.lam
        def per_sample_objective_func(x) -> torch.Tensor:
            return lam * regulariser(x.unsqueeze(dim=0))

        def func(*x: torch.Tensor) -> torch.Tensor:
            return torch.vmap(per_sample_objective_func)(torch.cat(x, dim=0))

    return func

def build_gradient_func(func: Callable, batch_optim: bool, unrolling: bool=False) -> Callable:
    if batch_optim:
        def grad_func(x: torch.Tensor) -> List[torch.Tensor]:
            with torch.enable_grad():
                x_ = x.detach().clone().requires_grad_(True)
                loss = func(x_)
            return list(torch.autograd.grad(outputs=loss, inputs=[x_], create_graph=unrolling))
    else:
        def grad_func(*x: torch.Tensor):
            with torch.enable_grad():
                x = [x_.detach().clone().requires_grad_(True) for x_ in x]
                loss = torch.sum(func(*x), dim=0)
            return list(torch.autograd.grad(outputs=loss, inputs=x, create_graph=unrolling))

    return grad_func

def solve_lower(energy: Energy, method: str,
                options: Dict[str, Any]) -> LowerProblemResult:
    batch_optim = options.get('batch_optimisation', True)

    if method == 'nag':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim=batch_optim)

        param_groups = assemble_param_groups_nag(energy.measurement_model.get_noisy_observation().detach().clone(),
                                                 **options)

        nag_result = optimise_nag(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'napg':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=True)
        grad_func = build_gradient_func(func, batch_optim=batch_optim)

        u_noisy = energy.measurement_model.get_noisy_observation().detach().clone()
        param_groups = assemble_param_groups_nag(u_noisy, **options)

        prox_operator = options.get('prox', None)
        add_prox(prox_operator, param_groups, u_noisy)

        nag_result = optimise_nag(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'adam':
        param_groups = assemble_param_groups_adam(energy.measurement_model.get_noisy_observation().detach().clone(),
                                                  **options)

        func = lambda *z: torch.sum(build_objective_func(energy, batch_optim=batch_optim, use_prox=False)(torch.cat(z, dim=0)))

        adam_result = optimise_adam(func, param_groups, **options)
        lower_prob_result = parse_result(adam_result, **options)
    elif method == 'nag_unrolling':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim, unrolling=True)
        param_groups = assemble_param_groups_nag(u_, **options)

        nag_result = optimise_nag_unrolling(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'napg_unrolling':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=True)
        grad_func = build_gradient_func(func, batch_optim, unrolling=True)
        param_groups = assemble_param_groups_nag(u_, **options)

        prox_operator = options.get('prox', None)
        add_prox(prox_operator, param_groups, u_, energy, batch_optim)

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


