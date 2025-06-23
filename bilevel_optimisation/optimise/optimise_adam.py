import torch
from typing import Dict, List, Callable, Any

from bilevel_optimisation.data import OptimiserResult
from bilevel_optimisation.optimise.optimise_nag import flatten_groups, compute_relative_error

DEFAULTS = {'proj': None}

def create_projected_optimiser(base_optimiser: type[torch.optim.Optimizer]) -> type[torch.optim.Optimizer]:

    class ProjectedOptimiser(base_optimiser):
        def __init__(self, params, *args, **kwargs):
            super().__init__(params, *args, **kwargs)

        def step(self, closure=None):
            loss = super().step(closure)

            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if not p.requires_grad:
                            continue

                        if group['proj']:
                            p.data.copy_(group['proj'](p.data))

            return loss

    ProjectedOptimiser.__name__ = 'Projected{:s}'.format(base_optimiser.__name__)
    return ProjectedOptimiser

def harmonise_param_groups_adam(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    param_groups_ = []
    for group in param_groups:
        group_ = {'params': [p for p in group['params']],
                  'history': [p.detach().clone().requires_grad_(True) for p in group['params']]}
        for key in group.keys():
            if key != 'params':
                group_[key] = group[key]
        for key, value in DEFAULTS.items():
            group_[key] = group.get(key, value)
        param_groups_.append(group_)
    return param_groups_

def step_adam(optimiser: torch.optim.Optimizer, func: Callable, param_groups: List[Dict[str, Any]]) -> torch.Tensor:
    for group in param_groups:
        for p, p_old in zip([p_ for p_ in group['params']], [p_ for p_ in group['history']]):
            p_old.data.copy_(p.data.clone())

    optimiser.zero_grad()
    params_flat = flatten_groups(param_groups)
    with torch.enable_grad():
        loss = func(*params_flat)
    loss.backward()
    optimiser.step()

    print(compute_relative_error(param_groups))

    return loss

def optimise_adam(func: Callable, param_groups: List[Dict[str, Any]], max_num_iterations: int,
                  rel_tol: float, **unknown_options) -> OptimiserResult:
    num_iterations = max_num_iterations

    param_groups_ = harmonise_param_groups_adam(param_groups)
    optimiser = create_projected_optimiser(torch.optim.Adam)(param_groups_)

    for k in range(0, max_num_iterations):
        _ = step_adam(optimiser, func, param_groups_)

        if rel_tol:
            rel_error = compute_relative_error(param_groups_)
            if rel_error <= rel_tol:
                num_iterations = k + 1
                break

    result = OptimiserResult(solution=param_groups_, num_iterations=num_iterations,
                             loss=func(*flatten_groups(param_groups_)))

    return result


