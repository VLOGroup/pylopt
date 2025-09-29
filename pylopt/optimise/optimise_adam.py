import torch
from typing import Dict, List, Callable, Any

from pylopt.data import OptimiserResult
from pylopt.optimise.optimise_nag import flatten_groups, compute_relative_error

LR_KEY = 'lr'
DEFAULTS_GROUP_ADAM = {LR_KEY: 1e-4, 'betas': (0.9, 0.999), 'weight_decay': 0.0, 'eps': 1e-8}

def create_projected_optimiser(base_optimiser: type[torch.optim.Optimizer]) -> type[torch.optim.Optimizer]:

    # TODO
    #   > take list of projections as input

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
                        if hasattr(p, 'orthogonal_projection'):
                            p.data.copy_(p.orthogonal_projection(p))
                        
                        if hasattr(p, 'unit_ball_projection'):
                            p.data.copy_(p.unit_ball_projection(p))

                        if hasattr(p, 'zero_mean_projection'):
                            p.data.copy_(p.zero_mean_projection(p.data))
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

        group_[LR_KEY] = group[LR_KEY] if group[LR_KEY] is not None else DEFAULTS_GROUP_ADAM[LR_KEY]
        group_['betas'] = group['betas'] if group['betas'] is not None else DEFAULTS_GROUP_ADAM['betas']
        group_['weight_decay'] = group['weight_decay'] if group['weight_decay'] is not None \
            else DEFAULTS_GROUP_ADAM['weight_decay']
        group_['eps'] = DEFAULTS_GROUP_ADAM['eps']

        param_groups_.append(group_)
    return param_groups_

def step_adam(optimiser: torch.optim.Optimizer, func: Callable, param_groups: List[Dict[str, Any]]) -> torch.Tensor:
    # update history
    for group in param_groups:
        for p, p_old in zip([p_ for p_ in group['params']], [p_ for p_ in group['history']]):
            p_old.data.copy_(p.data.clone())

    # make adam step
    optimiser.zero_grad()
    params_flat = flatten_groups(param_groups)
    with torch.enable_grad():
        loss = func(*params_flat)
    loss.backward()
    optimiser.step()

    return loss

def optimise_adam(func: Callable, param_groups: List[Dict[str, Any]], max_num_iterations: int,
                  rel_tol: float, **unknown_options) -> OptimiserResult:
    num_iterations = max_num_iterations

    param_groups_ = harmonise_param_groups_adam(param_groups)
    optimiser = create_projected_optimiser(torch.optim.SGD)(param_groups_)


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


