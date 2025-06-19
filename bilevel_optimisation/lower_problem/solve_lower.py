import torch
from typing import Optional, List, Dict, Any

from bilevel_optimisation.data import LowerProblemResult, OptimiserResult
from bilevel_optimisation.energy import InnerEnergy
from bilevel_optimisation.optimise import optimise_nag

def build_param_groups(u: torch.Tensor, alpha: Optional[float], beta: Optional[float],
                       lip_const: Optional[float], max_num_backtracking_iterations: Optional[int],
                       batch_optimisation: bool) -> List[Dict[str, Any]]:
    if batch_optimisation:
        param_groups = []
        group = {'params': [torch.nn.Parameter(u, requires_grad=True)]}
        if alpha:
            group['alpha'] = alpha
        if beta:
            group['beta'] = beta
        if lip_const:
            group['lip_const'] = lip_const
        if max_num_backtracking_iterations:
            group['max_num_backtracking_iterations'] = max_num_backtracking_iterations
        param_groups.append(group)
    else:
        param_groups = []
        # TODO
        #   > iterate over batch dimension and add each batch to its own parameter group.
        pass

    return param_groups

def parse_nag_result(result: OptimiserResult, max_num_iterations: int) -> LowerProblemResult:
    solution_tensor = build_solution_tensor_from_param_groups(result.solution)
    lower_prob_result = LowerProblemResult(solution=solution_tensor, num_iterations=result.num_iterations,
                                           loss=result.loss,
                                           message='Converged' if result.num_iterations < max_num_iterations
                                           else 'Max. number of iterations reached')
    return lower_prob_result

def build_solution_tensor_from_param_groups(param_groups: List[Dict[str, Any]]):
    solution_list = [group['params'][-1].data for group in param_groups]
    return torch.cat(solution_list, dim=0)


def solve_lower(u_noisy: torch.Tensor, inner_energy: InnerEnergy, method: str,
                max_num_iterations: int=1000, alpha: Optional[float]=None, beta: Optional[float]=None,
                lip_const: Optional[float]=None, max_num_backtracking_iterations: Optional[int]=None,
                rel_tol: Optional[float]=None, batch_optimisation: bool=True) -> LowerProblemResult:

    u_ = u_noisy.detach().clone()
    param_groups = build_param_groups(u_, alpha, beta, lip_const, max_num_backtracking_iterations,
                                      batch_optimisation)
    if method == 'nag':
        func = lambda x: inner_energy(x)
        def grad_func(x):
            with torch.enable_grad():
                loss = func(x)
            return list(torch.autograd.grad(outputs=loss, inputs=[x]))

        nag_result = optimise_nag(func, grad_func, param_groups, max_num_iterations, rel_tol)
        lower_prob_result = parse_nag_result(nag_result, max_num_iterations)

    elif method == 'napg':
        prox_map = lambda x, tau: (((tau / inner_energy.measurement_model.noise_level) * u_noisy + x)
                                   / (1 + (tau / inner_energy.measurement_model.noise_level)))
        for group in param_groups:
            group['prox'] = prox_map

        func = lambda x: inner_energy.lam * inner_energy.regulariser(x)
        def grad_func(x):
            with torch.enable_grad():
                loss = func(x)
            return list(torch.autograd.grad(outputs=loss, inputs=[x]))

        nag_result = optimise_nag(func, grad_func, param_groups, max_num_iterations, rel_tol)
        lower_prob_result = parse_nag_result(nag_result, max_num_iterations)
    elif method == 'adam':
        pass
    elif method == 'nag_unrolling':
        pass
        # alpha = 1 / lip_const if lip_const is not None else None
        # beta = momentum
        #
        # optimiser = UnrollingNAGOptimiser([u_], alpha, beta, lip_const, max_num_bt_iterations)
        #
        # loss_func = lambda z: inner_energy.forward(z)
        # loss, map_estimate = optimiser.step(loss_func = loss_func,
        #                                     num_iterations=max_num_iterations)

    elif method == 'napg_unrolling':
        pass
        # f =
        # grad_f =
        # setattr(u_, 'prox') = ...
    else:
        raise NotImplementedError

    return lower_prob_result


