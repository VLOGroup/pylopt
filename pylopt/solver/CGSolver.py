import torch
from typing import Callable
from typing import Tuple, Dict

from bilevel_optimisation.data import SolverResult, LinearSolverStats
from bilevel_optimisation.solver.LinearSystemSolver import IterativeLinearSystemSolver

def scalar_product_l2(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.sum(x1 * x2)

class CGSolver(IterativeLinearSystemSolver):

    def __init__(self, max_num_iterations: int=500, abs_tol: float = 1e-5, scalar_product: Callable = None) -> None:
        super().__init__()

        self._max_num_iterations = max_num_iterations
        self._absolute_tolerance = abs_tol

        self._scalar_product = scalar_product if scalar_product is not None else scalar_product_l2

    def _stopping_criteria_met(self, curr_residual: torch.Tensor,
                              curr_iteration_idx: int, max_num_iterations: int) -> Tuple[bool, Dict]:
        residual_norm = torch.sqrt(self._scalar_product(curr_residual, curr_residual)).cpu().item()
        info = {'max_num_iterations_reached': curr_iteration_idx >= max_num_iterations - 1,
                'residual_accuracy_reached': bool(residual_norm <= self._absolute_tolerance)}
        return info['max_num_iterations_reached'] or info['residual_accuracy_reached'], info

    @torch.no_grad()
    def solve(self, linear_operator: Callable, b: torch.Tensor, x0: torch.Tensor = None) -> SolverResult:
        max_num_iterations = self._max_num_iterations if self._max_num_iterations is not None else b.numel()
        x = torch.zeros_like(b) if x0 is None else x0
        solution = None

        r0 = b - linear_operator(x)
        r = r0.clone()
        p = r.clone()
        residual_norm_list = []
        iterate_list = []
        info = {}
        for k in range(0, max_num_iterations):
            tmp = linear_operator(p)

            alpha = self._scalar_product(r, r) / self._scalar_product(p, tmp)
            x_new = x + alpha * p
            r_new = r - alpha * tmp
            beta = self._scalar_product(r_new, r_new) / self._scalar_product(r, r)
            p_new = r_new + beta * p

            curr_res = torch.sqrt(self._scalar_product(r_new, r_new)).cpu().item()
            residual_norm_list.append(curr_res)
            iterate_list.append(x_new)
            stop, info = self._stopping_criteria_met(r_new, k, max_num_iterations)

            if stop:
                solution = x_new.clone()
                solution = solution.detach()
                break

            x = x_new.clone()
            p = p_new.clone()
            r = r_new.clone()

        cg_stats = LinearSolverStats(num_iterations=len(iterate_list), iterate_list=iterate_list,
                                     residual_norm_list=residual_norm_list)
        return SolverResult(solution=solution, info=info, stats=cg_stats)