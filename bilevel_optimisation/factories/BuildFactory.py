from typing import List, Tuple, Callable
import torch

from bilevel_optimisation.data.OptimiserSpec import OptimiserSpec
from bilevel_optimisation.data.SolverSpec import SolverSpec
from bilevel_optimisation.optimiser.StoppingCriterion import StoppingCriterion
from bilevel_optimisation.solver.CGSolver import LinearSystemSolver

def build_optimiser_factory(optimiser_spec: OptimiserSpec) -> Callable:
    def optimiser_factory(x: List[torch.Tensor]) -> Tuple[torch.optim.Optimizer, StoppingCriterion, Callable]:
        return (optimiser_spec.optimiser_class(x, **optimiser_spec.optimiser_params),
                optimiser_spec.stopping_class(**optimiser_spec.stopping_params), optimiser_spec.prox_map_factory)
    return optimiser_factory

def build_solver_factory(solver_spec: SolverSpec) -> Callable:
    def solver_factory() -> LinearSystemSolver:
        return solver_spec.solver_class(**solver_spec.solver_params)
    return solver_factory

def build_prox_map_factory(prox_map: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]):
    def prox_map_factory(x: torch.Tensor) -> Callable[[torch.Tensor, float], torch.Tensor]:
        return lambda z, tau: prox_map(z, x, tau)
    return prox_map_factory
