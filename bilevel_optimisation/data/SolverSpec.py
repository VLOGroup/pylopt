from dataclasses import dataclass
from typing import Dict, Any, Optional

from bilevel_optimisation.solver.CGSolver import LinearSystemSolver

@dataclass
class SolverSpec:
    solver_class: type[LinearSystemSolver]
    solver_params: Optional[Dict[str, Any]]