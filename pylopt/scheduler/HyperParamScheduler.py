from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from math import cos, pi

from bilevel_optimisation.bilevel_problem.parameter_groups import PARAM_GROUP_NAME_KEY
from bilevel_optimisation.optimise import LIP_CONST_KEY

class HyperParamScheduler(ABC):

    def __init__(self):
        self.step_counter = 0

    @abstractmethod
    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def step(self, **kwargs: Dict[str, Any]) -> None:
        pass

class AdaptiveLRRestartScheduler(HyperParamScheduler):
    """
    Class implementing adaptive restarts handled by means of the callable condition_func, which
    takes kwargs as input only, and must return a boolean value indicating if a restart shall
    be performed.

    NOTE
    ----
        > The authors of [1] indicate several suitable restart policies
            * Fixed interval restart
            * Adaptive restart
                - Loss-based: Restart the scheme if over a predefined number of steps (patience) the
                    loss does not decrease
                - Gradient-based: Restart if grad * (x_{k} - x_{k - 1}) > 0 over a predefined number of
                    steps (patience), i.e. current search direction is not a descent direction.

    References
    ----------
    [1] O’donoghue, B. and Candes, E., 2015. Adaptive restart for accelerated gradient schemes.
        Foundations of computational mathematics, 15(3), pp.715-732.
    """

    def __init__(self, condition_func: Callable, gamma: float=0.9999, warm_up_period: int=100, patience: int=10,
                 lr_key: str='lr') -> None:
        super().__init__()

        self.condition_func = condition_func
        self.gamma = gamma
        self.warm_up_period = warm_up_period
        self.patience = patience
        self.lr_key = lr_key

        self.num_bad_iterations = 0

        self.param_groups = None
        self.base_values = None

    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = param_groups
        self.base_values = []
        for group in param_groups:
            self.base_values.append({self.lr_key: group.get(self.lr_key, None)})

    def _update_param_group(self) -> None:
        for idx, group in enumerate(self.param_groups):
            if self.lr_key in group.keys():
                group[self.lr_key] *= self.gamma

    def _perform_restart(self) -> None:
        for idx, group in enumerate(self.param_groups):
            if self.lr_key in group.keys() and self.base_values[idx][self.lr_key] is not None:
                group[self.lr_key] = self.base_values[idx][self.lr_key]

    def step(self, **kwargs: Dict[str, Any]) -> None:
        self.step_counter += 1

        if self.condition_func(self.param_groups, **kwargs) and self.step_counter >= self.warm_up_period:
            self.num_bad_iterations += 1
            if self.num_bad_iterations == self.patience:
                self._perform_restart()
                self.num_bad_iterations = 0

        self._update_param_group()

class CosineAnnealingLRScheduler(HyperParamScheduler):
    """
    Implementation of cosine annealing lr scheduler with optional fixed interval restarts. Annealing is applied
    only within iteration window which needs to be specified.

    NOTE
    ----
        > This scheduler is particularly suited for optimisers from torch.optim like Adam, SGD, ...
        > Updates are performed for the learning rate of all (!) parameter groups.
    """
    def __init__(self, step_begin: int, step_end: int, lr_min: float=1e-7, lr_key: str='lr',
                 restart_cycle: Optional[int]=None, restart_cycle_multi: Optional[float]=None) -> None:
        """
        Initialisation of cosine annealing lr scheduler.

        :param step_begin: Begin of annealing window
        :param step_end: End of annealing window
        :param lr_min: Minimal learning rate
        :param lr_key: Key of learning within parameter groups
        :param restart_cycle: Optional integer indicating the after how many iterations (within the annealing window)
            a restart shall be performed; if None no restarts were performed.
        :param restart_cycle_multi: Multiplier to adjust length of the restart cycle over training; if None
            the restart cycles keep the same length.
        """
        super().__init__()

        self.step_begin = step_begin
        self.step_end = step_end
        self.lr_min = lr_min
        self.lr_key = lr_key
        self.decay_cycle_len = restart_cycle if restart_cycle is not None else self.step_end - self.step_begin
        self.decay_cycle_multi = 1 if (restart_cycle is None or restart_cycle_multi is None) \
            else restart_cycle_multi

        self.steps_since_restart_counter = 0

        self.param_groups = None
        self.base_values = None

    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = param_groups
        self.base_values = []
        for group in param_groups:
            self.base_values.append({self.lr_key: group.get(self.lr_key, None)})

    def _update_param_group(self):
        for idx, group in enumerate(self.param_groups):
            if self.lr_key in group.keys() and self.base_values[idx][self.lr_key] is not None:
                group[self.lr_key] = (self.lr_min +
                                      0.5 * (self.base_values[idx][self.lr_key] - self.lr_min) *
                                      (1 + cos((self.steps_since_restart_counter / self.decay_cycle_len) * pi)))

    def step(self,  **kwargs: Dict[str, Any]) -> None:
        self.step_counter += 1
        if self.step_begin <= self.step_counter <= self.step_end:
            self.steps_since_restart_counter += 1

            if self.steps_since_restart_counter % self.decay_cycle_len == 0:
                self.steps_since_restart_counter = 0
                self.decay_cycle_len = int(self.decay_cycle_multi * self.decay_cycle_len)
            self._update_param_group()

class NAGBacktrackingScheduler(HyperParamScheduler):
    pass

class NAGRestartScheduler(HyperParamScheduler):
    # TODO
    #   > Review me!
    def __init__(self, restart_freq: int, reset_keys: List[str]) -> None:
        super().__init__()

        self.restart_freq = restart_freq
        self.reset_keys = reset_keys

        self.param_groups = None
        self.base_values = None

    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = param_groups

        self.base_values = []
        for group in param_groups:
            self.base_values.append({key: group.get(key, None) for key in self.reset_keys})

    def step(self) -> None:
        self.step_counter += 1

        if self.step_counter % self.restart_freq == 0:
            for idx, group in enumerate(self.param_groups):
                for key in self.base_values[idx]:
                    if key in group.keys():
                        group[key] = self.base_values[idx].get(key, group[key])

class NAGLipConstGuard(HyperParamScheduler):
    """
    Scheduler for NAG Lipschitz constant. If the Lipschitz constant of all parameter groups or a specified parameter
    group is larger than a specified bound, it will set to 0.5 times this upper bound. Per default all parameter
    groups will be affected.
    """
    def __init__(self, lip_const_bound: float, lip_const_key: str=LIP_CONST_KEY,
                 group_name: Optional[str]=None) -> None:
        """
        Initialisation of Lipschitz constant guard

        :param lip_const_bound: Upper bound of Lipschitz constant
        :param lip_const_key: Key of Lipschitz constant within parameter group.
        :param group_name:
        """
        super().__init__()

        self.lip_const_bound = lip_const_bound
        self.lip_const_key = lip_const_key
        self.group_name = group_name if group_name is not None else 'all'

        self.param_groups = None

    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = param_groups

    def _find_param_group(self, group_name: str) -> Optional[Dict[str, Any]]:
        group = None
        for grp in self.param_groups:
            if hasattr(grp, PARAM_GROUP_NAME_KEY) and getattr(grp, PARAM_GROUP_NAME_KEY) == group_name:
                group = grp
        return group

    def _update_lip_const(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Function which updates the Lipschitz constant of a parameter group if the group does
        contain the corresponding key (self.lip_const_key). Otherwise, the group and its parameters
        won't be altered.

        :param group: Parameter group in terms of dictionary
        :return: Updated parameter group.
        """
        if self.lip_const_key in group.keys():
            group[self.lip_const_key] = [min(0.5 * self.lip_const_bound, item) for item in group[self.lip_const_key]]
        return group

    def step(self) -> None:
        self.step_counter += 1

        if self.group_name == 'all':
            for group in self.param_groups:
                self._update_lip_const(group)
        else:
            group = self._find_param_group(self.group_name)
            if group is None:
                raise ValueError('There is no parameter group with name {:s}'.format(self.group_name))
            self._update_lip_const(group)
