from abc import ABC, abstractmethod
from typing import List, Dict, Any
from math import cos, pi

class HyperParamScheduler(ABC):

    def __init__(self):
        self.step_counter = 0

    @abstractmethod
    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

class CosineAnnealingLRScheduler(HyperParamScheduler):
    def __init__(self, step_begin: int, step_end: int, max_num_iterations: int,
                 lr_min: float=0.0, lr_key: str='lr') -> None:
        super().__init__()

        self.step_begin = step_begin
        self.step_end = step_end
        self.max_num_iterations = max_num_iterations
        self.lr_min = lr_min
        self.lr_key = lr_key

        self.param_groups = None
        self.base_values = None

    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = param_groups
        self.base_values = []
        for group in param_groups:
            self.base_values.append({self.lr_key: group.get(self.lr_key, None)})

    def step(self) -> None:
        self.step_counter += 1

        for idx, group in enumerate(self.param_groups):
            if self.lr_key in group.keys() and self.base_values[idx][self.lr_key] is not None:
                group[self.lr_key] = (self.lr_min +
                                      0.5 * (self.base_values[idx][self.lr_key] - self.lr_min) *
                                      (1 + cos((self.step_counter / self.max_num_iterations) * pi)))

class NAGRestartScheduler(HyperParamScheduler):
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

class NAGLipschitzDelimiter(HyperParamScheduler):
    def __init__(self, lip_const_bound: float, lip_const_key: str='lip_const'):
        super().__init__()

        self.lip_const_bound = lip_const_bound
        self.lip_const_key = lip_const_key

        self.param_groups = None

    def bind(self, param_groups: List[Dict[str, Any]]) -> None:
        self.param_groups = param_groups

    def step(self) -> None:
        self.step_counter += 1

        for idx, group in self.param_groups:
            if self.lip_const_key in group.keys():
                group[self.lip_const_key] = min(self.lip_const_bound, group[self.lip_const_key])
