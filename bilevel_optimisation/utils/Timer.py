import torch
import time

class Timer:
    """
    Class which is used to measure elapsed on CPU or GPU. Time is measured in unit [ms].
    """
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._delta_time = -1.0

    def __enter__(self):
        if self._device.type == 'cuda':
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start.record()
        else:
            self._time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._device.type == 'cuda':
            self._end.record()
            torch.cuda.synchronize()
            self._delta_time = self._start.elapsed_time(self._end)
        else:
            self._delta_time = time.time() - self._time_start
            self._delta_time *= 1000

    def time_delta(self) -> float:
        return self._delta_time
