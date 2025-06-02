import torch

class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()

        self.delta_time = self.start.elapsed_time(self.end)