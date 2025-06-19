import torch
import logging

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

                        if hasattr(p, 'proj'):
                            logging.debug('[ProjectedOptimiser] apply projection')
                            p.data.copy_(p.proj(p.data))

            return loss

    ProjectedOptimiser.__name__ = 'Projected{:s}'.format(base_optimiser.__name__)
    return ProjectedOptimiser

