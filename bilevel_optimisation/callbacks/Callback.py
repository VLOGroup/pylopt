from typing import Optional
from abc import ABC
import torch
import os
import math
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps

from bilevel_optimisation.fields_of_experts import FieldsOfExperts

class Callback(ABC):

    def __init__(self):
        pass

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs) -> None:
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class SaveModel(Callback):
    def __init__(self, path_to_data_dir: str, save_freq: int = 2) -> None:
        super().__init__()

        self.path_to_model_dir = os.path.join(path_to_data_dir, 'models')
        self.save_freq = save_freq

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs):
        if step % self.save_freq and regulariser is not None:
            if not os.path.exists(self.path_to_model_dir):
                os.makedirs(self.path_to_model_dir, exist_ok=True)
            regulariser.get_image_filter().save(self.path_to_model_dir, 'filters_iter_{:d}.pt'.format(step))
            regulariser.get_potential().save(self.path_to_model_dir, 'potential_iter_{:d}.pt'.format(step))

class PlotFiltersAndPotentials(Callback):
    def __init__(self, path_to_data_dir: str, plotting_freq: int = 2):
        super().__init__()

        self.path_to_filter_plot_dir = os.path.join(path_to_data_dir, 'filters')
        self.path_to_potential_plot_dir = os.path.join(path_to_data_dir, 'potentials')
        self.plotting_freq = plotting_freq

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs):
        if step % self.plotting_freq and regulariser is not None:
            self._plot_filters(regulariser)
            self._plot_potentials()

    def _plot_filters(self, step: int, regulariser: FieldsOfExperts):
        filters = regulariser.get_image_filter().get_filter_tensor()
        num_filters = filters.shape[0]
        num_filters_sqrt = int(math.sqrt(num_filters)) + 1
        fig, axes = plt.subplots(num_filters_sqrt, num_filters_sqrt, figsize=(11, 11),
                                 gridspec_kw={'hspace': 0.9, 'wspace': 0.2})

        for i in range(0, num_filters_sqrt):
            for j in range(0, num_filters_sqrt):
                filter_idx = i * num_filters_sqrt + j
                if filter_idx < num_filters:
                    filter_norm = torch.linalg.norm(filters[filter_idx]).detach().cpu().item()

                    axes[i, j].imshow(filters[filter_idx, :, :, :].squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
                    title = 'idx={:d}, \nnorm={:.3f}'.format(filter_idx, filter_norm)
                    axes[i, j].set_title(title, fontsize=8)
                    axes[i, j].xaxis.set_visible(False)
                    axes[i, j].yaxis.set_visible(False)
                else:
                    fig.delaxes(axes[i, j])

        plt.savefig(os.path.join(self.path_to_filter_plot_dir, 'filters_iter_{:d}.png'.format(step)))
        plt.close(fig)

    def _plot_potentials(self):
        pass

class TensorBoardCallback(Callback):
    pass