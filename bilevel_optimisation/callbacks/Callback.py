from typing import Optional, Dict, Any, Callable, Tuple, List
from abc import ABC
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import math
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
import io
from PIL import Image
import logging
import numpy as np

from bilevel_optimisation.energy import Energy
from bilevel_optimisation.fields_of_experts import FieldsOfExperts
from bilevel_optimisation.lower_problem import solve_lower
from bilevel_optimisation.measurement_model import MeasurementModel
from bilevel_optimisation.utils.dataset_utils import collate_function
from bilevel_optimisation.utils.evaluation_utils import compute_psnr
from bilevel_optimisation.utils.Timer import Timer

def figure_to_tensor(fig: plt.Figure) -> torch.Tensor:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return transforms.ToTensor()(image)

def compute_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    lst = [np.nan for _ in range(0, window)]
    for i in range(0, len(data) - window + 1):
        lst.append(np.mean(data[i : i + window]))
    return np.array(lst)

class Callback(ABC):

    def __init__(self) -> None:
        pass

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs) -> None:
        pass

    def on_train_begin(self, regulariser: Optional[FieldsOfExperts]=None, **kwargs) -> None:
        pass

    def on_train_end(self) -> None:
        pass

class SaveModel(Callback):
    def __init__(self, path_to_data_dir: str, save_freq: int = 2) -> None:
        super().__init__()

        self.path_to_model_dir = os.path.join(path_to_data_dir, 'models')
        self.save_freq = save_freq

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs) -> None:
        if step % self.save_freq and regulariser is not None:
            if not os.path.exists(self.path_to_model_dir):
                os.makedirs(self.path_to_model_dir, exist_ok=True)
            regulariser.get_image_filter().save(self.path_to_model_dir, 'filters_iter_{:d}.pt'.format(step))
            regulariser.get_potential().save(self.path_to_model_dir, 'potential_iter_{:d}.pt'.format(step))

class PlotFiltersAndPotentials(Callback):
    def __init__(self, path_to_data_dir: str, plotting_freq: int = 2, tb_writer: Optional[SummaryWriter]=None) -> None:
        super().__init__()

        self.path_to_filter_plot_dir = os.path.join(path_to_data_dir, 'filters')
        self.path_to_potential_plot_dir = os.path.join(path_to_data_dir, 'potentials')

        self.tb_writer = tb_writer

        for pth in [self.path_to_filter_plot_dir, self.path_to_potential_plot_dir]:
            if not os.path.exists(pth):
                os.makedirs(pth, exist_ok=True)

        self.plotting_freq = plotting_freq

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs) -> None:
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)

        if step % self.plotting_freq and regulariser is not None:
            self._plot_filters(step, regulariser)
            self._plot_potentials(step, regulariser, device, dtype)

    @staticmethod
    def _normalise_filter(filter_tensor: torch.Tensor) -> torch.Tensor:
        filter_tensor = filter_tensor - torch.min(filter_tensor)
        filter_tensor = filter_tensor / torch.max(filter_tensor)
        return filter_tensor

    def _plot_filters(self, step: int, regulariser: FieldsOfExperts) -> None:
        filters = regulariser.get_image_filter().get_filter_tensor()
        filter_norms = [torch.linalg.norm(fltr).detach().cpu().item() for fltr in filters]

        filter_indices_sorted = np.argsort(filter_norms)[::-1].tolist()

        num_filters = filters.shape[0]
        num_filters_sqrt = int(math.sqrt(num_filters)) + 1
        fig, axes = plt.subplots(num_filters_sqrt, num_filters_sqrt, figsize=(11, 11),
                                 gridspec_kw={'hspace': 0.9, 'wspace': 0.2})

        for i in range(0, num_filters_sqrt):
            for j in range(0, num_filters_sqrt):
                filter_idx = i * num_filters_sqrt + j
                if filter_idx < num_filters:
                    idx = filter_indices_sorted[filter_idx]

                    fltr = self._normalise_filter(filters[idx, :, :, :].squeeze().detach().clone())
                    axes[i, j].imshow(fltr.cpu().numpy(), cmap=cmaps['gray'])

                    title = 'idx={:d}, \nnorm={:.3f}'.format(idx, filter_norms[idx])
                    axes[i, j].set_title(title, fontsize=8)
                    axes[i, j].axis('off')
                else:
                    fig.delaxes(axes[i, j])

        plt.savefig(os.path.join(self.path_to_filter_plot_dir, 'filters_iter_{:d}.png'.format(step)))
        if self.tb_writer:
            self.tb_writer.add_image('filters', figure_to_tensor(fig), step + 1)
        plt.close(fig)

    def _plot_potentials(self, step: int, regulariser: FieldsOfExperts, device: Optional[torch.device],
                         dtype: Optional[torch.dtype]) -> None:
        if device is not None and dtype is not None:
            filters = regulariser.get_image_filter().get_filter_tensor()
            filter_norms = [torch.linalg.norm(fltr).detach().cpu().item() for fltr in filters]
            filter_indices_sorted = np.argsort(filter_norms)[::-1].tolist()

            x_lower = -1.0
            x_upper = 1.0

            potential = regulariser.get_potential()
            potential_weight_tensor = potential.get_parameters()
            num_marginals = potential.get_num_marginals()
            num_marginals_sqrt = int(math.sqrt(num_marginals)) + 1

            fig, axes = plt.subplots(num_marginals_sqrt, num_marginals_sqrt, figsize=(11, 11),
                                     gridspec_kw={'hspace': 0.9, 'wspace': 0.2}, sharex=True, sharey=True)
            for i in range(0, num_marginals_sqrt):
                for j in range(0, num_marginals_sqrt):
                    potential_idx = i * num_marginals_sqrt + j
                    if potential_idx < potential.get_num_marginals():
                        idx = filter_indices_sorted[potential_idx]

                        t = torch.linspace(x_lower, x_upper, 101).to(device=device, dtype=dtype)
                        y = potential.forward_negative_log_marginal(t * filter_norms[idx], idx)

                        axes[i, j].plot(t.detach().cpu().numpy(), y.detach().cpu().numpy() -
                                        torch.min(y).detach().cpu().numpy(), color='blue')

                        potential_weight = potential_weight_tensor[potential_idx].detach().cpu().item()
                        axes[i, j].set_title('idx={:d}, \nweight={:.3f}'.format(idx, potential_weight),
                                             fontsize=8)
                        axes[i, j].set_xlim(x_lower, x_upper)
                    else:
                        fig.delaxes(axes[i, j])

            plt.savefig(os.path.join(self.path_to_potential_plot_dir, 'potentials_iter_{:d}.png'.format(step)))
            if self.tb_writer:
                self.tb_writer.add_image('potentials', figure_to_tensor(fig), step + 1)
            plt.close(fig)

class TrainingMonitor(Callback):

    def __init__(self, dataset: Dataset, config, method: str, options: Dict[str, Any],
                 loss_func: Callable, path_to_data_dir: str, evaluation_freq: int=2,
                 tb_writer: Optional[SummaryWriter]=None) -> None:
        super().__init__()
        self.test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                 collate_fn=lambda x: collate_function(x, crop_size=-1))

        self.config = config
        self.method = method
        self.options = options
        self.loss_func = loss_func

        self.path_to_data_dir = path_to_data_dir
        self.path_to_test_data_dir = os.path.join(self.path_to_data_dir, 'test')
        if not os.path.exists(self.path_to_test_data_dir):
            os.makedirs(self.path_to_test_data_dir, exist_ok=True)

        self.tb_writer = tb_writer

        self.evaluation_freq = evaluation_freq

        self.test_loss_list = []
        self.train_loss_list = []
        self.test_psnr_list = []
        self.potential_params_list = []

        self.lip_const_dict = {}

    def on_train_begin(self, regulariser: Optional[FieldsOfExperts]=None, **kwargs) -> None:
        logging.info('[{:s}] compute initial test loss and initial psnr'.format(self.__class__.__name__))
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)

        if regulariser is not None:
            self._evaluate_on_test_data(-1, regulariser, device, dtype)

    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs) -> None:
        if step % self.evaluation_freq:
            logging.info('[{:s}] evaluate on test dataset'.format(self.__class__.__name__))

            if loss:
                self.train_loss_list.append(loss.detach().cpu().numpy())
                if self.tb_writer:
                    self.tb_writer.add_scalar('loss/train', loss, step + 1)

            param_groups = kwargs.get('param_groups', None)
            if param_groups:
                for group in param_groups:
                    name = group.get('name', '')
                    lip_const = group.get('lip_const', [-1])
                    logging.info('[{:s}] lipschitz constant for group {:s}: '
                                 '{:.3f}'.format(self.__class__.__name__, name, lip_const[-1]))
                    if not name in self.lip_const_dict.keys():
                        self.lip_const_dict[name] = []
                    self.lip_const_dict[name].append(lip_const[-1])

                    if self.tb_writer:
                        self.tb_writer.add_scalar('lip_const/{:s}'.format(name), lip_const[-1], step + 1)

            if regulariser is not None:
                device = kwargs.get('device', None)
                dtype = kwargs.get('dtype', None)
                test_loss, test_psnr = self._evaluate_on_test_data(step, regulariser, device, dtype)
                for tag, value, value_list in zip(['loss/train', 'loss/test', 'psnr/test'],
                                                  [loss, test_loss, test_psnr],
                                                  [self.train_loss_list, self.test_loss_list, self.test_psnr_list]):
                    if value:
                        value_list.append(value if isinstance(value, float) else value.detach().cpu().numpy())
                        if self.tb_writer:
                            self.tb_writer.add_scalar(tag, value, step + 1)

                # TODO:
                #   > does this work as expected for all kind of potentials?
                #   > works fine for student-t potential
                potential_param = regulariser.potential.get_parameters().detach().cpu().numpy()
                self.potential_params_list.append(potential_param)
                if self.tb_writer:
                    self.tb_writer.add_scalars('potentials/weights',
                                               {'potential_{:d}'.format(i): np.exp(potential_param[i])
                                                for i in range(0, len(potential_param))}, step + 1)

    def on_train_end(self) -> None:
        self._visualise_training_stats()

    def _visualise_training_stats(self) -> None:
        moving_average = compute_moving_average(np.array(self.train_loss_list), 10)

        fig = plt.figure(figsize=(11, 11))

        ax_1 = fig.add_subplot(1, 2, 1)
        ax_1.set_title('training loss')
        ax_1.plot(np.arange(0, len(self.train_loss_list)),
                  self.train_loss_list, label='train loss')
        ax_1.plot(np.arange(0, len(moving_average)), moving_average, color='orange',
                  label='moving average of train loss')
        ax_1.plot(self.evaluation_freq * np.arange(0, len(self.test_loss_list)), self.test_loss_list,
                  color='cyan', label='test loss')
        ax_1.xaxis.get_major_locator().set_params(integer=True)
        ax_1.set_xlabel('iteration')
        ax_1.legend()

        ax_2 = fig.add_subplot(1, 2, 2)
        ax_2.set_title('average psnr over test set')
        ax_2.plot(self.evaluation_freq * np.arange(0, len(self.test_psnr_list)), self.test_psnr_list)
        ax_2.xaxis.get_major_locator().set_params(integer=True)
        ax_2.set_xlabel('iteration')

        plt.savefig(os.path.join(self.path_to_data_dir, 'training_stats.png'))
        plt.close(fig)

    def _evaluate_on_test_data(self, step: int, regulariser: FieldsOfExperts, device: Optional[torch.device],
                               dtype: Optional[torch.dtype]) -> Tuple[Optional[float], Optional[float]]:
        psnr = None
        loss_test = None
        if device is not None and dtype is not None:
            test_batch_clean = list(self.test_loader)[0]
            test_batch_clean_ = test_batch_clean.to(device=device, dtype=dtype)

            measurement_model = MeasurementModel(test_batch_clean_, config=self.config)
            energy = Energy(measurement_model, regulariser, self.config['energy']['lam'].get())
            energy.to(device=device, dtype=dtype)

            test_batch_noisy = measurement_model.get_noisy_observation()
            with Timer(device) as t:
                test_batch_denoised = solve_lower(energy, self.method, self.options).solution

            psnr = torch.mean(compute_psnr(energy.measurement_model.get_clean_data(), test_batch_denoised))
            psnr = psnr.detach().cpu().item()
            loss_test = self.loss_func(test_batch_clean_, test_batch_denoised)
            loss_test = loss_test.detach().cpu().item()

            logging.info('[{:s}]   > average psnr: {:.5f}'.format(self.__class__.__name__, psnr))
            logging.info('[{:s}]   > test loss: {:.5f}'.format(self.__class__.__name__, loss_test))
            logging.info('[{:s}]   > evaluation took [ms]: {:.5f}'.format(self.__class__.__name__, t.time_delta()))

            u_clean_splits = torch.split(test_batch_clean_, split_size_or_sections=1, dim=0)
            u_noisy_splits = torch.split(test_batch_noisy, split_size_or_sections=1, dim=0)
            u_denoised_splits = torch.split(test_batch_denoised, split_size_or_sections=1, dim=0)

            num_rows = len(u_clean_splits)
            num_cols = 3
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(11, 11),
                                     gridspec_kw={'hspace': 0.9, 'wspace': 0.2}, sharex=True, sharey=True)
            if num_rows == 1:
                axes = [axes]
            for idx, (item_clean, item_noisy, item_denoised) in (
                    enumerate(zip(u_clean_splits, u_noisy_splits, u_denoised_splits))):

                axes[idx][0].imshow(item_clean.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
                axes[idx][1].imshow(item_noisy.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
                axes[idx][2].imshow(item_denoised.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])

                if idx == 0:
                    axes[idx][0].set_title('clean')
                    axes[idx][1].set_title('noisy')
                    axes[idx][2].set_title('denoised')

                axes[idx][0].axis("off")
                axes[idx][1].axis("off")
                axes[idx][2].axis("off")

            plt.savefig(os.path.join(self.path_to_test_data_dir, 'triplet_iter_{:d}.png'.format(step)))
            if self.tb_writer:
                self.tb_writer.add_image('triplets/test', figure_to_tensor(fig), step + 1)
            plt.close(fig)

        return loss_test, psnr
