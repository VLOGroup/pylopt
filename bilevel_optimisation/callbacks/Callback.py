from typing import Optional, Dict, Any, Callable, Tuple, List
from abc import ABC, abstractmethod
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
from confuse import Configuration
import pandas as pd

from bilevel_optimisation.bilevel_problem.parameter_groups import PARAM_GROUP_NAME_KEY
from bilevel_optimisation.dataset.dataset_utils import collate_function
from bilevel_optimisation.dataset.ImageDataset import TestImageDataset
from bilevel_optimisation.energy import Energy
from bilevel_optimisation.fields_of_experts import FieldsOfExperts
from bilevel_optimisation.lower_problem import solve_lower
from bilevel_optimisation.measurement_model import MeasurementModel
from bilevel_optimisation.optimise import LIP_CONST_KEY
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
    """
    Base callback class.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def on_step(self, step: int, regulariser: Optional[FieldsOfExperts]=None,
                loss: Optional[torch.Tensor]=None, **kwargs) -> None:
        pass

    @abstractmethod
    def on_train_begin(self, regulariser: Optional[FieldsOfExperts]=None, **kwargs) -> None:
        pass

    @abstractmethod
    def on_train_end(self) -> None:
        pass

class SaveModel(Callback):
    """
    Callback which saves current model (image filter and potential) to disk on every
    training/learning step.
    """
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

    def on_train_begin(self, regulariser: Optional[FieldsOfExperts]=None, **kwargs) -> None:
        pass

    def on_train_end(self) -> None:
        pass

class PlotFiltersAndPotentials(Callback):
    """
    Callback which creates plots of current filters and potentials. The plots are saved to disk and
    written to tensorboard.

    NOTE
    ----
        > The j-th potential is plotted on the range [q_low, q_hiqh] of the j-th filter response, where
            q_low, q_high denote the 2.5% and 97.5% quantile respectively of the response of the j-th filter applied to a
            natural image.
    """
    def __init__(self, dataset: TestImageDataset, path_to_data_dir: str, plotting_freq: int = 2,
                 tb_writer: Optional[SummaryWriter]=None, p_low: float=0.025, p_high: float=0.975) -> None:
        """
        Initialisation of filter and potential plotting callback.

        :param dataset: Test image dataset containing natural images which is used to determine the quantile
            range on which the potentials are plotted.
        :param path_to_data_dir: Path to directory where data is stored.
        :param plotting_freq: Frequency on which plots are generated.
        :param tb_writer: Tensorboard writer.
        :param p_low: Probability corresponding to the quantile q_low
        :param p_high: Probability corresponding to the quantile q_high
        """
        super().__init__()
        self.test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                      collate_fn=lambda x: collate_function(x, crop_size=-1))
        self.p_low = p_low
        self.p_high = p_high

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
            image_filter = regulariser.get_image_filter()
            filter_tensors = image_filter.get_filter_tensor()
            filter_norms = [torch.linalg.norm(fltr).detach().cpu().item() for fltr in filter_tensors]
            filter_indices_sorted = np.argsort(filter_norms)[::-1].tolist()

            u_clean = list(self.test_loader)[0].to(dtype=dtype, device=device)
            filter_response = image_filter(u_clean)
            filter_response = torch.permute(filter_response, dims=(1, 0, 2, 3))
            filter_response_flat = torch.flatten(filter_response, start_dim=1, end_dim=3)
            quantiles = torch.quantile(filter_response_flat,
                                       q=torch.tensor([self.p_low, self.p_high]).to(device=device), dim=-1)

            potential = regulariser.get_potential()
            num_marginals = potential.get_num_marginals()
            num_marginals_sqrt = int(math.sqrt(num_marginals)) + 1

            t = torch.stack([torch.linspace(quantiles[0, i], quantiles[1, i], 111)
                             for i in range(0, num_marginals)]).unsqueeze(dim=1).unsqueeze(dim=0)
            t = t.to(device=device, dtype=dtype)
            rho = potential.forward_negative_log(t, reduce=False)

            fig, axes = plt.subplots(num_marginals_sqrt, num_marginals_sqrt, figsize=(13, 13),
                                     gridspec_kw={'hspace': 0.9, 'wspace': 0.4}, sharex=False, sharey=False)
            for i in range(0, num_marginals_sqrt):
                for j in range(0, num_marginals_sqrt):
                    potential_idx = i * num_marginals_sqrt + j
                    if potential_idx < potential.get_num_marginals():
                        idx = filter_indices_sorted[potential_idx]

                        axes[i, j].plot(t[0, idx, 0, :].detach().cpu().numpy(),
                                        rho[0, idx, 0, :].detach().cpu().numpy() -
                                        torch.min(rho[0, potential_idx, 0, :]).detach().cpu().numpy(), color='blue')
                        if potential.__class__.__name__ == 'StudentT':
                            potential_weight_tensor = potential.get_parameters()
                            potential_weight = potential_weight_tensor[idx].detach().cpu().item()
                            axes[i, j].set_title('idx={:d}, \nweight={:.3f}'.format(idx, potential_weight),
                                                 fontsize=8)
                        else:
                            axes[i, j].set_title('idx={:d}'.format(idx), fontsize=8)

                        axes[i, j].set_xlim(quantiles[0, idx].cpu().item(), quantiles[1, idx].cpu().item())
                        for x_label, y_label in zip(axes[i, j].get_xticklabels(), axes[i, j].get_yticklabels()):
                            x_label.set_fontsize(8)
                            y_label.set_fontsize(8)
                    else:
                        fig.delaxes(axes[i, j])

            plt.savefig(os.path.join(self.path_to_potential_plot_dir, 'potentials_iter_{:d}.png'.format(step)))
            if self.tb_writer:
                self.tb_writer.add_image('potentials', figure_to_tensor(fig), step + 1)
            plt.close(fig)

    def on_train_begin(self, regulariser: Optional[FieldsOfExperts]=None, **kwargs) -> None:
        pass

    def on_train_end(self) -> None:
        pass

class TrainingMonitor(Callback):

    TENSORBOARD_TAGS = {'loss': 'loss/train', 'test_loss': 'loss/test', 'test_psnr': 'test_psnr'}

    def __init__(self, dataset: Dataset, config: Configuration, method: str, options: Dict[str, Any],
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

        self.training_stats_dict_list = []
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
            logging.info('[{:s}] log statistics and hyperparameters'.format(self.__class__.__name__))

            param_groups = kwargs.get('param_groups', None)
            self._log_hyperparameters(step, param_groups)

            device = kwargs.get('device', None)
            dtype = kwargs.get('dtype', None)
            self._log_training_stats(step, regulariser, loss, device, dtype)

    def _log_hyperparameters(self, step: int, param_groups: Optional[List[Dict[str, Any]]]) -> None:
        if param_groups:
            for group in param_groups:
                name = group.get(PARAM_GROUP_NAME_KEY, '')
                lip_const = group.get(LIP_CONST_KEY, [-1])
                logging.info('[{:s}] lipschitz constant for group {:s}: '
                             '{:.3f}'.format(self.__class__.__name__, name, lip_const[-1]))
                if not name in self.lip_const_dict.keys():
                    self.lip_const_dict[name] = []
                self.lip_const_dict[name].append(lip_const[-1])

                if self.tb_writer:
                    self.tb_writer.add_scalar('{:s}/{:s}'.format(LIP_CONST_KEY, name), lip_const[-1], step + 1)

    def _log_training_stats(self, step: int, regulariser: Optional[FieldsOfExperts], loss: Optional[torch.Tensor],
                            device: Optional[torch.device], dtype: Optional[torch.dtype]) -> None:
        stats = {'step': step}
        if loss:
            loss_ = loss.detach().cpu().numpy()
            stats.update({'loss': loss_})
            if self.tb_writer:
                self.tb_writer.add_scalar(self.TENSORBOARD_TAGS['loss'], loss_, step + 1)

        if regulariser is not None:
            test_loss, test_psnr = self._evaluate_on_test_data(step, regulariser, device, dtype)

            for key, value in zip(['test_loss', 'test_psnr'], [test_loss, test_psnr]):
                if value:
                    stats.update({key: value})
                    if self.tb_writer:
                        self.tb_writer.add_scalar(self.TENSORBOARD_TAGS[key], value, step + 1)

            fitness = self._compute_model_fitness(loss, test_loss, test_psnr)
            if fitness:
                stats.update({'fitness': fitness})

            # TODO:
            #   > does this work as expected for all kind of potentials?
            #   > works fine for student-t potential
            potential_param = regulariser.potential.get_parameters().detach().cpu().numpy()
            self.potential_params_list.append(potential_param)
            if self.tb_writer:
                self.tb_writer.add_scalars('potentials/weights',
                                           {'potential_{:d}'.format(i): np.exp(potential_param[i])
                                            for i in range(0, len(potential_param))}, step + 1)

        self.training_stats_dict_list.append(stats)

    def on_train_end(self) -> None:
        df = pd.DataFrame.from_dict(self.training_stats_dict_list)

        train_loss_list = df['loss'].dropna().to_list()
        test_loss_list = df['test_loss'].dropna().to_list()
        test_psnr_list = df['test_psnr'].dropna().to_list()

        self._visualise_training_stats(train_loss_list, test_loss_list, test_psnr_list)
        self._export_model_ranking(df[['step', 'fitness']].dropna())

    def _visualise_training_stats(self, train_loss_list: List[float], test_loss_list: List[float],
                                  test_psnr_list: List[float]) -> None:

        moving_average = compute_moving_average(np.array(train_loss_list), 10)

        fig = plt.figure(figsize=(11, 11))

        ax_1 = fig.add_subplot(1, 2, 1)
        ax_1.set_title('training loss')
        ax_1.plot(np.arange(0, len(train_loss_list)), train_loss_list, label='train loss')
        ax_1.plot(np.arange(0, len(moving_average)), moving_average, color='orange',
                  label='moving average of train loss')
        ax_1.plot(self.evaluation_freq * np.arange(0, len(test_loss_list)), test_loss_list,
                  color='cyan', label='test loss')
        ax_1.xaxis.get_major_locator().set_params(integer=True)
        ax_1.set_xlabel('iteration')
        ax_1.legend()

        ax_2 = fig.add_subplot(1, 2, 2)
        ax_2.set_title('average psnr over test set')
        ax_2.plot(self.evaluation_freq * np.arange(0, len(test_psnr_list)), test_psnr_list)
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

    def _compute_model_fitness(self, train_loss: Optional[float], test_loss: Optional[float],
                               test_psnr: Optional[float], **kwargs: Optional[float]) -> Optional[float]:
        """
        Function which computes the so-called fitness of the current model. The fitness is intended to introduce a
        model ranking.

        NOTE
        ----
            > Currently, fitness is simply the psnr.

        :param train_loss: Current training loss
        :param test_loss: Current test loss
        :param test_psnr: Current psnr
        :param kwargs: Additional keyword arguments, which may be used when generalising, or updating the
            definition of fitness.
        :return: Float representing the fitness value
        """
        fitness = None
        if not any(item is None for item in [train_loss, test_loss, test_psnr]):
            fitness = 0.0 * train_loss + 0.0 * test_loss + 1.0 * test_psnr
        return fitness

    def _export_model_ranking(self, fitness: pd.DataFrame) -> None:
        fitness_sorted = fitness.sort_values('fitness', ascending=False, na_position='last', ignore_index=True)
        fitness_sorted.index += 1
        fitness_sorted.head(10).to_csv(os.path.join(self.path_to_data_dir, 'model_ranking.csv'))