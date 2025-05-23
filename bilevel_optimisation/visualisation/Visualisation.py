import os
import torch
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.ticker import MultipleLocator
from typing import List
import numpy as np

from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.potential import GaussianMixture


def visualise_test_triplets(u_clean: torch.Tensor, u_noisy: torch.Tensor, u_denoised: torch.Tensor,
                            fig_dir_path: str, file_name_pre_fix: str = 'test_triplet') -> None:
    u_clean_splits = torch.split(u_clean, split_size_or_sections=1, dim=0)
    u_noisy_splits = torch.split(u_noisy, split_size_or_sections=1, dim=0)
    u_denoised_splits = torch.split(u_denoised, split_size_or_sections=1, dim=0)

    for idx, (item_clean, item_noisy, item_denoised) in (
            enumerate(zip(u_clean_splits, u_noisy_splits, u_denoised_splits))):
        fig_images = plt.figure(figsize=(15, 5))

        ax_images_1 = fig_images.add_subplot(1, 3, 1)
        ax_images_1.set_title('clean')
        ax_images_1.xaxis.set_visible(False)
        ax_images_1.yaxis.set_visible(False)
        ax_images_1.imshow(item_clean.detach().cpu().numpy().squeeze(), cmap=cmaps['gray'])

        ax_images_2 = fig_images.add_subplot(1, 3, 2)
        ax_images_2.set_title('noisy')
        ax_images_2.xaxis.set_visible(False)
        ax_images_2.yaxis.set_visible(False)
        ax_images_2.imshow(item_noisy.detach().cpu().numpy().squeeze(), cmap=cmaps['gray'])

        ax_images_3 = fig_images.add_subplot(1, 3, 3)
        ax_images_3.set_title('denoised')
        ax_images_3.xaxis.set_visible(False)
        ax_images_3.yaxis.set_visible(False)
        ax_images_3.imshow(item_denoised.detach().cpu().numpy().squeeze(), cmap=cmaps['gray'])

        plt.savefig(os.path.join(fig_dir_path, '{:s}_{:d}.png'.format(file_name_pre_fix, idx)))
        plt.close(fig_images)

def compute_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    lst = [np.nan for _ in range(0, window)]
    for i in range(0, len(data) - window + 1):
        lst.append(np.mean(data[i : i + window]))
    return np.array(lst)

def visualise_training_stats(train_loss_list: List[float], test_loss_list: List[float],
                             psnr_list: List[float], test_freq: int, fig_dir_path: str,
                             file_name: str = 'training_stats.png') -> None:
    moving_average = compute_moving_average(np.array(train_loss_list), 10)
    test_iter_arr = test_freq * np.arange(0, len(psnr_list))

    fig = plt.figure(figsize=(11, 11))

    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.set_title('training loss')
    ax_1.plot(np.arange(1, len(train_loss_list) + 1), train_loss_list, label='train loss')
    ax_1.plot(np.arange(1, len(moving_average) + 1), moving_average, color='orange',
                    label='moving average of train loss')
    ax_1.plot(test_iter_arr, test_loss_list, color='cyan',
                    label='test loss')
    ax_1.set_xlabel('iteration')
    ax_1.legend()

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.set_title('average psnr over test set')
    ax_2.plot(test_iter_arr, psnr_list)
    ax_2.set_xlabel('iteration')
    ax_2.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.savefig(os.path.join(fig_dir_path, file_name))
    plt.close(fig)

def visualise_filters(filters: torch.Tensor, filter_weights: torch.Tensor, fig_dir_path: str,
                      file_name: str = 'filters.png') -> None:
    num_filters = filters.shape[0]
    num_filters_sqrt = int(np.sqrt(num_filters)) + 1
    fig, axes = plt.subplots(num_filters_sqrt, num_filters_sqrt, figsize=(11, 11),
                             gridspec_kw={'hspace': 0.9, 'wspace': 0.2})


    for i in range(0, num_filters_sqrt):
        for j in range(0, num_filters_sqrt):
            filter_idx = i * num_filters_sqrt + j
            if filter_idx < num_filters:
                filter_norm = torch.linalg.norm(filters[filter_idx]).detach().cpu().item()
                filter_weight = filter_weights[filter_idx].detach().cpu().item()

                axes[i, j].imshow(filters[filter_idx, :, :, :].squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
                title = '(idx={:d}, \nnorm={:.3f}, \nweight={:.3f})'.format(filter_idx,
                                                                            filter_norm, filter_weight)
                axes[i, j].set_title(title, fontsize=8)
                axes[i, j].xaxis.set_visible(False)
                axes[i, j].yaxis.set_visible(False)
            else:
                fig.delaxes(axes[i, j])

    plt.savefig(os.path.join(fig_dir_path, file_name))
    plt.close(fig)

def visualise_filter_stats(filters_list: List[torch.Tensor], filter_weights_list: List[torch.Tensor],
                           fig_dir_path: str, file_name: str = 'filter_stats.png') -> None:
    filters_norm_list = [torch.sqrt(torch.sum(fltr ** 2, dim=(-2, -1))).squeeze().detach().cpu().numpy()
                        for fltr in filters_list]
    filter_weights_list_ = [fltr_wght.detach().cpu().numpy() for fltr_wght in filter_weights_list]

    num_data_items = len(filters_norm_list)
    iter_arr = np.arange(0, num_data_items)

    fig = plt.figure(figsize=(11, 11))
    ax_1 = fig.add_subplot(1, 2, 1)

    ax_1.set_xlabel('iteration')
    ax_1.set_title('norm of filters')
    ax_1.plot(iter_arr, filters_norm_list)

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.plot(iter_arr, filter_weights_list_)
    ax_2.set_xlabel('iteration')
    ax_2.set_title('weights of filters')

    plt.savefig(os.path.join(fig_dir_path, file_name))
    plt.close(fig)

def visualise_gmm_potential(potential: GaussianMixture, device: torch.device, dtype: torch.dtype,
                            fig_dir_path: str, file_name: str = 'mixtures.png') -> None:

    x_lower = -3.0
    x_upper = 3.0
    steps = 101
    t = torch.linspace(x_lower, x_upper, steps=steps, device=device, dtype=dtype)

    num_mixtures = potential.get_number_of_mixtures()
    num_mixtures_sqrt = int(np.sqrt(num_mixtures)) + 1
    fig, axes = plt.subplots(num_mixtures_sqrt, num_mixtures_sqrt, figsize=(11, 11),
                             gridspec_kw={'hspace': 0.9, 'wspace': 0.2})

    for i in range(0, num_mixtures_sqrt):
        for j in range(0, num_mixtures_sqrt):
            mixture_idx = i * num_mixtures_sqrt + j
            if mixture_idx < num_mixtures:
                gmm = potential.get_single_mixture(mixture_idx)
                y = gmm.forward_negative_log(t)

                axes[i, j].plot(t.detach().cpu().numpy(), y.detach().cpu().numpy() -
                                torch.min(y).detach().cpu().numpy(), color='blue')
                axes[i, j].plot(t.detach().cpu().numpy(), torch.abs(t).detach().cpu().numpy(), color='orange')
                axes[i, j].set_title('idx={:d}'.format(mixture_idx), fontsize=8)
                axes[i, j].set_xlim(x_lower, x_upper)
                axes[i, j].axis('equal')
            else:
                fig.delaxes(axes[i, j])

    plt.savefig(os.path.join(fig_dir_path, file_name))
    plt.close(fig)
