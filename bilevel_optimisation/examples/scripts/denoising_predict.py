import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cmaps
import torch
from torch.utils.data import DataLoader
from confuse import Configuration
import argparse

from bilevel_optimisation.dataset.ImageDataset import TestImageDataset
from bilevel_optimisation.energy.InnerEnergy import UnrollingEnergy
from bilevel_optimisation.evaluation.Evaluation import compute_psnr
from bilevel_optimisation.utils.ConfigUtils import load_app_config, parse_datatype
from bilevel_optimisation.utils.DatasetUtils import collate_function
from bilevel_optimisation.utils.LoggingUtils import setup_logger
from bilevel_optimisation.utils.SeedingUtils import seed_random_number_generators
from bilevel_optimisation.utils.SetupUtils import (set_up_regulariser, set_up_measurement_model,
                                                   set_up_inner_energy)
from bilevel_optimisation.utils.TimerUtils import Timer
from bilevel_optimisation.visualisation.Visualisation import visualise_filter_responses


def load_data_from_file(root_path: str, file_name: str) -> torch.Tensor:
    file_path = os.path.join(root_path, file_name)
    return torch.load(file_path)

def denoise(config: Configuration):
    device = torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = parse_datatype(config)

    test_data_root_dir = config['data']['dataset']['test']['root_dir'].get()
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)
    test_loader = DataLoader(test_image_dataset, batch_size=len(test_image_dataset), shuffle=False,
                             collate_fn=lambda x: collate_function(x, crop_size=-1))

    regulariser = set_up_regulariser(config)
    regulariser = regulariser.to(device=device, dtype=dtype)

    test_batch = list(test_loader)[0]
    test_batch_ = test_batch.to(device=device, dtype=dtype)

    measurement_model_ = set_up_measurement_model(test_batch_, config)
    energy = set_up_inner_energy(measurement_model_, regulariser, config)
    energy.to(device=device, dtype=dtype)

    # ###
    num_inferences = 10
    time_list = []
    for _ in range(0, num_inferences):

        with Timer(device=device) as t:
            test_batch_denoised = energy.argmin(energy.measurement_model.obs_noisy)
            if type(energy).__name__ == UnrollingEnergy.__name__:
                num_unrolling_cycles = 10
                for i in range(0, num_unrolling_cycles):
                    test_batch_denoised = energy.argmin(test_batch_denoised)

        time_list.append(t.time_delta())

    import pandas as pd
    df = pd.DataFrame(time_list, columns=['Elapsed time [s]'])
    df.to_csv('out.csv')

    print(np.mean(time_list))




    # ###

    # visualise_filter_responses(regulariser, test_batch_denoised)

    print('denoising stats:')
    print(' > elapsed time [s] = {:.5f}'.format(t.time_delta()))

    u_clean_splits = torch.split(test_batch_, split_size_or_sections=1, dim=0)
    u_noisy_splits = torch.split(energy.measurement_model.obs_noisy, split_size_or_sections=1, dim=0)
    u_denoised_splits = torch.split(test_batch_denoised, split_size_or_sections=1, dim=0)
    for idx, (item_clean, item_noisy, item_denoised) in (
            enumerate(zip(u_clean_splits, u_noisy_splits, u_denoised_splits))):
        psnr = compute_psnr(item_clean, item_denoised)
        print(' > psnr [dB] = {:.5f}'.format(psnr.detach().cpu().item()))

        fig = plt.figure()
        ax_clean = fig.add_subplot(1, 3, 1)
        ax_clean.imshow(item_clean.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
        ax_clean.set_title('clean')
        ax_clean.xaxis.set_visible(False)
        ax_clean.yaxis.set_visible(False)

        ax_noisy = fig.add_subplot(1, 3, 2)
        ax_noisy.imshow(item_noisy.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
        ax_noisy.set_title('noisy')
        ax_noisy.xaxis.set_visible(False)
        ax_noisy.yaxis.set_visible(False)

        ax_denoised = fig.add_subplot(1, 3, 3)
        ax_denoised.imshow(item_denoised.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
        ax_denoised.set_title('denoised')
        ax_denoised.xaxis.set_visible(False)
        ax_denoised.yaxis.set_visible(False)

        plt.show()
        plt.close(fig)

def main():
    seed_random_number_generators(123)

    log_dir_path = './data'
    setup_logger(data_dir_path=log_dir_path, log_level_str='info')

    # specify config directory via command line argument
    #   1. Usage of custom configuration contained in bilevel_optimisation.config_data.custom
    #       python example_denoising_predict.py --configs example_prediction_I
    #   2. Usage of custom configuration in file system
    #       python example_denoising_predict.py --configs <path_to_custom_config_dir>
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs')
    args = parser.parse_args()
    app_name = 'bilevel_optimisation'
    configuring_module = '[DENOISING] predict'
    config = load_app_config(app_name, args.configs, configuring_module)

    denoise(config)

if __name__ == '__main__':
    main()