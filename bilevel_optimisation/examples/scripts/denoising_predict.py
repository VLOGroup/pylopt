import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
import torch
from torch.utils.data import DataLoader
from confuse import Configuration
import argparse

from bilevel_optimisation.dataset.ImageDataset import TestImageDataset
from bilevel_optimisation.energy import Energy
from bilevel_optimisation.fields_of_experts import FieldsOfExperts
from bilevel_optimisation.filters import ImageFilter
from bilevel_optimisation.lower_problem import solve_lower
from bilevel_optimisation.measurement_model import MeasurementModel
from bilevel_optimisation.potential import StudentT
from bilevel_optimisation.utils.config_utils import load_app_config, parse_datatype
from bilevel_optimisation.utils.dataset_utils import collate_function
from bilevel_optimisation.utils.evaluation_utils import compute_psnr
from bilevel_optimisation.utils.logging_utils import setup_logger
from bilevel_optimisation.utils.seeding_utils import seed_random_number_generators
from bilevel_optimisation.utils.Timer import Timer

def visualise_denoising_results(u_clean: torch.Tensor, u_noisy: torch.Tensor, u_denoised: torch.Tensor) -> None:
    u_clean_splits = torch.split(u_clean, split_size_or_sections=1, dim=0)
    u_noisy_splits = torch.split(u_noisy, split_size_or_sections=1, dim=0)
    u_denoised_splits = torch.split(u_denoised, split_size_or_sections=1, dim=0)
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

def denoise(config: Configuration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = parse_datatype(config)

    test_data_root_dir = config['data']['dataset']['test']['root_dir'].get()
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)
    test_loader = DataLoader(test_image_dataset, batch_size=len(test_image_dataset), shuffle=False,
                             collate_fn=lambda x: collate_function(x, crop_size=-1))

    image_filter = ImageFilter(config)
    potential = StudentT(image_filter.get_num_filters(), config)
    regulariser = FieldsOfExperts(potential, image_filter)

    u_clean = list(test_loader)[0]
    u_clean = u_clean.to(device=device, dtype=dtype)
    measurement_model = MeasurementModel(u_clean, config)
    lam = 0.5
    energy = Energy(measurement_model, regulariser, lam)
    energy.to(device=device, dtype=dtype)

    options_adam = {'max_num_iterations': 1000, 'rel_tol': 1e-4, 'lr': [1e-3, 1e-4], 'batch_optimisation': False}
    options_nag = {'max_num_iterations': 1000, 'rel_tol': 1e-5, 'beta': [0.71, 0.87], 'batch_optimisation': False}
    options_napg = {'max_num_iterations': 1000, 'rel_tol': 1e-5}
    options_nag_unrolling = {'max_num_iterations': 10, 'rel_tol': 1e-5, 'alpha': [1e-3, 1e-7], 'batch_optimisation': False}
    # options_nag_unrolling = {'max_num_iterations': 10, 'rel_tol': 1e-5, 'lip_const': [1e4]}

    with Timer(device=device) as t:
        lower_prob_result = solve_lower(energy=energy, method='nag_unrolling', options=options_nag_unrolling)

    print('denoising stats:')
    print(' > elapsed time [ms] = {:.5f}'.format(t.time_delta()))
    print(' > number of iterations = {:d}'.format(lower_prob_result.num_iterations))
    print(' > cause of termination = {:s}'.format(lower_prob_result.message))

    visualise_denoising_results(u_clean, measurement_model.get_noisy_observation(), lower_prob_result.solution)

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