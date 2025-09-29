import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

from pylopt.dataset.dataset_utils import collate_function
from pylopt.dataset.ImageDataset import TestImageDataset
from pylopt.energy import Energy, MeasurementModel
from pylopt.lower_problem import solve_lower
from pylopt.proximal_maps.ProximalOperator import DenoisingProx
from pylopt.regularisers import FieldsOfExperts, ImageFilter, StudentT, QuarticBSpline
from pylopt.utils.evaluation_utils import compute_psnr
from pylopt.utils.file_system_utils import get_repo_root_path
from pylopt.utils.logging_utils import setup_logger
from pylopt.utils.seeding_utils import seed_random_number_generators
from pylopt.utils.Timer import Timer
from pylopt.examples.scripts import PRETRAINED_FILTER_MODELS, PRETRAINED_POTENTIAL_MODELS

def visualise_denoising_results(u_clean: torch.Tensor, u_noisy: torch.Tensor, u_denoised: torch.Tensor) -> None:
    u_clean_splits = torch.split(u_clean, split_size_or_sections=1, dim=0)
    u_noisy_splits = torch.split(u_noisy, split_size_or_sections=1, dim=0)
    u_denoised_splits = torch.split(u_denoised, split_size_or_sections=1, dim=0)

                # import matplotlib.gridspec as gridspec
                # fig = plt.figure(figsize=(12, 4))
                
                # ax1 = fig.add_subplot(gs[0, 0])
                # ax2 = fig.add_subplot(gs[0, 1])
                # ax3 = fig.add_subplot(gs[0, 2])

                # plt.show()

    num_rows = len(u_clean_splits)
    num_cols = 3
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
                # gs = gridspec.GridSpec(1, 3, wspace=0.0005)



                # ax1 = fig.add_subplot(gs[0, 0])
                # ax2 = fig.add_subplot(gs[0, 1])
                # ax3 = fig.add_subplot(gs[0, 2])        

                    # from mpl_toolkits.axes_grid1 import ImageGrid


                    # fig = plt.figure(figsize=(4., 4.))
                    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    #                 nrows_ncols=(1, 3),  # creates 2x2 grid of Axes
                    #                 axes_pad=0.1,  # pad between Axes in inch.
                    #                 )


                    # for ax, im in zip(grid, u_clean_splits + u_noisy_splits + u_denoised_splits):
                    #     ax.imshow(im.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
                    #     ax.axis('off')


                    # plt.show()



    if num_rows == 1:
        axes = [axes]

    for idx, (item_clean, item_noisy, item_denoised) in (
            enumerate(zip(u_clean_splits, u_noisy_splits, u_denoised_splits))):
        psnr = compute_psnr(item_clean, item_denoised)
        print(' > psnr [dB] = {:.5f}'.format(psnr.detach().cpu().item()))

        axes[idx][0].imshow(item_clean.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
        axes[idx][1].imshow(item_noisy.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])
        axes[idx][2].imshow(item_denoised.squeeze().detach().cpu().numpy(), cmap=cmaps['gray'])

        axes[idx][0].axis('off')
        axes[idx][1].axis('off')
        axes[idx][2].axis('off')

        if idx == 0:
            axes[idx][0].set_title('clean')
            axes[idx][1].set_title('noisy')
            axes[idx][2].set_title('denoised')

    plt.show()
    plt.close(fig)

def denoise() -> None:
    """
        Function applying pretrained filters and potentials for image denoising on a set of
        test images.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    root_path = get_repo_root_path(Path(__file__))
    test_data_root_dir = os.path.join(root_path, 'data', 'images', 'test_images')
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)
    test_loader = DataLoader(test_image_dataset, 
                             batch_size=len(test_image_dataset), 
                             shuffle=False,
                             collate_fn=lambda x: collate_function(x, crop_size=-1))

    image_filter = ImageFilter.from_file(os.path.join(root_path, 'data', 'model_data', PRETRAINED_FILTER_MODELS['chen-ranftl-pock_2014_scaled_7x7']))
    potential = StudentT.from_file(os.path.join(os.getcwd(), 'data', 'model_data', PRETRAINED_POTENTIAL_MODELS['thaler_2025_student_t']))

    # potential = QuarticBSpline(num_marginals=48, multiplier=0.001, trainable=False)

    regulariser = FieldsOfExperts(potential, image_filter)

    u_clean = list(test_loader)[0]
    u_clean = u_clean.to(device=device, dtype=dtype)

    noise_level = 0.1
    lam = 10
    measurement_model = MeasurementModel(u_clean, operator=torch.nn.Identity(), noise_level=noise_level)
    energy = Energy(measurement_model, regulariser, lam=lam)
    energy.to(device=device, dtype=dtype)

    # energy.compile()

    method = 'adam'
    if method == 'nag':
        options = {'max_num_iterations': 1000, 
                   'rel_tol': 1e-5, 
                   'batch_optimisation': False, 
                   'lip_const': 1e1,
                   'resample_measurement_noise': False       # improves image reconstruction
                   }
    elif method == 'napg':
        prox = DenoisingProx(noise_level=noise_level)
        options = {'max_num_iterations': 1000, 
                   'rel_tol': 1e-5, 
                   'prox': prox, 
                   'batch_optimisation': False,
                   'lip_const': 1}
    elif method == 'adam':
        options = {'max_num_iterations': 1000, 
                   'rel_tol': 1e-5, 
                   'lr': 1e-3, 
                   'batch_optimisation': False, 
                   'resample_measurement_noise': False        # improves image reconstruction
                   }
    elif method == 'nag_unrolling':
        options = {'max_num_iterations': 35, 
                   'lip_const': 10, 
                   'batch_optimisation': False}
    elif method == 'napg_unrolling':
        options = {'max_num_iterations': 35, 
                   'lip_const': 10, 
                   'batch_optimisation': False}
    else:
        raise ValueError('Unknown solution method for lower level problem')

    with Timer(device=device) as t:
        lower_prob_result = solve_lower(energy=energy, method=method, options=options)

    print('denoising stats:')
    print(' > elapsed time [ms] = {:.5f}'.format(t.time_delta()))
    print(' > number of iterations = {:d}'.format(lower_prob_result.num_iterations))
    print(' > cause of termination = {:s}'.format(lower_prob_result.message))

    visualise_denoising_results(u_clean, measurement_model.get_noisy_observation(), lower_prob_result.solution)

def main():
    seed_random_number_generators(123)

    log_dir_path = './data'
    setup_logger(data_dir_path=log_dir_path, log_level_str='info')

    denoise()

if __name__ == '__main__':
    main()