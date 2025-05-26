import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from confuse import Configuration
import argparse

from bilevel_optimisation.dataset.ImageDataset import TestImageDataset, TrainingImageDataset
from bilevel_optimisation.evaluation.Evaluation import evaluate_on_test_data
from bilevel_optimisation.utils.DatasetUtils import collate_function
from bilevel_optimisation.utils.LoggingUtils import (setup_logger, log_trainable_params_stats,
                                                     log_gradient_norms)
from bilevel_optimisation.utils.SeedingUtils import seed_random_number_generators
from bilevel_optimisation.utils.FileSystemUtils import create_evaluation_dir, save_foe_model
from bilevel_optimisation.utils.SetupUtils import (set_up_regulariser, set_up_bilevel_problem,
                                                   set_up_measurement_model, set_up_inner_energy,
                                                   set_up_outer_loss)
from bilevel_optimisation.utils.ConfigUtils import load_configs, parse_datatype
from bilevel_optimisation.visualisation.Visualisation import (visualise_training_stats, visualise_filter_stats,
                                                              visualise_filters, visualise_gmm_potential)

def visualise_intermediate_results(regulariser: torch.nn.Module, device: torch.device, dtype: torch.dtype,
                                   curr_iter: int, path_to_data_dir: str, filter_image_subdir: str = 'filters',
                                   potential_image_subdir: str = 'potential') -> None:
    # visualise filters
    filter_images_dir_path = os.path.join(path_to_data_dir, filter_image_subdir)
    if not os.path.exists(filter_images_dir_path):
        os.makedirs(filter_images_dir_path, exist_ok=True)
    visualise_filters(regulariser.get_filters(), regulariser.get_filter_weights(),
                      fig_dir_path=filter_images_dir_path, file_name='filters_iter_{:d}.png'.format(curr_iter + 1))

    # visualise potential functions (per filter)
    potential = regulariser.get_potential()
    if type(potential).__name__ == 'GaussianMixture':
        potential_images_dir_path = os.path.join(path_to_data_dir, potential_image_subdir)
        if not os.path.exists(potential_images_dir_path):
            os.makedirs(potential_images_dir_path, exist_ok=True)

        visualise_gmm_potential(potential, device, dtype, fig_dir_path=potential_images_dir_path,
                                file_name='potential_iter_{:d}.png'.format(curr_iter + 1))

def train_bilevel(config: Configuration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = parse_datatype(config)

    train_data_root_dir = config['data']['dataset']['train']['root_dir'].get()
    train_image_dataset = TrainingImageDataset(root_path=train_data_root_dir, dtype=dtype)
    batch_size = config['data']['dataset']['train']['batch_size'].get()
    crop_size = config['data']['dataset']['train']['crop_size'].get()
    train_loader = DataLoader(train_image_dataset, batch_size=batch_size,
                              collate_fn=lambda x: collate_function(x, crop_size=crop_size))

    test_data_root_dir = config['data']['dataset']['test']['root_dir'].get()
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)
    test_loader = DataLoader(test_image_dataset, batch_size=len(test_image_dataset), shuffle=False,
                             collate_fn=lambda x: collate_function(x, crop_size=-1))

    regulariser = set_up_regulariser(config)
    regulariser = regulariser.to(device=device, dtype=torch.float64)
    bilevel = set_up_bilevel_problem(regulariser.parameters(), config)
    bilevel = bilevel.to(device=device, dtype=torch.float64)

    log_trainable_params_stats(regulariser, logging_module='train')
    path_to_eval_dir = create_evaluation_dir(config)

    logging.info('[TRAIN] compute initial test loss and initial psnr')
    psnr, test_loss = evaluate_on_test_data(test_loader, regulariser, config, device,
                                            dtype, -1, path_to_data_dir=None)

    train_loss_list = []
    test_loss_list = [test_loss]
    psnr_list = [psnr]

    filters_list = []
    filter_weights_list = []

    writer = SummaryWriter(log_dir=os.path.join(path_to_eval_dir, 'tensorboard'))

    evaluation_freq = 2
    max_num_iterations = 2000
    for k, batch in enumerate(train_loader):

        batch_ = batch.to(device=device, dtype=dtype)
        with torch.no_grad():
            measurement_model = set_up_measurement_model(batch_, config)
            inner_energy = set_up_inner_energy(measurement_model, regulariser, config)
            inner_energy = inner_energy.to(device=device, dtype=dtype)

            outer_loss = set_up_outer_loss(batch_, config)
            train_loss = bilevel.forward(outer_loss, inner_energy)

            train_loss_list.append(train_loss.detach().cpu().item())
            filters_list.append(regulariser.get_filters())
            filter_weights_list.append(regulariser.get_filter_weights())
            logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                         'loss = {:.5f}'.format(k + 1, max_num_iterations, train_loss.detach().cpu().item()))

            log_gradient_norms(regulariser, writer, k + 1)

            if (k + 1) % evaluation_freq == 0:
                logging.info('[TRAIN] evaluate on test dataset')

                psnr, test_loss = evaluate_on_test_data(test_loader, regulariser, config, device,
                                                        dtype, k, path_to_eval_dir)
                visualise_intermediate_results(regulariser, device, dtype, k, path_to_eval_dir)

                logging.info('[TRAIN] denoised test images')
                logging.info('[TRAIN]   > average psnr: {:.5f}'.format(psnr))
                logging.info('[TRAIN]   > test loss: {:.5f}'.format(test_loss))

                psnr_list.append(psnr)
                test_loss_list.append(test_loss)

            if (k + 1) == max_num_iterations:
                logging.info('[TRAIN] reached maximal number of iterations')
                writer.close()
                break
            else:
                k += 1

    save_foe_model(regulariser, path_to_eval_dir)

    visualise_training_stats(train_loss_list, test_loss_list, psnr_list, evaluation_freq, path_to_eval_dir)
    visualise_filter_stats(filters_list, filter_weights_list, path_to_eval_dir)

def main():
    seed_random_number_generators(123)

    log_dir_path = './data'
    setup_logger(data_dir_path=log_dir_path, log_level_str='info')

    # specify config directory by calling f.e.
    #   python example_denoising_train.py --configs example_training_IV
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs')
    args = parser.parse_args()
    default_config_dir_path = os.path.join('./data', 'configs', 'default')
    custom_config_dir_path = os.path.join('./data', 'configs', 'custom', args.configs)
    config = load_configs('[DENOISING] train', default_config_dir_path=default_config_dir_path,
                          custom_config_dir_path=custom_config_dir_path, configuring_module='train')
    train_bilevel(config)

if __name__ == '__main__':
    main()
