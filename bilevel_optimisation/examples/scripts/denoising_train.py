import torch
from confuse import Configuration
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

from bilevel_optimisation.bilevel_problem.solve_bilevel import BilevelOptimisation
from bilevel_optimisation.callbacks import SaveModel, PlotFiltersAndPotentials, TrainingMonitor
from bilevel_optimisation.dataset.ImageDataset import TestImageDataset, TrainingImageDataset
from bilevel_optimisation.fields_of_experts import FieldsOfExperts
from bilevel_optimisation.filters import ImageFilter
from bilevel_optimisation.potential import StudentT
from bilevel_optimisation.proximal_maps.ProximalOperator import DenoisingProx
from bilevel_optimisation.scheduler import NAGLipschitzDelimiter
from bilevel_optimisation.utils.logging_utils import setup_logger
from bilevel_optimisation.utils.seeding_utils import seed_random_number_generators
from bilevel_optimisation.utils.file_system_utils import create_experiment_dir
from bilevel_optimisation.utils.config_utils import parse_datatype, load_app_config

def l2_loss_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum((x - y) ** 2)

def bilevel_learn(config: Configuration):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = parse_datatype(config)

    train_data_root_dir = config['data']['dataset']['train']['root_dir'].get()
    test_data_root_dir = config['data']['dataset']['test']['root_dir'].get()
    train_image_dataset = TrainingImageDataset(root_path=train_data_root_dir, dtype=dtype)
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)

    image_filter = ImageFilter(config)
    potential = StudentT(image_filter.get_num_filters(), config)
    regulariser = FieldsOfExperts(potential, image_filter)

    method_lower = 'napg'
    if method_lower == 'nag':
        options_lower = {'max_num_iterations': 300, 'rel_tol': 1e-5, 'lip_const': 1e5, 'batch_optimisation': False}
    elif method_lower == 'napg':
        noise_level = config['measurement_model']['noise_level'].get()
        options_lower = {'max_num_iterations': 300, 'rel_tol': 1e-5, 'lip_const': 1e5,
                         'prox': DenoisingProx(noise_level=noise_level), 'batch_optimisation': True}
    elif method_lower == 'adam':
        options_lower = {'max_num_iterations': 1000, 'rel_tol': 5e-4, 'lr': 1e-3, 'batch_optimisation': True}
    else:
        raise ValueError('Unknown solution method for lower level problem.')

    path_to_eval_dir = create_experiment_dir(config)
    bilevel_optimisation = BilevelOptimisation(method_lower, options_lower, config, solver='cg',
                                               differentiation_method='implicit',
                                               options_solver={'max_num_iterations': 500},
                                               path_to_experiments_dir=path_to_eval_dir)
    lam = config['energy']['lam'].get()

    tb_writer = SummaryWriter(log_dir=os.path.join(path_to_eval_dir, 'tensorboard'))
    callbacks = [PlotFiltersAndPotentials(path_to_data_dir=path_to_eval_dir, plotting_freq=2, tb_writer=tb_writer),
                 SaveModel(path_to_data_dir=path_to_eval_dir, save_freq=2),
                 TrainingMonitor(test_image_dataset, config, method_lower, options_lower, l2_loss_func, path_to_eval_dir,
                                    evaluation_freq=2, tb_writer=tb_writer)]

    method_upper = 'nag'
    if method_upper == 'nag':
        options_upper = {'max_num_iterations': 3000, 'lip_const': [1, 1], 'alternating': True}
    elif method_upper == 'adam':
        options_upper = {'max_num_iterations': 3000, 'lr': [1e-3], 'alternating': True}
    elif method_upper == 'lbfgs':
        options_upper = {'max_num_iterations': 500, 'max_iter': 10, 'history_size': 10,
                         'line_search_fn': 'strong_wolfe'}
    else:
        raise ValueError('Unknown optimisation method for upper level problem.')

    schedulers = [NAGLipschitzDelimiter(lip_const_bound=2 ** 18, lip_const_key='lip_const')]
    bilevel_optimisation.learn(regulariser, lam, l2_loss_func, train_image_dataset,
                               optimisation_method_upper=method_upper, optimisation_options_upper=options_upper,
                               dtype=dtype, device=device, callbacks=callbacks, schedulers=schedulers)

def main():
    seed_random_number_generators(123)

    log_dir_path = './data'
    setup_logger(data_dir_path=log_dir_path, log_level_str='info')

    # specify config directory via command line argument
    #   1. Usage of custom configuration contained in bilevel_optimisation.config_data.custom
    #       python example_denoising_train.py --configs example_prediction_I
    #   2. Usage of custom configuration in file system
    #       python example_denoising_train.py --configs <path_to_custom_config_dir>
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs')
    args = parser.parse_args()
    app_name = 'bilevel_optimisation'
    configuring_module = '[DENOISING] train'
    config = load_app_config(app_name, args.configs, configuring_module)

    bilevel_learn(config)

if __name__ == '__main__':
    main()
