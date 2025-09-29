import torch
from confuse import Configuration
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from pylopt.bilevel_problem import BilevelOptimisation
from pylopt.callbacks import SaveModel, PlotFiltersAndPotentials, TrainingMonitor
from pylopt.dataset.ImageDataset import TestImageDataset, TrainingImageDataset
from pylopt.examples.scripts import PRETRAINED_FILTER_MODELS, PRETRAINED_POTENTIAL_MODELS
from pylopt.proximal_maps.ProximalOperator import DenoisingProx
from pylopt.regularisers.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from pylopt.regularisers.fields_of_experts.ImageFilter import ImageFilter
from pylopt.regularisers.fields_of_experts.potential import StudentT, Spline
from pylopt.scheduler import (NAGLipConstGuard, CosineAnnealingLRScheduler, AdaptiveLRRestartScheduler,
                                            restart_condition_loss_based, restart_condition_gradient_based)
from pylopt.utils.logging_utils import setup_logger
from pylopt.utils.seeding_utils import seed_random_number_generators
from pylopt.utils.file_system_utils import create_experiment_dir, get_repo_root_path
from pylopt.utils.config_utils import parse_datatype, load_app_config

def l2_loss_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum((x - y) ** 2)

def bilevel_learn(config: Configuration) -> None:
    """
        Function for training of filters and potentials in the context 
        of image denoising.

        NOTE
        ----
            > For training, it is NOT strictly required to use configs. All the
                parameters can be specified explicitely in the corresponding constructors. If no configs shall
                be used, this implementation has to be updated accordingely - see denoising_predict.py.
            > Configurations can be specified by means of yaml files, which are processed
                using the Python package confuse. 
            > If an instance of BilevelOptimisation, is configured using a configuration object, 
                a yaml file containing all the specified settings is written to file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = parse_datatype(config)

    train_data_root_dir = config['data']['dataset']['train']['root_dir'].get()
    test_data_root_dir = config['data']['dataset']['test']['root_dir'].get()
    train_image_dataset = TrainingImageDataset(root_path=train_data_root_dir, dtype=dtype)
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)

    # --- SETUP OF REGULARISER  

    # --- Ad example_training_I
    #   > Filters are loaded from file
    repo_root_path = get_repo_root_path(Path(__file__))

    image_filter = ImageFilter.from_config(config)
    # image_filter = ImageFilter.from_file(os.path.join(repo_root_path, 
    #                                                   'data', 'model_data',
    #                                                   PRETRAINED_FILTER_MODELS['chen-ranftl-pock_2014_scaled_7x7']))
    # image_filter.freeze()
    
    # --- Ad example_training_II
    #   > Filters will be configured by means of config
    # image_filter = ImageFilter.from_config(config)

    potential = StudentT.from_config(config)

    image_filter.unfreeze()
    potential.unfreeze()

    regulariser = FieldsOfExperts(potential, image_filter)

    method_lower = 'napg'
    if method_lower == 'nag':
        options_lower = {'max_num_iterations': 300, 'rel_tol': 1e-5, 'lip_const': 1e5, 'batch_optimisation': False}
    elif method_lower == 'napg':
        noise_level = config['measurement_model']['noise_level'].get()
        options_lower = {'max_num_iterations': 300, 'rel_tol': 1e-5, 'lip_const': 1e5,
                         'prox': DenoisingProx(noise_level=noise_level), 'batch_optimisation': False}
    elif method_lower == 'adam':
        options_lower = {'max_num_iterations': 1000, 'rel_tol': 5e-4, 'lr': 1e-3, 'batch_optimisation': True}
    else:
        raise ValueError('Unknown solution method for lower level problem.')

    path_to_eval_dir = create_experiment_dir(config)
    bilevel_optimisation = BilevelOptimisation(method_lower, 
                                               options_lower, 
                                               config=config, 
                                               differentiation_method='implicit',
                                               solver_name='cg',                            # implicit differentiation requires linear system solver
                                               options_solver={'max_num_iterations': 500},
                                               path_to_experiments_dir=path_to_eval_dir)
    lam = config['energy']['lam'].get()

    tb_writer = SummaryWriter(log_dir=os.path.join(path_to_eval_dir, 'tensorboard'))
    callbacks = [PlotFiltersAndPotentials(test_image_dataset, path_to_data_dir=path_to_eval_dir,
                                          plotting_freq=2, tb_writer=tb_writer),
                 SaveModel(path_to_data_dir=path_to_eval_dir, save_freq=2),
                 TrainingMonitor(test_image_dataset, config, method_lower, options_lower, l2_loss_func,
                                 path_to_eval_dir, evaluation_freq=1, tb_writer=tb_writer)]

    method_upper = 'adam'
    max_num_iterations = 10000
    if method_upper == 'nag':
        options_upper = {'max_num_iterations': max_num_iterations, 'lip_const': [1000], 'alternating': True}
    elif method_upper == 'adam':
        options_upper = {'max_num_iterations': max_num_iterations, 'lr': [1e-3, 1e-1], 'alternating': True}
    elif method_upper == 'lbfgs':
        options_upper = {'max_num_iterations': max_num_iterations, 'max_iter': 10, 'history_size': 10,
                         'line_search_fn': 'strong_wolfe'}
    else:
        raise ValueError('Unknown optimisation method for upper level problem.')

    # --- SCHEDULING ---

    # --- Schedulers for upper level optimisation affecting lip_const (NAG)
    #   
    # schedulers = [NAGLipConstGuard(lip_const_bound=2 ** 17, lip_const_key='lip_const')]

    # --- Schedulers for upper level optimisation affecting the learning rate (Adam, LBFGS)
    #
    schedulers = [CosineAnnealingLRScheduler(step_begin=5, 
                                             restart_cycle=None,
                                             step_end=8500,
                                             lr_min=1e-5)]

    # schedulers = [AdaptiveLRRestartScheduler(restart_condition_gradient_based, warm_up_period=2)]
    


    bilevel_optimisation.learn(regulariser, 
                               lam, 
                               l2_loss_func, 
                               train_image_dataset,
                               optimisation_method_upper=method_upper, 
                               optimisation_options_upper=options_upper,
                               dtype=dtype, device=device, 
                               callbacks=callbacks, 
                               schedulers=schedulers, 
                               do_compile=True)

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

    path_to_config_super_dir = os.path.join(Path(__file__).parent, 'config_data')   
    config = load_app_config(app_name, 
                             path_to_config_super_dir,              # super directory of 'defualt', 'custom'
                             os.path.join('custom', args.configs),  # one of 'example_training_I', ...
                             configuring_module)

    bilevel_learn(config)

if __name__ == '__main__':
    main()
