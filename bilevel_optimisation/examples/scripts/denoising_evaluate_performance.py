import torch
from torch.utils.data import DataLoader
from confuse import Configuration
import argparse
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps


from bilevel_optimisation.dataset.ImageDataset import TestImageDataset
from bilevel_optimisation.energy import Energy
from bilevel_optimisation.fields_of_experts import FieldsOfExperts
from bilevel_optimisation.filters import ImageFilter
from bilevel_optimisation.lower_problem import solve_lower
from bilevel_optimisation.measurement_model import MeasurementModel
from bilevel_optimisation.potential import StudentT
from bilevel_optimisation.proximal_maps.ProximalOperator import DenoisingProx
from bilevel_optimisation.utils.config_utils import load_app_config, parse_datatype
from bilevel_optimisation.utils.dataset_utils import collate_function
from bilevel_optimisation.utils.evaluation_utils import compute_psnr
from bilevel_optimisation.utils.logging_utils import setup_logger
from bilevel_optimisation.utils.seeding_utils import seed_random_number_generators
from bilevel_optimisation.utils.Timer import Timer

def evaluate_performance(config: Configuration):
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
    measurement_model = MeasurementModel(u_clean, config=config)
    lam = config['energy']['lam'].get()
    energy = Energy(measurement_model, regulariser, lam)
    energy.to(device=device, dtype=dtype)

    noise_level = config['measurement_model']['noise_level'].get()
    prox = DenoisingProx(noise_level=noise_level)
    options_napg = {'max_num_iterations': 300, 'rel_tol': 1e-5, 'prox': prox, 'batch_optimisation': False}

    with Timer(device=device) as t:
        lower_prob_result = solve_lower(energy=energy, method='napg', options=options_napg)

    psnr = torch.mean(compute_psnr(u_clean, lower_prob_result.solution))
    psnr = psnr.detach().cpu().item()

    print('denoising performance:')
    print(' > denoised {:d} images'.format(u_clean.shape[0]))
    print(' > elapsed time [ms] = {:.5f}'.format(t.time_delta()))
    print(' > number of iterations = {:d}'.format(lower_prob_result.num_iterations))
    print(' > cause of termination = {:s}'.format(lower_prob_result.message))
    print(' > average psnr: {:.5f}'.format(psnr))

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
    configuring_module = '[DENOISING] evaluate'
    config = load_app_config(app_name, args.configs, configuring_module)

    evaluate_performance(config)

if __name__ == '__main__':
    main()