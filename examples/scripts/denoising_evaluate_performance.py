import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os

from pylopt.dataset.ImageDataset import TestImageDataset
from pylopt.energy import Energy, MeasurementModel
from pylopt.regularisers.fields_of_experts import FieldsOfExperts, ImageFilter, StudentT
from pylopt.lower_problem import solve_lower
from pylopt.proximal_maps.ProximalOperator import DenoisingProx
from pylopt.dataset.dataset_utils import collate_function
from pylopt.utils.file_system_utils import get_repo_root_path
from pylopt.utils.evaluation_utils import compute_psnr
from pylopt.utils.logging_utils import setup_logger
from pylopt.utils.seeding_utils import seed_random_number_generators
from pylopt.utils.Timer import Timer

def evaluate_performance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    test_data_root_dir = '/home/florianthaler/Documents/data/image_data/BSDS68_rotated'
    test_image_dataset = TestImageDataset(root_path=test_data_root_dir, dtype=dtype)
    test_loader = DataLoader(test_image_dataset, batch_size=len(test_image_dataset), shuffle=False,
                             collate_fn=lambda x: collate_function(x, crop_size=-1))

    repo_root_path = get_repo_root_path(Path(__file__))
    image_filter = ImageFilter.from_file(os.path.join(repo_root_path, 'data', 'model_data', 'filters_iter_3796.pt'))
    potential = StudentT.from_file(os.path.join(repo_root_path, 'data', 'model_data', 'potential_iter_3796.pt'))
    regulariser = FieldsOfExperts(potential, image_filter)

    lam = 100
    noise_level = 0.1
    operator = torch.nn.Identity()
    u_clean = list(test_loader)[0]
    u_clean = u_clean.to(device=device, dtype=dtype)
    measurement_model = MeasurementModel(u_clean, operator=operator, noise_level=noise_level)
    
    energy = Energy(measurement_model, regulariser, lam)
    energy.to(device=device, dtype=dtype)

    prox = DenoisingProx(noise_level=noise_level)
    options_napg = {'max_num_iterations': 1000, 'rel_tol': 1e-7, 'prox': prox, 'batch_optimisation': False}

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
    evaluate_performance()

if __name__ == '__main__':
    main()