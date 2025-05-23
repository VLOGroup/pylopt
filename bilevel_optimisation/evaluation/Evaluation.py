import os
import torch
from confuse import Configuration
from typing import Tuple, Optional
import numpy as np

from bilevel_optimisation.utils.SetupUtils import set_up_measurement_model, set_up_inner_energy, \
    set_up_outer_loss
from bilevel_optimisation.visualisation.Visualisation import visualise_test_triplets

def compute_psnr(y_true: torch.Tensor, y_pred: torch.Tensor, max_pix_value: float = 1.0) -> torch.Tensor:
    mse = torch.mean(((y_true - y_pred) ** 2), dim=(-2, -1))
    return 20 * torch.log10(max_pix_value / torch.sqrt(mse))

def evaluate_on_test_data(test_loader: torch.utils.data.DataLoader, regulariser: torch.nn.Module,
                          config: Configuration, device: torch.device, dtype: torch.dtype,
                          curr_iter: int, path_to_data_dir: Optional[str],
                          denoised_image_subdir: str = 'denoised') -> Tuple[float, float]:
    test_batch = list(test_loader)[0]
    test_batch_ = test_batch.to(device=device, dtype=dtype)

    measurement_model_ = set_up_measurement_model(test_batch_, config)
    energy = set_up_inner_energy(measurement_model_, regulariser, config)
    energy.to(device=device, dtype=dtype)

    test_batch_denoised = energy.argmin(energy.measurement_model.obs_noisy)
    psnr = np.mean(compute_psnr(energy.measurement_model.obs_clean(), test_batch_denoised).detach().cpu().numpy())
    outer_loss = set_up_outer_loss(test_batch_, config)
    test_loss = outer_loss(test_batch_denoised).detach().cpu().numpy()

    if path_to_data_dir:
        denoised_images_dir_path = os.path.join(path_to_data_dir, denoised_image_subdir)
        if not os.path.exists(denoised_images_dir_path):
            os.makedirs(denoised_images_dir_path, exist_ok=True)
        visualise_test_triplets(test_batch_, energy.measurement_model.obs_noisy,
                                test_batch_denoised, fig_dir_path=denoised_images_dir_path,
                                file_name_pre_fix='test_triplet_iter_{:d}'.format(curr_iter + 1))

    return psnr, test_loss