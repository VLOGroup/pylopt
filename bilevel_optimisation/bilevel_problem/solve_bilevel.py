from typing import Callable, Any, Optional, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import math

from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.losses import BaseLoss
from bilevel_optimisation.energy.Energy import Energy
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.solver.CGSolver import CGSolver
from bilevel_optimisation.lower_problem import add_group_options, solve_lower
from bilevel_optimisation.optimise import step_adam, create_projected_optimiser, step_nag
from bilevel_optimisation.optimise.optimise_adam import harmonise_param_groups_adam
from bilevel_optimisation.optimise.optimise_nag import harmonise_param_groups_nag
from bilevel_optimisation.utils.dataset_utils import collate_function
from bilevel_optimisation.bilevel.gradients import OptimisationAutograd

def assemble_param_groups_base(regulariser: FieldsOfExperts, single_group_optimisation: bool=True):
    param_groups = []
    filter_params = regulariser.get_image_filter().parameters()
    filter_params_trainable = [p for p in filter_params if p.requires_grad]

    potential_params = regulariser.get_potential().parameters()
    potential_params_trainable = [p for p in potential_params if p.requires_grad]

    if single_group_optimisation:
        group = {'params': filter_params_trainable + potential_params_trainable}
        param_groups.append(group)
    else:
        for params in [filter_params_trainable, potential_params_trainable]:
            group = {'params': params}
            param_groups.append(group)

    return param_groups

def assemble_param_groups_adam(regulariser: FieldsOfExperts, lr: List[Optional[float]]=None,
                               betas: List[Optional[Tuple[float, float]]]=None, eps: List[Optional[float]]=None,
                               weight_decay: List[Optional[float]]=None,
                               single_group_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, single_group_optimisation)

    lr = [None for _ in range(0, len(param_groups))] if not lr else lr
    betas = [None for _ in range(0, len(param_groups))] if not betas else betas
    eps = [None for _ in range(0, len(param_groups))] if not eps else eps
    weight_decay = [None for _ in range(0, len(param_groups))] if not weight_decay else weight_decay
    add_group_options(param_groups, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})

    return param_groups

def assemble_param_groups_nag(regulariser: FieldsOfExperts, alpha: List[Optional[float]]=None,
                              beta: List[Optional[float]]=None,
                              lip_const: List[float]=None,
                              single_group_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, single_group_optimisation)

    alpha = [None for _ in range(0, len(param_groups))] if not alpha else alpha
    beta = [None for _ in range(0, len(param_groups))] if not beta else beta
    lip_const = [None for _ in range(0, len(param_groups))] if not lip_const else lip_const
    add_group_options(param_groups, {'alpha': alpha, 'beta': beta, 'lip_const': lip_const})

    return param_groups

def compute_psnr(y_true: torch.Tensor, y_pred: torch.Tensor, max_pix_value: float = 1.0) -> torch.Tensor:
    mse = torch.mean(((y_true - y_pred) ** 2), dim=(-2, -1))
    return 20 * torch.log10(max_pix_value / torch.sqrt(mse))

def arrange_filters_on_grid(regulariser: FieldsOfExperts) -> torch.Tensor:
    filter_tensor = regulariser.get_image_filter().get_filter_tensor()
    num_filters = filter_tensor.shape[0]
    grid_size = int(math.sqrt(num_filters)) + 1

    return make_grid(filter_tensor, nrow=grid_size, padding=2)

def arrange_potentials_on_grid(regulariser: FieldsOfExperts):

    potential = regulariser.get_potential()
    pass

class BilevelOptimisation:

    def __init__(self, forward_operator: torch.nn.Module, noise_level: float,
                 method_lower: str, options_lower: Dict[str, Any],
                 solver: Optional[str]='cg', options_solver: Optional[Dict[str, Any]]=None,
                 path_to_experiments_dir: Optional[str]=None) -> None:
        if solver == 'cg':
            options_solver = options_solver if options_solver is not None else {}
            self.solver = CGSolver(**options_solver)
        else:
            raise NotImplementedError

        self.backward_mode = 'optimisation'
        if 'unrolling' in method_lower:
            self.backward_mode = 'unrolling'

        self.forward_operator = forward_operator
        self.noise_level = noise_level

        self.method_lower = method_lower
        self.options_lower = options_lower

        path_to_experiments_dir = os.getcwd() if path_to_experiments_dir is None else path_to_experiments_dir
        self.writer = SummaryWriter(log_dir=os.path.join(path_to_experiments_dir, 'tensorboard'))

    def _loss_func_factory(self, upper_loss, energy, u_clean, denoising_func):
        if self.backward_mode == 'unrolling':
            pass

        else:
            return lambda params: OptimisationAutograd.apply(energy, denoising_func,
                                                                     lambda z: upper_loss(u_clean, z), self.solver,
                                                                     params)

    def evaluate(self, regulariser: FieldsOfExperts, lam: float, test_loader: DataLoader, upper_loss_func: Callable,
                 dtype: torch.dtype, device: torch.device, curr_iter_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        test_batch_clean = list(test_loader)[0]
        test_batch_clean_ = test_batch_clean.to(device=device, dtype=dtype)

        measurement_model = MeasurementModel(test_batch_clean_, self.forward_operator, self.noise_level)
        energy = Energy(measurement_model, regulariser, lam)
        energy.to(device=device, dtype=dtype)

        test_batch_noisy = measurement_model.obs_noisy
        test_batch_denoised = solve_lower(test_batch_noisy, energy, self.method_lower, self.options_lower).solution

        psnr = torch.mean(compute_psnr(energy.measurement_model.obs_clean(), test_batch_denoised))
        loss = upper_loss_func(test_batch_clean_, test_batch_denoised)
        triplets = torch.cat([test_batch_clean_, test_batch_noisy, test_batch_denoised], dim=3)

        self.writer.add_scalar('loss/test', loss, curr_iter_idx + 1)
        self.writer.add_scalar('psnr/test', psnr, curr_iter_idx + 1)
        self.writer.add_image('triplets/test', make_grid(triplets, nrow=1), curr_iter_idx + 1)
        self.writer.add_image('filters', arrange_filters_on_grid(regulariser), curr_iter_idx + 1)
        # self.writer.add_image('potentials')

        return psnr, loss

    def learn(self, regulariser: FieldsOfExperts, lam: float, upper_loss_func: Callable, dataset_train, dataset_test,
              optimisation_method_upper: str, optimisation_options_upper: Dict[str, Any],
              batch_size: int=32, crop_size: int=64, dtype: torch.dtype=torch.float32,
              device: Optional[torch.device]=None, evaluation_freq: int=2):

        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(dataset_train, batch_size=batch_size,
                                  collate_fn=lambda x: collate_function(x, crop_size=crop_size))
        test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False,
                                 collate_fn=lambda x: collate_function(x, crop_size=-1))

        if optimisation_method_upper == 'nag':
            self._learn_nag(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader, test_loader,
                            evaluation_freq, dtype, device)
        elif optimisation_method_upper == 'adam':
            self._learn_adam(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader, test_loader,
                            evaluation_freq, dtype, device)
        elif optimisation_method_upper == 'your_custom_method':
            pass
        else:
            raise NotImplementedError

    def _learn_nag(self, regulariser: FieldsOfExperts, lam: float, upper_loss_func: Callable,
                   optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader, evaluation_freq: int,
                   dtype: torch.dtype, device: torch.device):
       regulariser = regulariser.to(device=device, dtype=dtype)
       param_groups = assemble_param_groups_nag(regulariser, **optimisation_options_upper)
       param_groups_ = harmonise_param_groups_nag(param_groups)

       max_num_iterations = optimisation_options_upper['max_num_iterations']

       psnr, loss_test = self.evaluate(regulariser, lam, test_loader, upper_loss_func, dtype, device, k)
       logging.info('[TRAIN] inference on test images')
       logging.info('[TRAIN]   > average psnr: {:.5f}'.format(psnr.detach().cpu().item()))
       logging.info('[TRAIN]   > test loss: {:.5f}'.format(loss_test.detach().cpu().item()))

       for k, batch in enumerate(train_loader):
           with torch.no_grad():
               batch_ = batch.to(dtype=dtype, device=device)

               measurement_model = MeasurementModel(batch_, self.forward_operator, self.noise_level)
               energy = Energy(measurement_model, regulariser, lam)
               energy = energy.to(device=device, dtype=dtype)

               denoising_func = lambda z: solve_lower(z, energy, self.method_lower, self.options_lower).solution

               autograd_func = self._loss_func_factory(upper_loss_func, energy, batch_, denoising_func)

               def grad_func(trainable_params):
                   with torch.enable_grad():
                       loss = autograd_func(trainable_params)
                   return list(torch.autograd.grad(outputs=loss, inputs=[trainable_params]))

               loss_train = step_nag(autograd_func, grad_func, param_groups_)
               self.writer.add_scalar('loss/train', loss_train, k + 1)
               logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                            'loss = {:.5f}'.format(k + 1, max_num_iterations, loss_train.detach().cpu().item()))

               if (k + 1) % evaluation_freq == 0:
                   psnr, loss_test = self.evaluate(regulariser, lam, test_loader, upper_loss_func, dtype, device, k)
                   logging.info('[TRAIN] inference on test images')
                   logging.info('[TRAIN]   > average psnr: {:.5f}'.format(psnr.detach().cpu().item()))
                   logging.info('[TRAIN]   > test loss: {:.5f}'.format(loss_test.detach().cpu().item()))

               if (k + 1) == max_num_iterations:
                   logging.info('[TRAIN] reached maximal number of iterations')
                   break
               else:
                   k += 1

    def _learn_adam(self, regulariser: FieldsOfExperts, lam: float, upper_loss_func: Callable,
                    optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader, evaluation_freq: int,
                    dtype: torch.dtype, device: torch.device):
        param_groups = assemble_param_groups_adam(regulariser, **optimisation_options_upper)
        param_groups_ = harmonise_param_groups_adam(param_groups)

        optimiser = create_projected_optimiser(torch.optim.Adam)(param_groups_)

        max_num_iterations = optimisation_options_upper['max_num_iterations']

        for k, batch in enumerate(train_loader):
            with torch.no_grad():
                batch_ = batch.to(dtype=dtype, device=device)

                measurement_model = MeasurementModel(batch_, self.forward_operator, self.noise_level)
                energy = Energy(measurement_model, regulariser, lam)
                energy = energy.to(device=device, dtype=dtype)

                u_noisy = energy.measurement_model.obs_noisy
                u_denoised = solve_lower(u_noisy, energy, self.method_lower, self.options_lower).solution

                autograd_func = self._loss_func_factory(upper_loss_func, energy, batch_, u_denoised)
                loss_train = step_adam(optimiser, autograd_func, param_groups_)

                self.writer.add_scalar('loss/train', loss_train, k + 1)
                logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                             'loss = {:.5f}'.format(k + 1, max_num_iterations, loss_train.detach().cpu().item()))

                if (k + 1) % evaluation_freq == 0:
                    psnr, loss_test = self.evaluate(regulariser, lam, test_loader, upper_loss_func, dtype, device, k)
                    logging.info('[TRAIN] inference on test images')
                    logging.info('[TRAIN]   > average psnr: {:.5f}'.format(psnr.detach().cpu().item()))
                    logging.info('[TRAIN]   > test loss: {:.5f}'.format(loss_test.detach().cpu().item()))

                if (k + 1) == optimisation_options_upper['max_num_iterations']:
                    logging.info('[TRAIN] reached maximal number of iterations')
                    break
                else:
                    k += 1


