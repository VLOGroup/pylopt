from typing import Callable, Any, Optional, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
import logging
import os
from confuse import Configuration
from itertools import chain

from bilevel_optimisation.bilevel_problem.gradients import OptimisationAutogradFunction, UnrollingAutogradFunction
from bilevel_optimisation.callbacks import Callback
from bilevel_optimisation.energy.Energy import Energy
from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.lower_problem import add_group_options, solve_lower
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.optimise import step_adam, create_projected_optimiser, step_nag
from bilevel_optimisation.optimise.optimise_adam import harmonise_param_groups_adam
from bilevel_optimisation.optimise.optimise_nag import harmonise_param_groups_nag
from bilevel_optimisation.solver.CGSolver import CGSolver
from bilevel_optimisation.utils.dataset_utils import collate_function
from bilevel_optimisation.utils.file_system_utils import dump_config_file

def assemble_param_groups_base(regulariser: FieldsOfExperts, alternating: bool=False):
    param_dict = {}
    for child in regulariser.children():
        param_dict[child.__class__.__name__] = [p for p in child.parameters() if p.requires_grad]

    param_groups = []
    if not alternating:
        group = {'params': list(chain.from_iterable([param_list for param_list in param_dict.values()])),
                 'name': 'joint'}
        param_groups.append(group)
    else:
        for key in param_dict.keys():
            group = {'params': param_dict[key],
                     'name': key}
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
                              alternating: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, alternating)

    alpha = [None for _ in range(0, len(param_groups))] if not alpha else alpha
    beta = [None for _ in range(0, len(param_groups))] if not beta else beta
    lip_const = [None for _ in range(0, len(param_groups))] if not lip_const else lip_const
    add_group_options(param_groups, {'alpha': alpha, 'beta': beta, 'lip_const': lip_const})

    return param_groups


class BilevelOptimisation:
    def __init__(self, method_lower: str, options_lower: Dict[str, Any], config: Configuration,
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

        self.method_lower = method_lower
        self.options_lower = options_lower

        self.path_to_experiments_dir = os.getcwd() if path_to_experiments_dir is None else path_to_experiments_dir
        self.config = config
        dump_config_file(self.config, self.path_to_experiments_dir)

    def _loss_func_factory(self, upper_loss, energy, u_clean) -> Callable:
        if self.backward_mode == 'unrolling':
            # TODO
            #   > test me
            def func(*params: torch.nn.Parameter) -> torch.Tensor:
                return UnrollingAutogradFunction.apply(energy, self.method_lower, self.options_lower,
                                                       lambda z: upper_loss(u_clean, z), *params)
        else:
            def func(*params) -> torch.Tensor:
                upper_loss_func= lambda z: upper_loss(u_clean, z)
                return OptimisationAutogradFunction.apply(energy, self.method_lower, self.options_lower,
                                                          upper_loss_func, self.solver,
                                                          *params)

        return func

    def learn(self, regulariser: FieldsOfExperts, lam: float, upper_loss_func: Callable, dataset_train,
              optimisation_method_upper: str, optimisation_options_upper: Dict[str, Any],
              batch_size: int=32, crop_size: int=64, dtype: torch.dtype=torch.float32,
              device: Optional[torch.device]=None, callbacks=None):

        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(dataset_train, batch_size=batch_size,
                                  collate_fn=lambda x: collate_function(x, crop_size=crop_size))

        if optimisation_method_upper == 'nag':
            self._learn_nag(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, callbacks)
        elif optimisation_method_upper == 'adam':
            self._learn_adam(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, callbacks)
        elif optimisation_method_upper == 'your_custom_method':
            pass
        else:
            raise NotImplementedError

    def _learn_nag(self, regulariser: FieldsOfExperts, lam: float, upper_loss: Callable,
                   optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                   dtype: torch.dtype, device: torch.device, callbacks: Optional[List[Callback]]=None):
       if callbacks is None:
           callbacks = []
       regulariser = regulariser.to(device=device, dtype=dtype)
       param_groups = assemble_param_groups_nag(regulariser, **optimisation_options_upper)
       param_groups_ = harmonise_param_groups_nag(param_groups)

       for cb in callbacks:
           cb.on_train_begin(regulariser, device=device, dtype=dtype)

       max_num_iterations = optimisation_options_upper['max_num_iterations']
       jacobian_free = optimisation_options_upper.get('jacobian_free', False)

                   # params = [p for p in regulariser.parameters() if p.requires_grad]
                   # theta = params[0]
                   # filters = params[1]
                   # theta_old = theta.detach().clone()
                   # filters_old = filters.detach().clone()
                   # lip_const = 100

       try:
           for k, batch in enumerate(train_loader):
               with torch.no_grad():
                   batch_ = batch.to(dtype=dtype, device=device)
                   measurement_model = MeasurementModel(batch_, self.config)

                   energy = Energy(measurement_model, regulariser, lam)
                   energy = energy.to(device=device, dtype=dtype)

                   func = self._loss_func_factory(upper_loss, energy, batch_)
                   def grad_func(*params):
                       with torch.enable_grad():
                           loss_ = func(*params)
                       return list(torch.autograd.grad(outputs=loss_, inputs=params))
                   loss = step_nag(func, grad_func, param_groups_)

                   for cb in callbacks:
                       cb.on_step(k + 1, regulariser, loss, param_groups=param_groups_, device=device, dtype=dtype)

                   # ##########################################

                   # theta_curr = theta.detach().clone()
                   # theta.add_(0.71 * (theta - theta_old))
                   # theta_inter = theta.detach().clone()
                   # theta_old = theta_curr.detach().clone()
                   #
                   # denoised = solve_lower(energy, self.method_lower, self.options_lower).solution
                   #
                   # lagrange = compute_lagrange_multiplier(lambda z: upper_loss_func(z, batch_), energy, denoised, self.solver)
                   # grads = compute_hvp_mixed(energy, denoised, lagrange)[0]
                   # theta.sub_(1e-2 * grads)
                   #
                   #
                   # filters_curr = filters.detach().clone()
                   # filters.add_(0.71 * (filters - filters_old))
                   # filters_inter = filters.detach().clone()
                   # filters_old = filters_curr.detach().clone()
                   #
                   # denoised = solve_lower(energy, self.method_lower, self.options_lower).solution
                   # lagrange = compute_lagrange_multiplier(lambda z: upper_loss_func(z, batch_), energy, denoised,
                   #                                        self.solver)
                   # grads = compute_hvp_mixed(energy, denoised, lagrange)[1]
                   # filters.sub_(1e-2 * grads)
                   #
                   # loss = 0.5 * torch.sum((denoised - batch_) ** 2)



                   #
                   #
                   #
                   #
                   # loss = upper_loss_func(denoised, batch_)
                   # for l in range(0, 20):
                   #     step_size = 1 / lip_const
                   #     # for p in [p_ for p_ in energy.parameters() if p_.requires_grad]:
                   #     #     p.sub_(step_size * grads)
                   #     theta.sub_(step_size * grads)
                   #
                   #     denoised_new = solve_lower(energy, self.method_lower, self.options_lower).solution
                   #     loss_new = upper_loss_func(denoised_new, batch_)
                   #     quadr_approx = loss + torch.sum(grads * (theta - theta_inter)) + 0.5 * lip_const * torch.sum((theta - theta_inter) ** 2)
                   #     if loss_new <= quadr_approx:
                   #         lip_const *= 0.9
                   #         break
                   #     else:
                   #         lip_const *= 2.0
                   #         theta.copy_(theta_inter)
                   #
                   # print(lip_const)

                   #
                   # for p in [p_ for p_ in energy.parameters() if p_.requires_grad]:
                   #     p.sub_(1e-1 * grads)
                   #     # p.copy_(torch.clamp(p, min=1e-7))

                   # loss = upper_loss_func(denoised, batch_)

                   logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                                'loss = {:.5f}'.format(k + 1, max_num_iterations, loss.detach().cpu().item()))

                   for cb in callbacks:
                       cb.on_step(k + 1, regulariser=regulariser, loss=loss, device=device, dtype=dtype)

               if (k + 1) == max_num_iterations:
                   logging.info('[TRAIN] reached maximal number of iterations')
                   break
               else:
                   k += 1
       finally:
           for cb in callbacks:
               cb.on_train_end()

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


# def compute_lagrange_multiplier(outer_loss_func: torch.nn.Module, energy: Energy,
#                                 x_denoised: torch.Tensor, solver) -> torch.Tensor:
#
#     with torch.enable_grad():
#         x_ = x_denoised.detach().clone()
#         x_.requires_grad = True
#         outer_loss = outer_loss_func(x_)
#     grad_outer_loss = torch.autograd.grad(outputs=outer_loss, inputs=x_)[0]
#
#     lin_operator = lambda z: compute_hvp_state(energy, x_denoised, z)
#     lagrange_multiplier_result = solver.solve(lin_operator, -grad_outer_loss)
#     return lagrange_multiplier_result.solution
#
#     # return -grad_outer_loss
#
# def compute_hvp_state(energy: Energy, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     with torch.enable_grad():
#         x = u.detach().clone()
#         x.requires_grad = True
#
#         e = energy(x)
#         de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
#     return torch.autograd.grad(inputs=x, outputs=de_dx[0], grad_outputs=v)[0]
#
# def compute_hvp_mixed(energy: Energy, u: torch.Tensor, v: torch.Tensor) -> List[torch.Tensor]:
#     with torch.enable_grad():
#         x = u.detach().clone()
#         x.requires_grad = True
#
#         e = energy(x)
#         de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
#     d2e_mixed = torch.autograd.grad(inputs=[p for p in energy.parameters() if p.requires_grad],
#                                     outputs=de_dx, grad_outputs=v)
#     return list(d2e_mixed)
