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
                               parameterwise: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, parameterwise)

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
        elif optimisation_method_upper == 'debug':
            self._learn_debug(regulariser, lam, train_loader, optimisation_method_upper, optimisation_options_upper, dtype, device, callbacks)
        elif optimisation_method_upper == 'adam':
            self._learn_adam(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, callbacks)
        elif optimisation_method_upper == 'your_custom_method':
            pass
        else:
            raise NotImplementedError


    def _compute_grads(self, energy, denoised, lagrange, thetas, filters):
        with torch.enable_grad():
            denoised.requires_grad_(True)
            foe = energy.lam * energy.regulariser(denoised)
            grad_foe = torch.autograd.grad(foe, denoised, create_graph=True)[0]
        foe_grad_params = torch.autograd.grad(grad_foe, [thetas, filters], lagrange)

        return foe_grad_params

    def _learn_debug(self, regulariser: FieldsOfExperts, lam, train_loader, optimisation_method_upper: str,
                     optimisation_options_upper: Dict[str, Any], dtype, device, callbacks):

        regulariser = regulariser.to(device=device, dtype=dtype)
        param_groups = assemble_param_groups_nag(regulariser, **optimisation_options_upper)
        param_groups_ = harmonise_param_groups_nag(param_groups)

        thetas = regulariser.get_potential().get_parameters()
        filters = regulariser.get_image_filter().get_filter_tensor()

        L_thetas = 1
        L_filters = 1

        for cb in callbacks:
            cb.on_train_begin(regulariser, device=device, dtype=dtype)

        with torch.no_grad():
            for k, batch in enumerate(train_loader):
                ###################################################
                ### 1. Update thetas

                batch_ = batch.to(dtype=dtype, device=device)

                measurement_model = MeasurementModel(batch_, self.config)
                energy = Energy(measurement_model, regulariser, lam)
                energy = energy.to(device=device, dtype=dtype)



                denoised = solve_lower(energy, self.method_lower, self.options_lower).solution
                # foe_apgd(denoised, noisy, thetas, filters, verbose=0, maxit=apgd_it)

                # 2. solve for the Lagrange multipliers, initialize with the rhs
                lagrange = compute_lagrange_multiplier(lambda z: 0.5 * torch.sum((z - batch_) ** 2),
                                                       energy, denoised, self.solver)
                # CG(clean - denoised, compute_foe_hess_lag, denoised, thetas,
                #               filters, clean - denoised, maxit=cg_it)

                # 3. compute gradient for filters, thetas
                # grad_thetas, grad_filters = self._compute_grads(energy, denoised, lagrange, thetas, filters) # compute_foe_grad_params(denoised, thetas,
                #                                                     filters, lagrange)

                grad_thetas, grad_filters = compute_hvp_mixed(energy, denoised.detach(), lagrange)

                thetas_old = thetas.detach().clone()
                filters_old = filters.detach().clone()
                loss = 0.5 * torch.sum((batch_ - denoised) ** 2)

                for bt in range(20):
                    # gradient descent
                    # thetas_new = thetas - 1 / L_thetas * grad_thetas
                    thetas.sub_(1 / L_thetas * grad_thetas)

                    # 1. solve for \nabla_u E(u)=0
                    denoised_new = solve_lower(energy, self.method_lower, self.options_lower).solution

                    loss_new = 0.5 * torch.sum((batch_ - denoised_new) ** 2)

                    quad = loss + (grad_thetas * (thetas - thetas_old)).sum() + \
                           L_thetas / 2.0 * ((thetas - thetas_old) ** 2).sum()

                    if loss_new <= 1.01 * quad:
                        L_thetas = L_thetas / 1.1
                        break
                    else:
                        L_thetas = L_thetas * 2.0
                        thetas.copy_(thetas_old)


                ###################################################
                ### 2. Update filters

                # 1. solve for \nabla_u E(u)=0
                denoised = solve_lower(energy, self.method_lower, self.options_lower).solution

                # 2. solve for the Lagrange multipliers, initialize with the rhs
                lagrange = compute_lagrange_multiplier(lambda z: 0.5 * torch.sum((z - batch_) ** 2),
                                                       energy, denoised, self.solver)

                # 3. compute gradient for filters, thetas
                grad_thetas, grad_filters = compute_hvp_mixed(energy, denoised.detach(), lagrange)

                loss = 0.5 * torch.sum((batch_ - denoised) ** 2)

                for bt in range(20):

                    # gradient descent
                    filters.sub_(1 / L_filters * grad_filters)
                    filters.sub_(torch.mean(filters, axis=(2, 3), keepdims=True))


                    # 1. solve for \nabla_u E(u)=0
                    denoised_new = solve_lower(energy, self.method_lower, self.options_lower).solution

                    loss_new = 0.5 * torch.sum((batch_ - denoised_new) ** 2)

                    quad = loss + (grad_filters * (filters - filters_old)).sum() + \
                           L_filters / 2.0 * ((filters - filters_old) ** 2).sum()

                    if loss_new <= 1.01 * quad:
                        L_filters = L_filters / 1.1
                        break
                    else:
                        L_filters = L_filters * 2.0
                        filters.copy_(filters_old)

                for cb in callbacks:
                    cb.on_step(k + 1, regulariser, loss, param_groups=param_groups_, device=device, dtype=dtype)

                print("iter = ", k,
                      ", L_thetas = ", "{:3.3f}".format(L_thetas),
                      ", L_filters = ", "{:3.3f}".format(L_filters),
                      ", Loss = ", "{:3.6f}".format(loss.detach().cpu().numpy()),
                      ", PSNR = ", self.psnr(denoised.cpu().numpy(), batch_.cpu().numpy()),
                      end="\n")

    @staticmethod
    def psnr(u, g):
        import numpy as np
        return 20 * np.log10(1.0 / np.sqrt(np.mean((u - g) ** 2)))

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

                   logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                                'loss = {:.5f}'.format(k + 1, max_num_iterations, loss.detach().cpu().item()))

                   for group in param_groups_:
                       if group['lip_const'] > 1e6:
                           group['theta'] = 0
                           group['lip_const'] = 1
                           group['history'] = [p.detach().clone() for p in group['params']]
                           for p in group['params']:
                               p.add_(1e-5 * torch.randn_like(p))


               if (k + 1) == max_num_iterations:
                   logging.info('[TRAIN] reached maximal number of iterations')
                   break
               else:
                   k += 1
       finally:
           for cb in callbacks:
               cb.on_train_end()

    def _learn_adam(self, regulariser: FieldsOfExperts, lam: float, upper_loss: Callable,
                    optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                    dtype: torch.dtype, device: torch.device, callbacks: Optional[List[Callback]]=None):
        param_groups = assemble_param_groups_adam(regulariser, **optimisation_options_upper)
        param_groups_ = harmonise_param_groups_adam(param_groups)

        optimiser = create_projected_optimiser(torch.optim.Adam)(param_groups_)

        max_num_iterations = optimisation_options_upper['max_num_iterations']
        for cb in callbacks:
            cb.on_train_begin(regulariser, device=device, dtype=dtype)

        try:
            for k, batch in enumerate(train_loader):
                with torch.no_grad():
                    batch_ = batch.to(dtype=dtype, device=device)

                    measurement_model = MeasurementModel(batch_, self.config)
                    energy = Energy(measurement_model, regulariser, lam)
                    energy = energy.to(device=device, dtype=dtype)

                    func = self._loss_func_factory(upper_loss, energy, batch_)
                    loss = step_adam(optimiser, func, param_groups_)

                    for cb in callbacks:
                        cb.on_step(k + 1, regulariser, loss, param_groups=param_groups_, device=device, dtype=dtype)

                    logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                                 'loss = {:.5f}'.format(k + 1, max_num_iterations, loss.detach().cpu().item()))

                if (k + 1) == optimisation_options_upper['max_num_iterations']:
                    logging.info('[TRAIN] reached maximal number of iterations')
                    break
                else:
                    k += 1
        finally:
            for cb in callbacks:
                cb.on_train_end()


def compute_lagrange_multiplier(outer_loss_func: torch.nn.Module, energy: Energy,
                                x_denoised: torch.Tensor, solver) -> torch.Tensor:

    with torch.enable_grad():
        x_ = x_denoised.detach().clone()
        x_.requires_grad = True
        outer_loss = outer_loss_func(x_)
    grad_outer_loss = torch.autograd.grad(outputs=outer_loss, inputs=x_)[0]

    lin_operator = lambda z: compute_hvp_state(energy, x_denoised, z)
    lagrange_multiplier_result = solver.solve(lin_operator, -grad_outer_loss)
    return lagrange_multiplier_result.solution

    # return -grad_outer_loss

def compute_hvp_state(energy: Energy, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = u.detach().clone()
        x.requires_grad = True

        e = energy(x)
        de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
    return torch.autograd.grad(inputs=x, outputs=de_dx[0], grad_outputs=v)[0]

def compute_hvp_mixed(energy: Energy, u: torch.Tensor, v: torch.Tensor) -> List[torch.Tensor]:
    with torch.enable_grad():
        x = u.detach().clone()
        x.requires_grad = True

        e = energy(x)
        de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
    d2e_mixed = torch.autograd.grad(inputs=[p for p in energy.parameters() if p.requires_grad],
                                    outputs=de_dx, grad_outputs=v)
    return list(d2e_mixed)
