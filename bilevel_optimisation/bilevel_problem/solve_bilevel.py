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
from bilevel_optimisation.lower_problem import add_group_options
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
        for key, value in param_dict.items():
            if value:
                group = {'params': value,
                         'name': key}
                param_groups.append(group)

    return param_groups

def assemble_param_groups_adam(regulariser: FieldsOfExperts, lr: Optional[List[float]]=None,
                               betas: Optional[List[Tuple[float, float]]]=None, eps: Optional[List[float]]=None,
                               weight_decay: Optional[List[float]]=None,
                               parameterwise: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, parameterwise)

    lr = [None for _ in range(0, len(param_groups))] if not lr else lr
    betas = [None for _ in range(0, len(param_groups))] if not betas else betas
    eps = [None for _ in range(0, len(param_groups))] if not eps else eps
    weight_decay = [None for _ in range(0, len(param_groups))] if not weight_decay else weight_decay
    add_group_options(param_groups, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})

    return param_groups

def assemble_param_groups_nag(regulariser: FieldsOfExperts, alpha: Optional[List[float]]=None,
                              beta: Optional[List[float]]=None,
                              lip_const: Optional[List[float]]=None,
                              alternating: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, alternating)

    alpha = [None for _ in range(0, len(param_groups))] if not alpha else alpha
    beta = [None for _ in range(0, len(param_groups))] if not beta else beta
    lip_const = [None for _ in range(0, len(param_groups))] if not lip_const else lip_const
    add_group_options(param_groups, {'alpha': alpha, 'beta': beta, 'lip_const': lip_const})

    return param_groups

def assemble_param_groups_lbfgs(regulariser: FieldsOfExperts, lr: Optional[List[float]]=None,
                                max_iter: Optional[List[int]]=None,
                                max_eval: Optional[List[int]]=None, tolerance_grad: Optional[List[float]]=None,
                                tolerance_change: Optional[List[float]]=None, history_size: Optional[List[int]]=None,
                                line_search_fn: Optional[List[str]]=None, **unknown_options) -> List[Dict[str, Any]]:
    # NOTE
    #   > According to
    #           https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html,
    #       the PyTorch implementation currently supports only a single parameter group.

    param_groups = assemble_param_groups_base(regulariser, False)
    lr = [None for _ in range(0, len(param_groups))] if not lr else lr
    max_iter = [None for _ in range(0, len(param_groups))] if not max_iter else max_iter
    max_eval = [None for _ in range(0, len(param_groups))] if not max_eval else max_eval
    tolerance_grad = [None for _ in range(0, len(param_groups))] if not tolerance_grad else tolerance_grad
    tolerance_change = [None for _ in range(0, len(param_groups))] if not tolerance_change else tolerance_change
    history_size = [None for _ in range(0, len(param_groups))] if not history_size else history_size
    line_search_fn = [None for _ in range(0, len(param_groups))] if not line_search_fn else line_search_fn

    add_group_options(param_groups, {'lr': lr, 'max_iter': max_iter, 'max_eval': max_eval,
                                     'tolerance_grad': tolerance_grad, 'tolerance_chance': tolerance_change,
                                     'history_size': history_size, 'line_search_fn': line_search_fn})
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

    def _loss_func_factory(self, upper_loss: Callable, energy: Energy, u_clean: torch.Tensor) -> Callable:
        if self.backward_mode == 'unrolling':
            def func(*params: torch.nn.Parameter) -> torch.Tensor:
                upper_loss_func = lambda z: upper_loss(u_clean, z)
                return UnrollingAutogradFunction.apply(energy, self.method_lower, self.options_lower,
                                                       upper_loss_func, *params)
        else:
            def func(*params: torch.nn.Parameter) -> torch.Tensor:
                upper_loss_func = lambda z: upper_loss(u_clean, z)
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
        elif optimisation_method_upper == 'lbfgs':
            self._learn_lbfgs(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, callbacks)
        elif optimisation_method_upper == 'your_custom_method':
            pass
        else:
            raise NotImplementedError

    @torch.no_grad()
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

               if (k + 1) == max_num_iterations:
                   logging.info('[TRAIN] reached maximal number of iterations')
                   break
       finally:
           for cb in callbacks:
               cb.on_train_end()

    def _learn_adam(self, regulariser: FieldsOfExperts, lam: float, upper_loss: Callable,
                    optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                    dtype: torch.dtype, device: torch.device, callbacks: Optional[List[Callback]]=None):
        if callbacks is None:
            callbacks = []

        param_groups = assemble_param_groups_adam(regulariser, **optimisation_options_upper)
        param_groups_ = harmonise_param_groups_adam(param_groups)

        optimiser = create_projected_optimiser(torch.optim.Adam)(param_groups_)

        # TODO
        #   > make me configurable ...
        max_num_iterations = optimisation_options_upper['max_num_iterations']
        num_iterations_1 = min(5000, max_num_iterations)
        scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimiser, factor=1.0, total_iters=num_iterations_1)
        scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_num_iterations - num_iterations_1)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_1, scheduler_2], optimizer=optimiser)

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

                    # scheduler.step()

                    for cb in callbacks:
                        cb.on_step(k + 1, regulariser, loss, param_groups=param_groups_, device=device, dtype=dtype)

                    logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                                 'loss = {:.5f}'.format(k + 1, max_num_iterations, loss.detach().cpu().item()))

                if (k + 1) == max_num_iterations:
                    logging.info('[TRAIN] reached maximal number of iterations')
                    break
        finally:
            for cb in callbacks:
                cb.on_train_end()

    def _learn_lbfgs(self, regulariser: FieldsOfExperts, lam: float, upper_loss: Callable,
                    optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                    dtype: torch.dtype, device: torch.device, callbacks: Optional[List[Callback]]=None):
        if callbacks is None:
            callbacks = []

        param_groups_ = assemble_param_groups_lbfgs(regulariser, **optimisation_options_upper)
        optimiser = create_projected_optimiser(torch.optim.LBFGS)(param_groups_)

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

                    def closure():
                        optimiser.zero_grad()
                        func = self._loss_func_factory(upper_loss, energy, batch_)
                        with torch.enable_grad():
                            loss = func(*[p for p in energy.parameters() if p.requires_grad])
                            loss.backward()
                        return loss
                    loss = optimiser.step(closure)

                    for cb in callbacks:
                        cb.on_step(k + 1, regulariser, torch.Tensor(loss), param_groups=param_groups_, device=device, dtype=dtype)

                    logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                                 'loss = {:.5f}'.format(k + 1, max_num_iterations, loss))

                    if (k + 1) == max_num_iterations:
                        logging.info('[TRAIN] reached maximal number of iterations')
                        break
        finally:
            for cb in callbacks:
                cb.on_train_end()