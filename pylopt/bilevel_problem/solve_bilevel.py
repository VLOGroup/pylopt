from typing import Callable, Any, Optional, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
import logging
from confuse import Configuration
from itertools import chain

from pylopt.bilevel_problem.gradients import ImplicitAutogradFunction, UnrollingAutogradFunction
from pylopt.bilevel_problem.parameter_groups import get_param_group_name, PARAM_GROUP_NAME_KEY
from pylopt.callbacks import Callback
from pylopt.energy import Energy, MeasurementModel
from pylopt.optimise import step_adam, create_projected_optimiser, step_nag
from pylopt.optimise.optimise_adam import harmonise_param_groups_adam
from pylopt.optimise.optimise_lbfgs import harmonise_param_groups_lbfgs
from pylopt.optimise.optimise_nag import harmonise_param_groups_nag
from pylopt.regularisers import FieldsOfExperts, ImageFilter, Potential
from pylopt.regularisers.fields_of_experts import compile_regulariser
from pylopt.scheduler import HyperParamScheduler
from pylopt.solver.CGSolver import CGSolver
from pylopt.dataset.dataset_utils import collate_function
from pylopt.utils.file_system_utils import create_experiment_dir, dump_configs, \
    dump_bilevel_training_settings

def assemble_param_groups_base(regulariser: FieldsOfExperts, alternating: bool=False):
    param_dict = {}
    for child in regulariser.children():
        key = get_param_group_name(child)
        if key is None:
            raise ValueError('Regulariser is supposed to have only children of '
                             'type {:s}, {:s}'.format(ImageFilter.__name__, Potential.__name__))
        param_dict[key] = [p for p in child.parameters() if p.requires_grad]

    param_groups = []
    if not alternating:
        group = {'params': list(chain.from_iterable([param_list for param_list in param_dict.values()])),
                 PARAM_GROUP_NAME_KEY: 'joint'}
        param_groups.append(group)
    else:
        for key, value in param_dict.items():
            if value:
                group = {'params': value, PARAM_GROUP_NAME_KEY: key}
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

    for idx, group in enumerate(param_groups):
        group['lr'] = lr[idx]
        group['betas'] = betas[idx]
        group['eps'] = eps[idx]
        group['weight_decay'] = weight_decay[idx]

    return param_groups

def assemble_param_groups_nag(regulariser: FieldsOfExperts, alpha: Optional[List[float]]=None,
                              beta: Optional[List[float]]=None,
                              lip_const: Optional[List[float]]=None,
                              alternating: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    param_groups = assemble_param_groups_base(regulariser, alternating)

    alpha = [None for _ in range(0, len(param_groups))] if not alpha else alpha
    beta = [None for _ in range(0, len(param_groups))] if not beta else beta
    lip_const = [None for _ in range(0, len(param_groups))] if not lip_const else lip_const

    for idx, group in enumerate(param_groups):
        group['alpha'] = [alpha[idx]]
        group['beta'] = [beta[idx]]
        group['lip_const'] = [lip_const[idx]]

    return param_groups

def assemble_param_groups_lbfgs(regulariser: FieldsOfExperts, lr: Optional[List[float]]=None,
                                max_iter: Optional[int]=None,
                                max_eval: Optional[int]=None, tolerance_grad: Optional[float]=None,
                                tolerance_change: Optional[float]=None, history_size: Optional[int]=None,
                                line_search_fn: Optional[str]=None, **unknown_options) -> List[Dict[str, Any]]:
    # NOTE
    #   > According to
    #           https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html,
    #       the PyTorch implementation currently supports only a single parameter group.

    param_groups = assemble_param_groups_base(regulariser, False)
    for group in param_groups:
        group['lr'] = None if not lr else lr
        group['max_iter'] = None if not max_iter else max_iter
        group['max_eval'] = None if not max_eval else max_eval
        group['tolerance_grad'] = None if not tolerance_grad else tolerance_grad
        group['tolerance_change'] = None if not tolerance_change else tolerance_change
        group['history_size'] = None if not history_size else history_size
        group['line_search_fn'] = None if not line_search_fn else line_search_fn

    return param_groups

class BilevelOptimisation:
    """
    This class implements the routines for the training of filters and/or potentials of a FoE model. For the training
    gradient based methods are considered, where gradients are computed using implicit differentiation or
    an unrolling scheme. The unrolling scheme requires that the lower level problem is solved using NAG or NAPG.
    """
    def __init__(self,
                 method_lower: str, 
                 options_lower: Dict[str, Any], 
                 operator: Optional[torch.nn.Module]=None,
                 noise_level: Optional[float]=None,
                 config: Optional[Configuration]=None,
                 differentiation_method: str='implicit', 
                 solver_name: Optional[str]='cg',
                 options_solver: Optional[Dict[str, Any]]=None,
                 path_to_experiments_dir: Optional[str]=None
    ) -> None:
        """
        Initialisation of BilevelOptimisation.

        NOTE
        ----
            > For the setup of an instance of this class or the operator and the noise level 
                must be specified directly when calling the constructor, or by means of 
                a configuration object of type Configuration.
            > If a configuation object is provided, the config values are written to file.

        TODO
        ----
            > Make class completely configurable by config.

        :param method_lower: String indicating the solution method for the lower level problem.
        :param options_lower: Dictionary of options for the solution of the lower level problem
        :param operator: PyTorch module representing the forward operator
        :param noise_level: Float in the interval (0, 1) representing the noise level.
        :param config: Configuration object for the setup of measurement model and energy
        :param differentiation_method: String indicating which differentiation method is used:
            > 'implicit': Computation of derivative of upper level objective function using the implicit
                function theorem. This approach requires to solve a linear system of equations - hence a linear
                system solver must be specified if choosing this option.
            > 'hessian_free': Derivative of upper level objective function is computed by approximating
                the system matrix (involving the hessian of the energy function) of the implicit differentiation
                scheme using the identity matrix.
            > 'unrolling': This scheme is about performing a small number of gradient steps to solve the lower level
                problem while not breaking the corresponding computational graph. Derivatives of the resulting
                approximate solution of the lower level problem are then computed by means of autograd.
            Per default, implicit differentiation is applied.
        :param solver_name: String indicating the linear system solver which is used to compute gradients when implicit
            differentiation is applied. The default option is 'cg'.
        :param options_solver: Dictionary of options for the linear system solver. The cg solver takes the
            options
                max_num_iterations: int, optional
                    Maximal number of cg iterations; the default value equals 500.
                abs_tol: float, optional
                    Tolerance used for early stopping: Iteration is stopped if the cg residual is <= abs_tol. Per
                    default the tolerance value 1e-5 is used.
        :param path_to_experiments_dir: Path to experiment directory where intermediate results, optimisation stats, ecc.
            will be stored. If not provided, an experiment directory in the root directory of this python
            project is created.
        """
        self.path_to_experiments_dir = create_experiment_dir(self.config) if path_to_experiments_dir is None \
            else path_to_experiments_dir

        self.backward_mode = differentiation_method

        self.method_lower = method_lower
        self.options_lower = options_lower

        self.config = config
        if self.config is not None:
            dump_configs(config, self.path_to_experiments_dir)
        self.noise_level = noise_level
        self.operator = operator

        self.solver_name = solver_name
        if self.solver_name == 'cg':
            self.options_solver = options_solver if options_solver is not None else {}
            self.solver = CGSolver(**self.options_solver)
        elif self.solver_name == 'your_custom_solver':
            # Custom linear system solver goes here. Use the following structure
            #
            #    self.options_solver = ...
            #    self.solver = MySolver(**self.options_solver)
            pass
        else:
            raise ValueError('Unknown linear system solver')

    def _loss_func_factory(self, upper_loss: Callable, energy: Energy, u_clean: torch.Tensor) -> Callable:
        if self.backward_mode == 'unrolling':
            def func(*params: torch.nn.Parameter) -> torch.Tensor:
                upper_loss_func = lambda z: upper_loss(u_clean, z)
                return UnrollingAutogradFunction.apply(energy, self.method_lower, self.options_lower,
                                                       upper_loss_func, *params)
        else:
            def func(*params: torch.nn.Parameter) -> torch.Tensor:
                upper_loss_func = lambda z: upper_loss(u_clean, z)
                return ImplicitAutogradFunction.apply(energy, self.method_lower, self.options_lower,
                                                      upper_loss_func, self.solver,
                                                      self.backward_mode == 'hessian_free',
                                                      *params)
        return func

    def learn(self, regulariser: FieldsOfExperts, lam: float, upper_loss_func: Callable, dataset_train,
              optimisation_method_upper: str, optimisation_options_upper: Dict[str, Any],
              batch_size: int=32, crop_size: int=64, dtype: torch.dtype=torch.float32,
              device: Optional[torch.device]=None, callbacks: Optional[List[Callback]]=None,
              schedulers: Optional[List[HyperParamScheduler]]=None, do_compile: bool=False) -> None:

        upper_settings = {'method': optimisation_method_upper, 'options': optimisation_options_upper}
        options_lower_ = dict(self.options_lower)
        options_lower_.update({'prox': True if 'prox' in self.options_lower.keys() else False})
        lower_settings = {'method': self.method_lower, 'options': options_lower_}
        backward_settings = {'method': self.backward_mode}
        if self.backward_mode:
            backward_settings.update({'options': {'solver': self.solver_name, 'solver_options': self.options_solver}})
        dump_bilevel_training_settings(upper_settings, lower_settings, backward_settings, self.path_to_experiments_dir)

        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(dataset_train, batch_size=batch_size,
                                  collate_fn=lambda x: collate_function(x, crop_size=crop_size))

        if optimisation_method_upper == 'nag':
            self._learn_nag(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, do_compile, callbacks, schedulers)
        elif optimisation_method_upper == 'adam':
            self._learn_adam(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, callbacks, schedulers)
        elif optimisation_method_upper == 'lbfgs':
            self._learn_lbfgs(regulariser, lam, upper_loss_func, optimisation_options_upper, train_loader,
                            dtype, device, callbacks, schedulers)
        elif optimisation_method_upper == 'your_custom_method':
            pass
        else:
            raise ValueError('Unknown optimisation method for upper level problem.')

    @torch.no_grad()
    def _learn_nag(self, regulariser: FieldsOfExperts, lam: float, upper_loss: Callable,
                   optimisation_options_upper: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
                   dtype: torch.dtype, device: torch.device, do_compile: bool,
                   callbacks: Optional[List[Callback]]=None,
                   schedulers: Optional[List[HyperParamScheduler]]=None) -> None:
       if callbacks is None:
           callbacks = []
       if schedulers is None:
           schedulers = []

       regulariser = regulariser.to(device=device, dtype=dtype)
       if do_compile:
           train_iter = iter(train_loader)
           batch_0 = next(train_iter).to(dtype=dtype, device=device)
           # NOTE:
           #    > Use default compilation options here - in particular the dynamic shape option.
           #    > Dynamic shapes are important here since the same model is used for training and testing (on full
           #        images!)
           regulariser = compile_regulariser(regulariser, batch_0, dynamic=True)

       param_groups = assemble_param_groups_nag(regulariser, **optimisation_options_upper)
       param_groups_ = harmonise_param_groups_nag(param_groups)

       for sched in schedulers:
           sched.bind(param_groups_)

       for cb in callbacks:
           cb.on_train_begin(regulariser, device=device, dtype=dtype)

       max_num_iterations = optimisation_options_upper['max_num_iterations']

       try:
           for k, batch in enumerate(train_loader):
               batch_ = batch.to(dtype=dtype, device=device)
               measurement_model = MeasurementModel(batch_, config=self.config, 
                                                    operator=self.operator, noise_level=self.noise_level)

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

               for sched in schedulers:
                   sched.step(func=func, grad_func=grad_func)

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
                    dtype: torch.dtype, device: torch.device, callbacks: Optional[List[Callback]]=None,
                    schedulers: Optional[List[HyperParamScheduler]]=None) -> None:
        if callbacks is None:
            callbacks = []
        if schedulers is None:
            schedulers = []

        param_groups = assemble_param_groups_adam(regulariser, **optimisation_options_upper)
        param_groups_ = harmonise_param_groups_adam(param_groups)

        for sched in schedulers:
            sched.bind(param_groups_)

        for cb in callbacks:
            cb.on_train_begin(regulariser, device=device, dtype=dtype)

        optimiser = create_projected_optimiser(torch.optim.Adam)(param_groups_)
        max_num_iterations = optimisation_options_upper['max_num_iterations']
        try:
            for k, batch in enumerate(train_loader):
                with torch.no_grad():
                    batch_ = batch.to(dtype=dtype, device=device)

                    measurement_model = MeasurementModel(batch_, config=self.config,
                                                         operator=self.operator, noise_level=self.noise_level)
                    energy = Energy(measurement_model, regulariser, lam)
                    energy = energy.to(device=device, dtype=dtype)

                    func = self._loss_func_factory(upper_loss, energy, batch_)
                    loss = step_adam(optimiser, func, param_groups_)

                    def grad_func(*params):
                        with torch.enable_grad():
                            loss_ = func(*params)
                        return list(torch.autograd.grad(outputs=loss_, inputs=params))
                    for sched in schedulers:
                        sched.step(func=func, grad_func=grad_func)

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
                    dtype: torch.dtype, device: torch.device, callbacks: Optional[List[Callback]]=None,
                    schedulers: Optional[List[HyperParamScheduler]]=None) -> None:
        if callbacks is None:
            callbacks = []
        if schedulers is None:
            schedulers = []

        param_groups_ = assemble_param_groups_lbfgs(regulariser, **optimisation_options_upper)
        param_groups_ = harmonise_param_groups_lbfgs(param_groups_)
        optimiser = create_projected_optimiser(torch.optim.LBFGS)(param_groups_)

        for sched in schedulers:
            sched.bind(param_groups_)

        for cb in callbacks:
            cb.on_train_begin(regulariser, device=device, dtype=dtype)

        max_num_iterations = optimisation_options_upper['max_num_iterations']
        try:
            for k, batch in enumerate(train_loader):
                with torch.no_grad():
                    batch_ = batch.to(dtype=dtype, device=device)

                    measurement_model = MeasurementModel(batch_, config=self.config,
                                                         operator=self.operator, noise_level=self.noise_level)
                    energy = Energy(measurement_model, regulariser, lam)
                    energy = energy.to(device=device, dtype=dtype)

                    func = self._loss_func_factory(upper_loss, energy, batch_)
                    def closure():
                        optimiser.zero_grad()
                        with torch.enable_grad():
                            loss = func(*[p for p in energy.parameters() if p.requires_grad])
                            loss.backward()
                        return loss
                    loss = optimiser.step(closure)

                    for cb in callbacks:
                        cb.on_step(k + 1, regulariser, torch.Tensor(loss), param_groups=param_groups_,
                                   device=device, dtype=dtype)

                    def grad_func(*params):
                        with torch.enable_grad():
                            loss_ = func(*params)
                        return list(torch.autograd.grad(outputs=loss_, inputs=params))
                    for sched in schedulers:
                        sched.step(func=func, grad_func=grad_func)

                    logging.info('[TRAIN] iteration [{:d} / {:d}]: '
                                 'loss = {:.5f}'.format(k + 1, max_num_iterations, loss))

                    if (k + 1) == max_num_iterations:
                        logging.info('[TRAIN] reached maximal number of iterations')
                        break
        finally:
            for cb in callbacks:
                cb.on_train_end()