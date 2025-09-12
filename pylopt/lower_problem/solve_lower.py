import torch
from typing import Optional, List, Dict, Any, Tuple, Callable

from bilevel_optimisation.data import LowerProblemResult, OptimiserResult
from bilevel_optimisation.energy import Energy
from bilevel_optimisation.measurement_model import MeasurementModel
from bilevel_optimisation.optimise import optimise_nag, optimise_adam, optimise_nag_unrolling, LIP_CONST_KEY

def make_prox_map(prox_operator: torch.nn.Module, u: torch.Tensor) -> Callable:
    def prox_map(x: torch.Tensor, tau: float) -> torch.Tensor:
        return prox_operator(x, tau, u)
    return prox_map

def add_prox(prox_operator: Optional[torch.nn.Module], param_groups: List[Dict[str, Any]], u: torch.Tensor) -> None:
    u_ = u.detach().clone()
    prox_map = make_prox_map(prox_operator, u_)
    for group in param_groups:
        for p in group['params']:
            setattr(p, 'prox', prox_map)

def assemble_param_groups_nag(u: torch.Tensor, alpha: Optional[float]=None,
                              beta: Optional[float]=None, lip_const: Optional[float]=None,
                              batch_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    u_ = u.detach().clone()
    group = {'params': [torch.nn.Parameter(u_, requires_grad=True)]}

    num_optimisation_variables = 1 if batch_optimisation else u.shape[0]
    group['alpha'] = [alpha] * num_optimisation_variables if alpha is not None else [None] * num_optimisation_variables
    group['beta'] = [beta] * num_optimisation_variables if beta is not None else [None] * num_optimisation_variables
    group[LIP_CONST_KEY] = [lip_const] * num_optimisation_variables if lip_const is not None \
        else [None] * num_optimisation_variables

    return [group]

def assemble_param_groups_adam(u: torch.Tensor, lr: Optional[float]=None,
                               betas: Optional[Tuple[float, float]]=None,
                               weight_decay: Optional[float]=None,
                               batch_optimisation: bool=True, **unknown_options) -> List[Dict[str, Any]]:
    u_ = u.detach().clone()
    param_groups = []
    if batch_optimisation:
        group = {'params': [torch.nn.Parameter(u_, requires_grad=True)],
                 'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        param_groups.append(group)
    else:
        for i in range(0, u_.shape[0]):
            group = {'params': [torch.nn.Parameter(u_[i: i + 1, :, :, :].detach().clone(), requires_grad=True)],
                     'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
            param_groups.append(group)
    return param_groups

def parse_result(result: OptimiserResult, max_num_iterations: int, **unknown_options) -> LowerProblemResult:
    solution_tensor = build_solution_tensor_from_param_groups(result.solution)
    lower_prob_result = LowerProblemResult(solution=solution_tensor, num_iterations=result.num_iterations,
                                           loss=result.loss,
                                           message='Converged' if result.num_iterations < max_num_iterations
                                           else 'Max. number of iterations reached')
    return lower_prob_result

def build_solution_tensor_from_param_groups(param_groups: List[Dict[str, Any]]):
    solution_list = [group['params'][-1] for group in param_groups]
    return torch.cat(solution_list, dim=0)

def build_objective_func(energy: Energy, batch_optim: bool, use_prox: bool) -> Callable:
    if batch_optim and not use_prox:
        def func(x: torch.Tensor) -> torch.Tensor:
            return energy(x)
    elif batch_optim and use_prox:
        def func(x: torch.Tensor) -> torch.Tensor:
            return energy.lam * energy.regulariser(x)
    elif not batch_optim and not use_prox:
        operator = energy.measurement_model.operator
        noise_level = energy.measurement_model.noise_level

        regulariser = energy.regulariser
        lam = energy.lam

        per_sample_energy_models = []
        u_clean = energy.measurement_model.u_clean
        for i in range(u_clean.shape[0]):
            sample_measurement_model = MeasurementModel(u_clean[i:i + 1, :, :, :], operator=operator,
                                                        noise_level=noise_level)
            sample_energy = Energy(sample_measurement_model, regulariser, lam)
            per_sample_energy_models.append(sample_energy)

        # RECALL
        #   > torch.vmap cannot be used for vectorisation of a list of different maps.
        def func(*x: torch.Tensor) -> torch.Tensor:
            return torch.stack([sample_energy_model(sample.unsqueeze(dim=0)) for sample_energy_model, sample in
                                zip(per_sample_energy_models, torch.cat(x, dim=0))])

        return func
    else:
        # TODO
        #   > update me: using vmap here is rather bad ...
        #   > make profit of reduce=True/False implementation of potential functions ...

        regulariser = energy.regulariser
        lam = energy.lam
        def per_sample_objective_func(x) -> torch.Tensor:
            return lam * regulariser(x)

        def func(*x: torch.Tensor) -> torch.Tensor:
            return torch.vmap(per_sample_objective_func)(torch.cat(x, dim=0))


    return func

def build_gradient_func(func: Callable, batch_optim: bool, unrolling: bool=False) -> Callable:
    if batch_optim:
        def grad_func(x: torch.Tensor) -> List[torch.Tensor]:
            with torch.enable_grad():
                x_ = x.detach().clone().requires_grad_(True)
                loss = func(x_)
            return list(torch.autograd.grad(outputs=loss, inputs=[x_], create_graph=unrolling))
    else:
        def grad_func(*x: torch.Tensor):
            with torch.enable_grad():
                x = [x_.detach().clone().requires_grad_(True) for x_ in x]
                loss = torch.sum(func(*x), dim=0)
            return list(torch.autograd.grad(outputs=loss, inputs=x, create_graph=unrolling))

    return grad_func

def solve_lower(energy: Energy, method: str, options: Dict[str, Any]) -> LowerProblemResult:
    """
    :param energy: Instance of class Energy which represents the lower problem.
    :param method: String indicating which of the provided optimisation methods is used to solve the lower problem.
        The following methods are implemented:
        - 'nag' : Nestorov's accelerated gradient method.
        - 'napg': Proximal gradient method with Nesterov acceleration.
        - 'adam': Adam optimisation scheme
    :param options: Options need to be provided in terms of a dictionary. The options required depend on the chosen
        method.
        - method == 'nag'
          ---------------
          max_num_iterations: int
            Maximal number of iterations to perform
          rel_tol: float, optional
            Tolerance for early stopping of the optimisation scheme. If not provided, max_num_iterations iterations
            will be performed.
          batch_optimisation: bool, optional
            Flag indicating if batch-optimisation or per-sample-optimisation is performed. Batch-optimisation performs
            gradient steps on the full batch. Hence, only a single step size or a single Lipschitz constant, and a
            single momentum parameter are required. Per-sample-optimisation performs for each sample of the batch an
            independent gradient step. Consequently, each sample requires step size or Lipschitz constant, and a
            momentum parameter.
          alpha: float, optional
            Constant step size used in the optimisation scheme. If not specified, backtracking line search
            to obtain sufficient descent along the direction of the negative gradient is applied.
            If alpha is specified and batch_optimisation is set to False, the specified alpha is used for each sample.
          beta: float, optional
            Constant momentum parameter. If not specified, the momentum parameter (for the full batch
            and for each sample respectively) is computed at iteration level k as follows

                theta_{k + 1} = 0.5 * (1 + math.sqrt(1 + 4 * (theta_{k} ** 2)))
                beta_k = (theta_{k} - 1) / theta_{k + 1}

            If beta is specified, and batch_optimisation == False, the specified value is used for all the samples
            in the optimisation scheme.
          lip_const: float, optional
            Initial Lipschitz constant used within the backtracking line search. The specified value is used in case
            of batch-optimisation as well as in case of per-sample optimisation.
            If lip_const and alpha are not specified, the default value 1e5 for the full batch optimisation or
            per sample optimisation is used.
        - method == 'napg'
          ---------------
          max_num_iterations: int
            As for 'nag'
          rel_tol: float, optional
            As for 'nag'
          batch_optimisation: bool, optional
            As for 'nag'
          alpha: float, optional
            As for 'nag'
          beta: float, optional
            As for 'nag'
          lip_const: float, optional
            As for 'nag'
          prox: torch.nn.Module
            A PyTorch module which represents the proximal operator. Its forward function must take the arguments
              u: torch.Tensor representing the item at which the operator shall be evaluated
              tau: torch.Tensor representing the step size of the gradient step
              u_noisy: torch.Tensor representing the noisy observation
        - method == 'adam'
          ---------------
          max_num_iterations: int
            As for 'nag'
          batch_optimisation: bool, optional
            This flag indicates, similarly as for 'nag', 'napg', if each sample of the batch should be treated
            individually within the optimisation process. If set to True, for each sample separate hyperparameters (lr,
            betas, weight_decay) are used. Note that per-sample optimisation is organised in this context by means
            of different parameter groups, which in PyTorch are processed sequentially.
          lr: float, optional
            Learning rate, which is used for the single parameter group in batch-optimisation and for all the
            parameter groups in case of per-sample optimisation. Per default, 1e-4 is used if
            no specifications were made.
          betas: Tuple[float, float], optional
            Parameter tuple used to compute moving average of first and second moments of gradients. The same handling
            as for lr is applied. Per default the tuple (0.9, 0.999) is used.
          weight_decay: float, optional
            Weights for L^2 penalisation. Again, the same handling as for 'lr' is applied. Per default the
            value 0.0 is used.
    :return: Instance of class LowerProblemResult containing the solution tensor, the number of performed iterations,
        the final loss, and a message indicating why iteration terminated.
    """
    batch_optim = options.get('batch_optimisation', True)

    if method == 'nag':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim=batch_optim)

        param_groups = assemble_param_groups_nag(energy.measurement_model.get_noisy_observation().detach().clone(),
                                                 **options)

        nag_result = optimise_nag(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'napg':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=True)
        grad_func = build_gradient_func(func, batch_optim=batch_optim)

        u_noisy = energy.measurement_model.get_noisy_observation().detach().clone()
        param_groups = assemble_param_groups_nag(u_noisy, **options)

        prox_operator = options.get('prox', None)
        add_prox(prox_operator, param_groups, u_noisy)

        nag_result = optimise_nag(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'adam':
        param_groups = assemble_param_groups_adam(energy.measurement_model.get_noisy_observation().detach().clone(),
                                                  **options)

        func = lambda *z: torch.sum(build_objective_func(energy, batch_optim=batch_optim, use_prox=False)(torch.cat(z, dim=0)))

        adam_result = optimise_adam(func, param_groups, **options)
        lower_prob_result = parse_result(adam_result, **options)
    elif method == 'nag_unrolling':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim, unrolling=True)

        u = energy.measurement_model.get_noisy_observation().detach().clone()
        param_groups = assemble_param_groups_nag(u.requires_grad_(True), **options)

        nag_result = optimise_nag_unrolling(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'napg_unrolling':
        func = build_objective_func(energy, batch_optim=batch_optim, use_prox=False)
        grad_func = build_gradient_func(func, batch_optim, unrolling=True)

        u = energy.measurement_model.get_noisy_observation().detach().clone()
        param_groups = assemble_param_groups_nag(u.requires_grad_(True), **options)

        prox_operator = options.get('prox', None)
        add_prox(prox_operator, param_groups, u)

        nag_result = optimise_nag_unrolling(func, grad_func, param_groups, **options)
        lower_prob_result = parse_result(nag_result, **options)
    elif method == 'your_custom_method':
        # Custom method for solving the lower problem or by map
        # estimation of by mmse estimation goes here. Use the following
        # structure
        #
        #       solution = solution_method(...)
        #       lower_prob_result = LowerProblemResult(...)
        pass
    else:
        raise ValueError('Unknown solution method for lower level problem.')

    return lower_prob_result


