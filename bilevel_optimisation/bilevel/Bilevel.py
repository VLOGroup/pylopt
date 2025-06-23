from typing import Callable, Any, Optional, Dict, List, Tuple
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import os


from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.losses import BaseLoss
from bilevel_optimisation.energy.Energy import Energy
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.solver.CGSolver import CGSolver, LinearSystemSolver
from bilevel_optimisation.lower_problem import add_group_options, solve_lower
from bilevel_optimisation.optimise import step_adam, create_projected_optimiser, step_nag
from bilevel_optimisation.optimise.optimise_adam import harmonise_param_groups_adam
from bilevel_optimisation.optimise.optimise_nag import harmonise_param_groups_nag
from bilevel_optimisation.utils.dataset_utils import collate_function

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

def create_experiment_dir(path) -> str:
    experiment_list = sorted(os.listdir(export_path))
    if experiment_list:
        experiment_id = str(int(experiment_list[-1]) + 1).zfill(5)
    else:
        experiment_id = str(0).zfill(5)
    path_to_eval_dir = os.path.join(export_path, experiment_id)
    os.makedirs(path_to_eval_dir, exist_ok=True)

    dump_config_file(config, path_to_eval_dir)

    return path_to_eval_dir

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

        self.writer = SummaryWriter(log_dir=os.path.join(path_to_eval_dir, 'tensorboard'))


    def _loss_func_factory(self, upper_loss, energy, u_clean, denoising_func):
        if self.backward_mode == 'unrolling':
            pass

        else:
            return lambda params: OptimisationBackwardFunction.apply(energy, denoising_func,
                                                                     lambda z: upper_loss(u_clean, z), self.solver,
                                                                     params)

    def evaluate(self, regulariser: FieldsOfExperts, lam: float, test_loader: DataLoader, upper_loss_func: Callable,
                 dtype: torch.dtype, device: torch.device):
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

        return psnr, loss, make_grid(triplets, nrow=test_batch_clean_.shape[0])

    def learn(self, regulariser: FieldsOfExperts, lam: float, upper_loss_func: Callable, dataset_train, dataset_test,
              optimisation_method_upper: str, optimisation_options_upper: Dict[str, Any],
              batch_size: int=32, crop_size: int=64, dtype: torch.dtype=torch.float32,
              device: Optional[torch.device]=None, evaluation_freq: int=2):

        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        regulariser = regulariser.to(device=device, dtype=dtype)


        train_loader = DataLoader(dataset_train, batch_size=batch_size,
                                  collate_fn=lambda x: collate_function(x, crop_size=crop_size))

        test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False,
                                 collate_fn=lambda x: collate_function(x, crop_size=-1))

        if optimisation_method_upper == 'nag':

            param_groups = assemble_param_groups_nag(regulariser, **optimisation_options_upper)
            param_groups_ = harmonise_param_groups_nag(param_groups)

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

                    if (k + 1) % evaluation_freq == 0:

                        psnr, loss_test, test_triplet_grid = self.evaluate(regulariser, lam, test_loader, upper_loss_func, dtype, device)

                        self.writer.add_scalar('loss/test', loss_test, k + 1)
                        self.writer.add_scalar('psnr/test', psnr, k + 1)
                        self.writer.add_image('triplets/test', test_triplet_grid, k + 1)

                    if (k + 1) == optimisation_options_upper['max_num_iterations']:
                        logging.info('[TRAIN] reached maximal number of iterations')
                        break
                    else:
                        k += 1


        elif optimisation_method_upper == 'adam':
            param_groups = assemble_param_groups_adam(regulariser, **optimisation_options_upper)
            param_groups_ = harmonise_param_groups_adam(param_groups)

            optimiser = create_projected_optimiser(torch.optim.Adam)(param_groups_)

            for k, batch in enumerate(train_loader):
                with torch.no_grad():
                    batch_ = batch.to(dtype=dtype, device=device)

                    measurement_model = MeasurementModel(batch_, self.forward_operator, self.noise_level)
                    energy = Energy(measurement_model, regulariser, lam)
                    energy = energy.to(device=device, dtype=dtype)

                    u_noisy = energy.measurement_model.obs_noisy
                    u_denoised = solve_lower(u_noisy, energy, self.method_lower, self.options_lower).solution

                    autograd_func = self._loss_func_factory(upper_loss_func, energy, batch_, u_denoised)
                    loss = step_adam(optimiser, autograd_func, param_groups_)

                    print(loss)

                if (k + 1) == optimisation_options_upper['max_num_iterations']:
                    logging.info('[TRAIN] reached maximal number of iterations')
                    break
                else:
                    k += 1




class Bilevel(torch.nn.Module):
    """
    Class which represents the bilevel framework. Its building blocks are the outer
    loss function, and an optimiser to minimise the outer loss. The inner problem
    is modelled by means of a torch module; it is not held as a member of an object of class
    Bilevel. The forward call of an object of class Bilevel takes the inner problem (in terms
    of a module) as argument, and triggers a single optimisation step.

    The gradients w.r.t. to the outer optimisation problem will be determined
    by default using implicit differentiation. Alternatively one can use an unrolling
    procedure. Implicit differentiation is implemented by means of a custom
    implementation of the torch backward call.
    """

    def __init__(self, optimiser: torch.optim.Optimizer, solver_factory: Callable,
                 backward_mode: str = 'differentiation') -> None:
        """
        Initialisation of class Bilevel.

        :param optimiser:
        :param solver_factory:
        :param backward_mode:
        """
        super().__init__()

        self._optimiser = optimiser
        self._solver_factory = solver_factory if backward_mode == 'differentiation' else None

        self._loss_func_factory = self._build_loss_func_factory(backward_mode)
        self._closure_factory = self._build_closure_factory(self._loss_func_factory, optimiser)

    # @staticmethod
    # def _build_loss_func_factory(backward_mode: str) -> Callable:
    #     def loss_func_factory(inner_energy: InnerEnergy, outer_loss: BaseLoss,
    #                           solver: Optional[LinearSystemSolver]=None) -> Callable:
    #         trainable_parameters = [p for p in inner_energy.parameters() if p.requires_grad]
    #         if backward_mode == 'differentiation':
    #             def loss_func() -> Any:
    #                 return BilevelDifferentiation.apply(inner_energy, outer_loss, solver, *trainable_parameters)
    #         else:
    #             def loss_func() -> Any:
    #                 return BilevelUnrolling.apply(inner_energy, outer_loss, *trainable_parameters)
    #         return loss_func
    #
    #     return loss_func_factory
    #
    # @staticmethod
    # def _build_closure_factory(loss_func_factory, optimiser):
    #     def closure_factory(inner_energy: InnerEnergy, outer_loss: BaseLoss,
    #                         solver: Optional[LinearSystemSolver]=None):
    #         def closure() -> Any:
    #             optimiser.zero_grad()
    #             with torch.enable_grad():
    #                 loss = loss_func_factory(inner_energy, outer_loss, solver)()
    #                 loss.backward()
    #             return loss
    #         return closure
    #     return closure_factory

    def forward(self, outer_loss: BaseLoss, inner_energy: Energy) -> torch.Tensor:
        """

        :param outer_loss:
        :param inner_energy:
        :return:
        """
        logging.info('[BILEVEL] update parameter of regulariser')

        with torch.no_grad():
            # trainable_parameters = [p for p in inner_energy.parameters() if p.requires_grad]
            # solver = self._solver_factory()
            #
            # def closure():
            #     self._optimiser.zero_grad()
            #     with torch.enable_grad():
            #         loss = self._gradient_func.apply(inner_energy, outer_loss, solver, *trainable_parameters)
            #         loss.backward()
            #     return loss

            solver = self._solver_factory() if self._solver_factory is not None else None
            loss_func = self._loss_func_factory(inner_energy, outer_loss, solver)
            # closure = self._closure_factory(inner_energy, outer_loss, solver)

            t0 = time.time()
            # NOTE
            #   > closure is called at this stage to populate gradients
            #   > this is redundant for optimisers which use the closure (i.e. all NAG-type optimisers);
            #       for optimisers like Adam, etc. this call is absolutely necessary!
            # _ = closure()

            curr_loss = self._optimiser.step(loss_func)
            t1 = time.time()

            logging.info('[BILEVEL] performed update step')
            logging.info('[BILEVEL]  > elapsed time [s]: {:.5f}'.format(t1 - t0))
        return curr_loss


def compute_hvp_state(energy: Energy, u: torch.Tensor, v: torch.Tensor):
    with torch.enable_grad():
        x = u.detach().clone()
        x.requires_grad = True

        e = energy(x)
        de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
    return torch.autograd.grad(inputs=x, outputs=de_dx[0], grad_outputs=v)[0]

def compute_hvp_mixed(energy: Energy, u: torch.Tensor, v: torch.Tensor):
    with torch.enable_grad():
        x = u.detach().clone()
        x.requires_grad = True

        e = energy(x)
        de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
    d2e_mixed = torch.autograd.grad(inputs=[p for p in energy.parameters() if p.requires_grad],
                                    outputs=de_dx, grad_outputs=v)
    return list(d2e_mixed)

class OptimisationBackwardFunction(Function):
    """
    Subclass of torch.autograd.Function. It implements the implicit differentiation scheme
    to compute the gradients of optimiser of the inner problem w.r.t. to the parameters
    of the regulariser - as references for implicit differentiation see [1], [2]. For the
    custom implementation of the backward call see [3].

    References
    ----------
    [1] Chen, Y., Ranftl, R. and Pock, T., 2014. Insights into analysis operator learning: From patch-based
        sparse models to higher order MRFs. IEEE Transactions on Image Processing, 23(3), pp.1060-1072.
    [2] Samuel, K.G. and Tappen, M.F., 2009, June. Learning optimized MAP estimates in continuously-valued
        MRF models. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 477-484). IEEE.
    [3] https://docs.pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx: FunctionCtx, energy: Energy, denoising_func: Callable, loss_func: torch.nn.Module,
                solver: LinearSystemSolver, *params) -> torch.Tensor:
        """
        Function which needs to be implemented due to subclassing from torch.autograd.Function.
        It computes and provides data which is required in the backward step.

        :param ctx:
        :param energy:
        :param denoising_func:
        :param loss_func: PyTorch module representing the outer loss function
        :param solver: Linear system solver of class LinearSystemSolver
        :param params: List of PyTorch parameters whose gradients need to be computed.
        :return: Current outer loss
        """
        u_noisy = energy.measurement_model.obs_noisy
        u_denoised = denoising_func(u_noisy)
        ctx.save_for_backward(u_denoised.detach().clone())

        ctx.energy = energy
        ctx.loss_func = loss_func
        ctx.solver = solver

        return loss_func(u_denoised)

    @staticmethod
    def compute_lagrange_multiplier(outer_loss_func: torch.nn.Module, energy: Energy,
                                    x_denoised: torch.Tensor, solver: LinearSystemSolver) -> torch.Tensor:
        """
        Function which computes the Lagrange multiplier of the KKT-formulation of the bilevel problem.
        The Lagrange multiplier is required for the computation of the gradients of the outer loss w.r.t. to
        the parameters of the regulariser.

        :param outer_loss_func: PyTorch module representing the outer loss
        :param energy: PyTorch module representing the inner problem
        :param x_denoised: Result of denoising procedure
        :param solver: Linear system solver
        :return: Solution of linear system
        """
        with torch.enable_grad():
            x_ = x_denoised.detach().clone()
            x_.requires_grad = True
            outer_loss = outer_loss_func(x_)
        grad_outer_loss = torch.autograd.grad(outputs=outer_loss, inputs=x_)[0]

        lin_operator = lambda z: compute_hvp_state(energy, x_denoised, z)
        lagrange_multiplier_result = solver.solve(lin_operator, -grad_outer_loss)

        return lagrange_multiplier_result.solution

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_output: torch.Tensor) -> Any:
        """
        This function implements the custom backward step based on the principle of implicit differentiation.

        References
        ----------
        [1] https://docs.pytorch.org/docs/stable/notes/extending.html

        :param ctx: This kind of object is used to pass tensors, and other objects from the forward call
            to the backward call. For more details see [1]
        :param grad_output: Not used in this implementation
        :return: Gradients of outer loss w.r.t. the parameters of regulariser. For each input of the
            forward function there must be a return parameter. This is why for this implementation 5
            return values need to be specified. For more details see again [1].
        """
        u_denoised = ctx.saved_tensors[0]
        energy = ctx.energy
        outer_loss_func = ctx.loss_func
        solver = ctx.solver
        lagrange_multiplier = OptimisationBackwardFunction.compute_lagrange_multiplier(outer_loss_func, energy,
                                                                                       u_denoised, solver)
        grad_params = compute_hvp_mixed(energy, u_denoised.detach(), lagrange_multiplier)
        energy.zero_grad()

        return None, None, None, None, *grad_params

class UnrollingBackwardFunction(Function):
    """
    Subclass of torch.autograd.Function with the purpose to provide a custom backward
    function based on an unrolling scheme.
    """
    @staticmethod
    def forward(ctx: FunctionCtx, inner_energy: Energy,
                loss_func: torch.nn.Module, *params) -> torch.Tensor:
        x_noisy = inner_energy.measurement_model.obs_noisy
        x_denoised = inner_energy.argmin(x_noisy)

        with torch.enable_grad():
            loss = loss_func(x_denoised)
        grad_params = torch.autograd.grad(outputs=loss, inputs=[p for p in inner_energy.parameters() if p.requires_grad])

        ctx.grad_params = grad_params
        ctx.inner_energy = inner_energy
        return loss

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_output: torch.Tensor) -> Any:

        grad_params = ctx.grad_params
        inner_energy = ctx.inner_energy
        inner_energy.zero_grad()

        return None, None, *grad_params