from typing import Callable, Any, Optional, Dict
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import logging
import time

from bilevel_optimisation.fields_of_experts.FieldsOfExperts import FieldsOfExperts
from bilevel_optimisation.losses import BaseLoss
from bilevel_optimisation.energy.InnerEnergy import Energy
from bilevel_optimisation.measurement_model.MeasurementModel import MeasurementModel
from bilevel_optimisation.solver.CGSolver import LinearSystemSolver


class BilevelOptimisation:

    def __init__(self, forward_operator: torch.nn.Module, noise_level: float, method_lower: str, options_lower: Dict[str, Any]) -> None:
        self.forward_operator = forward_operator
        self.noise_level = noise_level

        self.method_lower = method_lower
        self.options_lower = options_lower




    def learn(self, regulariser: FieldsOfExperts, lam: float, loss_func: Callable, dataset_train, dataset_test, method_upper, options_upper):
        train_loader = DataLoader(dataset_train, batch_size=batch_size,
                                  collate_fn=lambda x: collate_function(x, crop_size=crop_size))

        test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False,
                                 collate_fn=lambda x: collate_function(x, crop_size=-1))

        trainable_params = [p for p in regulariser.parameters() if p.requires_grad]
        for k, batch in enumerate(train_loader):
            with torch.no_grad():
                batch_ = batch.to()

                measurement_model = MeasurementModel(batch_, self.forward_operator, self.noise_level)
                energy = Energy(measurement_model, regulariser, lam)

                if ...:
                    loss = OptimisationBackwardFunction.apply(energy, loss_func, cg_solver, *trainable_params)
                else:
                    loss = UnrollingBackwardFunction.apply(energy, loss_func, *trainable_params)



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

    @staticmethod
    def _build_loss_func_factory(backward_mode: str) -> Callable:
        def loss_func_factory(inner_energy: InnerEnergy, outer_loss: BaseLoss,
                              solver: Optional[LinearSystemSolver]=None) -> Callable:
            trainable_parameters = [p for p in inner_energy.parameters() if p.requires_grad]
            if backward_mode == 'differentiation':
                def loss_func() -> Any:
                    return BilevelDifferentiation.apply(inner_energy, outer_loss, solver, *trainable_parameters)
            else:
                def loss_func() -> Any:
                    return BilevelUnrolling.apply(inner_energy, outer_loss, *trainable_parameters)
            return loss_func

        return loss_func_factory

    @staticmethod
    def _build_closure_factory(loss_func_factory, optimiser):
        def closure_factory(inner_energy: InnerEnergy, outer_loss: BaseLoss,
                            solver: Optional[LinearSystemSolver]=None):
            def closure() -> Any:
                optimiser.zero_grad()
                with torch.enable_grad():
                    loss = loss_func_factory(inner_energy, outer_loss, solver)()
                    loss.backward()
                return loss
            return closure
        return closure_factory

    def forward(self, outer_loss: BaseLoss, inner_energy: InnerEnergy) -> torch.Tensor:
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
    def forward(ctx: FunctionCtx, inner_energy: Energy,
                loss_func: torch.nn.Module, solver: LinearSystemSolver, *params) -> torch.Tensor:
        """
        Function which needs to be implemented due to subclassing from torch.autograd.Function.
        It computes and provides data which is required in the backward step.

        :param ctx:
        :param inner_energy: PyTorch module representing the inner problem
        :param loss_func: PyTorch module representing the outer loss function
        :param solver: Linear system solver of class LinearSystemSolver
        :param params: List of PyTorch parameters whose gradients need to be computed.
        :return: Current outer loss
        """
        x_noisy = inner_energy.measurement_model.obs_noisy

        x_denoised = inner_energy.argmin(x_noisy)
        ctx.save_for_backward(x_denoised.detach().clone())

        ctx.inner_energy = inner_energy
        ctx.loss_func = loss_func
        ctx.solver = solver

        return loss_func(x_denoised)

    @staticmethod
    def compute_lagrange_multiplier(outer_loss_func: torch.nn.Module, inner_energy: InnerEnergy,
                                    x_denoised: torch.Tensor, solver: LinearSystemSolver) -> torch.Tensor:
        """
        Function which computes the Lagrange multiplier of the KKT-formulation of the bilevel problem.
        The Lagrange multiplier is required for the computation of the gradients of the outer loss w.r.t. to
        the parameters of the regulariser.

        :param outer_loss_func: PyTorch module representing the outer loss
        :param inner_energy: PyTorch module representing the inner problem
        :param x_denoised: Result of denoising procedure
        :param solver: Linear system solver
        :return: Solution of linear system
        """
        with torch.enable_grad():
            x_ = x_denoised.detach().clone()
            x_.requires_grad = True
            outer_loss = outer_loss_func(x_)
        grad_outer_loss = torch.autograd.grad(outputs=outer_loss, inputs=x_)[0]

        lin_operator = lambda z: inner_energy.hvp_state(x_denoised, z)
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
        x_denoised = ctx.saved_tensors[0]
        inner_energy = ctx.inner_energy
        outer_loss_func = ctx.loss_func
        solver = ctx.solver
        lagrange_multiplier = BilevelDifferentiation.compute_lagrange_multiplier(outer_loss_func, inner_energy,
                                                                                 x_denoised, solver)
        grad_params = inner_energy.hvp_mixed(x_denoised.detach(), lagrange_multiplier)
        inner_energy.zero_grad()

        return None, None, None, *grad_params

class UnrollingBackwardFunction(Function):
    """
    Subclass of torch.autograd.Function with the purpose to provide a custom backward
    function based on an unrolling scheme.
    """
    @staticmethod
    def forward(ctx: FunctionCtx, inner_energy: InnerEnergy,
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