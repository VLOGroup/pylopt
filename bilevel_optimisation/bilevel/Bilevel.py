from typing import Callable, List, Any
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import logging
import time

from bilevel_optimisation.losses import BaseLoss
from bilevel_optimisation.energy.InnerEnergy import InnerEnergy
from bilevel_optimisation.solver.CGSolver import LinearSystemSolver

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
        self._solver_factory = solver_factory

        self._backward_mode = backward_mode
        if self._backward_mode == 'differentiation':
            self._gradient_func = BilevelDifferentiation
        elif self._backward_mode == 'unrolling':
            self._gradient_func = BilevelUnrolling
        else:
            raise NotImplementedError('There is no backward mode named {:s}'.format(self._backward_mode))

    def forward(self, outer_loss: BaseLoss, inner_energy: InnerEnergy) -> torch.Tensor:
        """

        :param outer_loss:
        :param inner_energy:
        :return:
        """
        logging.info('[BILEVEL] update parameter of regulariser')

        with torch.no_grad():
            trainable_parameters = [p for p in inner_energy.parameters() if p.requires_grad]
            solver = self._solver_factory()

            def closure():
                self._optimiser.zero_grad()
                with torch.enable_grad():
                    loss = self._gradient_func.apply(inner_energy, outer_loss, solver, *trainable_parameters)
                    loss.backward()
                return loss

            t0 = time.time()
            # NOTE
            #   > closure is called at this stage to populate gradients
            #   > this is redundant for optimisers which use the closure (i.e. all NAG-type optimisers);
            #       for optimisers like Adam, etc. this call is absolutely necessary!
            _ = closure()
            curr_loss = self._optimiser.step(closure)
            t1 = time.time()

            logging.info('[BILEVEL] performed update step')
            logging.info('[BILEVEL]  > elapsed time [s]: {:.5f}'.format(t1 - t0))
        return curr_loss

class BilevelDifferentiation(Function):
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
    def forward(ctx: FunctionCtx, inner_energy: InnerEnergy,
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


class BilevelUnrolling(Function):
    """
    Subclass of torch.autograd.Function with the purpose to provide a custom backward
    function based on an unrolling scheme.
    """
    @staticmethod
    def forward(ctx: FunctionCtx, inner_energy: InnerEnergy,
                loss_func: torch.nn.Module, solver: LinearSystemSolver, *params) -> torch.Tensor:
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

        return None, None, None, *grad_params