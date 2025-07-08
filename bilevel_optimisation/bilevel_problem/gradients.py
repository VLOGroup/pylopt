from typing import Callable, Any, List, Dict
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from bilevel_optimisation.energy.Energy import Energy
from bilevel_optimisation.solver import LinearSystemSolver
from bilevel_optimisation.lower_problem.solve_lower import solve_lower

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


class OptimisationAutogradFunction(Function):
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
    def forward(ctx: FunctionCtx, energy: Energy, method_lower: str, options_lower: Dict[str, Any],
                loss_func: torch.nn.Module, solver: LinearSystemSolver, *params: torch.nn.Parameter) -> torch.Tensor:
        """
        Function which needs to be implemented due to subclassing from torch.autograd.Function.
        It computes and provides data which is required in the backward step.

        :param ctx:
        :param energy:
        :param method_lower:
        :param options_lower:
        :param loss_func: PyTorch module representing the outer loss function
        :param solver: Linear system solver of class LinearSystemSolver
        :param params: List of PyTorch parameters whose gradients need to be computed.
        :return: Current outer loss
        """
        u_denoised = solve_lower(energy, method_lower, options_lower).solution
        ctx.save_for_backward(u_denoised.detach().clone())

        ctx.energy = energy
        ctx.loss_func = loss_func
        ctx.solver = solver

        return loss_func(u_denoised)

    @staticmethod
    def compute_lagrange_multiplier(outer_loss_func: torch.nn.Module, energy: Energy,
                                    u_denoised: torch.Tensor, solver: LinearSystemSolver) -> torch.Tensor:
        """
        Function which computes the Lagrange multiplier of the KKT-formulation of the bilevel problem.
        The Lagrange multiplier is required for the computation of the gradients of the outer loss w.r.t. to
        the parameters of the regulariser.

        :param outer_loss_func: PyTorch module representing the outer loss
        :param energy: PyTorch module representing the inner problem
        :param u_denoised: Result of denoising procedure
        :param solver: Linear system solver
        :return: Solution of linear system
        """
        with torch.enable_grad():
            x = u_denoised.detach().clone()
            x.requires_grad = True
            outer_loss = outer_loss_func(x)
        grad_outer_loss = torch.autograd.grad(outputs=outer_loss, inputs=x)[0]

        lin_operator = lambda z: compute_hvp_state(energy, u_denoised, z)
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
        lagrange_multiplier = OptimisationAutogradFunction.compute_lagrange_multiplier(outer_loss_func, energy,
                                                                                       u_denoised, solver)
        grad_params = compute_hvp_mixed(energy, u_denoised.detach(), lagrange_multiplier)



        energy.zero_grad()

        return None, None, None, None, None, *grad_params

class UnrollingAutogradFunction(Function):
    """
    Subclass of torch.autograd.Function with the purpose to provide a custom backward
    function based on an unrolling scheme.
    """
    @staticmethod
    def forward(ctx: FunctionCtx, energy: Energy, method_lower: str, options_lower: Dict[str, Any],
                loss_func: torch.nn.Module, *params) -> torch.Tensor:

        with torch.enable_grad():
            u_denoised = solve_lower(energy, method_lower, options_lower).solution
            loss = loss_func(u_denoised)
        grad_params = torch.autograd.grad(outputs=loss, inputs=params)

        ctx.grad_params = grad_params
        ctx.energy = energy
        return loss

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_output: torch.Tensor) -> Any:
        grad_params = ctx.grad_params
        energy = ctx.energy
        energy.zero_grad()

        return None, None, None, None, *grad_params