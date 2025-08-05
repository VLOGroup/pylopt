from typing import Any, List, Dict
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from bilevel_optimisation.energy.Energy import Energy
from bilevel_optimisation.solver import LinearSystemSolver
from bilevel_optimisation.lower_problem.solve_lower import solve_lower

def compute_hvp_state(energy: Energy, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Function which is used to compute the Hessian vector product of the lower level energy function w.r.t.
    the state variable u.

    :param energy: PyTorch module representing the lower level energy funxtion
    :param u: Tensor at which second order derivative shall be computed
    :param v: Tensor at which the Hessian is applied
    :return: Tensor representing the result of the Hessian at u, applied to v.
    """
    with torch.enable_grad():
        x = u.detach().clone()
        x.requires_grad = True

        e = energy(x)
        de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
    return torch.autograd.grad(inputs=x, outputs=de_dx[0], grad_outputs=v)[0]

def compute_hvp_mixed(energy: Energy, u: torch.Tensor, v: torch.Tensor) -> List[torch.Tensor]:
    """
    Function for computing the mixed second order derivative of the energy function.

    :param energy: PyTorch module representing the lower level energy
    :param u: Tensor at which derivatives need to be computed.
    :param v: Tensor to which derivatives are applied.
    :return: Tensor representing the result of mixed Hessian at u, applied to v.
    """
    with torch.enable_grad():
        x = u.detach().clone()
        x.requires_grad = True

        e = energy(x)
        de_dx = torch.autograd.grad(inputs=x, outputs=e, create_graph=True)
    d2e_mixed = torch.autograd.grad(inputs=[p for p in energy.parameters() if p.requires_grad],
                                    outputs=de_dx, grad_outputs=v)
    return list(d2e_mixed)

class ImplicitAutogradFunction(Function):
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
                loss_func: torch.nn.Module, solver: LinearSystemSolver, hessian_free: bool,
                *params: torch.nn.Parameter) -> torch.Tensor:
        """
        Function which needs to be implemented due to subclassing from torch.autograd.Function.
        It computes and provides data which is required in the backward step.

        :param ctx:
        :param energy: PyTorch module representing the energy function of the lower level problem.
        :param method_lower: String indicating which method is used to solve the lower level problem.
        :param options_lower: Dictionary of options regarding the optimisation method for the lower
            level problem.
        :param loss_func: PyTorch module representing the upper loss function
        :param solver: Linear system solver of class LinearSystemSolver
        :param hessian_free: Boolean indicating if 0-th order approximation of the Hessian of the energy function
            is used for the gradient computation.
        :param params: List of PyTorch parameters whose gradients need to be computed.
        :return: Current upper loss
        """
        u_denoised = solve_lower(energy, method_lower, options_lower).solution
        ctx.save_for_backward(u_denoised.detach().clone())

        ctx.energy = energy
        ctx.loss_func = loss_func
        ctx.solver = solver
        ctx.hessian_free = hessian_free

        return loss_func(u_denoised)

    @staticmethod
    def compute_lagrange_multiplier(outer_loss_func: torch.nn.Module, energy: Energy,
                                    u_denoised: torch.Tensor, solver: LinearSystemSolver,
                                    hessian_free: bool=False) -> torch.Tensor:
        """
        Function which computes the Lagrange multiplier of the KKT-formulation of the bilevel problem.

        :param outer_loss_func: PyTorch module representing the outer loss
        :param energy: PyTorch module representing the inner problem
        :param u_denoised: Result of denoising procedure
        :param solver: Linear system solver
        :param hessian_free: Bool indicating if 0-th order approximation of the operator of the linear
            system shall be used or not. Per default the value is set to False
        :return: Solution of linear system in terms of a PyTorch tensor.
        """
        with torch.enable_grad():
            x = u_denoised.detach().clone()
            x.requires_grad = True
            outer_loss = outer_loss_func(x)
        grad_outer_loss = torch.autograd.grad(outputs=outer_loss, inputs=x)[0]

        if hessian_free:
            return -grad_outer_loss
        else:
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
            forward function there must be a return parameter. For more details see again [1].
        """
        u_denoised = ctx.saved_tensors[0]
        energy = ctx.energy
        outer_loss_func = ctx.loss_func
        solver = ctx.solver
        hessian_free = ctx.hessian_free
        lagrange_multiplier = ImplicitAutogradFunction.compute_lagrange_multiplier(outer_loss_func, energy,
                                                                                   u_denoised, solver,
                                                                                   hessian_free=hessian_free)
        grad_params = compute_hvp_mixed(energy, u_denoised.detach(), lagrange_multiplier)



        energy.zero_grad()

        return None, None, None, None, None, None, *grad_params

class UnrollingAutogradFunction(Function):
    """
    Subclass of torch.autograd.Function with the purpose to provide a custom backward
    function based on an unrolling scheme.
    """
    @staticmethod
    def forward(ctx: FunctionCtx, energy: Energy, method_lower: str, options_lower: Dict[str, Any],
                loss_func: torch.nn.Module, *params) -> torch.Tensor:
        """

        :param ctx:
        :param energy:
        :param method_lower:
        :param options_lower:
        :param loss_func:
        :param params:
        :return:
        """
        with torch.enable_grad():
            u_denoised = solve_lower(energy, method_lower, options_lower).solution
            loss = loss_func(u_denoised)
        grad_params = torch.autograd.grad(outputs=loss, inputs=params)

        ctx.grad_params = grad_params
        ctx.energy = energy
        return loss

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_output: torch.Tensor) -> Any:
        """

        :param ctx:
        :param grad_output:
        :return:
        """
        grad_params = ctx.grad_params
        energy = ctx.energy
        energy.zero_grad()

        return None, None, None, None, *grad_params