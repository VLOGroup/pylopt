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



                # def forward_debug(self, inner_energy):
                #     sigma = 0.1
                #     test_img_clean = ski.io.imread(
                #         '/home/florianthaler/Documents/data/image_data/some_images/watercastle.jpg') / 255.0
                #     test_img_clean = test_img_clean.mean(-1).astype(np.float32)
                #     test_img_clean = torch.from_numpy(test_img_clean).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                #
                #     test_img_noisy = test_img_clean + torch.randn_like(test_img_clean) * sigma
                #
                #
                #
                #     solver = CGSolver(max_num_iterations=500, rel_tol=1e-5)
                #
                #     thetas = inner_energy.regulariser.filter_weights.data.clone()
                #     filters = inner_energy.regulariser.filters.data.clone()
                #
                #     if self.thetas_old is None:
                #         self.thetas_old = thetas.clone()
                #         self.filters_old = filters.clone()
                #
                #     with torch.no_grad():
                #         thetas_i = thetas + 0.71 * (thetas - self.thetas_old)
                #         filters_i = filters + 0.71 * (filters - self.filters_old)
                #
                #         # debug
                #         clean = torch.load(
                #             '/home/florianthaler/Documents/research/stochastic_bilevel_optimisation/data/data_batches/batch_clean_{:d}.pt'.format(
                #                 0)).to(device=torch.device('cuda:0'), dtype=torch.float32)
                #         noisy = torch.load(
                #             '/home/florianthaler/Documents/research/stochastic_bilevel_optimisation/data/data_batches/batch_noisy_{:d}.pt'.format(
                #                 0)).to(device=torch.device('cuda:0'), dtype=torch.float32)
                #
                #         # 1. solve for \nabla_u E(u)=0
                #         inner_energy.regulariser.filters.data.copy_(filters_i)
                #         inner_energy.regulariser.filter_weights.data.copy_(thetas_i)
                #         denoised = inner_energy.argmin(noisy)
                #
                #         lagrange = BilevelDifferentiation.compute_lagrange_multiplier(self.loss, inner_energy, clean, denoised, solver)
                #
                #         # 3. compute gradient update for filetrs, thetas
                #         grad_filters, grad_thetas = inner_energy.hvp_mixed(denoised, lagrange)
                #         loss = 0.5 * torch.sum((clean - denoised) ** 2)
                #
                #         self.thetas_old = thetas.clone()
                #         self.filters_old = filters.clone()
                #
                #         for bt in range(10):
                #
                #             step_size = 1 / self.lip  # constant ...
                #
                #             thetas = thetas_i - step_size * grad_thetas
                #             thetas = torch.clamp(thetas, min=0.0)
                #
                #             filters = filters_i - step_size * grad_filters
                #             filters -= torch.mean(filters, axis=(2, 3), keepdims=True)
                #
                #             inner_energy.regulariser.filters.data.copy_(filters)
                #             inner_energy.regulariser.filter_weights.data.copy_(thetas)
                #             denoised = inner_energy.argmin(noisy)
                #
                #             loss_new = 0.5 * torch.sum((clean - denoised) ** 2)
                #
                #             quad = loss + torch.sum(grad_thetas * (thetas - thetas_i)) + \
                #                    torch.sum(grad_filters * (filters - filters_i)) + \
                #                    self.lip / 2.0 * torch.sum((thetas - thetas_i) ** 2) + \
                #                    self.lip / 2.0 * torch.sum((filters - filters_i) ** 2)
                #
                #             if loss_new <= quad:
                #                 self.lip = self.lip / 2.0
                #                 break
                #             else:
                #                 self.lip = self.lip * 2.0
                #
                #         # filters_list.append(filters.detach().clone())
                #         # thetas_list.append(thetas.detach().clone())
                #
                #         test_img_denoised = inner_energy.argmin(test_img_noisy)
                #
                #         psnr_ = psnr(test_img_denoised.cpu().numpy(), test_img_clean.cpu().numpy())
                #         print("iter = ", -1,
                #               ", Lip = ", "{:3.3f}".format(self.lip),
                #               ", Loss = ", "{:3.3f}".format(loss.cpu().numpy()),
                #               ", step_size = ", "{:3.7f}".format(step_size),
                #               ", psnr = ", "{:3.3f}".format(psnr_),
                #               ", norm_grad_filter_0 = ",
                #               "{:3.3f}".format(torch.linalg.norm(grad_filters[0, :, :, :]).detach().cpu().item()),
                #               ", norm_grad_filter_1 = ",
                #               "{:3.3f}".format(torch.linalg.norm(grad_filters[1, :, :, :]).detach().cpu().item()),
                #               ", norm_grad_filter_2 = ",
                #               "{:3.3f}".format(torch.linalg.norm(grad_filters[2, :, :, :]).detach().cpu().item()),
                #               end="\n")
                #
                #     return loss

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
                    loss = self._gradient_func.apply(inner_energy, outer_loss, solver,
                                                     *trainable_parameters)
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
            if hasattr(self._optimiser, 'param_lip_const_dict'):
                logging.info('[BILEVEL]  > lipschitz constants:')
                for key in self._optimiser.param_lip_const_dict.keys():
                    logging.info('[INNER]      * {:s}: {:.3f}'.format(key,
                                                                       self._optimiser.param_lip_const_dict[key]))
        return curr_loss

class BilevelUnrolling(Function):
    """
    Subclass of torch.autograd.Function with the purpose to provide a custom backward
    function based on an unrolling scheme.

    TODO:
        > implement me!
    """
    @staticmethod
    def forward(ctx: FunctionCtx, inner_energy: InnerEnergy, x_clean: torch.Tensor,
                loss_func: Callable, *params: List[torch.nn.Parameter]) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet')

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError('Not implemented yet')

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
        logging.debug('[BILEVEL] custom forward pass (differentiation)')
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
        logging.debug('[BILEVEL] linear system solver stats')
        logging.debug('[BILEVEL] > num_iterations: {:d}'.format(lagrange_multiplier_result.stats.num_iterations))
        final_residual = lagrange_multiplier_result.stats.residual_norm_list[-1]
        logging.debug('[BILEVEL] > norm of final residual: {:.9f}'.format(final_residual))

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
        logging.debug('[BILEVEL] compute gradients (implicit differentiation)')
        x_denoised = ctx.saved_tensors[0]
        inner_energy = ctx.inner_energy
        outer_loss_func = ctx.loss_func
        solver = ctx.solver

        lagrange_multiplier = BilevelDifferentiation.compute_lagrange_multiplier(outer_loss_func, inner_energy,
                                                                                 x_denoised, solver)
        grad_params = inner_energy.hvp_mixed(x_denoised.detach(), lagrange_multiplier)
        inner_energy.zero_grad()

        return None, None, None, *grad_params
