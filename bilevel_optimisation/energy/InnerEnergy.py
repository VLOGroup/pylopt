import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Any
import logging
import time

from bilevel_optimisation.measurement_model import MeasurementModel

class Energy(torch.nn.Module, ABC):
    """
    Class, inherited from torch.nn.Module, which is used as base class of the inner bilevel
    energy. Each class, inheriting from InnerEnergy must implement the methods sample() and
    argmin().
    """
    def __init__(self, measurement_model: MeasurementModel,
                 regulariser: torch.nn.Module, lam: float) -> None:
        """
        Initialisation of an object of class InnerEnergy

        :param measurement_model: Object of class MeasurementModel
        :param regulariser: Regulariser modelled as inheritance of class torch.nn.Module
        :param lam: Regularisation parameter scaling the regularisation term
        """
        super().__init__()
        self.measurement_model = measurement_model
        self.regulariser = regulariser
        self.lam = lam

    @abstractmethod
    def sample(self, num_sampling_steps: int) -> torch.Tensor:
        """
        Method which computes the MMSE based on a suitable sampling scheme
        and Monte Carlo estimation of expected values.

            >> TO BE IMPLEMENTED <<

        :param num_sampling_steps: Number of sampling steps
        :return: MMSE
        """
        raise NotImplementedError

    @abstractmethod
    def argmin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method which computes the MAP of the underlying probabilistic model by minimisation of
        the energy given the noisy observation x. The denoised data is returned to the caller

        Note that stopping criterion for the minimisation procedure is assumed to be provided
        in the subclasses for example by means of objects of class StoppingCriterion.

        :param x: Noisy observation
        :return: Denoised observation
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns sum of data fidelty and scaled regularisation term

        :param x: Tensor at which data fidelty and regulariser are evaluated
        :return: Tensor representing the energy at the input x
        """
        return self.measurement_model(x) + self.lam * self.regulariser(x)

class OptimisationEnergy(Energy):
    """
    Inheritance from class InnerEnergy which solves the inner problem by means of MAP estimation.
    The MAP is computed using an optimiser inheriting from torch.optim.Optimizer. Optimisers will be
    retrieved from an optimiser factory, which takes as input the variables/parameters to be optimised
    and returns an optimiser optimising exactly these variables.
    """

    def __init__(self, measurement_model: MeasurementModel, regulariser: torch.nn.Module, lam: float) -> None:
        """
        Initialisation of an object of class InnerEnergy

        :param measurement_model: See base class
        :param regulariser: See base class
        :param lam: See base class
        """
        super().__init__(measurement_model, regulariser, lam)

    def hvp_state(self, x: torch.Tensor, v: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns the hessian matrix of the energy as a function of the state at the point x applied to v.
        Derivatives are built w.r.t. by calling torch.autograd.grad(...).

        :param x: State argument of Hessian matrix
        :param v: Tensor the Hessian matrix is applied to
        :return: torch.Tensor if gradients exist, otherwise None
        """
        with torch.enable_grad():
            x_ = x.detach().clone()
            x_.requires_grad = True

            e = self.forward(x_)
            de_dx = torch.autograd.grad(inputs=x_, outputs=e, create_graph=True)
        return torch.autograd.grad(inputs=x_, outputs=de_dx[0], grad_outputs=v)[0]

    def hvp_mixed(self, x: torch.Tensor, v: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """
        Returns the mixed Hessian of the energy at x as function of the state and the trainable parameters, applied
        to v. More precise: The mixed Hessian refers to

            \nabla_{\theta} \nabla_{x} E(x, \theta)

        In both cases derivatives are computed by means of torch.autograd.grad().

        :param x: State argument of hessian
        :param v: Tensor the Hessian is applied to
        :return: List of gradients w.r.t. trainable parameters of gradient of the energy w.r.t. state.
        """
        with torch.enable_grad():
            x_ = x.clone()
            x_ = x_.detach()
            x_.requires_grad = True

            e = self.forward(x_)
            de_dx = torch.autograd.grad(inputs=x_, outputs=e, create_graph=True)
        d2e_mixed = torch.autograd.grad(inputs=[p for p in self.parameters() if p.requires_grad],
                                        outputs=de_dx, grad_outputs=v)
        return list(d2e_mixed)

    def argmin(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function computes the MAP estimate using the optimiser specified by
        self._optimiser_factory.

        :param x: Tensor corresponding to the image to be reconstructed
        :return: MAP estimate in terms of a tensor
        """
        logging.info('[INNER] perform argmin to compute MAP estimate')

        x_ = x.detach().clone()
        x_.requires_grad = True
        optimiser, stopping, prox_map_factory = self._optimiser_factory([x_])
        if prox_map_factory is not None:
            setattr(x_, 'prox', prox_map_factory(x))
        loss_func_factory = self._build_loss_func_factory()
        closure_factory = self._build_closure_factory(loss_func_factory, optimiser)

        k = 0
        stop = False
        t0 = time.time()
        while not stop:
            x_old = x_.detach().clone()

            # TODO
            #   > do we want to use PyTorch optimisers here too?!?! if so, then the calls
            #       optimiser.zero_grad(), loss = ..., loss.backward() are required.

            closure = closure_factory(x_)
            optimiser.step(closure)

            stop = stopping.stop_iteration(k + 1, x_old=x_old, x_curr=x_)
            if not stop:
                k += 1
        t1 = time.time()

        logging.info('[INNER] computed MAP estimate')
        logging.info('[INNER]  > number of iterations: {:d}'.format(k + 1))
        logging.info('[INNER]  > elapsed time [s]: {:.5f}'.format(t1 - t0))

        return x_

    def sample(self, num_sampling_steps: int) -> torch.Tensor:
        raise NotImplementedError('Sampling is not implemented.')

class UnrollingEnergy(Energy):
    """
    Class which solves the inner problem by means of an unrolling scheme.
    """

    def __init__(self, measurement_model: MeasurementModel, regulariser: torch.nn.Module, lam: float,
                 optimiser_factory: Callable) -> None:
        super().__init__(measurement_model, regulariser, lam)
        self._optimiser_factory = optimiser_factory

    def argmin(self, x: torch.Tensor, num_iterations: int=5) -> torch.Tensor:
        """
        Image reconstruction using an NAG-based unrolling scheme.

        Notes
        -----
        > For the unrolling scheme only NAG-type update steps were considered.
        >

        :param x:
        :param num_iterations:
        :return:
        """
        logging.info('[INNER] perform argmin to compute MAP estimate')

        x_ = x.detach().clone()
        x_.requires_grad = True
        optimiser, stopping, prox_map_factory = self._optimiser_factory([x_])
        if prox_map_factory is not None:
            setattr(x_, 'prox', prox_map_factory(x))
            loss_func = lambda z: self.lam * self.regulariser(z)
        else:
            loss_func = lambda z: self(z)
        optimiser.zero_grad()

        t0 = time.time()
        _, x_unrolled = optimiser.step_unroll(loss_func, num_iterations)
        t1 = time.time()

        logging.info('[INNER] computed MAP estimate')
        logging.info('[INNER]  > number of iterations: {:d}'.format(num_iterations))
        logging.info('[INNER]  > elapsed time [s]: {:.5f}'.format(t1 - t0))

        return x_unrolled[0]

    def sample(self, num_sampling_steps: int) -> torch.Tensor:
        raise NotImplementedError('Sampling is not implemented.')