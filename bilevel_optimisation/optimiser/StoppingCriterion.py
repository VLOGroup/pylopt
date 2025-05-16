from abc import ABC, abstractmethod
import torch
import logging

class StoppingCriterion(ABC):
    """
    Abstract class which manages stopping of optimisation procedures. All classes
    subclassing from StoppingCriterion must implement the method stop_iteration().
    """

    def __init__(self, max_num_iterations: int):
        """
        Initialisation of class StoppingCriterion

        :param max_num_iterations: Maximal number of iterations to be performed
        """
        self._max_num_iterations = max_num_iterations

    def get_max_num_iterations(self) -> int:
        return self._max_num_iterations

    @abstractmethod
    def stop_iteration(self, curr_iteration_idx: int, **kwargs) -> bool:
        pass

class FixedIterationsStopping(StoppingCriterion):
    """
    Class, inheriting from StoppingCriterion, which stops an optimisation procedure
    iff the maximal number of iterations is reached.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialisation of class FixedIterationsStopping. If the key 'max_num_iterations'
        is provided, the maximal number of iterations to be performed will be set
        accordingely.

        :param kwargs: Keyword parameters
        """

        self._max_num_iterations = kwargs['max_num_iterations'] if 'max_num_iterations' in kwargs.keys() else 1000
        super().__init__(self._max_num_iterations)

    def stop_iteration(self, curr_iteration_idx: int, **kwargs) -> bool:
        return curr_iteration_idx >= self._max_num_iterations


class EarlyStopping(StoppingCriterion):
    """
    Class, inheriting from StoppingCriterion, which applies early stopping by
    stopping an optimisation procedure as soon as the euclidian norm of
    two consecutive iterates is less than a specified tolerance.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialisation of class EarlyStopping. If max_num_iterations, rel_tol are provided,
        the parameters self._max_num_iterations, self._rel_tol are set accordingely.

        :param kwargs: Keyword parameters
        """
        self._max_num_iterations = kwargs['max_num_iterations'] if 'max_num_iterations' in kwargs.keys() else 1000
        self._rel_tol = kwargs['rel_tol'] if 'rel_tol' in kwargs.keys() else 1e-5
        super().__init__(self._max_num_iterations)

    def stop_iteration(self, curr_iteration_idx: int, **kwargs) -> bool:
        """
        Function which checks if the maximal number of iterations is reached
        or the euclidian distance of x_old, x_curr is less than self._rel_tol.

        :param curr_iteration_idx: Current iteration index
        :param kwargs: Keyword parameters
        :return: Boolean value indicating if procedure shall be stopped.
        """
        bool_1 = curr_iteration_idx >= self._max_num_iterations

        x_old = kwargs['x_old']
        x_curr = kwargs['x_curr']
        bool_2 = (torch.linalg.norm(x_old - x_curr).detach().cpu().item() < self._rel_tol)

        logging.debug('[STOPPING] max_num_iterations_reached: {:s}, min_rel_tol_reached: {:s}'.format(str(bool_1),
                                                                                           str(bool_2)))
        return bool_1 or bool_2