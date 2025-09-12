from typing import List, Dict, Any
import torch

def restart_condition_loss_based(param_groups: List[Dict[str, Any]], **kwargs: Dict[str, Any]) -> bool:
    """
    Function implementing loss based restart condition.

    NOTE
    ----
        > Each parameter group must contain the key 'history' of parameter values of the previous iteration.

    :param param_groups: List of dictionaries representing the parameter groups.
    :param kwargs: Keyword arguments - must contain the key-value pair ('func', func), where func is a
        callable representing the loss function.
    :return: Flag indicating if restart condition is satisfied.
    """
    func = kwargs.get('func', None)
    if func is not None:
        f_old = func(*[p for group in param_groups for p in group['history']])
        f_curr = func(*[p for group in param_groups for p in group['params']])

    return f_curr > f_old

def restart_condition_gradient_based(param_groups: List[Dict[str, Any]], **kwargs) -> bool:
    """
    Function implementing gradient based restart condition.

    :param param_groups:
    :param kwargs:
    :return:
    """
    grad_func = kwargs.get('grad_func', None)

    ret_val = False
    if grad_func is not None:
        grads = [g.clone() for g in grad_func(*[p for group in param_groups for p in group['params']])]

        tmp = torch.zeros(1).to(device=grads[-1].device)
        for idx, group in enumerate(param_groups):
            tmp += torch.sum(grads[idx] * (group['params'][-1] - group['history'][-1].to(device=grads[-1].device)))
        ret_val = tmp > 0

    return ret_val