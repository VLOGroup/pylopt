import torch

def apply_prox_map(p: torch.nn.Parameter, step_size: float) -> torch.nn.Parameter:
    """
    Function which applies prox map to parameter provided that the parameter has an
    attribute called 'prox'. By assumption this attribute is a callable taking the
    value of the parameter and the step size to be applied as arguments.

    :param p:
    :param step_size:
    :return: Parameter whose value is updated.
    """
    if hasattr(p, 'prox'):
        p.data.copy_(p.prox(p.data, step_size))
    return p

def apply_projection(p: torch.nn.Parameter) -> torch.nn.Parameter:
    """
    This function, applies a projection to the value of the given parameter. This is
    implemented similarly as for _apply_prox_map, i.e. the projection map has to be provided as callable
    via attribute 'proj' of the parameter.

    :param p:
    :return: Parameter whose value is updated.
    """
    if hasattr(p, 'proj'):
        p.data.copy_(p.proj(p.data))
    return p