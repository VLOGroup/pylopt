from typing import Dict, Any
import torch
import logging
import numpy as np

from pylopt.regularisers.fields_of_experts import FieldsOfExperts
from pylopt.utils.Timer import Timer

def warm_up(regulariser: FieldsOfExperts, x: torch.Tensor, num_calls: int = 100) -> None:
    logging.info('[WARM UP] Perform warm-up forward and backward calls after compilation.')
    x_ = x.detach().clone().requires_grad_(True)

    _ = regulariser(x_)

    timings = []
    for _ in range(0, num_calls):
        with Timer(x.device) as t:
            with torch.enable_grad():
                y = regulariser(x_)
            _ = torch.autograd.grad(inputs=x_, outputs=y)
        timings.append(t.time_delta())

    logging.info('[WARM UP] mean elapsed time [ms]: {:.5f}'.format(np.mean(timings)))
    logging.info('[WARM UP] median elapsed time [ms]: {:.5f}'.format(np.median(timings)))

def compile_regulariser(regulariser: FieldsOfExperts, x: torch.Tensor,
                        **compile_options: Dict[str, Any]) -> FieldsOfExperts:
    """
    This function compiles the regulariser using torch.compile() for improved memory management and
    speed-up in forward and backward calls. After compilation a few warm-up iterations were performed.

    NOTE
    ----
        > **IMPORTANT** Call this method only after assigning the module to a device:

            ...
            energy = Energy(...)
            energy.to(device=device)
            energy.compile_regulariser(...)
            ...

        > Compilation is not only suitable for GPU-based inference or training - also models on CPU can
            be compiled.
        > Compilation takes some time in the beginning - this is why warm-up forward and backward calls
            were performed right after compilation.
        > Per default the most aggressive optimisation options were applied. For all the available options see
            https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

    :param regulariser: Instance of FieldsOfExperts to be compiled.
    :param x: PyTorch tensor which is used in the warm-up procedure to perform several forward and backward calls.
    :param compile_options: Optional dictionary of compiling options - for the available options see
        https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    :return: compiled FieldsOfExperts model
    """

    torch._functorch.config.donated_buffer = False

    mode = compile_options.get('mode', 'max-autotune')
    backend = compile_options.get('backend', 'inductor')
    dynamic = compile_options.get('dynamic', True)
    fullgraph = compile_options.get('fullgraph', True)

    regulariser = torch.compile(regulariser, mode=mode, backend=backend, dynamic=dynamic, fullgraph=fullgraph)
    warm_up(regulariser, x, num_calls=100)
    return regulariser
