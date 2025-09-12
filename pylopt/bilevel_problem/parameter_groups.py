import torch
from typing import Optional

from pylopt.filters import ImageFilter
from pylopt.potential import Potential

PARAM_GROUP_NAME_KEY = 'name'
PARAM_GROUP_NAME_DICT = {Potential.__name__: 'potentials', ImageFilter.__name__: 'filters'}

def get_param_group_name(module: torch.nn.Module) -> Optional[str]:
    group_name = None
    for cls in type(module).__mro__:
        if cls.__name__ in PARAM_GROUP_NAME_DICT.keys():
            group_name = PARAM_GROUP_NAME_DICT[cls.__name__]
    return group_name