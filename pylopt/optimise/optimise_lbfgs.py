from typing import List, Dict, Any

from pylopt.optimise.optimise_adam import LR_KEY

DEFAULTS_GROUP_LBFGS = {LR_KEY: 1, 'max_iter': 20, 'max_eval': 25, 'tolerance_grad': 1e-7,
                        'tolerance_change': 1e-9, 'history_size': 20, 'line_search_fn': None}

def harmonise_param_groups_lbfgs(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    param_groups_ = []
    for group in param_groups:
        group_ = {'params': [p for p in group['params']],
                  'history': [p.detach().clone().requires_grad_(True) for p in group['params']]}
        for key in group.keys():
            if key != 'params':
                group_[key] = group[key]

        for key in DEFAULTS_GROUP_LBFGS.keys():
            group_[key] = group[key] if group[key] is not None else DEFAULTS_GROUP_LBFGS[key]

        param_groups_.append(group_)
    return param_groups_