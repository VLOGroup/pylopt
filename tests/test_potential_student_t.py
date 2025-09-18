import os
import torch
import pathlib
from confuse import Configuration

from pylopt.potential import StudentT

def equal(potential_0: StudentT, potential_1: StudentT) -> bool:
    return (potential_0.num_marginals == potential_1.num_marginals
            and torch.allclose(potential_0.weight_tensor, potential_1.weight_tensor))

def test_save_and_load(tmp_path: pathlib.Path):
    dtype = torch.float64

    file_name = 'potential.pt'

    potential_0 = StudentT().to(dtype=dtype)
    potential_0.save(str(tmp_path), file_name)

    model_path = os.path.join(tmp_path, file_name)
    potential_1 = StudentT.from_file(model_path).to(dtype=dtype)

    assert equal(potential_0, potential_1)

def test_load_from_config():
    dtype = torch.float64

    config = Configuration('unit_test')
    config.set_file(os.path.join('./tests/data', 'test_config.yaml'))

    torch.manual_seed(123)
    potential_0 = StudentT(num_marginals=config['potential']['student_t']['num_marginals'].get(),
                           initialisation_mode=config['potential']['student_t']['initialisation']['mode'].get(),
                           multiplier=config['potential']['student_t']['initialisation']['multiplier'].get(),
                           trainable=config['potential']['student_t']['trainable'].get()).to(dtype=dtype)

    torch.manual_seed(123)
    potential_1 = StudentT.from_config(config).to(dtype=dtype)

    assert equal(potential_0, potential_1)