import os
import pathlib
import torch
from confuse import Configuration

from pylopt.filters import ImageFilter

def equal(filter_0: ImageFilter, filter_1:ImageFilter) -> bool:
    return (filter_0.filter_dim == filter_1.filter_dim
            and filter_0.padding == filter_1.padding
            and filter_0.padding_mode == filter_1.padding_mode
            and filter_0.orthogonality_options == filter_1.orthogonality_options
            and torch.allclose(filter_0.filter_tensor, filter_1.filter_tensor))

def test_save_and_load(tmp_path: pathlib.Path):
    dtype = torch.float64

    image_filter_0 = ImageFilter().to(dtype=dtype)
    image_filter_0.save(str(tmp_path), 'image_filter.pt')

    model_path = os.path.join(tmp_path, 'image_filter.pt')
    image_filter_1 = ImageFilter.from_file(model_path).to(dtype=dtype)

    assert equal(image_filter_0, image_filter_1)

def test_load_from_config():
    dtype = torch.float64

    config = Configuration('unit_test')
    config.set_file(os.path.join('./tests/data', 'test_config.yaml'))

    torch.manual_seed(123)
    image_filter_0 = ImageFilter(
        filter_dim=config['image_filter']['filter_dim'].get(),
        padding=config['image_filter']['padding'].get(),
        padding_mode=config['image_filter']['padding_mode'].get(),
        initialisation_mode=config['image_filter']['initialisation']['mode'].get(),
        multiplier=config['image_filter']['initialisation']['multiplier'].get(),
        normalise=config['image_filter']['initialisation']['normalise'].get(),
        trainable=config['image_filter']['trainable'].get(),
        orthogonality_options={
            'enable': config['image_filter']['orthogonality']['enable'].get(),
            'max_num_iterations': config['image_filter']['orthogonality']['max_num_iterations'].get()
        }
    ).to(dtype=dtype)

    torch.manual_seed(123)
    image_filter_1 = ImageFilter.from_config(config).to(dtype=dtype)

    assert equal(image_filter_0, image_filter_1)
