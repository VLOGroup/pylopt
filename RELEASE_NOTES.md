# Release notes

## [Unreleased]

### Features

- Use CUDA kernel to speed up training and prediction with gmm potentials
- Use filter_weights as parameters of the StudentT potential only
- Improve handling of proximal maps

## [1.1.0]

### Features

- Decreased required Python version from Python 3.12 to Python 3.11. This makes sure that
the package can be used in Google Colab.
- Moved config and model directory from project directory into root directory of package
- Moved examples from project directory into root directory of package
- Made usage of proximal gradient method for NAG-type optimisers configurable. 
- Use MANIFEST.in to include all the non *.py files belonging to the package to the Python wheel.

### Fixes

- Removed unnecessary imports

## [1.0.0]

### Features

- First stable version for training of filters, filter-weights and gmm potentials in bilevel
framework. 
- Example scripts for prediction and training
- Pretrained filters and filter-weights
- Sphinx documentation
