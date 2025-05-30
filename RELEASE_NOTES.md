# Release notes

## [Unreleased]

### Features

- Usage of CUDA kernel to speed up training and prediction with gmm potentials
- Improvement of setup utilities: Currently paths to filters, weights, ecc. need to
provided  in terms of absolute oaths

## [1.1.0]

### Features

- Decrease required Python version from Python 3.12 to Python 3.11. This makes sure that
the package can be used in Google Colab.
- Moved config and model directory into root directory of package 

### Fixes

- Remove unnecessary imports

## [1.0.0]

### Features

- First stable version for training of filters, filter-weights and gmm potentials in bilevel
framework. 
- Example scripts for prediction and training
- Pretrained filters and filter-weights
- Sphinx documentation
