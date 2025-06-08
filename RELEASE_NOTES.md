# Release notes

## [Ideas]

- Wrapper for Optimisers
  - Current NAG-type implementations use closure functions for reevaluation of the loss and gradient computation.
  - The backtracking approach does not need multiple gradient computations, but multiple loss evaluations. Doing
    both every time is expansive.
  - Instead of passing closures, one could instead pass a callable representing the loss function. As a consequence, 
    gradients would need to be computed in the step method of the optimiser ... Disadvantages?!?

## [Unreleased]

### Features

- Use CUDA kernel to speed up training and prediction with gmm potentials
- Implement inner energy using unrolling scheme
- Improved handling of proximal maps
- Deal properly with param_groups in NAGOptimiser

## [2.0.0]

### Features

- Improved setup handling
  - Filter module, which allows to save and load filters using state dictionaries
  - Filter weights considered as part of Student-T potential
- Extended visualisations
  - Visualisation of negative log Student-T potentials
  - Visualisation of filter responses by means of violon plots
- Extension of Readme file

### Fixes

- Test data loader couldn't load images of different size - introduced padding
to fix this issue.

## [1.1.3]

### Feature

- Speed improvements by using torch.float32; additionally remove several debug log messages
and retain computational graph only if really necessary.

## [1.1.2]

### Fixes

- In denoising notebook use custom config which is already checked in the repository.

## [1.1.1]

### Fixes

- Enable loading of pre-trained filters, filter weights, etc. also for CPU.

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
