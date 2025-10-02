# Release notes

## [Ideas/ToDos]

- Improved handling of proximal maps
- Rudimentary tests which ensure the basic functionality of the package (train/predict) 
- Solving lower-level problem by means of fixed-point iterations (e.g. TorchDEQ)
- Lazy import for quartic_bspline_extension package
- Improve dumping of configs when training: Make sure that all the settings are stored in config and written to file. 
- Make visualisations look better.
- Backward-hooks for handling of projections and prox-mappings
- Improve error handling
- Improve handling of parameter groups:
    * Dictionaries to clearly map hyperparameter values to units which shall be trained, e.g. {'filters': {'lr': ...}, 'potentials': {'lip_const': ...}}
- Improve usage of hyperparameter schedulers:
    * Setup of scheduler per hyperparameter
    * Currently only the schedulers for NAG allow to specify the parameter group on which scheduling is applied (Adam schedulers
        apply scheduling by default to all of the parameter groups.)
- Registry of schedulers, optimisers, regularisers:
    * Plug in schedulers, etc. at runtime without need to change source code of package.

## [0.1.1]

### Fixes

- Update readme.
- Fix AdaptiveNAGRestartScheduler


## [0.1.0]

### Features

- First stable public version
- Support of StudenT-potentials, spline potentials
- Example scripts for prediction and training
- Pretrained filters and filter-weights
- Callbacks and schedulers for training
- Sphinx documentation
