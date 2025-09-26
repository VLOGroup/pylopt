# PyLevel Optimisation (PyLOpt)

PyLOpt is a PyTorch-based library for learning hyperparameters $\theta$ within the context of image reconstruction by means of solving the bilevel problem

$$(P_{bilevel}) ~~~~~\inf_{\theta} F(u^{*}(\theta), u^{(0)}) ~~~ \text{s.t.}  
    ~~~ u^{*}(\theta)\in\operatorname{arginf}_{u}E(u, u^{(\delta)}, \theta)$$

The function $F$ refers to the upper loss function quantifying the goodness of the learned $\theta$ w.r.t. groundtruth data $u^{(0)}$. $E$ denotes the lower cost or energy function, which is used to reconstruct clean data $u^{(0)}$ from noisy observations $u^{(\delta)}$. We assume that $E$ is of the form

$$E(u, u^{(\delta)}, \theta) = 
    \frac{1}{2\sigma^{2}}\|u - u^{(\delta)}\|_{2}^{2} + \lambda R(u, \theta)$$

where $\sigma$ indicates the noise level, $R$ refers to a regulariser and $\lambda>0$ denotes a regularisation parameter. We consider trainable regulariser modelled by means of a Fields of Experts (FoE) model: Let $k_{1}, \ldots, k_{m}$ denote quadratic image filters and let $\rho(\cdot| \gamma)$ be a parameter-dependent potential function. For the potential parameters $\gamma_{1}, \ldots, \gamma_{m}$, and the merged parameter $\theta$ comprising filters and potential parameters, we define

$$R(u, \theta) = \sum_{j}\sum_{i}-\log\rho([k_{j} * u]_{i}|\gamma_{j}),$$

where in the inner sum we sum up all elements of the $j$-th filter response.

#### Objective

PyLOpt aims to serve as a toolbox for scientists and engineers to address bilevel problems in imaging supporting different gradient-based solution methods. The package is modular and extendable by design and follows familiar interfaces from pupular Python packages such as SciPy ([[2]](#2)), scikit-learn ([[8]](#8)) and PyTorch ([[9]](#9)).

## Table of contents

- [Objective](#objective)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [References](#references)

## Features

### Current features

- Image reconstruction using pretrained filter and potential models by solving the lower level problem $\operatorname{arginf}_{u}E(u, u^{(\delta)}, \theta)$ using one of the following gradient-based methods
  - NAG: Nesterov accelerated gradient method - see f.e. [[5]](#5)
  - NAPG: Proximal gradient method with Nesterov acceleration - see f.e. [[4]](#4)
  - Adam: See [[6]](#6)
  - Unrolling: Both NAG and NAPG are implemented as well using an unrolled approach. This allows to solve the upper-level problem using automatic differentiation.
- Training of filters and/or potentials of an FoE regualriser model by solving the bilevel problem $P_{bilevel}$. The training relies on the application of one the following gradient-based methods onto the upper problem:
  - NAG
  - Adam
  - LBFGS: Quasi-Newton method - see [[7]](#7)

  Gradients of solutions of the lower level problem w.r.t. the parameter $\theta$ are computed or by implicit differentiation, or automatic differentiation provided the lower problem is solved by means of an unrolling scheme. 
- Modularity and extensibility: The package is modular by design. Its architecture allows easy customization and extension. Each of the core components
  - ImageFilter
  - Potential
  - FieldsOfExperts
  - Energy

  is encapsulated in its own module. Thus, all of these components can be exchanged easily without any need to modify the core logic. In addition, methods for the solution of the lower problem (`solve_lower.py`) and the upper problem (`solve_bilevel.py`) can easily be added.
- The repository contains pretrained models and sample scripts and notebooks showing the application of the package for image denoising.

### Upcoming features

- Sampling based approach for solving inner problem

## Installation

- Through pip

    ```
    pip install pylopt
    ```

- From source

    ```
    git clone https://github.com/VLOGroup/pylopt.git
    ```

## Core components 

The FoE regulariser is implemented via the `FieldsOfExperts` class. It combines an `ImageFilter`, which defines the convolutional filters applied to the image, and a subclass of `Potential`, which models the corresponding potential functions. The lower problem is modelled by the PyTorch module `Energy`, which represents the energy function to be minimised. An object of this class contains an `M̀easurementModel` instance, a PyTorch module modeling the measurement process, and a `FieldsOfExperts` instance as its components. 

Image reconstruction or the solution of the lower level problem is carried out by the function `solve_lower()`. The training of filters and potentials is managed by the class `BilevelOptimisation`. 

For the usage of the package and its methods see section [Usage](#usage).

## Usage

### Conceptual

The interface of the function `solve_lower()` which is used to solve the lower level problem is designed to closely follow the conventions of SciPy optimisation routines. Given an `Energy` instance, the corresponding lower level problem can be solved using Nesterov's accelerated gradient method (NAG) via

```python
lower_prob_result = solve_lower(energy=energy, method='nag', 
                                options={'max_num_iterations': 1000, 'rel_tol': 1e-5, 'batch_optimisation': False})
```

The upper-level optimisation, i.e. training of filters and potential parameters, follows conventions of scikit-learn for interface design and usability. Training these parameters using Adam for the upper level optimisation and NAPG for the lower level optimisation is obtained via

```python

prox = DenoisingProx(noise_level=noise_level)
bilevel_optimisation = BilevelOptimisation(method_lower='napg',
                                           options_lower={'max_num_iterations': 1000, 'rel_tol': 1e-5, 'prox': prox, 
                                                          'batch_optimisation': False}, 
                                           operator=torch.nn.Identity(),
                                           noise_level=0.1, 
                                           solver='cg', options_solver={'max_num_iterations': 500},
                                           path_to_experiments_dir=path_to_eval_dir)

bilevel_optimisation.learn(regulariser, lam, l2_loss_func, train_image_dataset,
                           optimisation_method_upper='adam', 
                           optimisation_options_upper={'max_num_iterations': 15000, 'lr': [1e-3, 1e-1], 
                                                       'parameterwise': True},
                           dtype=dtype, device=device, callbacks=callbacks, schedulers=schedulers)

```

### Concrete

Concrete and executable code for training and prediction is contained in `bilevel_optimisation/examples`. Please note that reproducibility of training results can be obtained only when using the datatype `torch.float64`. However, this comes at the cost of increased computation time. 

#### Denoising using pretrained models

- **Example I**
  - Filters: Pretrained filters from [[1]](#1)
  - Potential: 
    - Type: Student-t
    - Weights: Optimised using `BilevelOptimisation`

  To run the script, execute

    ```
    python examples/scripts/denoising_predict.py
    ```
  
  Alternatively, run the Jupyter notebook `example_denoising_predict.ipynb`. Denoising the images `watercastle`
  and `giraffe` of the well known BSDS300 dataset (see [[3]](#3)), we obtain

  | Method  | Options                                                                                                 | mean PSNR of denoised images [dB] |
  |:-------:|:--------------------------------------------------------------------------------------------------------|:---------------------------------:|
  |  'nag'  | ``` {'max_num_iterations': 1000, 'rel_tol': 1e-5, 'batch_optimisation': True} ```                       |             28.79780              |
  | 'napg'  | ``` {'max_num_iterations': 1000, 'rel_tol': 1e-5, 'prox': ..., 'batch_optimisation': False} ```         |             28.77631              |
  | 'adam'  | ``` {'max_num_iterations': 1000, 'rel_tol': 1e-4, 'lr': [1e-3, 1e-3], 'batch_optimisation': False} ```  |             29.24183              |

  and, when using the Adam optimiser:

  ![](images/results/prediction_I/results.png)

- **Example II** 

  ...

#### Training of FoE models

- **Example I**
  - Filters: 
    - Pretrained filters from [[1]](#1)
    - Non-trainable
  - Potential: 
    - Type: Student-t
    - Weights:
      - Uniform initialisation
      - Trainable 
  - Lower level 
  - Upper level
  To run the training script, execute   

    ```
    python examples/scripts/denoising_train.py --configs example_training_I 
    ```

  |                   Training stats                   |                Potential weight stats                |                    Test triplet                    |
  |:--------------------------------------------------:|:----------------------------------------------------:|:--------------------------------------------------:|
  |  ![](images/results/training_I/training_stats.png) | ![](images/results/training_I/student_t_stats.png)   | ![](images/results/training_I/test_triplets.jpg)   |

  Note that already after very iterations, we obtain a notable denoising performance. The main 
  reason for this behaviour is the nearly optimal initialisation of the filters. 

- **Example II**
  - Filters:
    - Pretrained filters from [[1]](#1)
    - Trainable
  - Potential:
    - Type: Student-t
    - Weights: 
      - Uniform initialisation
      - Trainable    
  - Inner energy: `OptimisationEnergy`
  - Optimiser:
    - Inner: NAGOptimiser
    - Outer: NAGOptimiser
  
  To run the corresponding training script, execute  

  ```
  python examples/scripts/denoising_train.py --configs example_training_II
  ```

  The training results are consistent with the results obtained in Example I. Notably, although the filters 
  are trainable, their $l^{2}$-norm remains nearly constant throughout the training process:

  |                   Training stats                   |                   Filter stats                    |               Student-t potential                |
  |:--------------------------------------------------:|:-------------------------------------------------:|:------------------------------------------------:|
  | ![](images/results/training_II/training_stats.png) |  ![](images/results/training_II/filter_stats.png) | ![](images/results/training_II/potentials.png)   |

- **Example III**
  - Filters:
    - 7x7 DCT filters
    - Trainable
  - Potential: 
    - Type: Student-t
    - Weights:
      - Uniform initialisation
      - Trainable
  - Inner energy: `OptimisationEnergy`
  - Optimiser:
    - Inner: NAGOptimiser
    - Outer: Adam

  To train filters and filter weights with these specifications, call
  ```
  python examples/scripts/denoising_train.py --configs example_training_III
  ```

  We obtain similar denoising results as before:

  
  |                    Training stats                    |                   Filter stats                    |                Potential weight stats                |                Student-t potential                |
  |:----------------------------------------------------:|:-------------------------------------------------:|:----------------------------------------------------:|:-------------------------------------------------:|
  |  ![](images/results/training_III/training_stats.png) | ![](images/results/training_III/filter_stats.png) | ![](images/results/training_III/student_t_stats.png) | ![](images/results/training_III/potentials.png)   |

- **Example IV** 
  - Filters:
    - Uniform on [-1, 1]
    - Trainable
  - Potential: 
    - Type: Student-t
    - Weights:
      - Uniform initialisation
      - Trainable
  - Inner energy: `OptimisationEnergy`
  - Optimiser:
    - Inner: NAGOptimiser
    - Outer: Adam

  Results: ...

- **Example V**
  - Filters:
    - Uniform on [-1, 1]
    - Trainable
  - Potential: 
    - Type: Student-t
    - Weights:
      - Uniform initialisation
      - Trainable
  - Inner energy: `UnrollingEnergy`
  - Optimiser:
    - Inner: UnrollingNAGOptimiser
    - Outer: NAGOptimiser

  To train the Student-t potential weights with this configuration run   
  ```
  python examples/scripts/denoising_train.py --configs example_training_V
  ```
  
  Using the unrolling scheme provided in `UnrollingNAGOptimiser` we obtain descent results:


  |                   Training stats                    |                    Test triplet                    |
  |:---------------------------------------------------:|:--------------------------------------------------:|
  | ![](./images/results/training_V/training_stats.png) | ![](./images/results/training_V/test_triplets.jpg) |


- **Example VI**
  - Potential: Gaussian mixture
  - Filters:
    - Pretrained filters from [[1]](#1)
    - Non-trainable
  - Filter weights:
    - Pretrained filters using configuration in Example I
    - Non-trainable
 - Optimiser:
    - Inner: NAGOptimiser
    - Outer: Adam

  Results: ...

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## References

<a id="1">[1]</a> 
Chen, Y., Ranftl, R. and Pock, T., 2014. 
Insights into analysis operator learning: From patch-based sparse models to
higher order MRFs. 
IEEE Transactions on Image Processing, 23(3), pp.1060-1072.

<a id="2">[2]</a>
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, 
David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, 
Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, 
Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, 
CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, 
Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, 
Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, 
Paul van Mulbregt, and SciPy 1.0 Contributors, 2020.
SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.
Nature Methods, 17(3), 261-272.

<a id="3">[3]</a>
Martin, D., Fowlkes, C., Tal, D. and Malik, J., 2001, July. A database of human segmented natural images and 
its application to evaluating segmentation algorithms and measuring ecological statistics. In Proceedings 
eighth IEEE international conference on computer vision. ICCV 2001 (Vol. 2, pp. 416-423). IEEE.

<a id="4">[4]</a>
Beck, A., 2017. First-order methods in optimization. Society for Industrial and Applied Mathematics.

<a id="5">[5]</a>
d’Aspremont A, Scieur D, Taylor A. Acceleration methods. Foundations and Trends® in Optimization. 2021 Dec 14;5(1-2):1-245.

<a id="6">[6]</a>
Kingma DP. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980. 2014.

<a id="7">[7]</a>
Nocedal, J. and Wright, S.J., 2006. Numerical optimization. New York, NY: Springer New York.

<a id="8">[8]</a>
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, pp.2825-2830.

<a id="9">[9]</a>
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and Desmaison, A., 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

## License

~~~~MIT License~~~~

