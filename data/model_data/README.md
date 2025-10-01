## Pretrained models

This directory contains pretrained filters and potentials for FoE (Fields of Experts) regularizers.

#### List of models

- **filters_7x7_chen-ranftl-pock_2014_scaled.pt**  
  Filter model from [1](#1), scaled to be applicable for images in the range [0, 1].

- **student_t_potential_pylopt_2025_I.pt**  
  Model trained with PyLOpt using pretrained and frozen filters from [1](#1).

- **student_t_potential_pylopt_2025_II.pt** and **filters_7x7_pylopt_2025_II.pt**  
  Pair of models (filters and Student-t potentials) trained with PyLOpt.

#### References

<a id="1">[1]</a> 
Chen, Y., Ranftl, R. and Pock, T., 2014. 
Insights into analysis operator learning: From patch-based sparse models to
higher order MRFs. 
IEEE Transactions on Image Processing, 23(3), pp.1060-1072.