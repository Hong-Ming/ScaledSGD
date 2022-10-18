# Accelerating SGD for Highly Ill-Conditioned Huge-Scale Online Matrix Completion
#### Authors: [Gavin Zhang](https://jialun-zhang.github.io), [Hong-Ming Chiu](https://hong-ming.github.io/), [Richard Y. Zhang](https://ryz.ece.illinois.edu)
#### [Link to Paper](https://arxiv.org/abs/2208.11246)
#### Citation:
```
@article{zhang2022accelerating,
  title={Accelerating SGD for Highly Ill-Conditioned Huge-Scale Online Matrix Completion},
  author={Zhang, Gavin and Chiu, Hong-Ming and Zhang, Richard Y},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
<!-- ## Table of Contents
* [Intorduction](#intorduction)
* [Directory Tree](#directory-tree)
* [Requirements](#requirements)
* [Author](#author) -->

## Intorduction
This MATLAB program contains the implementation of scaled stochastic gradient descent (ScaledSGD) algorithm proposed in our paper "Accelerating SGD for Highly Ill-Conditioned Huge-Scale Online Matrix Completion". This program also consists of the source code for all 9 experiments (Figure 1 ~ Figure 9) in the paper.

## Directory Tree
<!-- DIRSTRUCTURE_START_MARKER -->
<pre>
ScaledSGD/
├─ Data/ ................... Store data for experiments.
├─ Functions/ .............. Functions for data generation and plots.
├─ Exp1_RMSE.m ............. Run experiment in Figure 1.
├─ Exp2_EDM.m .............. Run experiment in Figure 2.
├─ Exp3_CF_Huge.m .......... Run experiment in Figure 3.
├─ Exp4_1bit.m ............. Run experiment in Figure 4.
├─ Exp5_RMSE_Noise.m ....... Run experiment in Figure 5.
├─ Exp6_1bit_Noise.m ....... Run experiment in Figure 6.
├─ Exp7_CF_Small.m ......... Run experiment in Figure 7.
├─ Exp8_CF_Medium.m ........ Run experiment in Figure 8.
├─ Exp9_CF_Large.m ......... Run experiment in Figure 9.
├─ Generate_Data.m ......... Generate data for Exp1 ~ Exp9.
├─ Plot_Figures.m .......... Plot all Figures 1 ~ Figure 9 in paper.
├─ scaledsgd.m ............. ScaledSGD algorithm.  
├─ bpr_scaledsgd.m ......... ScaledSGD algorithm optimized for BPR loss.
└─ bpr_npmaximum.m ......... Compute NP-Maximum in the paper.
</pre>
<!-- DIRSTRUCTURE_END_MARKER -->

## Requirements
- MATLAB version R2021a or later.
- MATLAB Parallel Computing Toolbox.
    
## Author
Name  : Hong-Ming Chiu

Email : hmchiu2@illinois.edu

Website : [https://hong-ming.github.io](https://hong-ming.github.io/)

## License
MIT License

