## Failure-informed adaptive sampling for PINNs

This is the code for the paper "failure-informed adaptive sampling for PINNs". To run the experiment, you just need to run the file "one_peak.py" 
in the folder "experiments". For other experiments, the code is similiar, you can implement it easily. 
## Abstract
Physics-informed neural networks (PINNs) have emerged as an effective technique for solving PDEs in a wide range of domains. It is noticed, however, the performance of PINNs can vary dramatically with different sampling procedures. For instance, a fixed set of (prior chosen) training points may fail to capture the effective solution region (especially for problems with singularities). To overcome this issue, we present in this work an adaptive strategy, termed the failure-informed PINNs (FI-PINNs), which is inspired by the viewpoint of reliability analysis. The key idea is to define an effective failure probability based on the residual, and then, with the aim of placing more samples in the failure region, the FI-PINNs employs a failure-informed enrichment technique to adaptively add new collocation points to the training set, such that the numerical accuracy is dramatically improved. In short, similar as adaptive finite element methods, the proposed FI-PINNs adopts the failure probability as the posterior error indicator to generate new training points. We prove rigorous error bounds of FI-PINNs and illustrate its performance through several problems.

## Citation

```
  @article{doi:10.1137/22M1527763,
        author = {Gao, Zhiwei and Yan, Liang and Zhou, Tao},
        title = {Failure-Informed Adaptive Sampling for PINNs},
        journal = {SIAM Journal on Scientific Computing},
        volume = {45},
        number = {4},
        pages = {A1971-A1994},
        year = {2023}}
```

