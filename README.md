# Gaussian process-aided transfer learning for probabilistic load forecasting against anomalous events

This repository contains the official implementation of the paper  
**“[Gaussian process-aided transfer learning for probabilistic load forecasting against anomalous events]”**  
📄 [Read the paper here](](https://ieeexplore.ieee.org/abstract/document/10068293/))

The dataset used in this work is publicly available at:  
📊 [Dataset link](https://your-dataset-link.com)

---

## Overview

This project implements a two-stage forecasting framework:

1. A residual multi-layer perceptron (Res-MLP) for primary load prediction  
2. A sparse variational Gaussian Process (SVGP) for residual correction and uncertainty quantification  

The final prediction is obtained by combining the neural network output with the GP posterior mean, while predictive variance is used to construct probabilistic intervals.

---

## Environment

- Python 3.8+
- PyTorch
- GPyTorch

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourcitation,
  title={...},
  author={...},
  journal={...},
  year={...}
}
