# Financial Tail-Risk Simulation using Generative Adversarial Networks

A PyTorch framework dedicated to synthetic financial time-series generation, extreme downside scenario simulation, and empirical tail-risk quantification. 

This repository implements, benchmarks, and evaluates several Generative Adversarial Network (GAN) architectures, addressing the convergence challenges, mode collapse, and computational bottlenecks inherent to high-dimensional heavy-tailed financial data.

---

## Overview

Traditional financial time-series generation methods frequently fail to replicate leptokurtosis, volatility clustering, and extreme joint tail risk. This project implements multiple generative models to assess their distributional fidelity, focusing specifically on Value-at-Risk ($\text{VaR}$) and Expected Shortfall ($\text{ES}$) metrics across multi-asset portfolios.

### Benchmarked Architectures
* **WGAN-GP (MLP Baseline):** Multi-Layer Perceptron architecture utilizing the Wasserstein objective with a Gradient Penalty constraint.
* **TCN-WGAN-GP:** Temporal Convolutional Network with causal, dilated convolutions designed to model multi-scale temporal dependencies without look-ahead bias.
* **TCN-WGAN-GP + EVT:** Hybrid pipeline combining temporal neural generation with Extreme Value Theory (Peaks-Over-Threshold via Generalized Pareto Distribution) for enhanced marginal tail calibration.
* **Multivariate Baselines (WA-GAN & Tail-GAN):** Joint architectures regularized via explicit tail objectives and fourth-moment (kurtosis) penalties.
* **Copula-TCN-WGAN-GP+EVT (Decoupled Reference Architecture):** A modular framework anchored in Sklar's Theorem. It isolates univariate marginal learning via TCN-WGAN-GP+EVT and reconstructs joint multi-asset dependencies through an empirical copula, preventing high-dimensional adversarial optimization failure.

