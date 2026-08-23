# Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![IEEE](https://img.shields.io/badge/IEEE-WCL-green.svg)](https://ieeexplore.ieee.org/)

Official PyTorch simulation framework for the manuscript:
**"Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks"** (*IEEE Wireless Communications Letters*).

---

##  Abstract Overview
In 6G indoor environments, pedestrian movement across doorways induces spatial concept drift and non-stationary channel statistics. Conventional Multi-Agent Reinforcement Learning (MARL) updates overwrite historical policies, triggering **catastrophic forgetting**. 

This framework proposes **Continual MAPPO**, integrating:
- **3D Indoor Metasurface Model:** 3GPP InH-inspired multi-wall path attenuation with Energy-Splitting STAR-RIS and Gauss-Markov mobility.
- **Autonomous Drift Monitoring:** Statistical 2-Wasserstein Bures distance tracking over sliding-window channel covariances.
- **Actor-Critic EWC Regularization:** Dynamic parameter consolidation via empirical Fisher Information Matrices to preserve historical policies without periodic retraining.

---

##  Installation

Clone the repository and install required dependencies:
```bash
git clone https://github.com/SepSR/Continual-MAPPO-Multi-STAR-RIS-6G.git
cd Continual-MAPPO-Multi-STAR-RIS-6G
pip install -r requirements.txt

 Reproducing Manuscript Results

1. Run Multi-Seed Continual MARL Training:

python train_evaluate.py

Executes 5-seed Monte Carlo simulations across non-stationary mobility phases
(Task A \to Task B \to Task A).

2. Generate Publication-Quality Figures:

python plot_figures.py

Generates and saves fig_continual_resilience.png matching Figure 2 in the
letter.

Citation

@article{rahmani2026continual,
  title={Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks},
  author={Rahmani, Sepehr and Rahmani, Sana and Rahmani, Sina},
  journal={IEEE Wireless Communications Letters},
  year={2026}
}

Corresponding Author: Sepehr Rahmani (sepsr2679@gmail.com)


---
