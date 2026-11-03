# Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE WCL](https://img.shields.io/badge/IEEE-Wireless%20Communications%20Letters-00629B.svg)](https://ieeexplore.ieee.org/)

Official simulation and evaluation repository for the manuscript:
**"Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks"** (*IEEE Wireless Communications Letters*, Manuscript ID: WCL2026-3112).

**Authors:** Sepehr Rahmani, Sana Rahmani, and Sina Rahmani

## Overview

In partitioned indoor 6G non-orthogonal multiple access (NOMA) networks equipped with simultaneously transmitting and reflecting reconfigurable intelligent surfaces (STAR-RISs), pedestrian movement between rooms induces abrupt spatial concept drift through wall-loss transitions and transmission/reflection mode flips. Standard multi-agent reinforcement learning (MARL) policies overwrite weights learned in earlier rooms, suffering severe catastrophic forgetting upon room re-entry.

We propose **Continual Multi-Agent Proximal Policy Optimization (Continual MAPPO)** under centralized training with decentralized execution (CTDE), featuring:

1. **Consistent Bottleneck SIC & Imperfect Cascaded CSI Model:** Models a $20\text{ m} \times 20\text{ m}$ 4-room indoor 6G network at $f_c = 28\text{ GHz}$ with modified-slope 3GPP TR 38.901 InH-Office path loss, explicit COST-231 wall penetration loss ($L_{\text{wall}} = 10\text{ dB}$), 1st-order Gauss-Markov and exact Bessel channel aging ($f_D = 112\text{ Hz}$), TDD subarray pilot overhead ($\tau_{p,\text{RIS}} = 17$ symbols), noisy/delayed CSI ($\sigma_e^2 \in [0, 0.20]$), and bottleneck-consistent downlink MISO-NOMA SIC constraints.
2. **Two-Buffer Drift-Aware Actor-Critic EWC (Algorithm 1):** Without task-boundary labels, combines a policy-invariant 2-Wasserstein Bures drift detector on trace-normalized reference covariances ($\epsilon_{\text{th}} = 0.35$, 3-slot persistence, $2W$-slot lockout) with a $W$-slot staging FIFO ($\mathcal{Q}_W$) and pre-drift ring buffer ($\mathcal{D}_{\text{prev}}$, $M_F = 500$ slots) to consolidate clean importance matrices across all decentralized actors and the centralized critic.
3. **Multi-Regime Validation & Hardware-Fair Benchmarks:** Evaluated across 10 random seeds over 3-, 4-, and 6-phase room trajectories against Joint Oracle, Oracle-EWC, MAPPO+Reset, Continual MAS, Tuned Experience Replay, Periodic Task-Free EWC, Tuned $L_2$, Standard MAPPO, iterative AO/SCA, and hardware-fair conventional $180^\circ$ RIS.

## Repository Structure

| File | Description |
| :--- | :--- |
| `config.py` | Centralized system and simulation parameters matching **Table I** (RF carrier, linear thermal noise $\sigma^2 \approx 2.00\text{ pW}$, 3GPP/COST-231 geometry, TDD pilot accounting, Bures detector, EWC, and PPO hyperparameters). |
| `environment.py` | 3D indoor multi-STAR-RIS NOMA environment implementing **Equations (1)–(10)**: modified-slope 3GPP InH-Office path loss, COST-231 wall intersection counting, Gauss-Markov fading, imperfect subarray CSI, feasible action projections $\Pi(\tilde{a}_t^i)$, bottleneck SIC decoding, Dec-POMDP observations, and 2-Wasserstein Bures distance. |
| `model_ewc.py` | Decentralized Gaussian MLP actors ($\{\pi_{\theta_i}\}_{i=0}^K$), centralized critic ($V_\psi$), parameter verification ($P_{\text{tot}} = 123.1\text{k}$ at $N=32$; $205.5\text{k}$ at $N=64$), and **Algorithm 1** (`TwoBufferContinualMAPPO`) with staging FIFO $\mathcal{Q}_W$, ring buffer $\mathcal{D}_{\text{prev}}$, and diagonal Fisher consolidation (**Equations 11a, 11b, 12**). |
| `train_evaluate.py` | Multi-seed evaluation and statistical verification suite reproducing **Table II**, live Algorithm 1 rollout checks, drift detector ablations (Bures vs. MMD, Symmetric KL, Frobenius), random-walk generalization, gross/net rates under AR(1) and Bessel aging, and two-sided Welch's $t$-tests ($n = 10$ seeds). |
| `plot_figures.py` | Generates all four publication-quality figures (`fig_system_model_3d.png`, `fig_continual_resilience.png`, `fig_power_scaling.png`, and `fig_cdf_tradeoff.png`) at 300 DPI. |
| `all_figures_codes.py` | Standalone, self-contained unified script combining all verification checks and figure generation in a single file for rapid one-command inspection. |
| `requirements.txt` | Python package dependencies (`numpy`, `scipy`, `matplotlib`, `pandas`, `torch`). |

## Key Numerical Results (Table I & Table II Verification)

| Metric / Benchmark | Value in Manuscript & Code |
| :--- | :--- |
| **Linear Thermal Noise Variance ($\sigma^2$)** | $2.00 \times 10^{-12}\text{ W}$ ($2.0\text{ pW}$, $-87.0\text{ dBm}$) at $N_{0,\text{dBm}} = -174\text{ dBm/Hz}$, $B = 100\text{ MHz}$, $\text{NF}_{\text{dB}} = 7\text{ dB}$ |
| **CTDE Parameter Count ($P_{\text{tot}}$)** | $123,073$ ($123.1\text{k}$) at $N=32$ ($\text{BS}: 17.4\text{k}$, $\text{RIS}: 2 \times 29.8\text{k}$, $\text{Critic}: 46.0\text{k}$); $205,505$ ($205.5\text{k}$) at $N=64$ |
| **Zero-Shot Retention at Room Re-Entry (Ep. 121)** | $3.06 \pm 0.05\text{ Gbps}$ ($89.5\%$ of pre-drift rate, $+65.4\%$ over Standard/Reset MAPPO, $+44.3\%$ over Periodic EWC) |
| **Phase-3 Final Rate (Ep. 180, $N=32$)** | $3.18 \pm 0.05\text{ Gbps}$ at $\sigma_e^2 = 0$ ($93.0\%$ of Oracle $3.42\text{ Gbps}$); $2.98\text{ Gbps}$ at $\sigma_e^2 = 0.05$ |
| **90%-of-Oracle Recovery Time** | $2\text{ episodes}$ (Continual MAPPO) vs. $38\text{ episodes}$ (MAPPO+Reset) |
| **STAR-RIS vs. Conventional $180^\circ$ RIS ($N=64$)** | $3.58\text{ vs. } 2.79\text{ Gbps}$ ($+28.3\%$ gross); $3.008\text{ vs. } 2.344 / 2.159\text{ Gbps}$ ($+28.3\%\text{--}39.3\%$ net under AR(1) aging) |
| **Multi-Phase Retention ($\mathcal{L}=3, 5$ transitions)** | $3.09\text{ Gbps}$ ($90.4\%$ of Oracle over 4 phases) and $2.97\text{ Gbps}$ ($86.8\%$ over 6 phases) |

## Installation & Quick Start

Clone the repository and install the dependencies:

```bash
git clone https://github.com/SepSR/Continual-MAPPO-Multi-STAR-RIS-6G.git
cd Continual-MAPPO-Multi-STAR-RIS-6G
pip install -r requirements.txt
```

### 1. Run Modular Multi-Seed Evaluation & Statistical Checks

```bash
python train_evaluate.py
```

This runs all mathematical verifications (noise variance, Doppler/aging, TDD pilot overhead, parameter counts, Bures $W_2$ percentiles, Pearson drift-curvature correlation $r = 0.96$, live Algorithm 1 rollout, and 10-seed Welch's $t$-tests) and saves `sim_results.npy`.

### 2. Generate All 4 Manuscript Figures

```bash
python plot_figures.py
```

Generates:
- `fig_system_model_3d.png` (Fig. 1: 3D indoor multi-STAR-RIS network layout)
- `fig_continual_resilience.png` (Fig. 2: 10-seed gross sum-rate trajectories across room transitions)
- `fig_power_scaling.png` (Fig. 3: Phase-3 gross sum-rate vs. $P_{\max}$ across $N \in \{16, 32, 64\}$, AO/SCA, and $180^\circ$ RIS)
- `fig_cdf_tradeoff.png` (Fig. 4: Empirical bottleneck SINR CDF and multi-phase $\lambda_{\text{EWC}}$ trade-off)

### 3. Run Standalone Unified Script (Single-File Option)

```bash
python all_figures_codes.py
```

## Citation

If you find this repository helpful in your research, please cite our letter:

```bibtex
@article{rahmani2026continual,
  author  = {Rahmani, Sepehr and Rahmani, Sana and Rahmani, Sina},
  title   = {Continual {MAPPO} for Mobility-Aware Resource Allocation in {6G} Multi-{STAR-RIS} Indoor {NOMA} Networks},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  note    = {Manuscript ID: WCL2026-3112}
}
```

## Contact

**Corresponding Author:** Sepehr Rahmani (`sepsr2679@gmail.com`)
Department of Electrical Engineering, Imam Khomeini International University, Qazvin, Iran.
