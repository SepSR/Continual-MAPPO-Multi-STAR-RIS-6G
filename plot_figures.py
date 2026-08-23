# Publication-quality figure generation for IEEE LaTeX manuscript
import numpy as np
import matplotlib.pyplot as plt

data = np.load("sim_results.npy", allow_pickle=True).item()
episodes = np.arange(1, 181)

plt.figure(figsize=(8, 4.5), dpi=300)
curves = [
    ("EWC", "#1f77b4", "-", "Proposed Continual MAPPO (EWC)"),
    ("Replay", "#2ca02c", "-.", "Experience Replay Baseline"),
    ("L2", "#ff7f0e", ":", "$L_2$ Regularization Baseline"),
    ("Standard", "#d62728", "--", "Standard MAPPO (Catastrophic Forgetting)")
]

for key, color, ls, label in curves:
    mean = np.mean(data[key], axis=0)
    std = np.std(data[key], axis=0)
    plt.plot(episodes, mean, color=color, lw=2.0, linestyle=ls, label=label)
    plt.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.12)

plt.axvline(x=60, color="gray", linestyle=":", alpha=0.7)
plt.axvline(x=120, color="gray", linestyle=":", alpha=0.7)
plt.title("Multi-Seed Sum-Rate Resilience with $\pm 1\sigma$ Confidence Intervals", fontsize=11, fontweight="bold")
plt.xlabel("Training Episodes", fontsize=10)
plt.ylabel("Network Sum-Rate (Gbps)", fontsize=10)
plt.legend(loc="lower left", fontsize=8.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("fig_continual_resilience.png", dpi=300)
print("[DONE] Figure saved as fig_continual_resilience.png")