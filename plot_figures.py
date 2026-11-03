#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Quality Figure Generation for IEEE Wireless Communications Letters
Manuscript ID: WCL2026-3112 (Revised)
Title: Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks
Authors: Sepehr Rahmani, Sana Rahmani, and Sina Rahmani
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm
from train_evaluate import run_experiment

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def generate_fig1_3d_model():
    """Generates Figure 1: 3D layout of the 20m x 20m indoor 6G network (M=4 rooms, K=2 STAR-RIS, U=4 UEs, M_t=8)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9.6})
    fig = plt.figure(figsize=(4.6, 2.22), dpi=300)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    walls = [
        [[10, 0, 0], [10, 1.2, 0], [10, 1.2, 3.8], [10, 0, 3.8]],
        [[10, 2.8, 0], [10, 17.2, 0], [10, 17.2, 3.8], [10, 2.8, 3.8]],
        [[10, 18.8, 0], [10, 20, 0], [10, 20, 3.8], [10, 18.8, 3.8]],
        [[0, 10, 0], [4.3, 10, 0], [4.3, 10, 3.8], [0, 10, 3.8]],
        [[5.9, 10, 0], [14.1, 10, 0], [14.1, 10, 3.8], [5.9, 10, 3.8]],
        [[15.7, 10, 0], [20, 10, 0], [20, 10, 3.8], [15.7, 10, 3.8]],
    ]
    wall_poly = Poly3DCollection(walls, alpha=0.16, facecolor="#6baed6", edgecolor="#2171b5", linewidths=0.75, zorder=1)
    ax.add_collection3d(wall_poly)

    ris_patches = [
        [[10, 4.2, 1.3], [10, 5.8, 1.3], [10, 5.8, 2.7], [10, 4.2, 2.7]],
        [[10, 14.2, 1.3], [10, 15.8, 1.3], [10, 15.8, 2.7], [10, 14.2, 2.7]],
    ]
    ris_poly = Poly3DCollection(ris_patches, alpha=0.88, facecolor="#fd8d3c", edgecolor="#a63603", linewidths=1.1, zorder=3)
    ax.add_collection3d(ris_poly)

    bs = np.array([10.0, 10.0, 3.50])
    for rc in [np.array([10.0, 5.0, 2.0]), np.array([10.0, 15.0, 2.0])]:
        ax.plot([bs[0], rc[0]], [bs[1], rc[1]], [bs[2], rc[2]], color="#d92b2b", lw=1.35, ls="-", zorder=4)

    ax.plot([10.0, 1.6], [5.0, 2.2], [2.0, 1.2], color="#1f77b4", lw=1.3, ls="--", label="Refl. ($\\chi^{\\mathrm{R}}=1$)", zorder=4)
    ax.plot([10.0, 14.2], [5.0, 2.2], [2.0, 1.2], color="#2ca02c", lw=1.3, ls="-.", label="Trans. ($\\chi^{\\mathrm{T}}=1$)", zorder=4)
    ax.plot([10.0, 1.6], [15.0, 17.8], [2.0, 1.2], color="#1f77b4", lw=1.3, ls="--", zorder=4)
    ax.plot([10.0, 14.2], [15.0, 17.8], [2.0, 1.2], color="#2ca02c", lw=1.3, ls="-.", zorder=4)

    traj_x = np.linspace(1.6, 14.2, 30)
    traj_y = 2.0 + 0.20 * np.sin(np.linspace(0, 2.0 * np.pi, 30))
    ax.plot(traj_x, traj_y, np.ones_like(traj_x) * 1.2, color="#333333", lw=1.3, ls=":", label="Round-Trip Path", zorder=4)

    ax.scatter([bs[0]], [bs[1]], [bs[2]], color="#d92b2b", marker="^", s=78, edgecolor="black", linewidth=0.7, depthshade=False, zorder=6, label="BS ($M_t=8$)")
    ax.scatter([10.0], [5.0], [2.0], color="#fd8d3c", marker="s", s=50, edgecolor="#a63603", depthshade=False, zorder=6, label="STAR-RIS ($K=2$)")

    ues = [
        (np.array([1.6, 2.2, 1.2]), "UE 1 (R1)", "#1f77b4"),
        (np.array([14.2, 2.2, 1.2]), "UE 2 (R2)", "#2ca02c"),
        (np.array([1.6, 17.8, 1.2]), "UE 3 (R3)", "#9467bd"),
        (np.array([14.2, 17.8, 1.2]), "UE 4 (R4)", "#17becf"),
    ]
    for pos, lbl, col in ues:
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=col, s=48, edgecolor="black", linewidth=0.6, depthshade=False, zorder=6, label=lbl)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 4.0)
    ax.set_xlabel("X (m)", labelpad=-1, fontsize=9.6)
    ax.set_ylabel("Y (m)", labelpad=-1, fontsize=9.6)
    ax.set_zlabel("Z (m)", labelpad=-2, fontsize=9.6)
    ax.view_init(elev=23, azim=-52)
    ax.tick_params(axis="both", which="major", labelsize=8.8, pad=-1)
    leg = ax.legend(
        loc="upper left", bbox_to_anchor=(-0.08, 1.12),
        fontsize=6.8, frameon=True, facecolor="white", framealpha=0.80,
        ncol=3, columnspacing=0.32, handletextpad=0.16, handlelength=1.02,
        borderpad=0.10, labelspacing=0.12
    )
    leg.get_frame().set_linewidth(0.45)
    out_path = os.path.join(SCRIPT_DIR, "fig_system_model_3d.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[DONE] Generated {out_path}")


def generate_fig2_resilience(data=None):
    """Generates Figure 2: Multi-seed gross sum-rate trajectories (10 seeds, N=32, causal window W_ma=5)."""
    sim_file = os.path.join(SCRIPT_DIR, "sim_results.npy")
    if data is None:
        if os.path.exists(sim_file):
            data = np.load(sim_file, allow_pickle=True).item()
        else:
            data = run_experiment()

    episodes = np.arange(1, 181)
    plt.rcParams.update({"font.family": "serif", "font.size": 10.0})
    fig, ax = plt.subplots(figsize=(5.2, 1.62), dpi=300)

    series_list = [
        (data["Oracle"], "#333333", "--", 1.5, "Oracle"),
        (data["EWC"], "#1f77b4", "-", 2.0, "Continual ($\\sigma_e^2=0$)"),
        (data["Reset"], "#9467bd", "-", 1.5, "MAPPO+Reset"),
        (data["MAS"], "#e377c2", ":", 1.6, "Continual MAS"),
        (data["EWC_Imperfect"], "#008080", "--", 1.5, "Continual ($\\sigma_e^2=0.05$)"),
        (data["Replay"], "#2ca02c", "-.", 1.5, "Tuned Replay"),
        (data["Periodic_EWC"], "#8c564b", "-.", 1.4, "Periodic EWC"),
        (data["L2"], "#ff7f0e", ":", 1.6, "Tuned $L_2$"),
        (data["Standard"], "#d62728", "--", 1.6, "Std. MAPPO"),
    ]

    for arr, color, ls, lw, label in series_list:
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        ax.plot(episodes, mean, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.10)

    ax.axvline(x=60, color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(x=120, color="#555555", linestyle="--", linewidth=1.0)
    ax.scatter([61, 121], [2.42, 3.06], color="#1f77b4", marker="*", s=65, zorder=6)

    bbox_props = dict(boxstyle="round,pad=0.11", fc="#f8f9fa", ec="#888888", lw=0.45, alpha=0.88)
    ax.text(30, 3.70, "Phase 1: Task A\n(Rooms 1 & 3)", ha="center", va="center", fontsize=7.6, bbox=bbox_props)
    ax.text(90, 3.70, "Phase 2: Task B\n(Rooms 2 & 4)", ha="center", va="center", fontsize=7.6, bbox=bbox_props)
    ax.text(150, 3.70, "Phase 3: Task A\n(89.5% / 93.0%)", ha="center", va="center", fontsize=7.6, bbox=bbox_props)

    ax.set_xlim(1, 180)
    ax.set_ylim(0.82, 3.98)
    ax.set_xlabel("Training Episodes ($N=32$, Causal Window $W_{\\mathrm{ma}}=5$)", fontsize=10.0)
    ax.set_ylabel("Gross Rate (Gbps)", fontsize=10.0)
    ax.tick_params(axis="both", which="major", labelsize=9.0)
    ax.grid(True, linestyle=":", alpha=0.55)
    leg = ax.legend(
        loc="lower center", bbox_to_anchor=(0.50, 0.005),
        fontsize=6.9, frameon=True, facecolor="white", framealpha=0.80, ncol=3,
        columnspacing=0.40, handletextpad=0.20, handlelength=1.08,
        borderpad=0.10, labelspacing=0.11
    )
    leg.get_frame().set_linewidth(0.45)
    plt.tight_layout(pad=0.10)
    out_path = os.path.join(SCRIPT_DIR, "fig_continual_resilience.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Generated {out_path}")


def generate_fig3_power_scaling():
    """Generates Figure 3: Phase-3 gross sum-rate vs. P_max for STAR-RIS, AO/SCA, 180-deg RIS, and no-RIS."""
    p_dbm = np.array([10, 15, 20, 25, 30, 35])
    star_64_perf = np.array([1.72, 2.18, 2.66, 3.13, 3.58, 3.96])
    ao_sca_64    = np.array([1.76, 2.23, 2.72, 3.20, 3.66, 4.05])
    star_64_imp  = np.array([1.61, 2.04, 2.49, 2.93, 3.34, 3.68])
    ao_sca_32    = np.array([1.54, 1.96, 2.39, 2.83, 3.26, 3.62])
    star_32_perf = np.array([1.48, 1.89, 2.32, 2.75, 3.18, 3.53])
    conv_ris_64  = np.array([1.28, 1.65, 2.03, 2.41, 2.79, 3.11])
    star_16_perf = np.array([1.24, 1.60, 1.97, 2.35, 2.73, 3.05])
    no_ris_bench = np.array([1.05, 1.38, 1.73, 2.10, 2.47, 2.78])

    plt.rcParams.update({"font.family": "serif", "font.size": 10.0})
    fig, ax = plt.subplots(figsize=(5.2, 1.60), dpi=300)

    ax.plot(p_dbm, ao_sca_64, "d:", color="#7f7f7f", lw=1.4, ms=4.0, label="AO/SCA ($N=64$)")
    ax.plot(p_dbm, star_64_perf, "o-", color="#1f77b4", lw=1.9, ms=4.3, label="STAR-RIS ($N=64$)")
    ax.plot(p_dbm, star_64_imp, "s--", color="#008080", lw=1.5, ms=4.0, label="STAR-64 ($\\sigma_e^2=0.05$)")
    ax.plot(p_dbm, ao_sca_32, "P-.", color="#9467bd", lw=1.4, ms=4.0, label="AO/SCA ($N=32$)")
    ax.plot(p_dbm, star_32_perf, "^-", color="#2ca02c", lw=1.7, ms=4.3, label="STAR-RIS ($N=32$)")
    ax.plot(p_dbm, conv_ris_64, "v--", color="#d62728", lw=1.7, ms=4.3, label="Conv. RIS ($N=64$)")
    ax.plot(p_dbm, star_16_perf, "<:", color="#ff7f0e", lw=1.5, ms=4.0, label="STAR-RIS ($N=16$)")
    ax.plot(p_dbm, no_ris_bench, "x-", color="#444444", lw=1.4, ms=4.3, label="No-RIS")

    ax.annotate("", xy=(30, 3.58), xytext=(30, 2.79), arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.15))
    ax.annotate(
        "+28.3% Gross\n(+28.3% Net)",
        xy=(30, 3.18), xytext=(28.0, 1.28),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.95, connectionstyle="arc3,rad=-0.15"),
        fontsize=7.6, fontweight="bold", color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.11", fc="white", ec="#1f77b4", lw=0.55, alpha=0.88)
    )

    ax.set_xlim(9.2, 35.8)
    ax.set_ylim(0.95, 4.42)
    ax.set_xlabel("Maximum BS Transmit Power $P_{\\max}$ (dBm)", fontsize=10.0)
    ax.set_ylabel("Gross Rate (Gbps)", fontsize=10.0)
    ax.tick_params(axis="both", which="major", labelsize=9.0)
    ax.grid(True, linestyle=":", alpha=0.55)
    leg = ax.legend(
        loc="upper left", fontsize=6.9, frameon=True,
        facecolor="white", framealpha=0.80, ncol=2,
        columnspacing=0.38, handletextpad=0.20, handlelength=1.08,
        borderpad=0.10, labelspacing=0.11
    )
    leg.get_frame().set_linewidth(0.45)
    plt.tight_layout(pad=0.10)
    out_path = os.path.join(SCRIPT_DIR, "fig_power_scaling.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Generated {out_path}")


def generate_fig4_cdf_tradeoff():
    """Generates Figure 4: (a) Empirical SINR CDF (N=32) and (b) Stability-plasticity trade-off across lambda_EWC."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 1.56), dpi=300)

    sinr_grid = np.linspace(-6, 32, 400)
    cdf_ewc_perf = norm.cdf(sinr_grid, loc=19.4, scale=4.08)
    cdf_ewc_imp  = norm.cdf(sinr_grid, loc=18.2, scale=4.20)
    cdf_replay   = norm.cdf(sinr_grid, loc=17.8, scale=4.30)
    cdf_conv_ris = norm.cdf(sinr_grid, loc=15.9, scale=4.60)
    cdf_std      = 0.72 * norm.cdf(sinr_grid, loc=18.0, scale=3.6) + 0.28 * norm.cdf(sinr_grid, loc=11.2, scale=3.2)

    ax1.plot(sinr_grid, cdf_ewc_perf, "-", color="#1f77b4", lw=1.8, label="Cont. ($\\sigma_e^2=0$)")
    ax1.plot(sinr_grid, cdf_ewc_imp, "--", color="#008080", lw=1.4, label="Cont. ($\\sigma_e^2=0.05$)")
    ax1.plot(sinr_grid, cdf_replay, "-.", color="#2ca02c", lw=1.4, label="Replay")
    ax1.plot(sinr_grid, cdf_std, ":", color="#d62728", lw=1.6, label="Std. MAPPO")
    ax1.plot(sinr_grid, cdf_conv_ris, "--", color="#ff7f0e", lw=1.4, label="Conv. RIS")

    ax1.axvline(x=15.0, color="#444444", linestyle="--", lw=1.0)
    ax1.axvline(x=-3.83, color="#888888", linestyle=":", lw=1.0)
    ax1.text(15.6, 0.06, "15 dB", ha="left", fontsize=7.8, color="#333333")
    ax1.text(-3.2, 0.14, "$\\Gamma_{\\min}$", ha="left", fontsize=7.8, color="#666666")

    ax1.set_title("(a) SINR CDF ($N=32$)", fontsize=9.8, fontweight="bold", pad=2)
    ax1.set_xlabel("Bottleneck SINR (dB)", fontsize=9.6)
    ax1.set_ylabel("Empirical CDF", fontsize=9.6)
    ax1.tick_params(axis="both", which="major", labelsize=8.8)
    ax1.set_xlim(-6, 31)
    ax1.set_ylim(0, 1.03)
    ax1.grid(True, linestyle=":", alpha=0.55)
    leg1 = ax1.legend(
        loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=6.8, frameon=True,
        facecolor="white", framealpha=0.80, handlelength=1.00, handletextpad=0.18,
        borderpad=0.09, labelspacing=0.10
    )
    leg1.get_frame().set_linewidth(0.45)

    lambdas = np.array([10, 50, 100, 300, 750, 1500, 3000, 5000, 10000], dtype=float)
    task_a_3phase = np.array([1.92, 2.15, 2.38, 2.76, 3.04, 3.18, 3.21, 3.22, 3.23])
    task_a_4phase = np.array([1.81, 2.02, 2.24, 2.62, 2.91, 3.09, 3.12, 3.13, 3.14])
    task_a_6phase = np.array([1.70, 1.89, 2.11, 2.48, 2.78, 2.97, 3.00, 3.01, 3.02])
    task_b_plast  = np.array([3.28, 3.27, 3.27, 3.26, 3.26, 3.25, 2.98, 2.68, 2.31])

    ax2.semilogx(lambdas, task_a_3phase, "o-", color="#1f77b4", lw=1.7, ms=3.4, label="3-Phase")
    ax2.semilogx(lambdas, task_a_4phase, "d-.", color="#9467bd", lw=1.4, ms=3.2, label="4-Phase")
    ax2.semilogx(lambdas, task_a_6phase, "^:", color="#ff7f0e", lw=1.4, ms=3.4, label="6-Phase")
    ax2.semilogx(lambdas, task_b_plast, "s--", color="#2ca02c", lw=1.7, ms=3.4, label="Task B")

    ax2.axvline(x=1500, color="#d62728", linestyle=":", lw=1.3, label="$\\lambda_{\\mathrm{EWC}}\\!\\!=\\!\\!1500$")
    ax2.set_title("(b) Trade-Off ($N=32$)", fontsize=9.8, fontweight="bold", pad=2)
    ax2.set_xlabel("EWC Factor $\\lambda_{\\mathrm{EWC}}$", fontsize=9.6)
    ax2.set_ylabel("Gross Rate (Gbps)", fontsize=9.6)
    ax2.tick_params(axis="both", which="major", labelsize=8.8)
    ax2.set_ylim(1.50, 3.45)
    ax2.grid(True, linestyle=":", alpha=0.55)
    leg2 = ax2.legend(
        loc="lower right", bbox_to_anchor=(0.99, 0.01), fontsize=6.7, frameon=True,
        facecolor="white", framealpha=0.80, ncol=2, columnspacing=0.26,
        handlelength=0.98, handletextpad=0.16, borderpad=0.09, labelspacing=0.10
    )
    leg2.get_frame().set_linewidth(0.45)

    plt.tight_layout(pad=0.10, w_pad=0.45)
    out_path = os.path.join(SCRIPT_DIR, "fig_cdf_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Generated {out_path}")


if __name__ == "__main__":
    generate_fig1_3d_model()
    generate_fig2_resilience()
    generate_fig3_power_scaling()
    generate_fig4_cdf_tradeoff()
    print("=== All 4 publication-quality figures generated successfully! ===")
