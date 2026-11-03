#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Configuration and Simulation Parameters (Table I)
Manuscript ID: WCL2026-3112 (Revised)
Title: Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks
Authors: Sepehr Rahmani, Sana Rahmani, and Sina Rahmani
Journal: IEEE Wireless Communications Letters
"""

import numpy as np
from scipy.special import j0


class SystemConfig:
    """Centralized simulation parameters strictly aligned with Table I of the revised manuscript."""

    # --- RF Carrier, Bandwidth & Linear Thermal Noise (Section II-B, Reviewer 3 Comment 6) ---
    FC_GHZ = 28.0                                  # Carrier frequency f_c = 28 GHz
    FC = FC_GHZ * 1e9                              # Carrier frequency in Hz (28e9 Hz)
    C_LIGHT = 3.0e8                                # Speed of light c = 3e8 m/s
    LAMBDA_M = C_LIGHT / FC                        # Wavelength lambda_c = 10.71 mm (lambda_c/2 = 5.35 mm)
    BW_HZ = 100e6                                  # System bandwidth B = 100 MHz
    BANDWIDTH = BW_HZ                              # Alias for backward compatibility
    N0_DBM_HZ = -174.0                             # Thermal noise power spectral density N_{0,dBm} = -174 dBm/Hz
    NOISE_DENSITY = N0_DBM_HZ                      # Alias
    NF_DB = 7.0                                    # Receiver noise figure NF_dB = 7 dB
    NOISE_FIGURE = NF_DB                           # Alias

    # Explicit linear scale conversion before multiplication:
    # sigma^2 = 10^{(N_{0,dBm} - 30)/10} * B * 10^{NF_dB / 10} ≈ 2.00e-12 W (2.0 pW)
    N0_LINEAR_W_HZ = 10.0 ** ((N0_DBM_HZ - 30.0) / 10.0)
    NF_LINEAR = 10.0 ** (NF_DB / 10.0)
    NOISE_VAR_W = N0_LINEAR_W_HZ * BW_HZ * NF_LINEAR
    NOISE_POWER = NOISE_VAR_W                      # Alias

    # --- Transmit Power, Path Loss & Wall Attenuation (Section II-A, Reviewer 2 Comments 7 & 8) ---
    P_MAX_DBM = 30.0                               # Max BS transmit power P_max = 30 dBm (1 W)
    P_MAX_W = 10.0 ** ((P_MAX_DBM - 30.0) / 10.0)
    P_MAX = P_MAX_W                                # Alias
    WALL_LOSS_DB = 10.0                            # COST-231 penetration loss L_wall = 10 dB per wall
    D_MIN_M = 1.0                                  # Far-field reference / exclusion bound d_min = 1.0 m

    # --- 3D Indoor Topology & Hardware Dimensions (Section II, Fig. 1, Table I) ---
    ROOM_DIM = 20.0                                # 20 m x 20 m floor plan divided into 4 rooms of 10 m x 10 m
    NUM_ROOMS = 4                                  # M = 4 rooms
    NUM_RIS = 2                                    # K = 2 wall-mounted STAR-RIS panels
    NUM_STAR_RIS = NUM_RIS                         # Alias
    NUM_USERS = 4                                  # U = 4 single-antenna mobile UEs
    NUM_BS_ANT = 8                                 # M_t = 8 BS antennas
    NUM_BS_ANTENNAS = NUM_BS_ANT                   # Alias
    DEFAULT_N = 32                                 # Default N = 32 (4x8 UPA); evaluated for N in {16, 32, 64}
    NUM_ELEMENTS = DEFAULT_N                       # Alias
    N_OPTIONS = (16, 32, 64)                       # Supported UPA sizes: 16 (4x4), 32 (4x8), 64 (8x8)
    UE_HEIGHT_M = 1.2                              # z_u = 1.2 m pedestrian handset height

    # --- QoS & Bottleneck-Consistent SIC Parameters (Section II-B & II-C, Equations 6-9) ---
    R_MIN_BPS_HZ = 0.5                             # Minimum spectral efficiency r_min = 0.5 bps/Hz
    GAMMA_MIN = (2.0 ** R_MIN_BPS_HZ) - 1.0        # QoS/SIC floor Gamma_min = 2^{r_min} - 1 ≈ 0.4142 (-3.83 dB)
    GAMMA_MIN_DB = 10.0 * np.log10(GAMMA_MIN)      # -3.828 dB
    COVERAGE_TARGET_DB = 15.0                      # 15 dB (5 bps/Hz, 500 Mbps) 6G coverage benchmark in Fig. 4(a)
    PENALTY_KAPPA = 10.0                           # QoS violation penalty factor kappa = 10 in Eq. (9)

    # --- Pedestrian Mobility, Time Compression & Channel Aging (Section II-A, Equation 2) ---
    VELOCITY_M_S = 1.2                             # Pedestrian walking speed v = 1.2 m/s
    PEDESTRIAN_VELOCITY = VELOCITY_M_S             # Alias
    SLOT_DURATION_S = 0.4e-3                       # Slot duration Delta_t = 0.4 ms
    SYM_DURATION_US = 8.92                         # OFDM symbol duration T_sym ≈ 8.92 us at 120 kHz SCS
    SYMS_PER_SLOT = 44                             # 44 OFDM symbols per 0.4 ms slot
    EPISODE_SLOTS = 50                             # T_ep = 50 slots per episode (20 ms physical duration)
    WAYPOINT_STEP_M = 0.14                         # Per-episode waypoint step Delta_d_ep = 0.14 m
    TIME_COMPRESSION = WAYPOINT_STEP_M / (VELOCITY_M_S * EPISODE_SLOTS * SLOT_DURATION_S)  # 5.833x (~5.8x)

    DOPPLER_HZ = VELOCITY_M_S * FC / C_LIGHT       # Maximum Doppler shift f_D = v * f_c / c = 112.0 Hz
    RHO_JAKES_EXACT = float(j0(2.0 * np.pi * DOPPLER_HZ * SLOT_DURATION_S))  # J_0(2*pi*f_D*Delta_t) ≈ 0.9803
    RHO_GM = 0.98                                  # 1st-order Gauss-Markov correlation rho = 0.98 (1 - rho^2 = 0.04)
    RHO_AGING = RHO_GM                             # Alias
    JAKES_HALF_CORR_MS = (1.520 / (2.0 * np.pi * DOPPLER_HZ)) * 1e3          # ≈ 2.16 ms (2.2 ms)
    AR1_HALF_LIFE_MS = (np.log(0.5) / np.log(RHO_GM)) * SLOT_DURATION_S * 1e3  # ≈ 13.72 ms (13.7 ms)

    # --- TDD Subarray Pilot Accounting & Imperfect CSI (Section II-A, Equation 5) ---
    NUM_SUBGROUPS = 8                              # N_g = 8 subarrays per panel
    PILOT_SYMS_RIS = NUM_RIS * NUM_SUBGROUPS + 1   # tau_{p,RIS} = K * N_g + 1 = 17 pilot symbols (~152 us)
    PILOT_SYMS_CONV_INDEP = 2 * NUM_RIS * NUM_SUBGROUPS + 1  # 2*K*N_g + 1 = 33 symbols (independent half-panels)
    PILOT_SYMS_NORIS = 1                           # tau_{p,no-RIS} = 1 symbol without RIS
    PILOT_BLOCK_SLOTS = 5                          # Pilots repeated every T_blk = 5 slots (2.0 ms)
    CSI_NMSE_GRID = (0.0, 0.01, 0.05, 0.10, 0.20)  # Evaluated CSI error variances sigma_e^2

    # --- Policy-Invariant Bures Drift Detector & Two-Buffer FIFO/Ring Architecture (Algorithm 1, Eq. 10) ---
    WINDOW_W = 20                                  # Sliding covariance window W = 20 slots (8.0 ms)
    WINDOW_SIZE = WINDOW_W                         # Alias
    WARMUP_SLOTS = 2 * WINDOW_W                    # Initial detector warm-up 2W = 40 slots
    LOCKOUT_SLOTS = 2 * WINDOW_W                   # Post-trigger lockout timer T_lock = 2W = 40 slots
    PERSISTENCE_SLOTS = 3                          # 3-slot persistence check tau in {t-2, t-1, t}
    RESERVOIR_M = 500                              # Pre-drift ring buffer D_prev capacity M_F = 500 slots (10 episodes)
    TIKHONOV_EPS = 1e-6                            # Trace-normalized covariance regularization eps_reg = 1e-6
    DRIFT_THRESH = 0.35                            # 2-Wasserstein Bures threshold eps_th = 0.35 (99.9th intra-room pct)
    DRIFT_THRESHOLD = DRIFT_THRESH                 # Alias

    # --- Continual MAPPO & Actor-Critic EWC Hyperparameters (Section III-B, Table I, Equations 11-12) ---
    MLP_HIDDEN = 64                                # H = 64 hidden units per 2-layer MLP
    EWC_LAMBDA = 1500.0                            # Actor EWC regularization weight lambda_EWC = 1500
    EWC_LAMBDA_PSI = 500.0                         # Critic EWC regularization weight lambda_psi = 500
    EWC_LAMBDA_PHI = EWC_LAMBDA_PSI                # Alias
    GAMMA_EWC = 0.95                               # Online EWC importance decay factor gamma_EWC = 0.95
    VF_COEF_C1 = 0.5                               # Critic value loss weight c_1 = 0.5 in Eq. (12)
    PPO_CLIP = 0.2                                 # PPO clipping ratio eps_clip = 0.2
    GAE_LAMBDA = 0.95                              # GAE parameter lambda_GAE = 0.95
    DISCOUNT_GAMMA = 0.99                          # Discount factor gamma = 0.99
    PPO_LR = 3e-4                                  # Adam learning rate alpha = 3e-4
    LR_ACTOR_CRITIC = PPO_LR                       # Alias
    PPO_EPOCHS = 4                                 # 4 PPO epochs per 50-slot episode
    MINIBATCH_SIZES = (13, 13, 12, 12)             # 4 mini-batches of 13/13/12/12 slots per 50-slot episode
    ENTROPY_COEF = 0.01                            # Entropy regularization weight 0.01

    # --- Evaluation Protocol (Section IV, Table I) ---
    NUM_SEEDS = 10                                 # 10 random test seeds
    VAL_SEEDS = 3                                  # 3 validation seeds for hyperparameter selection
    MA_WINDOW = 5                                  # Causal moving-average window W_ma = 5 episodes


# Module-level exports for convenient import (`import config; config.FC`, etc.)
FC_GHZ = SystemConfig.FC_GHZ
FC = SystemConfig.FC
C_LIGHT = SystemConfig.C_LIGHT
LAMBDA_M = SystemConfig.LAMBDA_M
BW_HZ = SystemConfig.BW_HZ
BANDWIDTH = SystemConfig.BANDWIDTH
N0_DBM_HZ = SystemConfig.N0_DBM_HZ
NOISE_DENSITY = SystemConfig.NOISE_DENSITY
NF_DB = SystemConfig.NF_DB
NOISE_FIGURE = SystemConfig.NOISE_FIGURE
NOISE_VAR_W = SystemConfig.NOISE_VAR_W
NOISE_POWER = SystemConfig.NOISE_POWER
P_MAX_DBM = SystemConfig.P_MAX_DBM
P_MAX_W = SystemConfig.P_MAX_W
P_MAX = SystemConfig.P_MAX
WALL_LOSS_DB = SystemConfig.WALL_LOSS_DB
D_MIN_M = SystemConfig.D_MIN_M
ROOM_DIM = SystemConfig.ROOM_DIM
NUM_ROOMS = SystemConfig.NUM_ROOMS
NUM_RIS = SystemConfig.NUM_RIS
NUM_STAR_RIS = SystemConfig.NUM_STAR_RIS
NUM_USERS = SystemConfig.NUM_USERS
NUM_BS_ANT = SystemConfig.NUM_BS_ANT
NUM_BS_ANTENNAS = SystemConfig.NUM_BS_ANTENNAS
DEFAULT_N = SystemConfig.DEFAULT_N
NUM_ELEMENTS = SystemConfig.NUM_ELEMENTS
R_MIN_BPS_HZ = SystemConfig.R_MIN_BPS_HZ
GAMMA_MIN = SystemConfig.GAMMA_MIN
PENALTY_KAPPA = SystemConfig.PENALTY_KAPPA
VELOCITY_M_S = SystemConfig.VELOCITY_M_S
PEDESTRIAN_VELOCITY = SystemConfig.PEDESTRIAN_VELOCITY
SLOT_DURATION_S = SystemConfig.SLOT_DURATION_S
EPISODE_SLOTS = SystemConfig.EPISODE_SLOTS
WAYPOINT_STEP_M = SystemConfig.WAYPOINT_STEP_M
TIME_COMPRESSION = SystemConfig.TIME_COMPRESSION
DOPPLER_HZ = SystemConfig.DOPPLER_HZ
RHO_GM = SystemConfig.RHO_GM
RHO_AGING = SystemConfig.RHO_AGING
WINDOW_W = SystemConfig.WINDOW_W
WINDOW_SIZE = SystemConfig.WINDOW_SIZE
LOCKOUT_SLOTS = SystemConfig.LOCKOUT_SLOTS
PERSISTENCE_SLOTS = SystemConfig.PERSISTENCE_SLOTS
RESERVOIR_M = SystemConfig.RESERVOIR_M
TIKHONOV_EPS = SystemConfig.TIKHONOV_EPS
DRIFT_THRESH = SystemConfig.DRIFT_THRESH
DRIFT_THRESHOLD = SystemConfig.DRIFT_THRESHOLD
MLP_HIDDEN = SystemConfig.MLP_HIDDEN
EWC_LAMBDA = SystemConfig.EWC_LAMBDA
EWC_LAMBDA_PSI = SystemConfig.EWC_LAMBDA_PSI
EWC_LAMBDA_PHI = SystemConfig.EWC_LAMBDA_PHI
GAMMA_EWC = SystemConfig.GAMMA_EWC
PPO_CLIP = SystemConfig.PPO_CLIP
GAE_LAMBDA = SystemConfig.GAE_LAMBDA
DISCOUNT_GAMMA = SystemConfig.DISCOUNT_GAMMA
PPO_LR = SystemConfig.PPO_LR
LR_ACTOR_CRITIC = SystemConfig.LR_ACTOR_CRITIC
PPO_EPOCHS = SystemConfig.PPO_EPOCHS
MINIBATCH_SIZES = SystemConfig.MINIBATCH_SIZES
NUM_SEEDS = SystemConfig.NUM_SEEDS
MA_WINDOW = SystemConfig.MA_WINDOW
