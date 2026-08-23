# System configuration and simulation parameters
import numpy as np

# RF Carrier & Bandwidth
FC = 28e9                # 28 GHz mmWave carrier
BANDWIDTH = 100e6        # 100 MHz bandwidth
NOISE_DENSITY = -174     # dBm/Hz thermal noise
NOISE_FIGURE = 7.0       # dB receiver noise figure

# Power & Channel Attenuation
P_MAX_DBM = 30.0         # 30 dBm (1 W) transmit power limit
P_MAX = 10 ** ((P_MAX_DBM - 30) / 10)
NOISE_POWER = 10 ** ((NOISE_DENSITY + 10 * np.log10(BANDWIDTH) + NOISE_FIGURE - 30) / 10)
WALL_LOSS_DB = 10.0      # Penetration loss per structural wall

# Topology & Hardware Dimensions
ROOM_DIM = 20.0          # 20m x 20m indoor area
NUM_ROOMS = 4            # M = 4 partition rooms
NUM_USERS = 4            # U = 4 active mobile users
NUM_STAR_RIS = 2         # K = 2 STAR-RIS surfaces
NUM_ELEMENTS = 32        # N = 32 elements per surface
NUM_BS_ANTENNAS = 8      # Mt = 8 BS antennas

# Mobility & Channel Aging (Gauss-Markov)
PEDESTRIAN_VELOCITY = 1.2 # v = 1.2 m/s
RHO_AGING = 0.98          # Temporal correlation
SIGMA_AGE = 0.01          # Aging noise variance

# Continual MARL & EWC Hyperparameters
EWC_LAMBDA = 1500.0       # Actor regularization weight
EWC_LAMBDA_PHI = 500.0    # Critic regularization weight
GAMMA_EWC = 0.95          # Memory discount factor
LR_ACTOR_CRITIC = 3e-3    # Adam learning rate
DRIFT_THRESHOLD = 0.35    # 2-Wasserstein Bures threshold
WINDOW_SIZE = 20          # Sliding window length Delta
NUM_SEEDS = 5             # Monte Carlo random seeds