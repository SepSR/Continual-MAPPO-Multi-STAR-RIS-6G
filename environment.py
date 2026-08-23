# 3D propagation environment, InH path loss, and NOMA SIC decoding
import numpy as np
import config

class IndoorEnvironment:
    def __init__(self):
        self.bs_pos = np.array([10.0, 10.0, 3.5])
        self.ris_positions = np.array([
            [10.0, 5.0, 2.0],   # STAR-RIS panel 1
            [10.0, 15.0, 2.0]   # STAR-RIS panel 2
        ])
        self.K = config.NUM_USERS
        self.N = config.NUM_ELEMENTS
        self.Mt = config.NUM_BS_ANTENNAS
        self.reset_positions(phase="Task_A")

    def reset_positions(self, phase="Task_A"):
        if phase == "Task_A":
            self.user_positions = np.array([
                [4.0, 4.0, 1.0], [6.0, 3.0, 1.0], [4.0, 16.0, 1.0], [6.0, 17.0, 1.0]
            ])
        else:  # Task_B: Displaced across room boundaries
            self.user_positions = np.array([
                [16.0, 4.0, 1.0], [14.0, 6.0, 1.0], [16.0, 16.0, 1.0], [14.0, 15.0, 1.0]
            ])
        self.velocities = np.random.uniform(-config.PEDESTRIAN_VELOCITY, config.PEDESTRIAN_VELOCITY, (self.K, 3))
        self.velocities[:, 2] = 0.0

    def get_pathloss(self, dist, is_nlos=False):
        # 3GPP InH-Office TR 38.901
        pl_db = 32.4 + 17.3 * np.log10(max(dist, 1.0)) + 20 * np.log10(config.FC / 1e9)
        if is_nlos:
            pl_db += 14.6 * np.log10(max(dist, 1.0))
        return 10 ** (-pl_db / 10)

    def compute_cascaded_channels(self):
        h_eff = []
        for u in range(self.K):
            u_pos = self.user_positions[u]
            cross_x = (self.bs_pos[0] < 10 and u_pos[0] > 10) or (self.bs_pos[0] > 10 and u_pos[0] < 10)
            cross_y = (self.bs_pos[1] < 10 and u_pos[1] > 10) or (self.bs_pos[1] > 10 and u_pos[1] < 10)
            nw = int(cross_x) + int(cross_y)
            wall_attenuation = 10 ** (-nw * config.WALL_LOSS_DB / 20)

            # Direct channel link
            dist_d = np.linalg.norm(self.bs_pos - u_pos)
            pl_d = self.get_pathloss(dist_d, is_nlos=(nw > 0))
            h_d = wall_attenuation * np.sqrt(pl_d) * (np.random.randn(self.Mt, 1) + 1j * np.random.randn(self.Mt, 1)) / np.sqrt(2)

            # Cascaded STAR-RIS link
            dist_br = np.linalg.norm(self.bs_pos - self.ris_positions[0])
            dist_ru = np.linalg.norm(self.ris_positions[0] - u_pos)
            pl_cascaded = self.get_pathloss(dist_br) * self.get_pathloss(dist_ru)
            h_cascaded = np.sqrt(pl_cascaded) * (np.random.randn(self.Mt, 1) + 1j * np.random.randn(self.Mt, 1)) / np.sqrt(2)

            h_eff.append(h_d + h_cascaded)
        return h_eff

    def calculate_noma_sum_rate(self, h_eff, P_tx):
        # Norm-ordered successive interference cancellation
        gains = [np.linalg.norm(h)**2 for h in h_eff]
        sort_idx = np.argsort(gains)[::-1]
        
        rates = []
        for i, idx in enumerate(sort_idx):
            sig_power = (P_tx / self.K) * gains[idx]
            intra_interference = sum([(P_tx / self.K) * gains[sort_idx[j]] for j in range(i + 1, self.K)])
            sinr = sig_power / (intra_interference + config.NOISE_POWER)
            rate = (config.BANDWIDTH / 1e9) * np.log2(1 + max(sinr, 1e-9))
            rates.append(rate)
        return sum(rates)