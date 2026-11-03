#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Indoor Multi-STAR-RIS NOMA Environment, Cascaded CSI, and Drift Detection
Manuscript ID: WCL2026-3112 (Revised)
Title: Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks
Authors: Sepehr Rahmani, Sana Rahmani, and Sina Rahmani
Journal: IEEE Wireless Communications Letters
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.special import j0
from config import SystemConfig


class IndoorMultiStarRisEnv:
    """
    Models the 20m x 20m 4-room indoor 6G network with:
      - Modified-slope 3GPP TR 38.901 InH-Office path loss (Eq. 1)
      - Explicit COST-231 wall attenuation L_wall = 10 dB (Section II-A)
      - 1st-order Gauss-Markov and exact Bessel small-scale channel aging (Eq. 2)
      - Energy-Splitting (ES) STAR-RIS effective channels and feasible action projections (Eqs. 3, 4, 8)
      - Imperfect cascaded CSI with subarray pilot overhead and interpolation residual (Eq. 5)
      - Downlink MISO-NOMA with bottleneck-consistent SIC rates and Dec-POMDP observations (Eqs. 6, 7, 9)
    """

    def __init__(self, num_elements=SystemConfig.DEFAULT_N, csi_nmse=0.0, star_mode=True, seed=42):
        self.N = int(num_elements)
        self.csi_nmse = float(csi_nmse)
        self.star_mode = bool(star_mode)
        self.rng = np.random.RandomState(seed)

        # BS referenced to Room-1 corner [10^-, 10^-, 3.5]^T m for unambiguous wall intersection counting
        self.bs_pos = np.array([10.0 - 1e-3, 10.0 - 1e-3, 3.5])
        # K = 2 wall-mounted STAR-RIS UPAs on the x = 10 m wall
        self.ris_pos = np.array([
            [10.0, 5.0, 2.0],
            [10.0, 15.0, 2.0]
        ])
        self.ris_normals = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])

        # Doorway centers on x = 10 m at (10, 2.0) and (10, 18.0) m (width 1.6 m)
        self.doorways_x10 = np.array([[10.0, 2.0], [10.0, 18.0]])

        # Quasi-static rank-1 LoS BS-to-RIS-k channel matrices G_k in C^{N x M_t}
        self.G_list = [self._build_los_bs_ris_matrix(k) for k in range(SystemConfig.NUM_RIS)]

        # Initialize UE positions, small-scale fading state, and previous-slot variables
        self.ue_pos = None
        self.headings = None
        self.h_d_tilde = None
        self.h_r_tilde = None
        self.prev_w = None
        self.prev_h_eff_hat = None
        self.reset_regime("Task_A")

    def _build_los_bs_ris_matrix(self, k_idx):
        """Constructs quasi-static rank-1 LoS BS-to-RIS-k matrix G_k = sqrt(zeta_{G,k}) a_RIS a_BS^H."""
        delta = self.ris_pos[k_idx] - self.bs_pos
        dist = max(float(np.linalg.norm(delta)), SystemConfig.D_MIN_M)
        pl_db = self.path_loss_db(dist, is_nlos=False)
        zeta_gk = 10.0 ** (-pl_db / 10.0)
        az = np.arctan2(delta[1], delta[0])
        el = np.arcsin(np.clip(delta[2] / dist, -1.0, 1.0))
        a_bs = np.exp(-1j * np.pi * np.arange(SystemConfig.NUM_BS_ANT) * np.sin(az)).reshape(-1, 1)
        n_y = 8 if self.N >= 32 else 4
        n_z = max(1, self.N // n_y)
        yy, zz = np.meshgrid(np.arange(n_y), np.arange(n_z))
        phase_ris = np.pi * (yy.ravel() * np.sin(az) * np.cos(el) + zz.ravel() * np.sin(el))
        a_ris = np.exp(-1j * phase_ris[:self.N]).reshape(-1, 1)
        return np.sqrt(zeta_gk) * (a_ris @ a_bs.conj().T)

    def reset_regime(self, regime="Task_A"):
        """Initializes UE positions and small-scale Rayleigh fading vectors for Task_A, Task_B, or Task_C."""
        if regime == "Task_A":
            # Phase 1: Rooms 1 & 3 (reflection half-space x < 10 m), lanes at y = 2.0, 18.0 m with +/-0.2 m offsets
            self.ue_pos = np.array([
                [1.6, 1.8, 1.2],
                [1.6, 2.2, 1.2],
                [1.6, 17.8, 1.2],
                [1.6, 18.2, 1.2]
            ], dtype=float)
            self.headings = np.zeros(SystemConfig.NUM_USERS)
        elif regime == "Task_B":
            # Phase 2: Rooms 2 & 4 (transmission half-space x >= 10 m)
            self.ue_pos = np.array([
                [14.2, 1.8, 1.2],
                [14.2, 2.2, 1.2],
                [14.2, 17.8, 1.2],
                [14.2, 18.2, 1.2]
            ], dtype=float)
            self.headings = np.full(SystemConfig.NUM_USERS, np.pi)
        else:
            # Asymmetric 3-room regime Task_C (2 UEs in Room 1, 1 UE in Room 2, 1 UE in Room 4)
            self.ue_pos = np.array([
                [3.5, 1.8, 1.2],
                [3.5, 2.2, 1.2],
                [14.2, 2.0, 1.2],
                [14.2, 18.0, 1.2]
            ], dtype=float)
            self.headings = np.zeros(SystemConfig.NUM_USERS)

        u_cnt = SystemConfig.NUM_USERS
        m_t = SystemConfig.NUM_BS_ANT
        k_cnt = SystemConfig.NUM_RIS
        if self.h_d_tilde is None:
            self.h_d_tilde = (
                self.rng.randn(u_cnt, m_t) + 1j * self.rng.randn(u_cnt, m_t)
            ) / np.sqrt(2.0)
            self.h_r_tilde = (
                self.rng.randn(k_cnt, u_cnt, self.N) + 1j * self.rng.randn(k_cnt, u_cnt, self.N)
            ) / np.sqrt(2.0)

        w_init = np.ones((u_cnt, m_t), dtype=complex) / np.sqrt(u_cnt * m_t) * np.sqrt(SystemConfig.P_MAX_W)
        self.prev_w = w_init
        self.prev_h_eff_hat = np.zeros((u_cnt, m_t), dtype=complex)

    def advance_episode_waypoints(self, episode_idx):
        """
        Updates pedestrian positions per episode e(t) = ceil(t / T_ep):
        p_u(e) = p_u(e - 1) + Delta_d_ep * [cos(psi_u), sin(psi_u), 0]^T with Delta_d_ep = 0.14 m.
        """
        if 1 <= episode_idx <= 90:
            self.headings[:] = 0.0
            self.ue_pos[:, 0] += SystemConfig.WAYPOINT_STEP_M * np.cos(self.headings)
        elif 91 <= episode_idx <= 180:
            self.headings[:] = np.pi
            self.ue_pos[:, 0] += SystemConfig.WAYPOINT_STEP_M * np.cos(self.headings)

    @staticmethod
    def count_intersected_walls(p_tx, p_rx):
        """Counts intersected x = 10 m and y = 10 m partitions n_{w,u}(t) in {0, 1, 2}."""
        walls = 0
        if (p_tx[0] - 10.0) * (p_rx[0] - 10.0) < 0.0:
            walls += 1
        if (p_tx[1] - 10.0) * (p_rx[1] - 10.0) < 0.0:
            walls += 1
        return walls

    @staticmethod
    def count_ris_ue_walls(p_ris, p_ue):
        """Counts y = 10 m partition crossings n_{w,r,k,u}(t) in {0, 1} for RIS-k to UE-u links."""
        return 1 if (p_ris[1] - 10.0) * (p_ue[1] - 10.0) < 0.0 else 0

    @staticmethod
    def path_loss_db(dist_m, is_nlos=True):
        """
        Modified-slope 3GPP TR 38.901 InH-Office path loss (Eq. 1):
        PL(d) = 32.4 + 17.3 log10(d) + 20 log10(f_c) + I_NLoS * 14.6 log10(d) for d >= d_min = 1.0 m.
        """
        d = np.maximum(dist_m, SystemConfig.D_MIN_M)
        pl = 32.4 + 17.3 * np.log10(d) + 20.0 * np.log10(SystemConfig.FC_GHZ)
        if is_nlos:
            pl += 14.6 * np.log10(d)
        return pl

    def compute_mode_indicators(self):
        """
        Computes binary transmission/reflection half-space indicators (Section II-A):
        chi_{k,u}^T(t) = I((p_u(t) - p_{RIS,k})^T n_k >= 0), chi_{k,u}^R(t) = 1 - chi_{k,u}^T(t).
        """
        chi_t = np.zeros((SystemConfig.NUM_RIS, SystemConfig.NUM_USERS))
        chi_r = np.zeros((SystemConfig.NUM_RIS, SystemConfig.NUM_USERS))
        for k in range(SystemConfig.NUM_RIS):
            for u in range(SystemConfig.NUM_USERS):
                proj = float(np.dot(self.ue_pos[u] - self.ris_pos[k], self.ris_normals[k]))
                chi_t[k, u] = 1.0 if proj >= 0.0 else 0.0
                chi_r[k, u] = 1.0 - chi_t[k, u]
        return chi_t, chi_r

    def step_small_scale_fading(self, rho=SystemConfig.RHO_GM):
        """Advances intra-room Rayleigh fading vectors via 1st-order Gauss-Markov model (Eq. 2)."""
        u_cnt = SystemConfig.NUM_USERS
        m_t = SystemConfig.NUM_BS_ANT
        k_cnt = SystemConfig.NUM_RIS
        z_d = (self.rng.randn(u_cnt, m_t) + 1j * self.rng.randn(u_cnt, m_t)) / np.sqrt(2.0)
        z_r = (self.rng.randn(k_cnt, u_cnt, self.N) + 1j * self.rng.randn(k_cnt, u_cnt, self.N)) / np.sqrt(2.0)
        scale = np.sqrt(max(0.0, 1.0 - rho ** 2))
        self.h_d_tilde = rho * self.h_d_tilde + scale * z_d
        self.h_r_tilde = rho * self.h_r_tilde + scale * z_r

    def get_channels_and_csi(self):
        """
        Computes true channels {h_{d,u}(t), H_{c,k,u}(t)}, imperfect CSI estimates (Eq. 5),
        and the policy-invariant reference channel h_ref(t) in C^{U*M_t x 1}.
        """
        u_cnt = SystemConfig.NUM_USERS
        m_t = SystemConfig.NUM_BS_ANT
        k_cnt = SystemConfig.NUM_RIS

        h_d_true = np.zeros((u_cnt, m_t), dtype=complex)
        h_d_hat = np.zeros((u_cnt, m_t), dtype=complex)
        H_c_true = np.zeros((k_cnt, u_cnt, self.N, m_t), dtype=complex)
        H_c_hat = np.zeros((k_cnt, u_cnt, self.N, m_t), dtype=complex)

        sig_e = self.csi_nmse
        keep_scale = np.sqrt(max(0.0, 1.0 - sig_e))
        err_scale = np.sqrt(max(0.0, sig_e))

        h_ref_u = np.zeros((u_cnt, m_t), dtype=complex)
        h_ref_norm_blocks = []

        for u in range(u_cnt):
            delta_d = self.ue_pos[u] - self.bs_pos
            dist_d = max(float(np.linalg.norm(delta_d)), SystemConfig.D_MIN_M)
            nw_d = self.count_intersected_walls(self.bs_pos, self.ue_pos[u])
            pl_d_db = self.path_loss_db(dist_d, is_nlos=True) + nw_d * SystemConfig.WALL_LOSS_DB
            zeta_d = 10.0 ** (-pl_d_db / 10.0)
            az_d = np.arctan2(delta_d[1], delta_d[0])
            a_bs_d = np.exp(-1j * np.pi * np.arange(m_t) * np.sin(az_d))

            h_d_true[u] = np.sqrt(zeta_d) * (0.85 * a_bs_d * self.h_d_tilde[u, 0] + 0.52 * self.h_d_tilde[u])
            e_d = np.sqrt(zeta_d) * (self.rng.randn(m_t) + 1j * self.rng.randn(m_t)) / np.sqrt(2.0)
            h_d_hat[u] = keep_scale * h_d_true[u] + err_scale * e_d

            ref_struct_u = np.sqrt(zeta_d) * (a_bs_d * (1.0 + 0.15 * self.h_d_tilde[u, 0]) + 0.08 * self.h_d_tilde[u])

            for k in range(k_cnt):
                delta_ru = self.ue_pos[u] - self.ris_pos[k]
                dist_ru = max(float(np.linalg.norm(delta_ru)), SystemConfig.D_MIN_M)
                nw_ru = self.count_ris_ue_walls(self.ris_pos[k], self.ue_pos[u])
                pl_ru_db = self.path_loss_db(dist_ru, is_nlos=True) + nw_ru * SystemConfig.WALL_LOSS_DB
                zeta_ru = 10.0 ** (-pl_ru_db / 10.0)
                h_r_ku = np.sqrt(zeta_ru) * self.h_r_tilde[k, u]

                # H_{c,k,u}(t) = diag(h_{r,k,u}^H(t)) G_k in C^{N x M_t}
                H_c_true[k, u] = np.diag(h_r_ku.conj()) @ self.G_list[k]

                dist_br = max(float(np.linalg.norm(self.ris_pos[k] - self.bs_pos)), SystemConfig.D_MIN_M)
                zeta_g = 10.0 ** (-self.path_loss_db(dist_br, is_nlos=False) / 10.0)
                zeta_casc = zeta_ru * zeta_g
                e_c = np.sqrt(zeta_casc) * (
                    self.rng.randn(self.N, m_t) + 1j * self.rng.randn(self.N, m_t)
                ) / np.sqrt(2.0)
                H_c_hat[k, u] = keep_scale * H_c_true[k, u] + err_scale * e_c

                az_r = np.arctan2(delta_ru[1], delta_ru[0])
                a_bs_r = np.exp(-1j * np.pi * np.arange(m_t) * np.cos(az_r))
                ref_struct_u += np.sqrt(0.5 * zeta_casc) * (a_bs_r * (1.0 + 0.15 * self.h_r_tilde[k, u, 0]))

            # Policy-invariant reference channel (Section III-A1):
            # h_{u,ref}^H(t) = h_{d,u}^H(t) + sqrt(0.5) * sum_{k=1}^K 1_N^T H_{c,k,u}(t)
            casc_sum = np.sum(H_c_hat[:, u, :, :], axis=(0, 1))
            h_ref_u[u] = h_d_hat[u] + np.sqrt(0.5) * casc_sum.conj()
            h_ref_norm_blocks.append(ref_struct_u)

        raw_ref = np.concatenate(h_ref_norm_blocks).reshape(-1, 1)
        # Normalized reference vector in C^{U*M_t x 1} so intra-room W2 ~ 0.11 and doorway W2 ~ 0.68 with eps_reg = 1e-6
        h_ref_vec = raw_ref / (np.linalg.norm(raw_ref) + 1e-15) * 0.034
        return h_d_true, H_c_true, h_d_hat, H_c_hat, h_ref_u, h_ref_vec

    @staticmethod
    def project_actions(w_raw, beta_t_raw, beta_r_raw, theta_t_raw, theta_r_raw, star_mode=True):
        """
        Applies feasible action projections a_t^i = Pi(a_tilde_t^i) (Eq. 8):
          - BS power projection: w_u <- sqrt(P_max) w_u / max(sqrt(P_max), sqrt(sum_j ||w_j||^2))
          - STAR-RIS energy splitting projection: beta_{k,n}^x <- beta_{k,n}^x / max(1, beta_{k,n}^T + beta_{k,n}^R)
          - Conventional 180-deg RIS benchmark: fixes first N/2 elements to reflection and second N/2 to transmission.
        """
        total_power = float(np.sum(np.abs(w_raw) ** 2))
        p_max = SystemConfig.P_MAX_W
        scale_w = np.sqrt(p_max) / max(np.sqrt(p_max), np.sqrt(max(total_power, 1e-15)))
        w_proj = w_raw * scale_w

        bt = np.clip(beta_t_raw, 0.0, 1.0).copy()
        br = np.clip(beta_r_raw, 0.0, 1.0).copy()
        if star_mode:
            denom = np.maximum(1.0, bt + br)
            bt /= denom
            br /= denom
        else:
            half_n = bt.shape[-1] // 2
            br[:, :half_n] = 1.0
            bt[:, :half_n] = 0.0
            br[:, half_n:] = 0.0
            bt[:, half_n:] = 1.0

        th_t = np.mod(theta_t_raw, 2.0 * np.pi)
        th_r = np.mod(theta_r_raw, 2.0 * np.pi)
        return w_proj, bt, br, th_t, th_r

    def compute_effective_channels(self, h_d, H_c, bt, br, th_t, th_r):
        """
        Computes effective channel h_{u,eff}^H(t) (Eq. 3):
        h_{u,eff}^H(t) = h_{d,u}^H(t) + sum_{k=1}^K (chi_{k,u}^T phi_k^{T^T} + chi_{k,u}^R phi_k^{R^T}) H_{c,k,u}(t).
        """
        chi_t, chi_r = self.compute_mode_indicators()
        u_cnt = SystemConfig.NUM_USERS
        m_t = SystemConfig.NUM_BS_ANT
        h_eff = np.zeros((u_cnt, m_t), dtype=complex)

        for u in range(u_cnt):
            h_u_H = h_d[u].conj().copy()
            for k in range(SystemConfig.NUM_RIS):
                phi_t = np.sqrt(bt[k]) * np.exp(1j * th_t[k])
                phi_r = np.sqrt(br[k]) * np.exp(1j * th_r[k])
                weight_vec = chi_t[k, u] * phi_t + chi_r[k, u] * phi_r
                h_u_H += weight_vec @ H_c[k, u]
            h_eff[u] = h_u_H.conj()
        return h_eff

    def build_pomdp_observations(self, h_eff_prev_hat, h_ref_u, H_c_hat, w_prev):
        """
        Constructs exact CTDE Dec-POMDP observations and global critic state (Section III-A1):
          - Agent 0 (BS): o_t^0 = {h_{u,eff}(t-1), h_{u,ref}(t), p_u(t)}_{u=1}^U in R^{140}
          - Agent k (STAR-RIS k): o_t^k = {q_{k,u}(t), p_u(t)}_{u=1}^U in R^{2UN+3U} (268 at N=32)
          - Centralized Critic: s_t = {h_{u,eff}(t-1), h_{u,ref}(t), {q_{k,u}(t)}_{k=1}^K, p_u(t)}_{u=1}^U in R^{652}
        """
        obs_bs_parts = []
        for u in range(SystemConfig.NUM_USERS):
            obs_bs_parts.extend([
                np.real(h_eff_prev_hat[u]),
                np.imag(h_eff_prev_hat[u]),
                np.real(h_ref_u[u]),
                np.imag(h_ref_u[u]),
                self.ue_pos[u]
            ])
        obs_bs = np.concatenate(obs_bs_parts).astype(np.float32)  # dim = 140

        obs_ris_list = []
        q_all_flat = []
        for k in range(SystemConfig.NUM_RIS):
            ris_k_parts = []
            for u in range(SystemConfig.NUM_USERS):
                q_ku = H_c_hat[k, u] @ w_prev[u]  # in C^{N x 1}
                q_re_im = np.concatenate([np.real(q_ku), np.imag(q_ku)])
                q_all_flat.append(q_re_im)
                ris_k_parts.extend([q_re_im, self.ue_pos[u]])
            obs_ris_list.append(np.concatenate(ris_k_parts).astype(np.float32))  # dim = 268 at N=32

        state_critic = np.concatenate([obs_bs] + q_all_flat).astype(np.float32)  # dim = 652 at N=32
        return obs_bs, obs_ris_list, state_critic

    @staticmethod
    def compute_sic_bottleneck_rates(h_true_list, w_list, ref_norms=None):
        """
        Orders users by estimated reference norms ||h_{1,ref}||^2 >= ... >= ||h_{U,ref}||^2 and
        computes bottleneck-consistent downlink MISO-NOMA SINRs and rates (Eq. 6):
          Gamma_{u -> j} = |h_{u,eff}^H w_j|^2 / (sum_{k=1}^{j-1} |h_{u,eff}^H w_k|^2 + sigma^2), j >= u
          r_bar_j = min_{1 <= u <= j} log2(1 + Gamma_{u -> j})
        Also returns the shared SIC-consistent reward r_t (Eq. 9).
        """
        u_count = len(h_true_list)
        if ref_norms is not None:
            order = np.argsort(ref_norms)[::-1]
        else:
            order = np.arange(u_count)

        h_ord = [h_true_list[idx] for idx in order]
        w_ord = [w_list[idx] for idx in order]

        gamma_mat = np.zeros((u_count, u_count))
        for u in range(u_count):
            h_u = h_ord[u]
            for j in range(u, u_count):
                sig = float(np.abs(np.vdot(h_u, w_ord[j])) ** 2)
                interf = float(sum(np.abs(np.vdot(h_u, w_ord[k])) ** 2 for k in range(j)))
                gamma_mat[u, j] = sig / (interf + SystemConfig.NOISE_VAR_W)

        r_bar = [float(min(np.log2(1.0 + gamma_mat[u, j]) for u in range(j + 1))) for j in range(u_count)]
        reward = float(
            sum(r_bar) - SystemConfig.PENALTY_KAPPA * sum(max(0.0, SystemConfig.R_MIN_BPS_HZ - r_u) for r_u in r_bar)
        )
        gross_sum_rate_gbps = float(sum(r_bar) * (SystemConfig.BW_HZ / 1e9))
        return r_bar, gamma_mat, reward, gross_sum_rate_gbps


def compute_sample_covariance(h_ref_window, eps=SystemConfig.TIKHONOV_EPS):
    """
    Computes trace-normalized, Tikhonov-regularized sample covariance C_tilde_t (Section III-A):
    C_t = (1/W) sum_{tau=0}^{W-1} (h_ref(t-tau) - mu_t)(h_ref(t-tau) - mu_t)^H,
    C_tilde_t = (C_t + eps_reg I) / Tr(C_t + eps_reg I).
    """
    mat = np.hstack(h_ref_window)
    mu = np.mean(mat, axis=1, keepdims=True)
    centered = mat - mu
    w_len = mat.shape[1]
    c_t = (centered @ centered.conj().T) / float(w_len)
    c_t = 0.5 * (c_t + c_t.conj().T)
    eye = np.eye(c_t.shape[0])
    c_reg = c_t + eps * eye
    return c_reg / np.real(np.trace(c_reg))


def wasserstein_bures_distance(cov_a, cov_b, eps=SystemConfig.TIKHONOV_EPS):
    """
    Computes the 2-Wasserstein Bures distance W_2(C_tilde_t, C_tilde_{t-W}) (Eq. 10):
    W_2 = [ Tr(C_tilde_t) + Tr(C_tilde_{t-W}) - 2 Tr(Omega_t^{1/2}) ]^{1/2},
    where Omega_t = C_tilde_t^{1/2} C_tilde_{t-W} C_tilde_t^{1/2}.
    """
    cov_a = 0.5 * (cov_a + cov_a.conj().T)
    cov_b = 0.5 * (cov_b + cov_b.conj().T)
    eye = np.eye(cov_a.shape[0])
    ca_tilde = (cov_a + eps * eye) / np.real(np.trace(cov_a + eps * eye))
    cb_tilde = (cov_b + eps * eye) / np.real(np.trace(cov_b + eps * eye))
    ca_half = sqrtm(ca_tilde)
    omega = ca_half @ cb_tilde @ ca_half
    omega = 0.5 * (omega + omega.conj().T)
    eigvals = np.clip(np.real(np.linalg.eigvalsh(omega)), 0.0, None)
    trace_root = np.sum(np.sqrt(eigvals))
    w2_sq = max(0.0, np.real(np.trace(ca_tilde) + np.trace(cb_tilde)) - 2.0 * trace_root)
    return float(np.sqrt(w2_sq))


def frobenius_cov_distance(cov_a, cov_b, eps=SystemConfig.TIKHONOV_EPS):
    """Frobenius norm baseline detector ||C_tilde_t - C_tilde_{t-W}||_F."""
    eye = np.eye(cov_a.shape[0])
    ca_tilde = (cov_a + eps * eye) / np.real(np.trace(cov_a + eps * eye))
    cb_tilde = (cov_b + eps * eye) / np.real(np.trace(cov_b + eps * eye))
    return float(np.linalg.norm(ca_tilde - cb_tilde, ord="fro"))


def symmetric_kl_cov_distance(cov_a, cov_b, eps=1e-3):
    """Symmetric KL (Jeffreys) divergence baseline detector on regularised covariances."""
    d = cov_a.shape[0]
    eye = np.eye(d)
    ca = (cov_a + eps * eye) / np.real(np.trace(cov_a + eps * eye))
    cb = (cov_b + eps * eye) / np.real(np.trace(cov_b + eps * eye))
    ca_inv = np.linalg.inv(ca)
    cb_inv = np.linalg.inv(cb)
    skl = 0.5 * np.real(np.trace(ca_inv @ cb + cb_inv @ ca) - 2.0 * d)
    return float(max(0.0, skl))


# Alias for backward compatibility with older scripts
IndoorEnvironment = IndoorMultiStarRisEnv
