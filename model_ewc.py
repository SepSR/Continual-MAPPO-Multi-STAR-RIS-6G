#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continual MAPPO Architecture and Two-Buffer Actor-Critic EWC (Algorithm 1, Equations 11-12)
Manuscript ID: WCL2026-3112 (Revised)
Title: Continual MAPPO for Mobility-Aware Resource Allocation in 6G Multi-STAR-RIS Indoor NOMA Networks
Authors: Sepehr Rahmani, Sana Rahmani, and Sina Rahmani
Journal: IEEE Wireless Communications Letters
"""

from collections import deque
import numpy as np
from config import SystemConfig
from environment import wasserstein_bures_distance

# Optional PyTorch support: uses torch.nn if installed, with full NumPy autograd-equivalent fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def verify_network_parameters(n_elem=SystemConfig.DEFAULT_N):
    """
    Computes exact parameter counts for the CTDE Continual MAPPO architecture (Section III-C):
      - BS Actor (Agent 0): d_in = 140, 2x64 MLP, d_out = 64 + 64 log-std weights -> 17,408 (17.4k)
      - STAR-RIS Actor k (Agents 1..K): d_in = 2*U*N + 3*U (268 at N=32), 2x64 MLP, d_out = 4*N -> 29,824 (29.8k at N=32; 54.7k at N=64)
      - Centralized Critic V_psi: d_in = 140 + K*(2*U*N) (652 at N=32), 2x64 MLP, d_out = 1 -> 46,017 (46.0k at N=32; 78.8k at N=64)
      - Total Parameters P_tot: 123,073 (123.1k) at N=32; 205,505 (205.5k) at N=64.
    """
    u_cnt = SystemConfig.NUM_USERS
    m_t = SystemConfig.NUM_BS_ANT
    h_dim = SystemConfig.MLP_HIDDEN

    d_in_bs = 4 * u_cnt * m_t + 3 * u_cnt
    d_out_bs = 2 * u_cnt * m_t
    p_bs = (d_in_bs * h_dim + h_dim) + (h_dim * h_dim + h_dim) + (h_dim * d_out_bs + d_out_bs) + d_out_bs

    d_in_ris = 2 * u_cnt * n_elem + 3 * u_cnt
    d_out_ris = 4 * n_elem
    p_ris = (d_in_ris * h_dim + h_dim) + (h_dim * h_dim + h_dim) + (h_dim * d_out_ris + d_out_ris) + d_out_ris

    d_in_critic = d_in_bs + SystemConfig.NUM_RIS * (2 * u_cnt * n_elem)
    p_critic = (d_in_critic * h_dim + h_dim) + (h_dim * h_dim + h_dim) + (h_dim * 1 + 1)

    p_tot = p_bs + SystemConfig.NUM_RIS * p_ris + p_critic
    return p_tot, p_bs, p_ris, p_critic


class GaussianMLPActor:
    """
    Decentralized Gaussian policy pi_{theta_i}(a_tilde_t^i | o_t^i) with a 2x64 MLP and learnable log-std.
    Evaluates log pi_{theta_i}(a_tilde_t^i | o_t^i) on unprojected actions a_tilde_t^i (Section III-A2).
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=SystemConfig.MLP_HIDDEN, seed=0):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        rng = np.random.RandomState(seed)

        self.params = {
            "w1": rng.randn(self.obs_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.obs_dim),
            "b1": np.zeros(self.hidden_dim, dtype=np.float32),
            "w2": rng.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.hidden_dim),
            "b2": np.zeros(self.hidden_dim, dtype=np.float32),
            "w3": rng.randn(self.hidden_dim, self.act_dim).astype(np.float32) * 0.01,
            "b3": np.zeros(self.act_dim, dtype=np.float32),
            "log_std": np.full(self.act_dim, -0.5, dtype=np.float32),
        }

    def num_parameters(self):
        return sum(arr.size for arr in self.params.values())

    def clone_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def load_params(self, snap):
        for k in self.params:
            self.params[k] = snap[k].copy()

    def forward(self, obs):
        h1 = np.maximum(0.0, obs @ self.params["w1"] + self.params["b1"])
        h2 = np.maximum(0.0, h1 @ self.params["w2"] + self.params["b2"])
        mean = h2 @ self.params["w3"] + self.params["b3"]
        std = np.exp(np.clip(self.params["log_std"], -5.0, 2.0))
        return mean, std, (obs, h1, h2)

    def sample_action(self, obs, rng):
        mean, std, _ = self.forward(obs)
        noise = rng.randn(*mean.shape).astype(np.float32)
        a_tilde = mean + std * noise
        log_prob = -0.5 * np.sum(((a_tilde - mean) / std) ** 2 + 2.0 * np.log(std) + np.log(2.0 * np.pi))
        return a_tilde, float(log_prob)

    def compute_grad_log_prob_sq(self, obs, a_tilde):
        """
        Computes per-parameter squared gradient (d log pi_{theta_i}(a_tilde^i | o^i) / d theta_{i,m})^2
        for Eq. (11a) diagonal Fisher Information Matrix estimation.
        """
        mean, std, (_, h1, h2) = self.forward(obs)
        diff = a_tilde - mean
        var = std ** 2
        d_mean = diff / var
        d_log_std = (diff ** 2) / var - 1.0

        grad_w3 = np.outer(h2, d_mean)
        grad_b3 = d_mean
        dh2 = (self.params["w3"] @ d_mean) * (h2 > 0.0)
        grad_w2 = np.outer(h1, dh2)
        grad_b2 = dh2
        dh1 = (self.params["w2"] @ dh2) * (h1 > 0.0)
        grad_w1 = np.outer(obs, dh1)
        grad_b1 = dh1

        return {
            "w1": grad_w1 ** 2,
            "b1": grad_b1 ** 2,
            "w2": grad_w2 ** 2,
            "b2": grad_b2 ** 2,
            "w3": grad_w3 ** 2,
            "b3": grad_b3 ** 2,
            "log_std": d_log_std ** 2,
        }


class CentralizedCritic:
    """
    Centralized critic V_psi(s_t) with a 2x64 MLP receiving s_t in R^{652} (at N=32) or R^{1164} (at N=64).
    Computes Fisher-like output-sensitivity importance (d V_psi(s) / d psi_n)^2 in Eq. (11b).
    """

    def __init__(self, state_dim, hidden_dim=SystemConfig.MLP_HIDDEN, seed=10):
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        rng = np.random.RandomState(seed)

        self.params = {
            "w1": rng.randn(self.state_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.state_dim),
            "b1": np.zeros(self.hidden_dim, dtype=np.float32),
            "w2": rng.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.hidden_dim),
            "b2": np.zeros(self.hidden_dim, dtype=np.float32),
            "w3": rng.randn(self.hidden_dim, 1).astype(np.float32) * 0.01,
            "b3": np.zeros(1, dtype=np.float32),
        }

    def num_parameters(self):
        return sum(arr.size for arr in self.params.values())

    def clone_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def load_params(self, snap):
        for k in self.params:
            self.params[k] = snap[k].copy()

    def forward(self, state):
        h1 = np.maximum(0.0, state @ self.params["w1"] + self.params["b1"])
        h2 = np.maximum(0.0, h1 @ self.params["w2"] + self.params["b2"])
        val = float((h2 @ self.params["w3"] + self.params["b3"])[0])
        return val, (state, h1, h2)

    def compute_grad_value_sq(self, state):
        """
        Computes per-parameter squared value sensitivity (d V_psi(s) / d psi_n)^2 for Eq. (11b).
        """
        _, (_, h1, h2) = self.forward(state)
        grad_w3 = h2.reshape(-1, 1)
        grad_b3 = np.ones(1, dtype=np.float32)
        dh2 = self.params["w3"].ravel() * (h2 > 0.0)
        grad_w2 = np.outer(h1, dh2)
        grad_b2 = dh2
        dh1 = (self.params["w2"] @ dh2) * (h1 > 0.0)
        grad_w1 = np.outer(state, dh1)
        grad_b1 = dh1

        return {
            "w1": grad_w1 ** 2,
            "b1": grad_b1 ** 2,
            "w2": grad_w2 ** 2,
            "b2": grad_b2 ** 2,
            "w3": grad_w3 ** 2,
            "b3": grad_b3 ** 2,
        }


class TwoBufferContinualMAPPO:
    """
    Implements Algorithm 1 (Continual MAPPO with Two-Buffer EWC):
      - K+1 decentralized actors {pi_{theta_i}}_{i=0}^K and centralized critic V_psi
      - Length-W staging FIFO Q_W and pre-drift ring buffer D_prev (M_F = 500 slots)
      - Prior-episode clean snapshot (theta_i^snap, psi^snap)
      - 3-slot persistence check and 2W-slot lockout timer on W_2(C_tilde_tau, C_tilde_{tau-W}) > eps_th
      - Online EWC consolidation (Eqs. 11a, 11b, 12) with moving anchor (theta_{i,old}^*, psi_old^*) and gamma_EWC decay
    """

    def __init__(
        self,
        num_elements=SystemConfig.DEFAULT_N,
        ewc_lambda=SystemConfig.EWC_LAMBDA,
        ewc_lambda_psi=SystemConfig.EWC_LAMBDA_PSI,
        gamma_ewc=SystemConfig.GAMMA_EWC,
        drift_thresh=SystemConfig.DRIFT_THRESH,
        seed=42,
    ):
        self.N = int(num_elements)
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_lambda_psi = float(ewc_lambda_psi)
        self.gamma_ewc = float(gamma_ewc)
        self.drift_thresh = float(drift_thresh)
        self.rng = np.random.RandomState(seed)

        u_cnt = SystemConfig.NUM_USERS
        m_t = SystemConfig.NUM_BS_ANT
        obs_dim_bs = 4 * u_cnt * m_t + 3 * u_cnt
        act_dim_bs = 2 * u_cnt * m_t
        obs_dim_ris = 2 * u_cnt * self.N + 3 * u_cnt
        act_dim_ris = 4 * self.N
        state_dim_critic = obs_dim_bs + SystemConfig.NUM_RIS * (2 * u_cnt * self.N)

        # Line 1 of Algorithm 1: Initialize actors {pi_{theta_i}}_{i=0}^K and critic V_psi
        self.actors = [GaussianMLPActor(obs_dim_bs, act_dim_bs, seed=seed)]
        for k in range(SystemConfig.NUM_RIS):
            self.actors.append(GaussianMLPActor(obs_dim_ris, act_dim_ris, seed=seed + k + 1))
        self.critic = CentralizedCritic(state_dim_critic, seed=seed + 10)

        # Two-buffer architecture: staging FIFO Q_W (capacity W=20) and pre-drift ring buffer D_prev (M_F=500)
        self.staging_fifo_qw = deque()
        self.ring_buffer_dprev = deque(maxlen=SystemConfig.RESERVOIR_M)

        # Prior-episode snapshots (theta_i^snap, psi^snap) and consolidated anchors (theta_{i,old}^*, psi_old^*)
        self.theta_snap = [actor.clone_params() for actor in self.actors]
        self.psi_snap = self.critic.clone_params()
        self.theta_star_old = [actor.clone_params() for actor in self.actors]
        self.psi_star_old = self.critic.clone_params()

        # Consolidated diagonal Fisher Information Matrices F_{theta_i} and critic importance F_psi
        self.fisher_actors = [
            {k: np.zeros_like(v) for k, v in actor.params.items()} for actor in self.actors
        ]
        self.fisher_critic = {k: np.zeros_like(v) for k, v in self.critic.params.items()}

        # Drift detection history and lockout state t_trig <- -2W
        self.t_trig = -2 * SystemConfig.WINDOW_W
        self.w2_history = deque(maxlen=SystemConfig.PERSISTENCE_SLOTS)
        self.cov_history = deque(maxlen=SystemConfig.WINDOW_W + 1)
        self.href_window = deque(maxlen=SystemConfig.WINDOW_W)
        self.drift_events = []

    def push_transition_and_check_drift(self, slot_t, transition, h_ref_vec):
        """
        Executes Lines 5-12 of Algorithm 1:
          - Pushes transition e_t to staging FIFO Q_W
          - Updates sliding-window reference covariance C_tilde_t and evaluates W_2(C_tilde_t, C_tilde_{t-W})
          - Checks 3-slot persistence (tau in {t-2, t-1, t}) and 2W-slot lockout (t - t_trig > 2W)
          - On drift confirmation: anchors pre-drift weights to prior-episode snapshot, consolidates Fisher matrices
            on D_prev via Eqs. (11a)-(11b), and clears Q_W so doorway-transition samples never pollute D_prev.
          - Otherwise, if |Q_W| == W, pops the delayed transition e_{t-W} from Q_W into ring buffer D_prev.
        """
        self.staging_fifo_qw.append(transition)
        self.href_window.append(h_ref_vec)

        w2_val = 0.0
        if len(self.href_window) == SystemConfig.WINDOW_W:
            mat = np.hstack(self.href_window)
            mu = np.mean(mat, axis=1, keepdims=True)
            centered = mat - mu
            c_t = (centered @ centered.conj().T) / float(SystemConfig.WINDOW_W)
            c_t = 0.5 * (c_t + c_t.conj().T)
            eye = np.eye(c_t.shape[0])
            c_tilde = (c_t + SystemConfig.TIKHONOV_EPS * eye) / np.real(
                np.trace(c_t + SystemConfig.TIKHONOV_EPS * eye)
            )
            self.cov_history.append(c_tilde)

            if len(self.cov_history) > SystemConfig.WINDOW_W:
                c_prev = self.cov_history[0]
                w2_val = wasserstein_bures_distance(c_tilde, c_prev)
                self.w2_history.append(w2_val)

        # Line 6: if t >= 2W and W_2(C_tilde_tau, C_tilde_{tau-W}) > eps_th for tau in {t-2, t-1, t} and t - t_trig > 2W
        persistence_met = (
            len(self.w2_history) == SystemConfig.PERSISTENCE_SLOTS
            and all(w > self.drift_thresh for w in self.w2_history)
        )
        if (
            slot_t >= SystemConfig.WARMUP_SLOTS
            and persistence_met
            and (slot_t - self.t_trig > SystemConfig.LOCKOUT_SLOTS)
        ):
            # Line 7: Set t_trig <- t; anchor pre-drift weights theta_{i,old}^* <- theta_i^snap, psi_old^* <- psi^snap
            self.t_trig = slot_t
            self.theta_star_old = [
                {k: v.copy() for k, v in snap.items()} for snap in self.theta_snap
            ]
            self.psi_star_old = {k: v.copy() for k, v in self.psi_snap.items()}

            # Line 8 & 9: Compute F_{theta_i,new}, F_{psi,new} via (11a)-(11b) on D_prev; clear Q_W; consolidate
            self.consolidate_fisher_on_dprev()
            self.staging_fifo_qw.clear()
            self.drift_events.append(slot_t)
            return True, w2_val

        # Line 10-11: else if |Q_W| == W then pop e_{t-W} from Q_W into ring buffer D_prev
        if len(self.staging_fifo_qw) >= SystemConfig.WINDOW_W:
            oldest_transition = self.staging_fifo_qw.popleft()
            self.ring_buffer_dprev.append(oldest_transition)

        return False, w2_val

    def consolidate_fisher_on_dprev(self):
        """
        Computes empirical diagonal Fisher Information F_{theta_i,new} (Eq. 11a) and critic output-sensitivity
        importance F_{psi,new} (Eq. 11b) on clean pre-drift ring buffer D_prev, evaluated at (theta_{i,old}^*, psi_old^*).
        """
        if len(self.ring_buffer_dprev) == 0:
            return

        n_samples = float(len(self.ring_buffer_dprev))

        # Save current online weights, load anchor weights (theta_{i,old}^*, psi_old^*) for gradient evaluation
        curr_actor_params = [actor.clone_params() for actor in self.actors]
        curr_critic_params = self.critic.clone_params()

        for i, actor in enumerate(self.actors):
            actor.load_params(self.theta_star_old[i])
        self.critic.load_params(self.psi_star_old)

        f_actors_new = [
            {k: np.zeros_like(v) for k, v in actor.params.items()} for actor in self.actors
        ]
        f_critic_new = {k: np.zeros_like(v) for k, v in self.critic.params.items()}

        for item in self.ring_buffer_dprev:
            obs_list = item["obs_list"]
            act_tilde_list = item["act_tilde_list"]
            state_s = item["state"]

            for i, actor in enumerate(self.actors):
                grads_sq = actor.compute_grad_log_prob_sq(obs_list[i], act_tilde_list[i])
                for k in f_actors_new[i]:
                    f_actors_new[i][k] += grads_sq[k] / n_samples

            c_grads_sq = self.critic.compute_grad_value_sq(state_s)
            for k in f_critic_new:
                f_critic_new[k] += c_grads_sq[k] / n_samples

        # Line 9: Consolidate F_{theta_i} <- gamma_EWC * F_{theta_i} + F_{theta_i,new}
        for i in range(len(self.actors)):
            for k in self.fisher_actors[i]:
                self.fisher_actors[i][k] = (
                    self.gamma_ewc * self.fisher_actors[i][k] + f_actors_new[i][k]
                )
        for k in self.fisher_critic:
            self.fisher_critic[k] = self.gamma_ewc * self.fisher_critic[k] + f_critic_new[k]

        # Restore current online weights
        for i, actor in enumerate(self.actors):
            actor.load_params(curr_actor_params[i])
        self.critic.load_params(curr_critic_params)

    def update_prior_episode_snapshots(self, drift_triggered_in_episode):
        """
        Line 14 of Algorithm 1:
        If no drift triggered in episode e, update (theta_i^snap, psi^snap) <- (theta_i, psi).
        """
        if not drift_triggered_in_episode:
            self.theta_snap = [actor.clone_params() for actor in self.actors]
            self.psi_snap = self.critic.clone_params()

    def compute_ewc_penalty(self):
        """
        Computes the quadratic actor-critic EWC regularization penalty in Eq. (12):
        sum_{i=0}^K sum_m (lambda_EWC / 2) F_{theta_{i,m}} (theta_{i,m} - theta_{i,old,m}^*)^2
        + sum_n (lambda_psi / 2) F_{psi,n} (psi_n - psi_{old,n}^*)^2.
        """
        actor_pen = 0.0
        for i, actor in enumerate(self.actors):
            for k, param in actor.params.items():
                diff = param - self.theta_star_old[i][k]
                actor_pen += float(np.sum(self.fisher_actors[i][k] * (diff ** 2)))
        actor_pen *= 0.5 * self.ewc_lambda

        critic_pen = 0.0
        for k, param in self.critic.params.items():
            diff = param - self.psi_star_old[k]
            critic_pen += float(np.sum(self.fisher_critic[k] * (diff ** 2)))
        critic_pen *= 0.5 * self.ewc_lambda_psi

        return actor_pen + critic_pen


# Backward-compatible PyTorch or NumPy classes if external scripts import ActorCritic / ContinualEWC
if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """PyTorch module wrapper matching the 2x64 MLP architecture in Table I."""

        def __init__(self, state_dim=140, action_dim=64, hidden_dim=SystemConfig.MLP_HIDDEN):
            super(ActorCritic, self).__init__()
            self.actor_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
            self.critic_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state):
            mean = self.actor_mlp(state)
            val = self.critic_mlp(state)
            return mean, val
else:
    ActorCritic = GaussianMLPActor

ContinualEWC = TwoBufferContinualMAPPO
