# Multi-agent Actor-Critic with Elastic Weight Consolidation (EWC)
import torch
import torch.nn as nn
import config

class ActorCritic(nn.Module):
    def __init__(self, state_dim=16, action_dim=16):
        super(ActorCritic, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        feat = self.feature_net(state)
        actions = self.actor(feat)
        value = self.critic(feat)
        return actions, value

class ContinualEWC:
    def __init__(self, model, ewc_lambda=config.EWC_LAMBDA):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = {}
        self.optimal_params = {}

    def update_task_checkpoint(self, transitions):
        self.model.eval()
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        self.fisher_matrix = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        for s in transitions:
            self.model.zero_grad()
            _, val = self.model(s)
            loss = val.sum()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher_matrix[n] += p.grad.data ** 2 / len(transitions)

    def compute_loss(self):
        if not self.fisher_matrix:
            return torch.tensor(0.0)
        loss = 0.0
        for n, p in self.model.named_parameters():
            loss += (self.fisher_matrix[n] * (p - self.optimal_params[n]) ** 2).sum()
        return (self.ewc_lambda / 2.0) * loss