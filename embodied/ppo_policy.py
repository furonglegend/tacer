"""
PPO Policy Network for Graph Trajectory Generation
Implements actor-critic architecture for learning explanation trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, List, Optional
import numpy as np


class TrajectoryPolicy(nn.Module):
    """
    Actor-Critic policy network for trajectory-based explanations.
    Uses LSTM to encode trajectory history and predict next actions.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 lstm_layers: int = 2):
        """
        Initialize policy network.

        Args:
            obs_dim: Observation dimension
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
            lstm_layers: Number of LSTM layers
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # LSTM for trajectory history
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize LSTM hidden state
        self.lstm_layers = lstm_layers

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, obs_dim] or [batch, seq, obs_dim]
            hidden: LSTM hidden state tuple
            action_mask: Mask for invalid actions [batch, action_dim]

        Returns:
            action_logits, value, new_hidden
        """
        # Encode observation
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension

        batch_size = obs.size(0)
        device = obs.device

        # Encode observations
        encoded = self.obs_encoder(obs)

        # LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        lstm_out, new_hidden = self.lstm(encoded, hidden)

        # Use last LSTM output
        last_out = lstm_out[:, -1, :]

        # Actor: action logits
        action_logits = self.actor(last_out)

        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

        # Critic: state value
        value = self.critic(last_out)

        return action_logits, value.squeeze(-1), new_hidden

    def get_action(self, obs: torch.Tensor, hidden: Optional[Tuple] = None,
                   action_mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample action from policy.

        Args:
            obs: Observation
            hidden: LSTM hidden state
            action_mask: Mask for invalid actions
            deterministic: If True, take argmax action

        Returns:
            action, log_prob, value, new_hidden
        """
        action_logits, value, new_hidden = self.forward(obs, hidden, action_mask)

        # Create distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value, new_hidden

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         hidden: Optional[Tuple] = None,
                         action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observations [batch, obs_dim]
            actions: Actions taken [batch]
            hidden: LSTM hidden state
            action_mask: Action mask

        Returns:
            log_probs, values, entropy
        """
        action_logits, values, _ = self.forward(obs, hidden, action_mask)

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


class PPOTrainer:
    """
    PPO trainer for trajectory policy.
    """

    def __init__(self, policy: TrajectoryPolicy, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5):
        """
        Initialize PPO trainer.

        Args:
            policy: Policy network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip range
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping threshold
        """
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool], next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for final state

        Returns:
            advantages, returns
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + np.array(values)
        return advantages, returns

    def update(self, rollout: Dict[str, torch.Tensor], n_epochs: int = 4,
               batch_size: int = 64) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            rollout: Dictionary with observations, actions, log_probs, values, advantages, returns
            n_epochs: Number of update epochs
            batch_size: Mini-batch size

        Returns:
            Training statistics
        """
        obs = rollout['observations']
        actions = rollout['actions']
        old_log_probs = rollout['log_probs']
        advantages = rollout['advantages']
        returns = rollout['returns']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.size(0)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(n_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
