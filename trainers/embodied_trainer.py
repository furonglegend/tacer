"""
Embodied Agent Trainer
Training loop for PPO-based trajectory explanation generation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from embodied.graph_env import GraphEnv, create_graph_env_from_data
from embodied.ppo_policy import TrajectoryPolicy, PPOTrainer

try:
    from dataset import CORA
except ImportError:
    CORA = None

try:
    from log import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


def collect_rollout(env: GraphEnv, policy: TrajectoryPolicy, n_steps: int,
                    device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Collect rollout data from environment.

    Args:
        env: Graph environment
        policy: Policy network
        n_steps: Number of steps to collect
        device: Torch device

    Returns:
        Rollout dictionary
    """
    observations = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []

    obs, info = env.reset()
    hidden = None

    for _ in range(n_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, value, hidden = policy.get_action(obs_tensor, hidden)

        action_np = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action_np)

        observations.append(obs)
        actions.append(action_np)
        log_probs.append(log_prob.cpu().numpy()[0])
        values.append(value.cpu().numpy()[0])
        rewards.append(reward)
        dones.append(terminated or truncated)

        obs = next_obs

        if terminated or truncated:
            obs, info = env.reset()
            hidden = None

    return {
        'observations': torch.FloatTensor(np.array(observations)),
        'actions': torch.LongTensor(np.array(actions)),
        'log_probs': torch.FloatTensor(np.array(log_probs)),
        'values': torch.FloatTensor(np.array(values)),
        'rewards': rewards,
        'dones': dones
    }


def run_embodied(args):
    """
    Main training function for Embodied Agent.

    Args:
        args: Command line arguments
    """
    logger.info("Starting Embodied Agent Training...")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if args.dataset == 'cora':
        data, texts, n_class = CORA.load(args.seed)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Create environment
    env = create_graph_env_from_data(data, node_id=0)

    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize policy
    policy = TrajectoryPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lstm_layers=2
    ).to(device)

    # Initialize trainer
    trainer = PPOTrainer(
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5
    )

    # Training parameters
    total_steps = 100000
    rollout_steps = 256
    n_epochs = 4
    batch_size = 64
    log_interval = 10

    # Training loop
    n_updates = total_steps // rollout_steps
    all_rewards = []

    for update in range(n_updates):
        # Collect rollout
        rollout = collect_rollout(env, policy, rollout_steps, device)

        # Compute advantages and returns
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(np.array([env._get_observation()])).to(device)
            _, _, next_value, _ = policy.get_action(obs_tensor)
            next_value = next_value.cpu().numpy()[0]

        advantages, returns = trainer.compute_gae(
            rollout['rewards'],
            rollout['values'].numpy().tolist(),
            rollout['dones'],
            next_value
        )

        rollout['advantages'] = torch.FloatTensor(advantages)
        rollout['returns'] = torch.FloatTensor(returns)

        # Move to device
        rollout['observations'] = rollout['observations'].to(device)
        rollout['actions'] = rollout['actions'].to(device)
        rollout['log_probs'] = rollout['log_probs'].to(device)
        rollout['advantages'] = rollout['advantages'].to(device)
        rollout['returns'] = rollout['returns'].to(device)

        # PPO update
        stats = trainer.update(rollout, n_epochs=n_epochs, batch_size=batch_size)

        # Logging
        episode_rewards = sum(rollout['rewards'])
        all_rewards.append(episode_rewards)

        if update % log_interval == 0:
            avg_reward = np.mean(all_rewards[-100:]) if all_rewards else 0
            logger.info(
                f"Update {update}/{n_updates} | "
                f"Reward: {episode_rewards:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Policy Loss: {stats['policy_loss']:.4f} | "
                f"Value Loss: {stats['value_loss']:.4f}"
            )

    # Save model
    ckpt_dir = Path('ckpts')
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), ckpt_dir / 'trajectory_policy.pt')

    logger.info("Embodied Agent training completed!")
    return policy
