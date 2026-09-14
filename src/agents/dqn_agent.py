"""
DQN agent wrapper using Stable-Baselines3.
"""
import os
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


def load_dqn_config(config_path="configs/drl_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dqn_agent(env, config_path="configs/drl_config.yaml", tensorboard_log=None):
    """Create a DQN agent with configured hyperparameters."""
    cfg = load_dqn_config(config_path)
    dqn_cfg = cfg.get("dqn", {})

    model = DQN(
        policy=dqn_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=dqn_cfg.get("learning_rate", 1e-4),
        buffer_size=dqn_cfg.get("buffer_size", 100000),
        learning_starts=dqn_cfg.get("learning_starts", 5000),
        batch_size=dqn_cfg.get("batch_size", 128),
        gamma=dqn_cfg.get("gamma", 0.99),
        train_freq=dqn_cfg.get("train_freq", 4),
        target_update_interval=dqn_cfg.get("target_update_interval", 1000),
        exploration_fraction=dqn_cfg.get("exploration_fraction", 0.2),
        exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.05),
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model


def train_dqn(env, save_path, config_path="configs/drl_config.yaml",
              total_timesteps=None, checkpoint_dir=None):
    """Train DQN agent and save."""
    cfg = load_dqn_config(config_path)
    dqn_cfg = cfg.get("dqn", {})
    timesteps = total_timesteps or dqn_cfg.get("total_timesteps", 300000)

    tb_log = os.path.dirname(save_path) + "/tensorboard/"
    model = create_dqn_agent(env, config_path, tensorboard_log=tb_log)

    callbacks = []
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=50000,
            save_path=checkpoint_dir,
            name_prefix="dqn_edge_security",
        ))

    print(f"Training DQN for {timesteps} timesteps ...")
    model.learn(total_timesteps=timesteps, callback=callbacks if callbacks else None)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"DQN saved: {save_path}")

    return model
