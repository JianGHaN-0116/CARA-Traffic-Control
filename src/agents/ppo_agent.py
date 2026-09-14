"""
PPO agent wrapper using Stable-Baselines3.
"""
import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def load_ppo_config(config_path="configs/drl_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_ppo_agent(env, config_path="configs/drl_config.yaml", tensorboard_log=None):
    """Create a PPO agent with configured hyperparameters."""
    cfg = load_ppo_config(config_path)
    ppo_cfg = cfg.get("ppo", {})

    model = PPO(
        policy=ppo_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 128),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model


def train_ppo(env, save_path, config_path="configs/drl_config.yaml",
              total_timesteps=None):
    """Train PPO agent and save."""
    cfg = load_ppo_config(config_path)
    ppo_cfg = cfg.get("ppo", {})
    timesteps = total_timesteps or ppo_cfg.get("total_timesteps", 300000)

    tb_log = os.path.dirname(save_path) + "/tensorboard/"
    model = create_ppo_agent(env, config_path, tensorboard_log=tb_log)

    print(f"Training PPO for {timesteps} timesteps ...")
    model.learn(total_timesteps=timesteps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"PPO saved: {save_path}")

    return model
