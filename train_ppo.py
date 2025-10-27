import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import torch
import gymnasium as gym
import numpy as np


class ImageObservationWrapper(gym.ObservationWrapper):
    """Wrapper to extract image observation from dictionary."""
    def __init__(self, env, image_key="agentview_image"):
        super().__init__(env)
        self.image_key = image_key
        # Update observation space to be just the image
        self.observation_space = env.observation_space[image_key]
    
    def observation(self, obs):
        return obs[self.image_key]


def train_ppo_lift(policy_type):
    device = "cpu" if policy_type == "mlp" else "cuda"
    
    use_camera_obs = policy_type == "cnn"
    
    env_config = {
        "env_name": "Lift",
        "robots": "Panda",
        "has_renderer": False,
        "has_offscreen_renderer": use_camera_obs,
        "use_camera_obs": use_camera_obs,
        "reward_shaping": True,
        "horizon": 500,
        "control_freq": 20,
    }
    
    # Add camera configuration for CNN policy
    if use_camera_obs:
        env_config.update({
            "camera_names": "agentview",
            "camera_heights": 84,
            "camera_widths": 84,
        })
    
    env = robosuite.make(**env_config)

    # For CNN, use keys parameter to get image observations
    if use_camera_obs:
        env = GymWrapper(env, keys=["agentview_image"], flatten_obs=False)
        env = ImageObservationWrapper(env, image_key="agentview_image")
    else:
        env = GymWrapper(env)
    
    env = DummyVecEnv([lambda: env])
    
    policy = "MlpPolicy" if policy_type == "mlp" else "CnnPolicy"
    
    model = PPO(policy, env, verbose=1, device=device)
    model.learn(total_timesteps=1_000_000)
    model.save(f"./policy/{policy_type}_ppo_lift")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Train PPO on Lift task")
    parser.add_argument(
        "--policy", 
        type=str, 
        choices=["mlp", "cnn"], 
        default="mlp",
        help="Policy type to use: 'mlp' for MlpPolicy (CPU) or 'cnn' for CnnPolicy (GPU)"
    )
    
    args = parser.parse_args()
    train_ppo_lift(args.policy)

if __name__ == "__main__":
    main()