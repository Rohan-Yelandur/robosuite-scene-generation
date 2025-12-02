import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import torch
import json
from robomimic.config.base_config import config_factory as ConfigFactory

image_size = 84  # Reduced from 84 to save memory

class AgentviewWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.H, self.W = shape
        self.C = 3
        # SB3 expects (C, H, W) for images
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.C, self.H, self.W), dtype=np.uint8
        )
    
    def observation(self, observation):
        # Reshape to (H, W, C) then transpose to (C, H, W)
        # Ensure it is uint8
        return observation.reshape(self.H, self.W, self.C).transpose(2, 0, 1).astype(np.uint8)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = GymWrapper(
            robosuite.make(
                env_id,
                robots=["Panda"],
                reward_shaping=True,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_object_obs=True,
                use_camera_obs=True,
                camera_names=['agentview'],
                camera_heights=image_size,
                camera_widths=image_size,
                control_freq=20,
            ),
            keys=['agentview_image']
        )
        env = AgentviewWrapper(env, shape=(image_size, image_size))
        return env
    set_random_seed(seed)
    return _init

def save_robomimic_checkpoint(model, save_path, env_name="NutAssemblySquare"):
    # 1. Create Config
    # We use BC config as a template, though we won't use the algo part
    config = ConfigFactory(algo_name="bc") 
    config.experiment.rollout.horizon = 400
    
    # Ensure observation modalities are set correctly in the config
    with config.values_unlocked():
        config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        config.observation.modalities.obs.rgb = ["agentview_image"]
    
    # 2. Create Env Metadata
    env_metadata = {
        "env_name": env_name,
        "type": 1, # EnvType.ROBOSUITE_TYPE
        "env_kwargs": {
            "env_name": env_name,
            "robots": ["Panda"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "camera_names": ['agentview'],
            "camera_heights": image_size,
            "camera_widths": image_size,
            "control_freq": 20,
            "reward_shaping": True,
        }
    }

    # 3. Create Checkpoint Dict
    ckpt_dict = {
        "algo_name": "SB3_PPO", # Custom algo name recognized by train_discrete.py
        "config": config.dump(), # JSON string
        "model": model.policy.state_dict(),
        "env_metadata": env_metadata,
        "shape_metadata": {
            "use_images": True,
            "use_depths": False,
            "all_shapes": {}, 
            "ac_dim": 0 
        }
    }
    
    torch.save(ckpt_dict, save_path)
    print(f"Saved robomimic-compatible checkpoint to {save_path}")

def train_ppo_square():
    num_cpu = 2
    env_id = "NutAssemblySquare"
    
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    n_steps = 2048
    batch_size = 256
    seed = np.random.randint(0, 1000000)
    print(f"Training on {num_cpu} CPUs with seed {seed}")
    print(f"n_steps per env: {n_steps}")    
    print(f"Total buffer size: {n_steps * num_cpu}")

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=n_steps,      # Dynamic n_steps based on num_cpu
        batch_size=batch_size,
        seed=seed,
        tensorboard_log="./policy/tensorboard_logs/"
    )
    model.learn(total_timesteps=2_000_000)
    
    os.makedirs("./policy", exist_ok=True)
    # Save standard SB3 format
    model.save("./policy/cnn_ppo_square")
    
    # Save Robomimic format
    save_robomimic_checkpoint(model, "./policy/cnn_ppo_square_robomimic.pth", env_name=env_id)
    
    env.close()

def main():
    train_ppo_square()

if __name__ == "__main__":
    main()