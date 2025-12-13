import argparse
import os
import json
import math
from pathlib import Path
import types

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robomimic.config.base_config import config_factory as ConfigFactory

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback

IMAGE_SIZE = 84

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
        return observation.reshape(self.H, self.W, self.C).transpose(2, 0, 1).astype(np.uint8)

class LiftRewardWrapper(gym.Wrapper):
    """Adds dense lift shaping while still highlighting fast, successful lifts."""
    def __init__(self, env, lift_bonus=100.0, success_bonus=50.0, delta_cap=0.01):
        super().__init__(env)
        self._lift_bonus = lift_bonus
        self._success_bonus = success_bonus
        self._delta_cap = delta_cap
        self._last_height = None
        self._table_height = None
        self._lifted = False
        self._grasp_height = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_height = self._cube_height()
        self._table_height = self._get_table_height()
        self._lifted = False
        self._grasp_height = None
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cube_h = self._cube_height()
        delta = cube_h - (self._last_height or cube_h)
        self._last_height = cube_h

        if info.get("is_grasped", False) and not self._lifted:
            if self._grasp_height is None:
                self._grasp_height = cube_h
            positive_delta = max(0.0, delta)
            capped_delta = min(positive_delta, self._delta_cap)
            reward += self._lift_bonus * capped_delta

        table_h = self._table_height if self._table_height is not None else self._get_table_height()
        if not self._lifted and cube_h > table_h + 0.04:
            reward += self._success_bonus
            self._lifted = True
            info["lift_success"] = True
            
        return obs, reward, terminated, truncated, info

    def _cube_height(self):
        rs_env = self._robosuite_env()
        return rs_env.sim.data.body_xpos[rs_env.cube_body_id][2]

    def _get_table_height(self):
        rs_env = self._robosuite_env()
        arena = getattr(rs_env.model, "mujoco_arena", None)
        if arena is not None:
            return arena.table_offset[2]
        # Fallback to default lift table height
        return 0.8

    def _robosuite_env(self):
        base_env = self.env.unwrapped
        return getattr(base_env, "env", base_env)

def make_env_cnn(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env with CNN observations.
    """
    def _init():
        env_seed = seed + rank
        set_random_seed(env_seed)
        env = GymWrapper(
            suite.make(
                env_id,
                robots=["Panda"],
                reward_shaping=True,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_object_obs=True,
                use_camera_obs=True,
                camera_names=['agentview'],
                camera_heights=IMAGE_SIZE,
                camera_widths=IMAGE_SIZE,
                control_freq=20,
            ),
            keys=['agentview_image']
        )
        env = AgentviewWrapper(env, shape=(IMAGE_SIZE, IMAGE_SIZE))
        env = LiftRewardWrapper(env)
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            seed_fn(env_seed)
        else:
            env.reset(seed=env_seed)
        return env
    return _init


def make_env_mlp(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env with low-dim observations.
    """
    def _init():
        env_seed = seed + rank
        set_random_seed(env_seed)
        env = GymWrapper(
            suite.make(
                env_id,
                robots=["Panda"],
                reward_shaping=True,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_object_obs=True,
                use_camera_obs=False,
                control_freq=20,
            ),
            keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
        )
        env = LiftRewardWrapper(env)
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            seed_fn(env_seed)
        else:
            env.reset(seed=env_seed)
        return env
    return _init


def build_vector_env(env_id, num_cpu, seed, start_method, policy_type='cnn'):
    if policy_type == 'cnn':
        env_fns = [make_env_cnn(env_id, i, seed) for i in range(num_cpu)]
    else:
        env_fns = [make_env_mlp(env_id, i, seed) for i in range(num_cpu)]
    
    base_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecMonitor(base_env)
    return vec_env

def save_robomimic_checkpoint(model, save_path, env_name):
    config = ConfigFactory(algo_name="bc") 
    config.experiment.rollout.horizon = 400
    
    with config.values_unlocked():
        config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        config.observation.modalities.obs.rgb = ["agentview_image"]
    
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
            "camera_heights": IMAGE_SIZE,
            "camera_widths": IMAGE_SIZE,
            "control_freq": 20,
            "reward_shaping": True,
        }
    }

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Square", choices=["Square", "Lift", "Thread", "Stack", "Can"], help="Environment to train on")
    parser.add_argument("--policy", type=str, default="cnn", choices=["cnn", "mlp"], help="Policy type to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_cpu", type=int, default=8, help="Number of CPU cores")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--save_freq", type=int, default=100_000, help="Checkpoint frequency")
    parser.add_argument("--chunk_size", type=int, default=200_000, help="Timesteps to run before recycling env workers")
    parser.add_argument("--start_method", type=str, default="spawn", choices=["spawn", "forkserver", "fork"], help="multiprocessing start method for SubprocVecEnv")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (e.g., policy/checkpoints/Lift_mlp/ppo_model_500000_steps.zip)")
    parser.add_argument("--resume_vecnormalize", type=str, default=None, help="Path to VecNormalize statistics to resume from")
    args = parser.parse_args()

    env_map = {
        "Square": "NutAssemblySquare",
        "Lift": "Lift",
        "Thread": "NutAssemblyRound",
        "Stack": "Stack",
        "Can": "PickPlaceCan"
    }
    

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    env_id = env_map[args.env]
    print(f"Training on {env_id} with {args.num_cpu} CPUs, {args.policy.upper()} policy, and seed {args.seed}")

    policy_dir = Path("policy")
    policy_dir.mkdir(exist_ok=True)
    checkpoint_dir = policy_dir / "checkpoints" / f"{args.env}_{args.policy}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = policy_dir / "tensorboard_logs" / f"{args.env}_{args.policy}"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    resume_vecnormalize_supplied = False
    derived_vecnormalize_path = None
    if args.resume:
        candidate_path = Path(args.resume).with_name(Path(args.resume).stem + "_vecnormalize.pkl")
        if candidate_path.exists():
            derived_vecnormalize_path = candidate_path

    if args.resume_vecnormalize:
        vecnormalize_path = Path(args.resume_vecnormalize)
        resume_vecnormalize_supplied = True
    elif derived_vecnormalize_path:
        vecnormalize_path = derived_vecnormalize_path
        resume_vecnormalize_supplied = True
    else:
        vecnormalize_dir = policy_dir / "vecnormalize"
        vecnormalize_dir.mkdir(parents=True, exist_ok=True)
        vecnormalize_path = vecnormalize_dir / f"{args.env}_{args.policy}.pkl"

    vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)

    if resume_vecnormalize_supplied and not vecnormalize_path.exists():
        print(f"Warning: VecNormalize stats not found at {vecnormalize_path}. A new normalizer will be initialized.")
        resume_vecnormalize_supplied = False

    base_n_steps = 2048
    cpu_scale = max(1, math.ceil(args.num_cpu / 8))
    adaptive_n_steps = base_n_steps * cpu_scale
    print(f"Using {adaptive_n_steps} rollout steps per environment (~{adaptive_n_steps * args.num_cpu} samples/update)")

    total_trained = 0
    model = None

    # Resume from checkpoint if specified
    if args.resume:
        import re
        # Extract timesteps from checkpoint filename (e.g., ppo_model_500000_steps.zip)
        match = re.search(r'_(\d+)_steps', args.resume)
        if match:
            total_trained = int(match.group(1))
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"Starting from timestep {total_trained}, training to {args.timesteps}")
        else:
            print(f"Warning: Could not extract timestep from checkpoint name '{args.resume}', starting from 0")

    while total_trained < args.timesteps:
        chunk_steps = min(args.chunk_size, args.timesteps - total_trained)
        vec_env = build_vector_env(
            env_id=env_id,
            num_cpu=args.num_cpu,
            seed=args.seed,
            start_method=args.start_method,
            policy_type=args.policy,
        )

        should_load_vecnormalize = vecnormalize_path.exists() and (total_trained > 0 or resume_vecnormalize_supplied)
        if should_load_vecnormalize:
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            vec_env.training = True
            vec_env.norm_reward = True
            vec_env.norm_obs = True
        else:
            vec_env = VecNormalize(
                vec_env,
                training=True,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // args.num_cpu, 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_model"
        )

        if model is None:
            policy_name = "CnnPolicy" if args.policy == "cnn" else "MlpPolicy"
            device = "auto" if args.policy == "cnn" else "cpu"
            
            if args.resume:
                # Load model from checkpoint
                model = PPO.load(
                    args.resume,
                    env=vec_env,
                    verbose=1,
                    tensorboard_log=str(tensorboard_dir),
                    device=device,
                    n_steps=adaptive_n_steps
                )
                # Set the num_timesteps so checkpoints are named correctly
                model.num_timesteps = total_trained
            else:
                if args.policy == 'cnn':
                    model = PPO(
                        policy_name,
                        vec_env,
                        verbose=1,
                        n_steps=adaptive_n_steps,
                        batch_size=256,
                        ent_coef=0.01,
                        seed=args.seed,
                        tensorboard_log=str(tensorboard_dir),
                        device=device
                    )
                else:
                    model = PPO(
                        policy_name,
                        vec_env,
                        verbose=1,
                        n_steps=adaptive_n_steps,
                        batch_size=256,
                        ent_coef=0.05,
                        target_kl=0.03,
                        seed=args.seed,
                        tensorboard_log=str(tensorboard_dir),
                        device=device,
                        policy_kwargs=dict(
                            net_arch=dict(pi=[256, 256], vf=[256, 256]),
                        )
                    )
        else:
            model.set_env(vec_env)

        print(f"[Chunk] Training next {chunk_steps} timesteps (progress {total_trained}/{args.timesteps})")
        model.learn(total_timesteps=chunk_steps, callback=checkpoint_callback, reset_num_timesteps=False)
        vec_env.save(str(vecnormalize_path))
        vec_env.close()
        total_trained += chunk_steps
        print(f"Completed {total_trained}/{args.timesteps} timesteps. Recycled env workers to release renderer memory.")

    final_model_path = policy_dir / f"{args.policy}_ppo_{args.env.lower()}"

    model.save(str(final_model_path))
    save_robomimic_checkpoint(model, f"./policy/{args.policy}_ppo_{args.env.lower()}_robomimic.pth", env_name=env_id)

if __name__ == "__main__":
    main()