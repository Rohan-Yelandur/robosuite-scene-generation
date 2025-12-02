import os
import argparse
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# robomimic imports
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# Local import of our environment
from env.can_env import CanEnv
from env.lift_env import LiftEnv
from env.thread_env import ThreadEnv
from env.square_env import SquareEnv
from env.stack_env import StackEnv

TASK_ENVS = {
    "can": CanEnv,
    "lift": LiftEnv,
    "thread": ThreadEnv,
    "square": SquareEnv,
    "stack": StackEnv,
}

class RobomimicPPOWrapper:
    """
    Wraps an SB3 PPO policy to mimic the interface of a Robomimic agent.
    """
    def __init__(self, policy):
        self.policy = policy

    def start_episode(self):
        pass

    def set_eval(self):
        self.policy.set_training_mode(False)

    def set_train(self):
        self.policy.set_training_mode(True)

    def get_action(self, obs_dict):
        # Extract image from robomimic observation dict
        if isinstance(obs_dict, dict) and 'agentview_image' in obs_dict:
            img = obs_dict['agentview_image']
            # Ensure (C, H, W)
            if img.ndim == 3 and img.shape[-1] == 3: # (H, W, C)
                img = img.transpose(2, 0, 1)
            # Ensure uint8
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            action, _ = self.policy.predict(img, deterministic=True)
            return action
        
        # Fallback if obs is already correct format
        action, _ = self.policy.predict(obs_dict, deterministic=True)
        return action

    def __call__(self, ob, goal=None):
        return self.get_action(ob)
    
    def state_dict(self):
        return self.policy.state_dict()

def main(args):
    # Prepare directories
    log_dir = f"{args.task_name}_logs"
    data_dir = f"{args.task_name}_rl_data"

    # Create required directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, args.save_path), exist_ok=True)

    if args.collect_data:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, args.save_path), exist_ok=True)

    # Print arguments for clarity
    print(f"Agent Path: {args.agent_path}")
    print(f"Video Record: {args.video_record}")
    print(f"RL Update Step: {args.rl_update_step}")
    print(f"RL Timesteps: {args.rl_timesteps}")
    print(f"Collect Data: {args.collect_data}")
    print(f"Render: {args.render}")

    # Load the policy checkpoint
    ckpt_path = args.agent_path
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Check if it is our custom SB3 PPO checkpoint
    try:
        ckpt_dict = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        ckpt_dict = {}

    if isinstance(ckpt_dict, dict) and ckpt_dict.get("algo_name") == "SB3_PPO":
        print("Detected SB3_PPO checkpoint. Loading custom policy...")
        
        # Hack: Temporarily set algo_name to 'bc' so robomimic can load the config/env
        ckpt_dict["algo_name"] = "bc"

        # Initialize ObsUtils so that the environment can properly process observations
        import robomimic.utils.obs_utils as ObsUtils
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        
        # Patch config to ensure agentview_image is in rgb modalities
        # This is necessary because the dummy config created in train_ppo.py might not have it set
        with config.values_unlocked():
            config.observation.modalities.obs.rgb = ["agentview_image"]

        ObsUtils.initialize_obs_utils_with_config(config)

        # Create environment first to get action space
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=None,
            render=args.render,
            render_offscreen=True,
            verbose=True
        )
        
        # Define observation space expected by the CNN policy (3, 84, 84)
        obs_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
        
        # Create action space manually since EnvRobosuite doesn't have it
        # Robosuite actions are continuous and typically normalized [-1, 1]
        action_dim = env.action_dimension
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Instantiate policy
        policy_net = ActorCriticCnnPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lambda _: 0.0
        )
        
        policy_net.load_state_dict(ckpt_dict["model"])
        policy_net.to(device)
        
        policy = RobomimicPPOWrapper(policy_net)
        
    else:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

        # Create environment from the checkpoint
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=None,
            render=args.render,
            render_offscreen=True,  # Enable offscreen rendering for image collection
            verbose=True
        )

    # Determine horizon (if not specified, read from config)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    rollout_horizon = config.experiment.rollout.horizon

        # Fix old controller configs that are incompatible with newer robosuite
    if 'env_metadata' in ckpt_dict and 'env_kwargs' in ckpt_dict['env_metadata']:
        env_kwargs = ckpt_dict['env_metadata']['env_kwargs']
        if 'controller_configs' in env_kwargs:
            ctrl = env_kwargs['controller_configs']
            if isinstance(ctrl, dict) and 'type' in ctrl and 'body_parts' not in ctrl:
                # Old format - remove it to use defaults
                del ckpt_dict['env_metadata']['env_kwargs']['controller_configs']
                print("[INFO] Removed incompatible old controller config, using defaults")

    # Instantiate our custom environment
    env_class = TASK_ENVS.get(args.task_name)
    if env_class is None:
        raise ValueError(f"Unknown task name: {args.task_name}")

    def make_rl_env():
        return env_class(
            env=env,
            policy=policy,
            rollout_horizon=rollout_horizon,
            video_record=args.video_record,
            collect_data=args.collect_data,
            save_path=args.save_path,
            device=device
        )

    # Create a VecEnv for stable-baselines
    vec_env = DummyVecEnv([make_rl_env])

    # Create the PPO model
    # Ensure batch_size divides n_steps to avoid warnings
    # Default n_steps is 300. 60 is a good divisor close to default 64.
    batch_size = 60
    if args.rl_update_step % batch_size != 0:
        batch_size = args.rl_update_step # Fallback to full batch if not divisible

    ppo_model = PPO("CnnPolicy", vec_env, verbose=1, n_steps=args.rl_update_step, batch_size=batch_size)
    ppo_model.learn(total_timesteps=args.rl_timesteps)

    # Save the trained model
    os.makedirs("trained_rl_models", exist_ok=True)
    save_model_path = f"trained_rl_models/{args.save_path}_ppo_model_{args.rl_timesteps}"
    ppo_model.save(save_model_path)
    print(f"Training completed. Model saved to {save_model_path}")

    # Example: get action log probabilities
    observation = vec_env.reset()
    observation = torch.tensor(observation).float().to(ppo_model.device)
    with torch.no_grad():
        dist = ppo_model.policy.get_distribution(observation)
    n_actions = vec_env.action_space.n
    all_actions = torch.arange(n_actions).to(ppo_model.device)
    log_probs = dist.log_prob(all_actions)
    print("Log Probabilities of All Actions:", log_probs)

    if args.save_logs:
        log_file = os.path.join(log_dir, args.save_path, "log_prob.txt")
        with open(log_file, "a") as file:
            file.write(str(log_probs) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent to find failure modes for robosuite task.")

    parser.add_argument("--task_name", type=str, default="can", help="Name of the task/environment. Used to form directory names (e.g., can_logs, can_rl_data).")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to load the agent checkpoint (.pt file).")
    parser.add_argument("--rl_timesteps", type=int, default=300, help="Number of training timesteps (default: 300)")
    parser.add_argument("--rl_update_step", type=int, default=300, help="Number of steps per PPO update (default: 300)")
    parser.add_argument("--video_record", action="store_true", help="Record training video if set.")
    parser.add_argument("--render", action="store_true", help="Render rollout if set.")
    parser.add_argument("--collect_data", action="store_true", help="If set, collect image data during rollouts.")
    parser.add_argument("--save_path", type=str, default="default_run", help="Folder name to save logs and data.")
    parser.add_argument("--save_logs", action="store_true", help="Save logs.")

    args = parser.parse_args()
    main(args)
