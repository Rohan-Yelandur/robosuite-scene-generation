import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import torch

def train_ppo_lift(policy_type):
    device = "cpu" if policy_type == "mlp" else "cuda"
    
    use_camera_obs = policy_type == "cnn"
    
    env = robosuite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=use_camera_obs,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
    )

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