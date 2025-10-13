
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
# stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def train_ppo_lift():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
    )

    # Wrap the environment
    env = GymWrapper(env)
    env = DummyVecEnv([lambda: env])

    # Define the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("ppo_lift")

    # Close the environment
    env.close()


def main():
    train_ppo_lift()

if __name__ == "__main__":
    main()