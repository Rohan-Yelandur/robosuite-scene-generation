import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_ppo_lift():
    env = robosuite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
    )

    env = GymWrapper(env)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_lift")
    env.close()

def main():
    train_ppo_lift()

if __name__ == "__main__":
    main()