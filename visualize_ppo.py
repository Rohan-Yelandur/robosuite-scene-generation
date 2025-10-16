import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO

def visualize_ppo_lift():
    """Visualize the trained PPO policy on the Lift task."""
    # Create environment with camera view enabled
    env = robosuite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,  # Enable on-screen rendering
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
    )

    env = GymWrapper(env)
    
    # Load the trained model
    model = PPO.load("./policy/ppo_lift", env=env)
    
    # Run episodes with the learned policy
    num_episodes = 3
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done:
            # Use the trained policy to predict actions
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        
        print(f"Episode {episode + 1} completed. Total reward: {total_reward:.2f}")
    
    env.close()
    print("Visualization complete!")

if __name__ == "__main__":
    visualize_ppo_lift()
