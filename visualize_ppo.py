import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
import os
import imageio

def visualize_ppo_lift():
    """Visualize the trained PPO policy on the Lift task."""
    # Create environment with offscreen rendering for video capture
    env = robosuite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,  # Disable on-screen rendering (use offscreen)
        has_offscreen_renderer=True,  # Enable offscreen rendering for video
        use_camera_obs=False,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
    )

    env = GymWrapper(env)
    
    # Load the trained model
    model = PPO.load("./policy/ppo_lift", env=env)
    
    # Create videos directory
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run episodes with the learned policy
    num_episodes = 3
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        frames = []
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done:
            # Capture frame from offscreen renderer
            frame = env.env.sim.render(height=512, width=512, camera_name="frontview")[::-1]
            frames.append(frame)
            
            # Use the trained policy to predict actions
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        # Save video
        video_path = os.path.join(output_dir, f"episode_{episode+1}.mp4")
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Episode {episode + 1} completed. Total reward: {total_reward:.2f}")
        print(f"Video saved to: {video_path}")
    
    env.close()
    print(f"\nVisualization complete! All videos saved in '{output_dir}/' directory")

if __name__ == "__main__":
    visualize_ppo_lift()
