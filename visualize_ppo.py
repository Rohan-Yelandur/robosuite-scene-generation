import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
import os
import imageio
import gymnasium as gym
from gymnasium import spaces
import numpy as np

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

def visualize_ppo_square():
    """Visualize the trained PPO policy on the NutAssemblySquare task."""
    # Create environment with offscreen rendering for video capture
    env = GymWrapper(
        robosuite.make(
            "NutAssemblySquare",
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
    
    # Load the trained model
    model = PPO.load("./policy/cnn_ppo_square", env=env)
    
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
    visualize_ppo_square()