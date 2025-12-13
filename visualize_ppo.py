import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
import os
import imageio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse

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

def visualize_ppo(env_name, env_id, policy_type='cnn'):
    """Visualize the trained PPO policy on the specified task."""
    # Create environment with offscreen rendering for video capture
    if policy_type == 'cnn':
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
    else:
        env = GymWrapper(
            robosuite.make(
                env_id,
                robots=["Panda"],
                reward_shaping=True,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_object_obs=True,
                use_camera_obs=False,
                control_freq=20,
            ),
            keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
        )
    
    # Load the trained model
    model = PPO.load(f"./policy/{policy_type}_ppo_{env_name.lower()}", env=env)
    
    # Create videos directory
    output_dir = f"videos/{env_name}_{policy_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run episodes with the learned policy
    num_episodes = 3
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        frames = []
        
        print(f"[{env_name}] Episode {episode + 1}/{num_episodes}")
        
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
        print(f"[{env_name}] Episode {episode + 1} completed. Total reward: {total_reward:.2f}")
        print(f"[{env_name}] Video saved to: {video_path}")
    
    env.close()
    print(f"\n[{env_name}] Visualization complete! All videos saved in '{output_dir}/' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, nargs='+', default=["square"], 
                        choices=["square", "lift", "thread", "stack", "can"], 
                        help="Environment(s) to visualize")
    parser.add_argument("--policy", type=str, default="cnn", choices=["cnn", "mlp"], 
                        help="Policy type to visualize")
    args = parser.parse_args()
    
    # Map short names to Robosuite env IDs
    env_map = {
        "square": "NutAssemblySquare",
        "lift": "Lift",
        "thread": "NutAssemblyRound",
        "stack": "Stack",
        "can": "PickPlaceCan"
    }
    
    for env_name in args.env:
        env_id = env_map[env_name]
        print(f"\n{'='*60}")
        print(f"Visualizing {env_name} ({env_id}) with {args.policy.upper()} policy")
        print(f"{'='*60}")
        visualize_ppo(env_name, env_id, args.policy)