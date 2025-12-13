import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import json
from datetime import datetime

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

def evaluate_policy(env_name, env_id, policy_type='cnn', num_trials=50):
    """Evaluate the trained PPO policy on the specified task."""
    # Create environment
    if policy_type == 'cnn':
        env = GymWrapper(
            robosuite.make(
                env_id,
                robots=["Panda"],
                reward_shaping=Truee,
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
    else:
        env = GymWrapper(
            robosuite.make(
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
    
    # Load the trained model
    model_path = f"./policy/{policy_type}_ppo_{env_name.lower()}"
    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        env.close()
        return None
    
    print(f"Evaluating {env_name} with {policy_type.upper()} policy over {num_trials} trials...")
    
    # Track metrics
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    for trial in range(num_trials):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        success = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Check for success
            if hasattr(env.env, 'is_success'):
                success_dict = env.env.is_success()
                if isinstance(success_dict, dict) and 'task' in success_dict:
                    success = success_dict['task']
                elif isinstance(success_dict, bool):
                    success = success_dict
            
            # Add max steps limit
            if steps >= 500:
                done = True
        
        episode_rewards.append(total_reward)
        episode_successes.append(1 if success else 0)
        episode_lengths.append(steps)
        
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{num_trials} - Success: {success}, Reward: {total_reward:.2f}, Steps: {steps}")
    
    env.close()
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = np.mean(episode_successes) * 100
    mean_length = np.mean(episode_lengths)
    
    results = {
        'env_name': env_name,
        'env_id': env_id,
        'policy_type': policy_type,
        'num_trials': num_trials,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'success_rate': float(success_rate),
        'mean_episode_length': float(mean_length),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_successes': [int(s) for s in episode_successes],
        'episode_lengths': [int(l) for l in episode_lengths],
        'timestamp': datetime.now().isoformat()
    }
    
    return results

def save_results(results, output_dir):
    """Save evaluation results to file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    env_name = results['env_name']
    policy_type = results['policy_type']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON with full details
    json_path = output_path / f"eval_{env_name}_{policy_type}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary to text file
    summary_path = output_path / f"eval_{env_name}_{policy_type}_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Environment: {results['env_name']} ({results['env_id']})\n")
        f.write(f"Policy Type: {results['policy_type'].upper()}\n")
        f.write(f"Number of Trials: {results['num_trials']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        f.write(f"Performance Metrics\n")
        f.write(f"-------------------\n")
        f.write(f"Success Rate: {results['success_rate']:.2f}%\n")
        f.write(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Mean Episode Length: {results['mean_episode_length']:.2f} steps\n\n")
        f.write(f"Individual Trial Results\n")
        f.write(f"------------------------\n")
        for i, (r, s, l) in enumerate(zip(results['episode_rewards'], 
                                           results['episode_successes'], 
                                           results['episode_lengths'])):
            f.write(f"Trial {i+1:3d}: Success={s}, Reward={r:7.2f}, Length={l:4d}\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    
    return json_path, summary_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, nargs='+', default=["square"], 
                        choices=["square", "lift", "thread", "stack", "can"], 
                        help="Environment(s) to evaluate")
    parser.add_argument("--policy", type=str, default="cnn", choices=["cnn", "mlp"], 
                        help="Policy type to evaluate")
    parser.add_argument("--trials", type=int, default=50, 
                        help="Number of evaluation trials per environment")
    parser.add_argument("--output_dir", type=str, default="eval_results", 
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    # Map short names to Robosuite env IDs
    env_map = {
        "square": "NutAssemblySquare",
        "lift": "Lift",
        "thread": "NutAssemblyRound",
        "stack": "Stack",
        "can": "PickPlaceCan"
    }
    
    # Evaluate each environment
    all_results = []
    for env_name in args.env:
        env_id = env_map[env_name]
        print(f"\n{'='*60}")
        print(f"Evaluating {env_name} ({env_id}) with {args.policy.upper()} policy")
        print(f"{'='*60}")
        
        results = evaluate_policy(env_name, env_id, args.policy, args.trials)
        
        if results:
            all_results.append(results)
            
            # Print summary
            print(f"\n{env_name} Results:")
            print(f"  Success Rate: {results['success_rate']:.2f}%")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Mean Episode Length: {results['mean_episode_length']:.2f} steps")
            
            # Save individual results
            save_results(results, args.output_dir)
    
    # Print overall summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Overall Summary")
        print(f"{'='*60}")
        for res in all_results:
            print(f"{res['env_name']:10s} - Success: {res['success_rate']:6.2f}% | Reward: {res['mean_reward']:7.2f}")