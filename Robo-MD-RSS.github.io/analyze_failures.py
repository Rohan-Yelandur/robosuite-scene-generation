#!/usr/bin/env python3
"""
RoboMD Failure Analysis Script

Analyzes training results from:
1. train_discrete.py - Discrete action RL results
2. train_continuous.py - Continuous latent action RL results
3. Computes success rates and generates visualizations

Usage:
    # For discrete training results:
    python analyze_failures.py --task lift --mode discrete --data_dir lift_rl_data/default_run
    
    # For continuous training results:
    python analyze_failures.py --task lift --mode continuous --log_dir lift_pipeline/latent_ppo_logs
    
    # For log probability analysis:
    python analyze_failures.py --task lift --mode logprob --log_file lift_logs/default_run/log_prob.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch

# Import action dictionaries from existing config file
from configs.action_dicts import ACTION_DICTS


def load_actions(actions_file):
    """Load action indices from actions.txt"""
    with open(actions_file, 'r') as f:
        actions = [int(line.strip()) for line in f if line.strip().isdigit()]
    return actions


def load_success_rates(success_file):
    """Load success rates from success_rate.txt"""
    with open(success_file, 'r') as f:
        successes = [float(line.strip()) > 0.5 for line in f if line.strip()]
    return successes


def load_stats(stats_file):
    """Load episode statistics from stats.txt and return list of success flags"""
    successes = []
    if not os.path.exists(stats_file):
        print(f"Warning: stats.txt not found at {stats_file}")
        return None
    
    with open(stats_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    stats_dict = eval(line.strip())
                    success = stats_dict.get('Success_Rate', 0.0)
                    successes.append(success > 0.5)
                except:
                    successes.append(False)
    return successes


def load_episode_stats(stats_file):
    """Load full episode statistics from episode_stats.txt"""
    stats_list = []
    if not os.path.exists(stats_file):
        return None
    
    with open(stats_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    stats_dict = eval(line.strip())
                    stats_list.append(stats_dict)
                except:
                    pass
    return stats_list


def load_log_probs(log_prob_file):
    """Load log probabilities from log_prob.txt (from train_discrete.py)"""
    if not os.path.exists(log_prob_file):
        return None
    
    with open(log_prob_file, 'r') as f:
        content = f.read()
        # Parse tensor string
        try:
            # Remove tensor wrapper and device info
            content = content.replace('tensor(', '').replace(')', '')
            content = content.split(', device=')[0]
            # Parse the values
            values = eval(content)
            return np.array(values)
        except:
            return None


def analyze_discrete_training(data_dir, task_name):
    """Analyze discrete RL training results"""
    actions_file = os.path.join(data_dir, "actions.txt")
    success_file = os.path.join(data_dir, "success_rate.txt")
    
    if not os.path.exists(actions_file) or not os.path.exists(success_file):
        print(f"Error: Missing required files in {data_dir}")
        return None
    
    print(f"Loading discrete training data from {data_dir}...")
    actions = load_actions(actions_file)
    successes = load_success_rates(success_file)
    
    if len(actions) != len(successes):
        print(f"Warning: Mismatch between actions ({len(actions)}) and successes ({len(successes)})")
        min_len = min(len(actions), len(successes))
        actions = actions[:min_len]
        successes = successes[:min_len]
    
    # Compute statistics
    total_episodes = len(actions)
    successful_episodes = sum(successes)
    failed_episodes = total_episodes - successful_episodes
    overall_success_rate = (successful_episodes / total_episodes) * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Successful: {successful_episodes} ({overall_success_rate:.1f}%)")
    print(f"  Failed: {failed_episodes} ({100 - overall_success_rate:.1f}%)")
    
    # Analyze per-action statistics
    action_dict = ACTION_DICTS.get(task_name, {})
    action_stats = {}
    
    for action_id in set(actions):
        # Find all episodes with this action
        action_mask = [a == action_id for a in actions]
        action_successes = [s for a, s in zip(actions, successes) if a == action_id]
        
        total_uses = len(action_successes)
        num_successes = sum(action_successes)
        num_failures = total_uses - num_successes
        success_rate = (num_successes / total_uses) * 100 if total_uses > 0 else 0
        
        action_stats[action_id] = {
            'total_uses': total_uses,
            'successes': num_successes,
            'failures': num_failures,
            'success_rate': success_rate,
            'failure_rate': 100 - success_rate
        }
    
    return {
        'mode': 'discrete',
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes,
        'failed_episodes': failed_episodes,
        'overall_success_rate': overall_success_rate,
        'action_stats': action_stats,
        'action_dict': action_dict,
        'task_name': task_name
    }


def analyze_continuous_training(log_dir, task_name):
    """Analyze continuous RL training results"""
    actions_file = os.path.join(log_dir, "actions.txt")
    stats_file = os.path.join(log_dir, "stats.txt")
    
    if not os.path.exists(actions_file):
        print(f"Error: actions.txt not found at {actions_file}")
        return None
    
    print(f"Loading continuous training data from {log_dir}...")
    actions = load_actions(actions_file)
    successes = load_stats(stats_file)
    
    if successes is None:
        print("Warning: No stats file found, assuming all episodes failed")
        successes = [False] * len(actions)
    
    if len(actions) != len(successes):
        print(f"Warning: Mismatch between actions ({len(actions)}) and successes ({len(successes)})")
        min_len = min(len(actions), len(successes))
        actions = actions[:min_len]
        successes = successes[:min_len]
    
    # Compute statistics
    total_episodes = len(actions)
    successful_episodes = sum(successes)
    failed_episodes = total_episodes - successful_episodes
    overall_success_rate = (successful_episodes / total_episodes) * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Successful: {successful_episodes} ({overall_success_rate:.1f}%)")
    print(f"  Failed: {failed_episodes} ({100 - overall_success_rate:.1f}%)")
    
    # Analyze per-action statistics
    action_dict = ACTION_DICTS.get(task_name, {})
    action_stats = {}
    
    for action_id in set(actions):
        action_successes = [s for a, s in zip(actions, successes) if a == action_id]
        
        total_uses = len(action_successes)
        num_successes = sum(action_successes)
        num_failures = total_uses - num_successes
        success_rate = (num_successes / total_uses) * 100 if total_uses > 0 else 0
        
        action_stats[action_id] = {
            'total_uses': total_uses,
            'successes': num_successes,
            'failures': num_failures,
            'success_rate': success_rate,
            'failure_rate': 100 - success_rate
        }
    
    return {
        'mode': 'continuous',
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes,
        'failed_episodes': failed_episodes,
        'overall_success_rate': overall_success_rate,
        'action_stats': action_stats,
        'action_dict': action_dict,
        'task_name': task_name
    }


def analyze_log_probabilities(log_prob_file, task_name):
    """Analyze log probabilities from trained discrete PPO model"""
    print(f"Loading log probabilities from {log_prob_file}...")
    log_probs = load_log_probs(log_prob_file)
    
    if log_probs is None:
        print("Error: Could not load log probabilities")
        return None
    
    action_dict = ACTION_DICTS.get(task_name, {})
    
    # Convert log probabilities to probabilities
    probs = np.exp(log_probs)
    probs = probs / probs.sum()  # Normalize
    
    # Create action statistics based on probabilities
    action_stats = {}
    for action_id, prob in enumerate(probs):
        action_stats[action_id] = {
            'log_prob': log_probs[action_id],
            'probability': prob * 100,  # As percentage
            'preference_rank': 0  # Will be set later
        }
    
    # Rank by probability
    sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['probability'], reverse=True)
    for rank, (action_id, stats) in enumerate(sorted_actions, 1):
        action_stats[action_id]['preference_rank'] = rank
    
    return {
        'mode': 'logprob',
        'action_stats': action_stats,
        'action_dict': action_dict,
        'task_name': task_name,
        'log_probs': log_probs,
        'probs': probs
    }


def create_short_labels(action_dict, action_list):
    """Create concise labels for visualization"""
    label_map = {}
    
    for action_id in action_list:
        description = action_dict.get(action_id, f"Action {action_id}")
        
        # Create short labels based on common patterns
        if "cylinder color to" in description.lower():
            color = description.split("to")[-1].strip().rstrip(".")
            label_map[action_id] = f"Cyl {color.capitalize()}"
        elif "cube color to" in description.lower():
            color = description.split("to")[-1].strip().rstrip(".")
            label_map[action_id] = f"Cube {color.capitalize()}"
        elif "can color to" in description.lower():
            color = description.split("to")[-1].strip().rstrip(".")
            label_map[action_id] = f"Can {color.capitalize()}"
        elif "table color to" in description.lower():
            color = description.split("to")[-1].strip().rstrip(".")
            label_map[action_id] = f"Table {color.capitalize()}"
        elif "robot color to" in description.lower():
            color = description.split("to")[-1].strip().rstrip(".")
            label_map[action_id] = f"Robot {color.capitalize()}"
        elif "lighting color to" in description.lower():
            color = description.split("to")[-1].strip().rstrip(".")
            label_map[action_id] = f"Light {color.capitalize()}"
        elif "resize" in description.lower() and "cylinder" in description.lower():
            label_map[action_id] = f"Cyl Size {action_id}"
        elif "resize" in description.lower() and "cube" in description.lower():
            label_map[action_id] = f"Cube Size {action_id}"
        elif "resize" in description.lower() and "table" in description.lower():
            label_map[action_id] = f"Table Size {action_id}"
        elif "resize" in description.lower() and "box" in description.lower():
            label_map[action_id] = f"Box Size {action_id}"
        elif "no perturbation" in description.lower():
            label_map[action_id] = "No Change"
        else:
            # Truncate long descriptions
            words = description.split()
            if len(words) > 3:
                label_map[action_id] = " ".join(words[:3]) + "..."
            else:
                label_map[action_id] = description[:20]
    
    return label_map


def create_radar_chart(analysis_results, output_file):
    """Create radar chart visualization"""
    task_name = analysis_results['task_name']
    action_dict = analysis_results['action_dict']
    action_stats = analysis_results['action_stats']
    mode = analysis_results['mode']
    
    # Determine what to plot based on mode
    if mode == 'logprob':
        # Plot probability distribution
        sorted_actions = sorted(action_stats.items(), 
                              key=lambda x: x[1]['probability'], 
                              reverse=True)
        top_n = min(12, len(sorted_actions))
        top_actions = [a[0] for a in sorted_actions[:top_n]]
        values = [action_stats[a]['probability'] for a in top_actions]
        title = f'{task_name.upper()} - Action Selection Probabilities'
        ylabel = 'Probability (%)'
    else:
        # Plot failure rates
        sorted_actions = sorted(action_stats.items(), 
                              key=lambda x: x[1]['failure_rate'], 
                              reverse=True)
        top_n = min(12, len(sorted_actions))
        top_actions = [a[0] for a in sorted_actions[:top_n]]
        values = [action_stats[a]['failure_rate'] for a in top_actions]
        title = f'{task_name.upper()} - Policy Failure Rates ({mode.capitalize()})'
        ylabel = 'Failure Rate (%)'
    
    # Get short labels
    label_map = create_short_labels(action_dict, top_actions)
    categories = [label_map[action] for action in top_actions]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Complete the circle
    values += values[:1]
    angles += angles[:1]
    categories += categories[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    color = 'green' if mode == 'logprob' else 'purple'
    ax.plot(angles, values, 'o-', linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=10)
    
    # Set y-axis limits
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_yticklabels([])
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add title
    ax.set_title(title, size=16, weight='bold', pad=20)
    
    # Add value labels
    threshold = max(values) * 0.1  # Only label values > 10% of max
    for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
        if value > threshold:
            ax.text(angle, value + max(values) * 0.05, f'{value:.1f}%', 
                   ha='center', va='center', size=9, weight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {output_file}")
    plt.close()


def generate_text_report(analysis_results, output_file):
    """Generate comprehensive text report"""
    task_name = analysis_results['task_name']
    action_dict = analysis_results['action_dict']
    action_stats = analysis_results['action_stats']
    mode = analysis_results['mode']
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{task_name.upper()} POLICY ANALYSIS - {mode.upper()} MODE\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        if mode in ['discrete', 'continuous']:
            f.write("Overall Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total episodes: {analysis_results['total_episodes']}\n")
            f.write(f"Successful episodes: {analysis_results['successful_episodes']} "
                   f"({analysis_results['overall_success_rate']:.1f}%)\n")
            f.write(f"Failed episodes: {analysis_results['failed_episodes']} "
                   f"({100 - analysis_results['overall_success_rate']:.1f}%)\n\n")
        
        # Per-action analysis
        f.write("Per-Action Analysis:\n")
        f.write("=" * 80 + "\n\n")
        
        if mode == 'logprob':
            sorted_actions = sorted(action_stats.items(), 
                                  key=lambda x: x[1]['probability'], 
                                  reverse=True)
            
            for rank, (action_id, stats) in enumerate(sorted_actions, 1):
                description = action_dict.get(action_id, f"Unknown action {action_id}")
                f.write(f"{rank}. Action {action_id} - {stats['probability']:.2f}% selection probability\n")
                f.write(f"   Log Probability: {stats['log_prob']:.4f}\n")
                f.write(f"   → {description}\n\n")
        else:
            sorted_actions = sorted(action_stats.items(), 
                                  key=lambda x: x[1]['failure_rate'], 
                                  reverse=True)
            
            for rank, (action_id, stats) in enumerate(sorted_actions, 1):
                description = action_dict.get(action_id, f"Unknown action {action_id}")
                f.write(f"{rank}. Action {action_id}\n")
                f.write(f"   Success Rate: {stats['success_rate']:.1f}% "
                       f"({stats['successes']}/{stats['total_uses']} episodes)\n")
                f.write(f"   Failure Rate: {stats['failure_rate']:.1f}% "
                       f"({stats['failures']}/{stats['total_uses']} episodes)\n")
                f.write(f"   → {description}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Text report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RoboMD training results")
    parser.add_argument("--task", "-t", type=str, required=True,
                       help="Task name (e.g., 'lift', 'can', 'square')")
    parser.add_argument("--mode", "-m", type=str, required=True,
                       choices=['discrete', 'continuous', 'logprob'],
                       help="Analysis mode: discrete, continuous, or logprob")
    
    # Mode-specific arguments
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory for discrete mode (e.g., lift_rl_data/default_run)")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Log directory for continuous mode (e.g., lift_pipeline/latent_ppo_logs)")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log probability file for logprob mode")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.mode == 'discrete' and not args.data_dir:
        parser.error("--data_dir required for discrete mode")
    if args.mode == 'continuous' and not args.log_dir:
        parser.error("--log_dir required for continuous mode")
    if args.mode == 'logprob' and not args.log_file:
        parser.error("--log_file required for logprob mode")
    
    # Run analysis
    if args.mode == 'discrete':
        results = analyze_discrete_training(args.data_dir, args.task)
        default_output_dir = os.path.join(os.path.dirname(args.data_dir), "analysis")
    elif args.mode == 'continuous':
        results = analyze_continuous_training(args.log_dir, args.task)
        default_output_dir = os.path.join(os.path.dirname(args.log_dir), "analysis")
    else:  # logprob
        results = analyze_log_probabilities(args.log_file, args.task)
        default_output_dir = os.path.join(os.path.dirname(args.log_file), "analysis")
    
    if results is None:
        print("Analysis failed!")
        return
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate outputs
    text_output = os.path.join(output_dir, f"{args.task}_{args.mode}_analysis.txt")
    chart_output = os.path.join(output_dir, f"{args.task}_{args.mode}_radar.png")
    
    generate_text_report(results, text_output)
    create_radar_chart(results, chart_output)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Text report: {text_output}")
    print(f"Radar chart: {chart_output}")


if __name__ == "__main__":
    main()
