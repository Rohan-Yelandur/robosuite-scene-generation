#!/usr/bin/env python3
"""
RoboMD Failure Analysis Script

Analyzes the output of train_continuous.py to generate:
1. Text summary of failure probabilities
2. Radar chart visualization of failure modes

Usage:
    python analyze_failures.py --task square --log_dir square_failure_diagnosis/latent_ppo_logs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Import action dictionaries from existing config file
from configs.action_dicts import ACTION_DICTS

def load_actions(actions_file):
    """Load action indices from actions.txt"""
    with open(actions_file, 'r') as f:
        actions = [int(line.strip()) for line in f if line.strip()]
    return actions

def load_stats(stats_file):
    """Load episode statistics from stats.txt and return list of success flags"""
    successes = []
    if not os.path.exists(stats_file):
        print(f"Warning: stats.txt not found at {stats_file}, assuming all episodes failed")
        return None
    
    with open(stats_file, 'r') as f:
        for line in f:
            if line.strip():
                # Parse dict-like string: {'Return': -1.0, 'Horizon': 400, 'Success_Rate': 0.0}
                try:
                    stats_dict = eval(line.strip())
                    success = stats_dict.get('Success_Rate', 0.0)
                    successes.append(success > 0.5)  # Consider >0.5 as success
                except:
                    successes.append(False)
    return successes

def analyze_actions(actions, successes, task_name):
    """Analyze failure rate per action (failures / total times action was used)"""
    action_dict = ACTION_DICTS.get(task_name, {})
    
    # Count total occurrences of each action
    total_action_counts = Counter(actions)
    
    # Filter to only failed episodes
    if successes is not None:
        failed_actions = [action for action, success in zip(actions, successes) if not success]
        print(f"Total episodes: {len(actions)}, Failed episodes: {len(failed_actions)}")
    else:
        # If no stats, assume all failed
        failed_actions = actions
        print(f"Total episodes (assumed failures): {len(actions)}")
    
    if not failed_actions:
        print("Warning: No failed episodes found!")
        failed_actions = actions  # Fallback to all actions
    
    # Count failures per action
    failed_action_counts = Counter(failed_actions)
    total_failures = len(failed_actions)
    
    # Compute failure rate for each action: (failures / total times used) * 100
    action_failure_rates = {}
    for action in total_action_counts.keys():
        failures = failed_action_counts.get(action, 0)
        total_uses = total_action_counts[action]
        failure_rate = (failures / total_uses) * 100
        action_failure_rates[action] = failure_rate
    
    # Sort by failure rate (descending)
    sorted_actions = sorted(action_failure_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Create stats dict with both failure rate and counts for reporting
    action_stats = {
        action: {
            'failure_rate': failure_rate,
            'failures': failed_action_counts.get(action, 0),
            'total_uses': total_action_counts[action]
        }
        for action, failure_rate in sorted_actions
    }
    
    return sorted_actions, action_failure_rates, total_failures, action_dict, action_stats

def create_short_labels(action_dict, action_list):
    """Create concise labels for radar chart"""
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
            label_map[action_id] = f"Action {action_id}"
    
    return label_map

def create_radar_chart(action_probs, action_dict, task_name, output_file):
    """Create a radar chart visualization of failure probabilities"""
    
    # Take top N actions for visualization (to avoid clutter)
    top_n = min(12, len(action_probs))
    top_actions = list(action_probs.keys())[:top_n]
    
    # Get short labels
    label_map = create_short_labels(action_dict, top_actions)
    
    # Prepare data
    categories = [label_map[action] for action in top_actions]
    values = [action_probs[action] for action in top_actions]
    
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
    ax.plot(angles, values, 'o-', linewidth=2, color='purple', label='Failure Probability')
    ax.fill(angles, values, alpha=0.25, color='purple')
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=10)
    
    # Set y-axis limits and hide radial tick labels
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_yticklabels([])  # Hide radial tick labels
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add title
    ax.set_title(f'{task_name.upper()} Policy Failure Rates', 
                 size=16, weight='bold', pad=20)
    
    # Add value labels on the plot
    for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
        if value > 10:  # Only label significant values (>10% failure rate)
            ax.text(angle, value + max(values) * 0.05, f'{value:.1f}%', 
                   ha='center', va='center', size=9, weight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {output_file}")
    
    return fig

def generate_text_report(sorted_actions, action_failure_rates, total_failures, action_dict, task_name, output_file, total_episodes=None, action_stats=None):
    """Generate detailed text report of failure analysis"""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"{task_name.upper()} POLICY FAILURE RATE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        if total_episodes:
            success_rate = ((total_episodes - total_failures) / total_episodes) * 100
            f.write(f"Total episodes analyzed: {total_episodes}\n")
            f.write(f"Failed episodes: {total_failures} ({100 - success_rate:.1f}%)\n")
            f.write(f"Successful episodes: {total_episodes - total_failures} ({success_rate:.1f}%)\n\n")
        else:
            f.write(f"Total failed episodes analyzed: {total_failures}\n\n")
        
        # Ranked list
        f.write("Failure Rate Rankings (Highest to Lowest):\n")
        f.write("-" * 80 + "\n\n")
        
        for rank, (action_id, failure_rate) in enumerate(sorted_actions, 1):
            description = action_dict.get(action_id, f"Unknown action {action_id}")
            
            if action_stats and action_id in action_stats:
                stats = action_stats[action_id]
                failures = stats['failures']
                total_uses = stats['total_uses']
                f.write(f"{rank}. Action {action_id} - {failure_rate:.1f}% failure rate ({failures}/{total_uses} uses)\n")
            else:
                f.write(f"{rank}. Action {action_id} - {failure_rate:.1f}% failure rate\n")
            
            f.write(f"   → {description}\n\n")
        
        f.write("=" * 80 + "\n\n")
        

    
    print(f"Text report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze RoboMD failure diagnosis results")
    parser.add_argument("--task", "-t", type=str, required=True,
                       help="Task name (e.g., 'square', 'lift', 'can')")
    parser.add_argument("--log_dir", "-l", type=str, required=True,
                       help="Path to latent_ppo_logs directory")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="Output directory for analysis results (default: <log_dir>/../data_analysis)")
    
    args = parser.parse_args()
    
    # Validate inputs
    actions_file = os.path.join(args.log_dir, "actions.txt")
    stats_file = os.path.join(args.log_dir, "stats.txt")
    
    if not os.path.exists(actions_file):
        print(f"Error: actions.txt not found at {actions_file}")
        return
    
    if args.task not in ACTION_DICTS:
        print(f"Warning: Task '{args.task}' not in predefined action dictionaries.")
        print(f"Available tasks: {list(ACTION_DICTS.keys())}")
    
    # Set output directory
    if args.output_dir is None:
        log_parent = os.path.dirname(args.log_dir)
        output_dir = os.path.join(log_parent, "data_analysis")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load actions and stats
    print(f"Loading actions from {actions_file}...")
    actions = load_actions(actions_file)
    
    print(f"Loading stats from {stats_file}...")
    successes = load_stats(stats_file)
    
    # Analyze
    print(f"Analyzing failure modes...")
    sorted_actions, action_failure_rates, failed_count, action_dict, action_stats = analyze_actions(actions, successes, args.task)
    
    # Generate text report
    text_output = os.path.join(output_dir, f"{args.task}_failure_summary.txt")
    total_episodes = len(actions) if successes else None
    generate_text_report(sorted_actions, action_failure_rates, failed_count, action_dict, args.task, text_output, total_episodes, action_stats)
    
    # Generate radar chart
    chart_output = os.path.join(output_dir, f"{args.task}_failure_radar.png")
    create_radar_chart(action_failure_rates, action_dict, args.task, chart_output)
    
    print(f"\nAnalysis complete!")
    print(f"  Text report: {text_output}")
    print(f"  Radar chart: {chart_output}")

if __name__ == "__main__":
    main()
