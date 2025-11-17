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

def analyze_actions(actions, task_name):
    """Analyze action frequencies and compute probabilities"""
    action_dict = ACTION_DICTS.get(task_name, {})
    
    # Count frequencies
    action_counts = Counter(actions)
    total = len(actions)
    
    # Sort by frequency (descending)
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Compute probabilities
    action_probs = {action: (count / total) * 100 for action, count in sorted_actions}
    
    return sorted_actions, action_probs, total, action_dict

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
    ax.set_title(f'{task_name.upper()} Policy Failure Modes', 
                 size=16, weight='bold', pad=20)
    
    # Add value labels on the plot
    for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
        if value > 5:  # Only label significant values
            ax.text(angle, value + max(values) * 0.05, f'{value:.1f}%', 
                   ha='center', va='center', size=9, weight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {output_file}")
    
    return fig

def generate_text_report(sorted_actions, action_probs, total, action_dict, task_name, output_file):
    """Generate detailed text report of failure analysis"""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"{task_name.upper()} POLICY FAILURE PROBABILITY MAP\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total episodes analyzed: {total}\n\n")
        
        # Ranked list
        f.write("Failure Mode Rankings (Highest to Lowest Probability):\n")
        f.write("-" * 80 + "\n\n")
        
        for rank, (action_id, count) in enumerate(sorted_actions, 1):
            probability = action_probs[action_id]
            description = action_dict.get(action_id, f"Unknown action {action_id}")
            
            f.write(f"{rank}. Action {action_id} - {probability:.1f}% ({count}/{total} episodes)\n")
            f.write(f"   → {description}\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Key insights section
        f.write("KEY INSIGHTS:\n")
        f.write("=" * 80 + "\n\n")
        
        # Categorize vulnerabilities
        critical = [(aid, prob) for aid, prob in action_probs.items() if prob >= 30]
        moderate = [(aid, prob) for aid, prob in action_probs.items() if 10 <= prob < 30]
        minor = [(aid, prob) for aid, prob in action_probs.items() if prob < 10]
        
        if critical:
            f.write("🔴 CRITICAL VULNERABILITIES (≥30% failure probability):\n")
            for action_id, prob in critical:
                desc = action_dict.get(action_id, f"Action {action_id}")
                f.write(f"   • {desc} ({prob:.1f}%)\n")
            f.write("\n")
        
        if moderate:
            f.write("🟡 MODERATE VULNERABILITIES (10-30% failure probability):\n")
            for action_id, prob in moderate:
                desc = action_dict.get(action_id, f"Action {action_id}")
                f.write(f"   • {desc} ({prob:.1f}%)\n")
            f.write("\n")
        
        if minor:
            f.write("🟢 MINOR VULNERABILITIES (<10% failure probability):\n")
            for action_id, prob in minor:
                desc = action_dict.get(action_id, f"Action {action_id}")
                f.write(f"   • {desc} ({prob:.1f}%)\n")
            f.write("\n")
        
        # Recommendations
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR POLICY IMPROVEMENT:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. DOMAIN RANDOMIZATION:\n")
        f.write("   Focus training data augmentation on the top 3 failure modes.\n\n")
        
        f.write("2. TARGETED DATA COLLECTION:\n")
        f.write("   Collect additional demonstrations under critical failure conditions.\n\n")
        
        f.write("3. FINE-TUNING:\n")
        f.write("   Fine-tune the policy specifically on scenarios involving:\n")
        for i, (action_id, prob) in enumerate(list(action_probs.items())[:3], 1):
            desc = action_dict.get(action_id, f"Action {action_id}")
            f.write(f"      {i}) {desc}\n")
        f.write("\n")
        
        f.write("4. RE-EVALUATION:\n")
        f.write("   After retraining, run RoboMD again to verify failure modes are reduced.\n")
        f.write("   Success = more uniform probability distribution across all actions.\n\n")
        
        f.write("=" * 80 + "\n")
    
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
    
    # Load and analyze actions
    print(f"Loading actions from {actions_file}...")
    actions = load_actions(actions_file)
    
    print(f"Analyzing {len(actions)} episodes...")
    sorted_actions, action_probs, total, action_dict = analyze_actions(actions, args.task)
    
    # Generate text report
    text_output = os.path.join(output_dir, f"{args.task}_failure_summary.txt")
    generate_text_report(sorted_actions, action_probs, total, action_dict, args.task, text_output)
    
    # Generate radar chart
    chart_output = os.path.join(output_dir, f"{args.task}_failure_radar.png")
    create_radar_chart(action_probs, action_dict, args.task, chart_output)
    
    print(f"\nAnalysis complete!")
    print(f"  Text report: {text_output}")
    print(f"  Radar chart: {chart_output}")

if __name__ == "__main__":
    main()
