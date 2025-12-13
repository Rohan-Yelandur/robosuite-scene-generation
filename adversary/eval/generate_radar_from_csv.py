#!/usr/bin/env python3
"""
Generate a radar plot from an existing sweep_results.csv file.

This mirrors the radar plot produced by eval_env_sweep.py but uses
concise labels similar to analyze_failures.py (e.g., "Table color green",
"Cube size 12").
"""

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_sweep_results(csv_path: str) -> List[Dict]:
    """Load sweep results from CSV into a list of dicts."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"sweep results not found at {csv_path}")

    results = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                action_id = int(row["action_id"])
                desc = row["description"]
                success_rate = float(row["success_rate"])
                results.append(
                    {
                        "action_id": action_id,
                        "desc": desc,
                        "success_rate": success_rate * 100.0,
                    }
                )
            except (KeyError, ValueError):
                continue
    return results


def _extract_color(description: str) -> str:
    """Helper to pull trailing color name."""
    color = description.split("to")[-1].strip().rstrip(".")
    return color.capitalize()


def create_short_label(description: str, action_id: int) -> str:
    """
    Build a concise label following analyze_failures.py style.
    """
    desc_lower = description.lower()

    if "cylinder color to" in desc_lower:
        return f"Cyl color {_extract_color(description)}"
    if "cube color to" in desc_lower:
        return f"Cube color {_extract_color(description)}"
    if "can color to" in desc_lower:
        return f"Can color {_extract_color(description)}"
    if "table color to" in desc_lower:
        return f"Table color {_extract_color(description)}"
    if "robot color to" in desc_lower:
        return f"Robot color {_extract_color(description)}"
    if "lighting color to" in desc_lower or "light color to" in desc_lower:
        return f"Light color {_extract_color(description)}"

    if "resize" in desc_lower and "cylinder" in desc_lower:
        return f"Cyl size {action_id}"
    if "resize" in desc_lower and "cube" in desc_lower:
        return f"Cube size {action_id}"
    if "resize" in desc_lower and "table" in desc_lower:
        return f"Table size {action_id}"
    if "resize" in desc_lower and "box" in desc_lower:
        return f"Box size {action_id}"
    if "no perturbation" in desc_lower or "no change" in desc_lower:
        return "No change"

    # Generic fallback: short prefix of the description
    words = description.split()
    if len(words) > 3:
        return " ".join(words[:3]) + "..."
    return description[:24]


def create_short_labels(entries: List[Dict]) -> Dict[int, str]:
    """Create a label map keyed by action_id."""
    return {entry["action_id"]: create_short_label(entry["desc"], entry["action_id"]) for entry in entries}


def create_radar_chart(entries: List[Dict], output_file: str, title: str):
    """Generate and save a radar chart of failure rates."""
    if not entries:
        raise ValueError("No entries found in sweep results.")

    # Sort by failure rate descending
    sorted_entries = sorted(entries, key=lambda e: (100.0 - e["success_rate"]), reverse=True)
    top_n = min(12, len(sorted_entries))
    top_entries = sorted_entries[:top_n]

    values = [100.0 - e["success_rate"] for e in top_entries]
    categories_ids = [e["action_id"] for e in top_entries]
    label_map = create_short_labels(top_entries)
    categories = [label_map[aid] for aid in categories_ids]

    if not values:
        raise ValueError("No values to plot.")

    # Close the loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    categories += categories[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.plot(angles, values, "o-", linewidth=2, color="purple")
    ax.fill(angles, values, alpha=0.25, color="purple")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=9)
    ax.set_ylim(0, max(values) * 1.1 if values else 1.0)
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_title(title, size=14, weight="bold", pad=15)

    threshold = max(values) * 0.1
    for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
        if value > threshold:
            ax.text(
                angle,
                value + max(values) * 0.05,
                f"{value:.1f}%",
                ha="center",
                va="center",
                size=8,
                weight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Radar chart saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate radar plot from sweep_results.csv.")
    parser.add_argument("--csv_path", type=str, default="adversary/eval/sweep_results.csv", help="Path to sweep_results.csv.")
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: sweep_radar.png next to CSV).")
    parser.add_argument("--title", type=str, default="Policy Failure Rates", help="Title for the radar chart.")
    args = parser.parse_args()

    csv_path = args.csv_path
    output_path = args.output
    if output_path is None:
        output_dir = os.path.dirname(csv_path) or "."
        output_path = os.path.join(output_dir, "sweep_radar.png")

    entries = read_sweep_results(csv_path)
    create_radar_chart(entries, output_path, args.title)


if __name__ == "__main__":
    main()
