import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from robomimic.utils import file_utils as FileUtils
from robomimic.algo.bc import BC_RNN_GMM
from robomimic.algo.algo import RolloutPolicy
from adversary_env import apply_action_to_xml

# Make configs importable when running from repo root
CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "Robo-MD-RSS.github.io"
if CONFIGS_ROOT.exists():
    sys.path.append(str(CONFIGS_ROOT))
from configs.action_dicts import ACTION_DICTS  # noqa: E402


def rollout(policy: BC_RNN_GMM, policy_wrapper: RolloutPolicy, env, horizon: int, device: torch.device):
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    total_reward = 0.0
    success = 0.0
    for t in range(horizon):
        # Use robomimic preprocessing to ensure image shapes / normalization match the encoder config.
        obs_t = policy_wrapper._prepare_observation(obs, batched_ob=False)
        dist, _ = policy.nets["policy"].forward_train_step(obs_t, rnn_state=None)
        action = dist.sample()
        action_np = action.squeeze(0).detach().cpu().numpy()

        obs, r, done, _ = env.step(action_np)
        total_reward += r
        success = float(env.is_success()["task"])
        if done or success:
            break
    return {"success": success, "return": total_reward, "steps": t + 1}


def apply_action(env, task_name: str, action_id: int):
    state = env.get_state()
    import xml.etree.ElementTree as ET

    root = ET.fromstring(state["model"])
    apply_action_to_xml(task_name, action_id, root)
    state["model"] = ET.tostring(root, encoding="unicode")
    env.reset_to(state)
    return env.reset()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Support both original BC checkpoints and adversary-produced checkpoints that
    # wrap the original path inside cfg.ckpt_path.
    raw_ckpt = torch.load(args.ckpt_path, map_location="cpu")
    base_ckpt_path = args.ckpt_path
    if isinstance(raw_ckpt, dict) and "cfg" in raw_ckpt and isinstance(raw_ckpt["cfg"], dict):
        base_ckpt_path = raw_ckpt["cfg"].get("ckpt_path", args.ckpt_path)

    try:
        policy_wrapped, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=base_ckpt_path,
            device=device,
            verbose=True,
        )
    except KeyError:
        # Older or custom checkpoints might miss algo_name; add a safe default.
        ckpt_loaded = FileUtils.maybe_dict_from_checkpoint(ckpt_path=base_ckpt_path, ckpt_dict=raw_ckpt)
        ckpt_loaded.setdefault("algo_name", "bc_rnn_gmm")
        policy_wrapped, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=None,
            ckpt_dict=ckpt_loaded,
            device=device,
            verbose=True,
        )

    # Fix legacy controller configs like we do in training to avoid missing registrations.
    if "env_metadata" in ckpt_dict and "env_kwargs" in ckpt_dict["env_metadata"]:
        env_kwargs = ckpt_dict["env_metadata"]["env_kwargs"]
        if "controller_configs" in env_kwargs:
            ctrl = env_kwargs["controller_configs"]
            if isinstance(ctrl, dict) and "type" in ctrl and "body_parts" not in ctrl:
                del ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"]
                print("[INFO] Removed incompatible old controller config, using defaults")

    # Keep RolloutPolicy for preprocessing just like in training.
    if isinstance(policy_wrapped, RolloutPolicy):
        policy_wrapper = policy_wrapped
        policy = policy_wrapped.policy
    else:
        policy_wrapper = RolloutPolicy(policy_wrapped)
        policy = policy_wrapped

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=args.render,
        render_offscreen=not args.render,
        verbose=True,
    )
    action_dict = ACTION_DICTS.get(args.task_name, {})

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "sweep_results.csv")

    # collect per-action stats for radar plot
    per_action_stats = []

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["action_id", "description", "success_rate", "avg_return"])

        for action_id, desc in action_dict.items():
            successes = []
            returns = []
            for _ in range(args.rollouts):
                apply_action(env, args.task_name, action_id)
                outcome = rollout(policy, policy_wrapper, env, args.horizon, device)
                successes.append(outcome["success"])
                returns.append(outcome["return"])
            success_rate = float(np.mean(successes)) if successes else 0.0
            avg_return = float(np.mean(returns)) if returns else 0.0
            writer.writerow(
                [
                    action_id,
                    desc,
                    success_rate,
                    avg_return,
                ]
            )
            print(f"Action {action_id}: success_rate={np.mean(successes):.2f}")
            per_action_stats.append(
                {
                    "action_id": action_id,
                    "desc": desc,
                    "success_rate": success_rate * 100.0,
                }
            )

    print(f"Saved sweep results to {results_path}")

    # Radar plot (failure rates, matching analyze_failures style)
    def _extract_color(description: str) -> str:
        color = description.split("to")[-1].strip().rstrip(".")
        return color.capitalize()

    def create_short_label(description: str, action_id: int) -> str:
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

        words = description.split()
        if len(words) > 3:
            return " ".join(words[:3]) + "..."
        return description[:20]

    def create_short_labels(action_dict_local, action_list):
        label_map = {}
        for action_id in action_list:
            description = action_dict_local.get(action_id, f"Action {action_id}")
            label_map[action_id] = create_short_label(description, action_id)
        return label_map

    def create_radar_chart(action_stats, action_dict_local, output_file, task_name):
        if not action_stats:
            return
        sorted_actions = sorted(action_stats, key=lambda x: (100.0 - x["success_rate"]), reverse=True)
        top_n = min(12, len(sorted_actions))
        top_actions = sorted_actions[:top_n]
        values = [100.0 - a["success_rate"] for a in top_actions]
        categories_ids = [a["action_id"] for a in top_actions]
        label_map = create_short_labels(action_dict_local, categories_ids)
        categories = [label_map[aid] for aid in categories_ids]

        if not values:
            return

        # close the loop
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
        ax.set_title(f"{task_name.upper()} - Policy Failure Rates", size=14, weight="bold", pad=15)

        threshold = max(values) * 0.1
        for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
            if value > threshold:
                ax.text(angle, value + max(values) * 0.05, f"{value:.1f}%", ha="center", va="center", size=8, weight="bold")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Radar chart saved to: {output_file}")

    radar_path = os.path.join(args.output_dir, "sweep_radar.png")
    create_radar_chart(per_action_stats, action_dict, radar_path, args.task_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BC-RNN-GMM over env modifications.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to policy checkpoint.")
    parser.add_argument("--task_name", type=str, default="lift")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--output_dir", type=str, default="adversary/eval")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)

