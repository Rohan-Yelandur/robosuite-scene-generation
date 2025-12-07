import argparse
import csv
import os
from typing import Dict

import numpy as np
import torch

from robomimic.utils import file_utils as FileUtils
from robomimic.algo.bc import BC_RNN_GMM
from robomimic.adversary.adversary_env import apply_action_to_xml
from configs.action_dicts import ACTION_DICTS


def rollout(policy: BC_RNN_GMM, env, horizon: int, device: torch.device):
    obs = env.reset()
    policy.reset()
    total_reward = 0.0
    success = 0.0
    for t in range(horizon):
        obs_t = {k: torch.as_tensor(v, device=device).float().unsqueeze(0) for k, v in obs.items()}
        for k, v in obs_t.items():
            if v.dtype == torch.uint8:
                obs_t[k] = v.float() / 255.0

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
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt_path,
        device=device,
        verbose=True,
    )
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

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["action_id", "description", "success_rate", "avg_return"])

        for action_id, desc in action_dict.items():
            successes = []
            returns = []
            for _ in range(args.rollouts):
                apply_action(env, args.task_name, action_id)
                outcome = rollout(policy, env, args.horizon, device)
                successes.append(outcome["success"])
                returns.append(outcome["return"])
            writer.writerow(
                [
                    action_id,
                    desc,
                    float(np.mean(successes)),
                    float(np.mean(returns)),
                ]
            )
            print(f"Action {action_id}: success_rate={np.mean(successes):.2f}")

    print(f"Saved sweep results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BC-RNN-GMM over env modifications.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to policy checkpoint.")
    parser.add_argument("--task_name", type=str, default="lift")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="robomimic/adversary/eval")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)

