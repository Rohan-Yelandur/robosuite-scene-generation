import copy
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional

import gym
import numpy as np

# Make configs importable when running from repo root
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent / "Robo-MD-RSS.github.io"
if REPO_ROOT.exists():
    sys.path.append(str(REPO_ROOT))

from configs.action_dicts import ACTION_DICTS  # noqa: E402


def _apply_lift_action(root: ET.Element, action: int) -> None:
    """
    Mutate XML to reflect the given lift action (mirrors RoboMD behavior).
    """
    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name", "")

        # Cube colors
        if action == 0 and name == "cube_g0_vis":
            geom.set("rgba", "1 0 0 1")
        elif action == 1 and name == "cube_g0_vis":
            geom.set("rgba", "0 1 0 1")
        elif action == 2 and name == "cube_g0_vis":
            geom.set("rgba", "0 0 1 1")
        elif action == 3 and name == "cube_g0_vis":
            geom.set("rgba", "0.5 0.5 0.5 1")

        # Table color
        elif action == 4 and name == "table_visual":
            geom.set("rgba", "0 1 0 1")
        elif action == 5 and name == "table_visual":
            geom.set("rgba", "0 0 1 1")
        elif action == 6 and name == "table_visual":
            geom.set("rgba", "1 0 0 1")
        elif action == 7 and name == "table_visual":
            geom.set("rgba", "0.7 0.7 0.7 1")

        # Table size
        elif action == 8 and name == "table_visual":
            geom.set("size", "0.8 0.2 0.025")
        elif action == 9 and name == "table_visual":
            geom.set("size", "0.2 0.8 0.025")

        # Cube size
        elif action == 10 and name == "cube_g0_vis":
            geom.set("size", "0.04 0.04 0.04")
        elif action == 11 and name == "cube_g0_vis":
            geom.set("size", "0.01 0.01 0.01")
        elif action == 12 and name == "cube_g0_vis":
            geom.set("size", "0.04 0.01 0.01")

        # Robot color
        elif action == 13 and "robot0_g" in name:
            geom.set("rgba", "1 0 0 1")
        elif action == 14 and "robot0_g" in name:
            geom.set("rgba", "0 1 0 1")
        elif action == 15 and "robot0_g" in name:
            geom.set("rgba", "0 1 1 1")
        elif action == 16 and "robot0_g" in name:
            geom.set("rgba", "0.5 0.5 0.5 1")

    lights = root.findall(".//light")
    if action == 17:
        for light in lights:
            light.set("diffuse", "1 0 0")
    elif action == 18:
        for light in lights:
            light.set("diffuse", "0 1 0")
    elif action == 19:
        for light in lights:
            light.set("diffuse", "0 0 1")
    elif action == 20:
        for light in lights:
            light.set("diffuse", "0.5 0.5 0.5")


def apply_action_to_xml(task_name: str, action: int, root: ET.Element) -> None:
    if task_name == "lift":
        _apply_lift_action(root, action)
    else:
        raise NotImplementedError(f"Task {task_name} not yet supported for adversary actions.")


class AdversaryEnv(gym.Env):
    """
    Gym wrapper for the adversary to choose environment modifications.
    Observation includes budget, recent performance, and action history.
    """

    def __init__(
        self,
        base_env,
        task_name: str,
        protagonist_rollout_fn: Callable[[object, Optional[dict], bool], Dict[str, float]],
        action_dict: Optional[Dict[int, str]] = None,
        max_modifications: int = 3,
        history_len: int = 10,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.base_env = base_env
        self.task_name = task_name
        self.protagonist_rollout_fn = protagonist_rollout_fn
        self.max_modifications = max_modifications
        self.history_len = history_len

        self.action_dict = action_dict or ACTION_DICTS.get(task_name, {})
        self.num_actions = len(self.action_dict)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # observation: [remaining, mean_success, mean_return, diversity] + history one-hot
        self.obs_dim = 4 + self.num_actions * self.history_len
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        defaults = dict(
            paired_return_coef=1.0,
            paired_success_coef=1.0,
            repeat_penalty=-0.2,
            step_cost=-0.05,
            diversity_bonus=0.05,
            speed_bonus_scale=0.5,
        )
        if reward_weights:
            defaults.update(reward_weights)
        self.rw = defaults

        self.reset()

    def reset(self):
        self.mod_count = 0
        self.history: List[int] = []
        self.success_hist: List[float] = []
        self.return_hist: List[float] = []
        self._env_state = self.base_env.reset()
        return self._obs()

    def _history_one_hot(self):
        vec = np.zeros((self.history_len, self.num_actions), dtype=np.float32)
        recent = self.history[-self.history_len :]
        for i, act in enumerate(recent):
            if 0 <= act < self.num_actions:
                vec[i, act] = 1.0
        return vec.flatten()

    def _obs(self):
        remaining = 1.0 - (self.mod_count / max(1, self.max_modifications))
        mean_success = float(np.mean(self.success_hist)) if self.success_hist else 0.0
        mean_return = float(np.mean(self.return_hist)) if self.return_hist else 0.0
        diversity = len(set(self.history)) / max(1, self.max_modifications)
        base = np.array([remaining, mean_success, mean_return, diversity], dtype=np.float32)
        return np.concatenate([base, self._history_one_hot()]).astype(np.float32)

    def _apply_action(self, base_state: dict, action: int) -> dict:
        """
        Apply env modification to a copy of base_state and return the new state.
        """
        robot_state = copy.deepcopy(base_state)
        root = ET.fromstring(robot_state["model"])
        apply_action_to_xml(self.task_name, action, root)
        robot_state["model"] = ET.tostring(root, encoding="unicode")
        return robot_state

    def step(self, action: int):
        self.mod_count += 1
        repeated = action in self.history
        self.history.append(action)
        action_desc = self.action_dict.get(action, "unknown")
        print(f"[Adversary] Applied action {action}: {action_desc} with remaining budget {self.max_modifications - self.mod_count}")
        print(f"[Adversary] Current history: {self.history}")

        base_state = self.base_env.get_state()
        # baseline rollout (no mods)
        baseline_outcome = self.protagonist_rollout_fn(self.base_env, start_state=base_state, collect=False)

        # apply adversary modification
        modified_state = self._apply_action(base_state, action)
        obs_mod = self.base_env.reset_to(modified_state)
        if obs_mod is None:
            obs_mod = self.base_env.reset()

        outcome = self.protagonist_rollout_fn(self.base_env, start_state=modified_state, collect=True)
        success = float(outcome.get("success", 0.0))
        total_return = float(outcome.get("return", 0.0))
        steps = float(outcome.get("steps", 1))

        baseline_success = float(baseline_outcome.get("success", 0.0))
        baseline_return = float(baseline_outcome.get("return", 0.0))

        self.success_hist.append(success)
        self.return_hist.append(total_return)

        done = self.mod_count >= self.max_modifications

        # PAIRED-style regret reward
        regret_return = max(0.0, baseline_return - total_return)
        regret_success = max(0.0, baseline_success - success)
        reward = (
            self.rw["paired_return_coef"] * regret_return
            + self.rw["paired_success_coef"] * regret_success
        )

        reward += self.rw["repeat_penalty"] * float(repeated)
        reward += self.rw["step_cost"]
        reward += self.rw["diversity_bonus"] * (len(set(self.history)) / max(1, self.max_modifications))
        reward += self.rw["speed_bonus_scale"] * (1.0 / max(1.0, steps))

        return self._obs(), reward, done, {
            "outcome": outcome,
            "baseline": baseline_outcome,
            "action_history": list(self.history),
            "action_desc": action_desc,
        }

