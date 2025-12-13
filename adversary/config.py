import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AdversaryConfig:
    """
    Configuration for adversarial finetuning.
    Tuned for lift; extendable to other tasks.
    """

    # core paths
    ckpt_path: str = ""
    log_dir: str = "robomimic/adversary/runs"
    ckpt_dir: str = "robomimic/adversary/checkpoints"

    # task / env
    task_name: str = "lift"
    max_modifications: int = 3
    rollout_horizon: int = 400
    terminate_on_success: bool = True
    render: bool = False

    # protagonist PPO
    protagonist_lr: float = 3e-5
    protagonist_clip: float = 0.2
    protagonist_ent_coef: float = 0.005
    protagonist_value_coef: float = 0.5
    protagonist_max_grad_norm: float = 0.5
    protagonist_batch_size: int = 64
    protagonist_epochs: int = 4
    protagonist_lr_warmup_epochs: int = 5
    freeze_visual_epochs: int = 0
    use_rnn_hidden_for_value: bool = True

    # adversary PPO
    adversary_lr: float = 1e-4
    adversary_clip: float = 0.2
    adversary_ent_coef: float = 0.01
    adversary_value_coef: float = 0.5
    adversary_max_grad_norm: float = 0.5
    adversary_batch_size: int = 64
    adversary_epochs: int = 4
    adversary_update_ratio: int = 3  # update adversary every N protagonist updates

    # adversary reward shaping
    # PAIRED-style regret coefficients
    paired_return_coef: float = 1.0
    paired_success_coef: float = 1.0
    reward_repeat_penalty: float = -0.2
    reward_step_cost: float = -0.02
    reward_diversity_bonus: float = 0.05
    reward_speed_bonus_scale: float = 0.3
    min_adversary_success_rate: float = 0.05
    max_adversary_success_rate: float = 0.95

    # data collection
    chunk_episodes: int = 20
    total_epochs: int = 50
    seed: int = 1
    device: str = "cuda"

    # adversary observation / history
    adversary_history_len: int = 10
    extra_actions: Optional[Dict[int, str]] = field(default=None)

    def ensure_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

