import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import tensor_utils as TensorUtils
from robomimic.algo.bc import BC_RNN_GMM
from robomimic.algo.algo import RolloutPolicy

from config import AdversaryConfig
from adversary_env import AdversaryEnv
from adversary_policy import DiscreteActorCritic

# Ensure configs are importable when running from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent / "Robo-MD-RSS.github.io"
if REPO_ROOT.exists():
    sys.path.append(str(REPO_ROOT))
from configs.action_dicts import ACTION_DICTS  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_low_dim(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten only non-image observations.
    """
    parts = []
    for k, v in obs.items():
        if v is None:
            continue
        if isinstance(v, np.ndarray) and v.dtype != np.uint8 and v.ndim <= 2:
            parts.append(v.reshape(-1))
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def stack_obs(obs_list: List[Dict[str, np.ndarray]], device: torch.device):
    """
    Stack obs into torch tensors, normalizing images to float.
    """
    stacked = {}
    keys = obs_list[0].keys()
    for k in keys:
        stacked[k] = np.stack([o[k] for o in obs_list], axis=0)
        tens = torch.as_tensor(stacked[k], device=device)
        if tens.dtype == torch.uint8:
            tens = tens.float() / 255.0
        else:
            tens = tens.float()
        stacked[k] = tens
    return stacked


class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs: List[Dict[str, np.ndarray]] = []
        self.actions: List[np.ndarray] = []
        self.logps: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.value_feats: List[np.ndarray] = []

    def add(self, obs, action, logp, reward, done, value, value_feat):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.value_feats.append(value_feat)

    def compute_advantages(self, gamma=0.99, lam=0.95):
        values = np.array(self.values + [0.0], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns


def estimate_entropy(dist, n_samples: int = 10):
    samples = dist.sample((n_samples,))
    log_probs = dist.log_prob(samples)
    return (-log_probs.mean(dim=0)).detach()


class ProtagonistPPO:
    def __init__(
        self,
        policy: BC_RNN_GMM,
        policy_wrapper: RolloutPolicy,
        value_dim: int,
        cfg: AdversaryConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.policy_wrapper = policy_wrapper
        self.device = device
        self.cfg = cfg
        self._rnn_state = None

        layers = [
            nn.Linear(value_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ]
        self.value_net = nn.Sequential(*layers).to(device)

        policy_params = list(self.policy.nets.parameters())
        self.optim = torch.optim.Adam(
            policy_params + list(self.value_net.parameters()),
            lr=cfg.protagonist_lr,
        )

    def reset_rnn(self):
        self._rnn_state = None

    def _value_feature(self, rnn_state, low_dim_flat):
        if not self.cfg.use_rnn_hidden_for_value or rnn_state is None:
            return low_dim_flat
        if isinstance(rnn_state, tuple):
            h = rnn_state[0]  # LSTM h
        else:
            h = rnn_state
        # take last layer hidden state and flatten
        feat = h[-1].detach().cpu().reshape(-1).numpy()
        if low_dim_flat.size > 0:
            feat = np.concatenate([feat, low_dim_flat], axis=0)
        return feat

    def _prepare_obs(self, obs: Dict[str, np.ndarray]):
        """
        Use robomimic's RolloutPolicy preprocessing to ensure image sizes / normalization
        match the policy's expected encoder configuration.
        """
        # _prepare_observation adds batch dimension when batched_ob=False
        return self.policy_wrapper._prepare_observation(obs, batched_ob=False)

    def _prepare_obs_batch(self, obs_batch: List[Dict[str, np.ndarray]]):
        processed = [self._prepare_obs(o) for o in obs_batch]
        keys = processed[0].keys()
        stacked = {}
        for k in keys:
            stacked[k] = torch.cat([p[k] for p in processed], dim=0)
        return stacked

    def act(self, obs: Dict[str, np.ndarray]):
        obs_t = self._prepare_obs(obs)

        dist, self._rnn_state = self.policy.nets["policy"].forward_train_step(
            obs_t, rnn_state=self._rnn_state
        )
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = estimate_entropy(dist)

        low_dim_flat = flatten_low_dim(obs)
        value_feat = self._value_feature(self._rnn_state, low_dim_flat)
        value = self.value_net(torch.as_tensor(value_feat, device=self.device, dtype=torch.float32)).item()

        return (
            action.squeeze(0).detach().cpu().numpy(),
            logp.item(),
            entropy.mean().item(),
            value,
            value_feat,
        )

    def evaluate(self, obs_batch: List[Dict[str, np.ndarray]], actions: np.ndarray):
        obs_t = self._prepare_obs_batch(obs_batch)
        dist, _ = self.policy.nets["policy"].forward_train_step(obs_t, rnn_state=None)
        act_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        logp = dist.log_prob(act_t)
        entropy = estimate_entropy(dist)
        return logp, entropy

    def value(self, feats: np.ndarray):
        feats_t = torch.as_tensor(feats, device=self.device, dtype=torch.float32)
        return self.value_net(feats_t).squeeze(-1)


class AdversaryPPO:
    def __init__(self, obs_dim: int, action_dim: int, cfg: AdversaryConfig, device: torch.device):
        self.model = DiscreteActorCritic(obs_dim, action_dim).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.adversary_lr)
        self.cfg = cfg
        self.device = device

    def act(self, obs_vec: np.ndarray):
        obs_t = torch.as_tensor(obs_vec, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, logp, value, entropy = self.model.act(obs_t)
        return action.item(), logp.item(), value.item(), entropy.mean().item()

    def evaluate(self, obs_batch: np.ndarray, act_batch: np.ndarray):
        obs_t = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        dist, values = self.model(obs_t)
        actions_t = torch.as_tensor(act_batch, device=self.device)
        logp = dist.log_prob(actions_t)
        entropy = dist.entropy()
        return logp, entropy, values


def _save_checkpoint(path: str, protagonist, adversary, cfg):
    # BC_RNN_GMM is an Algo, not an nn.Module; save its underlying networks.
    policy_state = protagonist.policy.nets.state_dict() if hasattr(protagonist.policy, "nets") else protagonist.policy.state_dict()
    torch.save(
        {
            "protagonist_policy": policy_state,
            "protagonist_value": protagonist.value_net.state_dict(),
            "adversary_model": adversary.model.state_dict(),
            "cfg": cfg.__dict__,
        },
        path,
    )


def train(cfg: AdversaryConfig):
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    policy_wrapped, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=cfg.ckpt_path,
        device=device,
        verbose=True,
    )
    # Keep RolloutPolicy for preprocessing (image sizing / normalization)
    if isinstance(policy_wrapped, RolloutPolicy):
        policy_wrapper = policy_wrapped
        policy = policy_wrapped.policy
    else:
        policy = policy_wrapped
        policy_wrapper = RolloutPolicy(policy)

    # Ensure policy is in training mode (file_utils sets eval by default)
    policy.set_train()
    if not isinstance(policy, BC_RNN_GMM):
        print(f"[WARN] Loaded policy type {type(policy)} is not BC_RNN_GMM; proceeding assuming compatible interface.")

    # Fix old controller configs that are incompatible with newer robosuite
    if "env_metadata" in ckpt_dict and "env_kwargs" in ckpt_dict["env_metadata"]:
        env_kwargs = ckpt_dict["env_metadata"]["env_kwargs"]
        if "controller_configs" in env_kwargs:
            ctrl = env_kwargs["controller_configs"]
            if isinstance(ctrl, dict) and "type" in ctrl and "body_parts" not in ctrl:
                # Old format - remove it to use defaults
                del ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"]
                print("[INFO] Removed incompatible old controller config, using defaults")

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=cfg.render,
        render_offscreen=not cfg.render,
        verbose=True,
    )

    action_dict = ACTION_DICTS.get(cfg.task_name, {})
    if cfg.extra_actions:
        action_dict = {**action_dict, **cfg.extra_actions}

    protagonist_buffer = PPOBuffer()
    adversary_buffer = PPOBuffer()
    epoch_pro_returns = []
    epoch_pro_success = []
    epoch_pro_lens = []
    epoch_adv_returns = []
    epoch_adv_lens = []
    epoch_baseline_returns = []
    epoch_mod_returns = []
    epoch_mod_success = []

    init_obs = env.reset()
    low_dim_feat = flatten_low_dim(init_obs)
    rnn_hidden = getattr(policy.algo_config.rnn, "hidden_dim", 0) if cfg.use_rnn_hidden_for_value else 0
    value_dim = max(1, rnn_hidden + low_dim_feat.shape[0])
    protagonist = ProtagonistPPO(policy, policy_wrapper, value_dim, cfg, device)
    adversary = AdversaryPPO(
        obs_dim=4 + len(action_dict) * cfg.adversary_history_len,
        action_dim=len(action_dict),
        cfg=cfg,
        device=device,
    )

    writer = SummaryWriter(log_dir=cfg.log_dir)

    protagonist_updates = 0
    best_return_mean = float("-inf")

    def protagonist_rollout_fn(env_instance, start_state=None, collect=True):
        if start_state is not None:
            env_instance.reset_to(start_state)
        obs = env_instance.reset()
        protagonist.reset_rnn()
        total_reward = 0.0
        success = 0.0
        for step_i in range(cfg.rollout_horizon):
            act, logp, entropy, value, value_feat = protagonist.act(obs)
            next_obs, r, done, _ = env_instance.step(act)

            if collect:
                protagonist_buffer.add(
                    obs=obs,
                    action=act,
                    logp=logp,
                    reward=r,
                    done=done,
                    value=value,
                    value_feat=value_feat,
                )

            total_reward += r
            success = float(env_instance.is_success()["task"])
            obs = next_obs
            if done or (success and cfg.terminate_on_success):
                break
        if collect:
            epoch_pro_returns.append(total_reward)
            epoch_pro_success.append(success)
            epoch_pro_lens.append(step_i + 1)
        return {"success": success, "return": total_reward, "steps": step_i + 1}

    reward_weights = dict(
        paired_return_coef=cfg.paired_return_coef,
        paired_success_coef=cfg.paired_success_coef,
        repeat_penalty=cfg.reward_repeat_penalty,
        step_cost=cfg.reward_step_cost,
        diversity_bonus=cfg.reward_diversity_bonus,
        speed_bonus_scale=cfg.reward_speed_bonus_scale,
    )

    adversary_env = AdversaryEnv(
        base_env=env,
        task_name=cfg.task_name,
        protagonist_rollout_fn=protagonist_rollout_fn,
        action_dict=action_dict,
        max_modifications=cfg.max_modifications,
        history_len=cfg.adversary_history_len,
        reward_weights=reward_weights,
    )

    def maybe_warmup_lr(epoch_idx: int):
        if epoch_idx < cfg.protagonist_lr_warmup_epochs and cfg.protagonist_lr_warmup_epochs > 0:
            scale = (epoch_idx + 1) / cfg.protagonist_lr_warmup_epochs
            for g in protagonist.optim.param_groups:
                g["lr"] = cfg.protagonist_lr * scale

    for epoch in range(cfg.total_epochs):
        # reset per-epoch trackers
        epoch_pro_returns.clear()
        epoch_pro_success.clear()
        epoch_pro_lens.clear()
        epoch_adv_returns.clear()
        epoch_adv_lens.clear()
        epoch_baseline_returns.clear()
        epoch_mod_returns.clear()
        epoch_mod_success.clear()
        maybe_warmup_lr(epoch)
        adversary_buffer.clear()
        protagonist_buffer.clear()

        # optional encoder freeze
        freeze_enc = cfg.freeze_visual_epochs > 0 and epoch < cfg.freeze_visual_epochs
        for p in policy.nets["policy"].nets["encoder"].parameters():
            p.requires_grad = not freeze_enc

        for _ in range(cfg.chunk_episodes):
            obs_adv = adversary_env.reset()
            done = False
            episode_reward = 0.0
            episode_len = 0
            while not done:
                a_act, a_logp, a_val, a_ent = adversary.act(obs_adv)
                next_obs, a_reward, done, info = adversary_env.step(a_act)

                adversary_buffer.add(
                    obs=obs_adv,
                    action=np.array(a_act, dtype=np.int64),
                    logp=a_logp,
                    reward=a_reward,
                    done=done,
                    value=a_val,
                    value_feat=None,
                )
                episode_reward += a_reward
                episode_len += 1
                # track protagonist outcomes for this adversary choice
                if info and "outcome" in info and "baseline" in info:
                    epoch_mod_returns.append(float(info["outcome"].get("return", 0.0)))
                    epoch_mod_success.append(float(info["outcome"].get("success", 0.0)))
                    epoch_baseline_returns.append(float(info["baseline"].get("return", 0.0)))
                obs_adv = next_obs
            epoch_adv_returns.append(episode_reward)
            epoch_adv_lens.append(episode_len)

        # protagonist PPO update (minibatch)
        adv_pg, ret_pg = protagonist_buffer.compute_advantages()
        obs_batch = protagonist_buffer.obs
        act_batch = np.stack(protagonist_buffer.actions, axis=0)
        logp_old = torch.as_tensor(protagonist_buffer.logps, device=device, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_pg, device=device, dtype=torch.float32)
        ret_t = torch.as_tensor(ret_pg, device=device, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        value_feats = np.stack(protagonist_buffer.value_feats, axis=0)

        indices = np.random.permutation(len(obs_batch))
        proto_policy_loss_sum = 0.0
        proto_value_loss_sum = 0.0
        proto_entropy_sum = 0.0
        proto_kl_sum = 0.0
        proto_batches = 0
        for _ in range(cfg.protagonist_epochs):
            for start in range(0, len(indices), cfg.protagonist_batch_size):
                mb_idx = indices[start : start + cfg.protagonist_batch_size]
                mb_obs = [obs_batch[i] for i in mb_idx]
                mb_act = act_batch[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                mb_feats = value_feats[mb_idx]

                logp_new, entropy = protagonist.evaluate(mb_obs, mb_act)
                values = protagonist.value(mb_feats)
                ratio = torch.exp(logp_new - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.protagonist_clip, 1 + cfg.protagonist_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_ret - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.protagonist_value_coef * value_loss + cfg.protagonist_ent_coef * entropy_loss
                approx_kl = (mb_logp_old - logp_new).mean()

                protagonist.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(protagonist.policy.nets.parameters()) + list(protagonist.value_net.parameters()),
                    cfg.protagonist_max_grad_norm,
                )
                protagonist.optim.step()
                proto_policy_loss_sum += policy_loss.item()
                proto_value_loss_sum += value_loss.item()
                proto_entropy_sum += entropy.mean().item()
                proto_kl_sum += approx_kl.item()
                proto_batches += 1

        protagonist_updates += 1

        # adversary PPO update (minibatch) every N protagonist updates
        adv_policy_loss_sum = 0.0
        adv_value_loss_sum = 0.0
        adv_entropy_sum = 0.0
        adv_kl_sum = 0.0
        adv_batches = 0
        if protagonist_updates % cfg.adversary_update_ratio == 0:
            adv_adv, adv_ret = adversary_buffer.compute_advantages()
            obs_adv_batch = np.stack(adversary_buffer.obs, axis=0)
            act_adv_batch = np.stack(adversary_buffer.actions, axis=0)
            logp_adv_old = torch.as_tensor(adversary_buffer.logps, device=device, dtype=torch.float32)
            adv_adv_t = torch.as_tensor(adv_adv, device=device, dtype=torch.float32)
            adv_ret_t = torch.as_tensor(adv_ret, device=device, dtype=torch.float32)
            adv_adv_t = (adv_adv_t - adv_adv_t.mean()) / (adv_adv_t.std() + 1e-8)

            indices_adv = np.random.permutation(len(obs_adv_batch))
            for _ in range(cfg.adversary_epochs):
                for start in range(0, len(indices_adv), cfg.adversary_batch_size):
                    mb_idx = indices_adv[start : start + cfg.adversary_batch_size]
                    mb_obs = obs_adv_batch[mb_idx]
                    mb_act = act_adv_batch[mb_idx]
                    mb_logp_old = logp_adv_old[mb_idx]
                    mb_adv = adv_adv_t[mb_idx]
                    mb_ret = adv_ret_t[mb_idx]

                    logp_new, entropy, values = adversary.evaluate(mb_obs, mb_act)
                    ratio = torch.exp(logp_new - mb_logp_old)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - cfg.adversary_clip, 1 + cfg.adversary_clip) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (mb_ret - values).pow(2).mean()
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + cfg.adversary_value_coef * value_loss + cfg.adversary_ent_coef * entropy_loss
                    approx_kl = (mb_logp_old - logp_new).mean()

                    adversary.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(adversary.model.parameters(), cfg.adversary_max_grad_norm)
                    adversary.optim.step()
                    adv_policy_loss_sum += policy_loss.item()
                    adv_value_loss_sum += value_loss.item()
                    adv_entropy_sum += entropy.mean().item()
                    adv_kl_sum += approx_kl.item()
                    adv_batches += 1

        # Logging
        pro_step_count = len(protagonist_buffer.rewards)
        adv_step_count = len(adversary_buffer.rewards)
        pro_ep_return_mean = float(np.mean(epoch_pro_returns)) if epoch_pro_returns else 0.0
        pro_success_rate = float(np.mean(epoch_pro_success)) if epoch_pro_success else 0.0
        pro_ep_len_mean = float(np.mean(epoch_pro_lens)) if epoch_pro_lens else 0.0
        pro_entropy_mean = proto_entropy_sum / max(1, proto_batches)
        pro_kl_mean = proto_kl_sum / max(1, proto_batches)
        pro_policy_loss_mean = proto_policy_loss_sum / max(1, proto_batches)
        pro_value_loss_mean = proto_value_loss_sum / max(1, proto_batches)
        pro_return_baseline_mean = float(np.mean(epoch_baseline_returns)) if epoch_baseline_returns else 0.0
        pro_return_mod_mean = float(np.mean(epoch_mod_returns)) if epoch_mod_returns else 0.0
        pro_success_mod_rate = float(np.mean(epoch_mod_success)) if epoch_mod_success else 0.0

        adv_ep_return_mean = float(np.mean(epoch_adv_returns)) if epoch_adv_returns else 0.0
        adv_ep_len_mean = float(np.mean(epoch_adv_lens)) if epoch_adv_lens else 0.0
        adv_entropy_mean = adv_entropy_sum / max(1, adv_batches)
        adv_kl_mean = adv_kl_sum / max(1, adv_batches)
        adv_policy_loss_mean = adv_policy_loss_sum / max(1, adv_batches)
        adv_value_loss_mean = adv_value_loss_sum / max(1, adv_batches)

        writer.add_scalar("protagonist/episode_return_mean", pro_ep_return_mean, epoch)
        writer.add_scalar("protagonist/step_reward_mean", float(np.mean(protagonist_buffer.rewards)) if protagonist_buffer.rewards else 0.0, epoch)
        writer.add_scalar("protagonist/success_rate", pro_success_rate, epoch)
        writer.add_scalar("protagonist/episode_length_mean", pro_ep_len_mean, epoch)
        writer.add_scalar("protagonist/policy_loss", pro_policy_loss_mean, epoch)
        writer.add_scalar("protagonist/value_loss", pro_value_loss_mean, epoch)
        writer.add_scalar("protagonist/entropy_mean", pro_entropy_mean, epoch)
        writer.add_scalar("protagonist/approx_kl", pro_kl_mean, epoch)
        writer.add_scalar("protagonist/baseline_return_mean", pro_return_baseline_mean, epoch)
        writer.add_scalar("protagonist/modified_return_mean", pro_return_mod_mean, epoch)
        writer.add_scalar("protagonist/modified_success_rate", pro_success_mod_rate, epoch)
        writer.add_scalar("protagonist/steps", pro_step_count, epoch)
        writer.add_scalar("protagonist/lr", protagonist.optim.param_groups[0]["lr"], epoch)

        writer.add_scalar("adversary/episode_return_mean", adv_ep_return_mean, epoch)
        writer.add_scalar("adversary/step_reward_mean", float(np.mean(adversary_buffer.rewards)) if adversary_buffer.rewards else 0.0, epoch)
        writer.add_scalar("adversary/episode_length_mean", adv_ep_len_mean, epoch)
        writer.add_scalar("adversary/policy_loss", adv_policy_loss_mean, epoch)
        writer.add_scalar("adversary/value_loss", adv_value_loss_mean, epoch)
        writer.add_scalar("adversary/entropy_mean", adv_entropy_mean, epoch)
        writer.add_scalar("adversary/approx_kl", adv_kl_mean, epoch)
        writer.add_scalar("adversary/steps", adv_step_count, epoch)

        # Save epoch checkpoint
        epoch_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch}.pt")
        _save_checkpoint(epoch_path, protagonist, adversary, cfg)

        # Track and save the best-performing checkpoint (by return mean)
        if pro_ep_return_mean > best_return_mean:
            best_return_mean = pro_ep_return_mean
            best_path = os.path.join(cfg.ckpt_dir, "best.pt")
            _save_checkpoint(best_path, protagonist, adversary, cfg)

    writer.close()

    # Save final checkpoint
    final_path = os.path.join(cfg.ckpt_dir, "final.pt")
    _save_checkpoint(final_path, protagonist, adversary, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial finetuning for BC-RNN-GMM with PPO.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to BC-RNN-GMM checkpoint.")
    parser.add_argument("--task_name", type=str, default="lift", help="robosuite task (lift by default).")
    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--chunk_episodes", type=int, default=20)
    parser.add_argument("--log_dir", type=str, default="adversary/runs")
    parser.add_argument("--ckpt_dir", type=str, default="adversary/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = AdversaryConfig(
        ckpt_path=args.ckpt_path,
        task_name=args.task_name,
        total_epochs=args.total_epochs,
        chunk_episodes=args.chunk_episodes,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
    )
    train(cfg)

