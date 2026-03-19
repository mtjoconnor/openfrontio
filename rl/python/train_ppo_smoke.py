#!/usr/bin/env python3
"""
Masked PPO smoke trainer with true multi-agent self-play.

This trainer uses one shared policy across all controlled clients.
Each env tick contributes one transition per agent, so metrics now reflect
actual self-play outcomes rather than one-agent-vs-passive-opponents.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    from .openfront_env import OpenFrontSelfPlayEnv, default_bridge_command
except ImportError:
    from openfront_env import OpenFrontSelfPlayEnv, default_bridge_command


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    invalid = mask < 0.5
    return logits.masked_fill(invalid, -1e9)


class PolicyValueNet(nn.Module):
    """Simple shared MLP actor-critic."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.body(obs)
        return self.policy_head(x), self.value_head(x).squeeze(-1)


@dataclass
class PPOConfig:
    updates: int
    rollout_steps: int
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    players: int
    max_ticks: int
    nations: str
    bots: int
    seed: int
    observation_mode: str
    log_every: int
    device: str
    bridge_command: Optional[Sequence[str]]
    checkpoint_dir: str
    save_every: int
    eval_every: int
    eval_episodes: int


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="OpenFront masked PPO smoke trainer")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument(
        "--nations",
        default="disabled",
        help='Nation config for bridge env: "disabled", "default", or integer count (1-400)',
    )
    parser.add_argument("--bots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--observation-mode", choices=["student", "teacher"], default="student"
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="rl/python/checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument(
        "--bridge-cmd",
        default="",
        help='Optional full bridge command, e.g. "node --import tsx src/scripts/rl-bridge.ts --players 4 --max-ticks 300"',
    )

    args = parser.parse_args()
    bridge_command = shlex.split(args.bridge_cmd) if args.bridge_cmd.strip() else None
    nations = parse_nations_arg(args.nations)
    return PPOConfig(
        updates=max(1, args.updates),
        rollout_steps=max(8, args.rollout_steps),
        ppo_epochs=max(1, args.ppo_epochs),
        minibatch_size=max(8, args.minibatch_size),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_coef=float(args.clip_coef),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        max_grad_norm=float(args.max_grad_norm),
        players=max(2, args.players),
        max_ticks=max(1, args.max_ticks),
        nations=nations,
        bots=max(0, min(400, args.bots)),
        seed=max(0, args.seed),
        observation_mode=args.observation_mode,
        log_every=max(1, args.log_every),
        device=args.device,
        bridge_command=bridge_command,
        checkpoint_dir=args.checkpoint_dir,
        save_every=max(1, args.save_every),
        eval_every=max(0, args.eval_every),
        eval_episodes=max(1, args.eval_episodes),
    )


def parse_nations_arg(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if normalized in {"disabled", "default"}:
        return normalized
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f'Invalid --nations value "{raw}". Use "disabled", "default", or integer 1-400.'
        ) from exc
    return str(max(1, min(400, parsed)))


def compute_gae(batch: Dict[str, np.ndarray], cfg: PPOConfig) -> Tuple[np.ndarray, np.ndarray]:
    rewards = batch["rewards"]
    values = batch["values"]
    dones = batch["dones"]
    next_values = batch["next_values"]

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_values[t] * non_terminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    batch: Dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    cfg: PPOConfig,
) -> Dict[str, float]:
    obs_t = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    actions_t = torch.tensor(batch["actions"], dtype=torch.int64, device=device)
    old_logprobs_t = torch.tensor(batch["logprobs"], dtype=torch.float32, device=device)
    masks_t = torch.tensor(batch["masks"], dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = obs_t.shape[0]
    mb = min(cfg.minibatch_size, n)

    approx_kl_values: List[float] = []
    entropy_values: List[float] = []
    policy_loss_values: List[float] = []
    value_loss_values: List[float] = []

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mb):
            idx = perm[start : start + mb]
            logits, values = model(obs_t[idx])
            masked_logits = apply_action_mask(logits, masks_t[idx])
            dist = Categorical(logits=masked_logits)

            new_logprobs = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()
            log_ratio = new_logprobs - old_logprobs_t[idx]
            ratio = torch.exp(log_ratio)

            adv = advantages_t[idx]
            pg_loss_1 = ratio * adv
            pg_loss_2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv
            policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * ((values - returns_t[idx]) ** 2).mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            approx_kl_values.append(float(approx_kl))
            entropy_values.append(float(entropy.item()))
            policy_loss_values.append(float(policy_loss.item()))
            value_loss_values.append(float(value_loss.item()))

    return {
        "policy_loss": float(np.mean(policy_loss_values)) if policy_loss_values else 0.0,
        "value_loss": float(np.mean(value_loss_values)) if value_loss_values else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "approx_kl": float(np.mean(approx_kl_values)) if approx_kl_values else 0.0,
    }


def rollout_multiagent(
    env: OpenFrontSelfPlayEnv,
    model: PolicyValueNet,
    device: torch.device,
    cfg: PPOConfig,
    obs_by_client: Dict[str, np.ndarray],
    mask_by_client: Dict[str, np.ndarray],
    episode_returns: Dict[str, float],
    episode_steps: int,
    completed_returns: Deque[float],
    completed_lengths: Deque[int],
    completed_placements: Deque[float],
    completed_wins: Deque[float],
    global_step: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, float],
    int,
    int,
    Dict[str, np.ndarray],
]:
    agent_ids = env.agent_ids
    num_agents = len(agent_ids)
    transitions = cfg.rollout_steps * num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    obs_buf = np.zeros((transitions, obs_dim), dtype=np.float32)
    action_buf = np.zeros((transitions,), dtype=np.int64)
    logprob_buf = np.zeros((transitions,), dtype=np.float32)
    value_buf = np.zeros((transitions,), dtype=np.float32)
    reward_buf = np.zeros((transitions,), dtype=np.float32)
    done_buf = np.zeros((transitions,), dtype=np.float32)
    next_value_buf = np.zeros((transitions,), dtype=np.float32)
    mask_buf = np.zeros((transitions, action_dim), dtype=np.float32)

    cursor = 0
    for _ in range(cfg.rollout_steps):
        obs_batch = np.stack([obs_by_client[cid] for cid in agent_ids], axis=0)
        mask_batch = np.stack([mask_by_client[cid] for cid in agent_ids], axis=0)
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits_t, value_t = model(obs_t)
            masked_logits = apply_action_mask(logits_t, mask_t)
            dist = Categorical(logits=masked_logits)
            action_t = dist.sample()
            logprob_t = dist.log_prob(action_t)

        actions = action_t.cpu().numpy().astype(np.int64)
        values = value_t.cpu().numpy().astype(np.float32)
        logprobs = logprob_t.cpu().numpy().astype(np.float32)

        sl = slice(cursor, cursor + num_agents)
        obs_buf[sl] = obs_batch
        mask_buf[sl] = mask_batch
        action_buf[sl] = actions
        value_buf[sl] = values
        logprob_buf[sl] = logprobs

        action_payload = {cid: int(actions[i]) for i, cid in enumerate(agent_ids)}
        next_obs, next_masks, rewards, done, _, info = env.step_multi(action_payload)

        for i, cid in enumerate(agent_ids):
            reward_buf[cursor + i] = float(rewards[cid])
            done_buf[cursor + i] = 1.0 if done else 0.0
            episode_returns[cid] += float(rewards[cid])
        episode_steps += 1
        global_step += num_agents

        if done:
            next_value_buf[sl] = 0.0
            placements = info.get("placements", {})
            wins = info.get("wins", {})
            for cid in agent_ids:
                completed_returns.append(episode_returns[cid])
                completed_lengths.append(episode_steps)
                completed_placements.append(float(placements.get(cid, cfg.players)))
                completed_wins.append(1.0 if wins.get(cid, False) else 0.0)
            reset_seed = cfg.seed + (global_step * 17)
            obs_by_client, mask_by_client, _ = env.reset_multi(seed=reset_seed)
            episode_returns = {cid: 0.0 for cid in env.agent_ids}
            episode_steps = 0
        else:
            next_obs_batch = np.stack([next_obs[cid] for cid in agent_ids], axis=0)
            next_obs_t = torch.tensor(next_obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, next_values_t = model(next_obs_t)
            next_value_buf[sl] = next_values_t.cpu().numpy().astype(np.float32)
            obs_by_client = next_obs
            mask_by_client = next_masks

        cursor += num_agents

    batch = {
        "obs": obs_buf,
        "actions": action_buf,
        "logprobs": logprob_buf,
        "values": value_buf,
        "rewards": reward_buf,
        "dones": done_buf,
        "next_values": next_value_buf,
        "masks": mask_buf,
    }
    return (
        obs_by_client,
        mask_by_client,
        episode_returns,
        episode_steps,
        global_step,
        batch,
    )


def evaluate_policy(
    model: PolicyValueNet,
    cfg: PPOConfig,
    device: torch.device,
    bridge_cmd: Sequence[str],
    episodes: int,
    seed_base: int,
) -> Dict[str, float]:
    env = OpenFrontSelfPlayEnv(
        players=cfg.players,
        max_ticks=cfg.max_ticks,
        seed=seed_base,
        observation_mode=cfg.observation_mode,
        bridge_command=bridge_cmd,
    )
    returns: List[float] = []
    lengths: List[int] = []
    placements: List[float] = []
    wins: List[float] = []
    try:
        for episode in range(episodes):
            obs_by_client, mask_by_client, _ = env.reset_multi(seed=seed_base + episode)
            episode_returns = {cid: 0.0 for cid in env.agent_ids}
            steps = 0
            while True:
                agent_ids = env.agent_ids
                obs_batch = np.stack([obs_by_client[cid] for cid in agent_ids], axis=0)
                mask_batch = np.stack([mask_by_client[cid] for cid in agent_ids], axis=0)
                obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits_t, _ = model(obs_t)
                    masked_logits = apply_action_mask(logits_t, mask_t)
                    actions = torch.argmax(masked_logits, dim=1).cpu().numpy()
                payload = {cid: int(actions[i]) for i, cid in enumerate(agent_ids)}
                next_obs, next_masks, rewards, done, _, info = env.step_multi(payload)
                for cid in agent_ids:
                    episode_returns[cid] += float(rewards[cid])
                steps += 1
                if done:
                    placement_map = info.get("placements", {})
                    win_map = info.get("wins", {})
                    for cid in agent_ids:
                        returns.append(episode_returns[cid])
                        lengths.append(steps)
                        placements.append(float(placement_map.get(cid, cfg.players)))
                        wins.append(1.0 if win_map.get(cid, False) else 0.0)
                    break
                obs_by_client = next_obs
                mask_by_client = next_masks
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_len": float(np.mean(lengths)) if lengths else 0.0,
        "mean_placement": float(np.mean(placements)) if placements else float(cfg.players),
        "win_rate": float(np.mean(wins)) if wins else 0.0,
    }


def save_checkpoint(
    checkpoint_dir: Path,
    update: int,
    global_step: int,
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    cfg: PPOConfig,
    train_metrics: Dict[str, float],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"ppo_smoke_update_{update:06d}.pt"
    payload = {
        "update": update,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "train_metrics": train_metrics,
        "saved_at_unix": time.time(),
    }
    torch.save(payload, path)
    return path


def append_eval_snapshot(
    checkpoint_dir: Path,
    update: int,
    global_step: int,
    metrics: Dict[str, float],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "eval_history.jsonl"
    record = {
        "update": update,
        "global_step": global_step,
        "timestamp_unix": time.time(),
        **metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    cfg = parse_args()
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)
    checkpoint_dir = Path(cfg.checkpoint_dir)

    bridge_cmd = (
        list(cfg.bridge_command)
        if cfg.bridge_command is not None
        else default_bridge_command(
            cfg.players,
            cfg.max_ticks,
            cfg.seed,
            nations=cfg.nations,
            bots=cfg.bots,
        )
    )
    print(
        "[ppo-smoke] starting "
        f"updates={cfg.updates} rollout_steps={cfg.rollout_steps} "
        f"players={cfg.players} max_ticks={cfg.max_ticks} nations={cfg.nations} bots={cfg.bots} seed={cfg.seed}"
    )
    print(f"[ppo-smoke] bridge_cmd={' '.join(bridge_cmd)}")

    env = OpenFrontSelfPlayEnv(
        players=cfg.players,
        max_ticks=cfg.max_ticks,
        seed=cfg.seed,
        observation_mode=cfg.observation_mode,
        bridge_command=bridge_cmd,
    )

    try:
        obs_by_client, mask_by_client, _ = env.reset_multi(seed=cfg.seed)
        model = PolicyValueNet(env.obs_dim, env.action_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

        global_step = 0
        episode_steps = 0
        episode_returns = {cid: 0.0 for cid in env.agent_ids}

        completed_returns: Deque[float] = deque(maxlen=512)
        completed_lengths: Deque[int] = deque(maxlen=512)
        completed_placements: Deque[float] = deque(maxlen=512)
        completed_wins: Deque[float] = deque(maxlen=512)

        train_start = time.time()
        for update in range(1, cfg.updates + 1):
            (
                obs_by_client,
                mask_by_client,
                episode_returns,
                episode_steps,
                global_step,
                batch,
            ) = rollout_multiagent(
                env=env,
                model=model,
                device=device,
                cfg=cfg,
                obs_by_client=obs_by_client,
                mask_by_client=mask_by_client,
                episode_returns=episode_returns,
                episode_steps=episode_steps,
                completed_returns=completed_returns,
                completed_lengths=completed_lengths,
                completed_placements=completed_placements,
                completed_wins=completed_wins,
                global_step=global_step,
            )

            advantages, returns = compute_gae(batch, cfg)
            losses = ppo_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                advantages=advantages,
                returns=returns,
                device=device,
                cfg=cfg,
            )

            elapsed = time.time() - train_start
            sps = global_step / max(1e-9, elapsed)
            mean_return = float(np.mean(completed_returns)) if completed_returns else 0.0
            mean_len = float(np.mean(completed_lengths)) if completed_lengths else 0.0
            mean_placement = (
                float(np.mean(completed_placements))
                if completed_placements
                else float(cfg.players)
            )
            win_rate = float(np.mean(completed_wins)) if completed_wins else 0.0
            train_metrics = {
                "mean_return": mean_return,
                "mean_len": mean_len,
                "mean_placement": mean_placement,
                "win_rate": win_rate,
                **losses,
                "sps": sps,
            }

            if update % cfg.log_every == 0 or update == cfg.updates:
                print(
                    "[ppo-smoke] "
                    f"update={update}/{cfg.updates} "
                    f"global_step={global_step} "
                    f"mean_return={mean_return:.4f} "
                    f"mean_len={mean_len:.1f} "
                    f"mean_placement={mean_placement:.2f} "
                    f"win_rate={win_rate:.3f} "
                    f"policy_loss={losses['policy_loss']:.4f} "
                    f"value_loss={losses['value_loss']:.4f} "
                    f"entropy={losses['entropy']:.4f} "
                    f"approx_kl={losses['approx_kl']:.6f} "
                    f"sps={sps:.1f}"
                )

            if update % cfg.save_every == 0 or update == cfg.updates:
                path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    update=update,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    train_metrics=train_metrics,
                )
                print(f"[ppo-smoke] checkpoint saved: {path}")

            if cfg.eval_every > 0 and (update % cfg.eval_every == 0 or update == cfg.updates):
                eval_metrics = evaluate_policy(
                    model=model,
                    cfg=cfg,
                    device=device,
                    bridge_cmd=bridge_cmd,
                    episodes=cfg.eval_episodes,
                    seed_base=cfg.seed + 100_000 + update * 1_000,
                )
                append_eval_snapshot(
                    checkpoint_dir=checkpoint_dir,
                    update=update,
                    global_step=global_step,
                    metrics=eval_metrics,
                )
                print(
                    "[ppo-smoke] eval "
                    f"update={update} "
                    f"mean_return={eval_metrics['mean_return']:.4f} "
                    f"mean_len={eval_metrics['mean_len']:.1f} "
                    f"mean_placement={eval_metrics['mean_placement']:.2f} "
                    f"win_rate={eval_metrics['win_rate']:.3f}"
                )

        print("[ppo-smoke] training complete")
    finally:
        env.close()


if __name__ == "__main__":
    main()
