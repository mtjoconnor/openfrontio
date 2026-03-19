#!/usr/bin/env python3
"""
Gym-style wrapper around the TypeScript self-play bridge.

This wrapper intentionally keeps the API minimal and familiar:
- reset(seed) -> (obs_vector, info)
- step(action_index) -> (obs_vector, reward, done, truncated, info)
- action mask is exposed via `current_action_mask()`
- reset_multi(seed) -> (obs_by_client, masks_by_client, info)
- step_multi(action_indices) -> (obs_by_client, masks_by_client, rewards_by_client, done, truncated, info)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .openfront_rl_bridge import BridgeConfig, OpenFrontRLBridge
except ImportError:
    from openfront_rl_bridge import BridgeConfig, OpenFrontRLBridge


def repo_root() -> Path:
    # .../OpenFront/rl/python/openfront_env.py -> parents[2] = OpenFront
    return Path(__file__).resolve().parents[2]


def default_bridge_command(
    players: int,
    max_ticks: int,
    seed: int,
    nations: str = "disabled",
    bots: int = 0,
) -> List[str]:
    # Use direct Node invocation instead of `npm run` so stdout remains pure
    # NDJSON protocol (npm adds banner lines that break JSON parsing).
    return [
        "node",
        "--import",
        "tsx",
        "src/scripts/rl-bridge.ts",
        "--players",
        str(players),
        "--max-ticks",
        str(max_ticks),
        "--seed",
        str(seed),
        "--nations",
        str(nations),
        "--bots",
        str(bots),
    ]


@dataclass(frozen=True)
class ActionLayout:
    """Fixed action-index layout derived from mask tensor sizes."""

    n_target_slots: int
    n_ratio_bins: int
    n_cancel_slots: int
    n_spawn_slots: int
    n_target_player_slots: int

    no_op_offset: int
    attack_offset: int
    cancel_offset: int
    spawn_offset: int
    target_player_offset: int
    action_dim: int

    @staticmethod
    def from_mask(mask: Dict[str, Any]) -> "ActionLayout":
        n_target_slots = len(mask["attack"]["target_slots"])
        n_ratio_bins = len(mask["attack"]["ratio_bins"])
        n_cancel_slots = len(mask["cancel_attack"]["attack_slots"])
        n_spawn_slots = len(mask["spawn"]["spawn_slots"])
        n_target_player_slots = len(mask["target_player"]["target_slots"])

        no_op_offset = 0
        attack_offset = 1
        attack_count = n_target_slots * n_ratio_bins
        cancel_offset = attack_offset + attack_count
        spawn_offset = cancel_offset + n_cancel_slots
        target_player_offset = spawn_offset + n_spawn_slots
        action_dim = target_player_offset + n_target_player_slots

        return ActionLayout(
            n_target_slots=n_target_slots,
            n_ratio_bins=n_ratio_bins,
            n_cancel_slots=n_cancel_slots,
            n_spawn_slots=n_spawn_slots,
            n_target_player_slots=n_target_player_slots,
            no_op_offset=no_op_offset,
            attack_offset=attack_offset,
            cancel_offset=cancel_offset,
            spawn_offset=spawn_offset,
            target_player_offset=target_player_offset,
            action_dim=action_dim,
        )


def sample_masked_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """
    Uniformly sample from legal action indices.

    The action mask is guaranteed to include `no_op`, but we still guard
    against pathological empty masks for robustness.
    """
    legal_indices = np.flatnonzero(mask > 0.5)
    if legal_indices.size == 0:
        return 0
    return int(rng.choice(legal_indices))


class OpenFrontSelfPlayEnv:
    """
    Single-agent environment over the multi-player TS simulator.

    Supports:
    - single-agent API (`reset`, `step`) for simple smoke runs
    - multi-agent API (`reset_multi`, `step_multi`) for true shared-policy self-play
    """

    def __init__(
        self,
        *,
        players: int = 4,
        max_ticks: int = 3_000,
        seed: int = 1337,
        nations: str = "disabled",
        bots: int = 0,
        observation_mode: str = "student",
        bridge_command: Optional[Sequence[str]] = None,
        controlled_client_id: Optional[str] = None,
        max_players_features: Optional[int] = None,
    ) -> None:
        self.players = players
        self.max_ticks = max_ticks
        self.default_seed = seed
        self.observation_mode = observation_mode
        self.controlled_client_id = controlled_client_id
        self.max_players_features = max_players_features

        command = (
            list(bridge_command)
            if bridge_command is not None
            else default_bridge_command(
                players,
                max_ticks,
                seed,
                nations=nations,
                bots=bots,
            )
        )
        self.bridge = OpenFrontRLBridge(
            BridgeConfig(command=command, cwd=str(repo_root()))
        )

        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_mask: Optional[Dict[str, Any]] = None
        self._last_obs_by_client: Dict[str, Dict[str, Any]] = {}
        self._last_mask_by_client: Dict[str, Dict[str, Any]] = {}
        self._controlled_client_ids: List[str] = []
        self._layout: Optional[ActionLayout] = None
        self._obs_dim: Optional[int] = None

    @property
    def action_dim(self) -> int:
        if self._layout is None:
            raise RuntimeError("Environment must be reset before reading action_dim")
        return self._layout.action_dim

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            raise RuntimeError("Environment must be reset before reading obs_dim")
        return self._obs_dim

    def close(self) -> None:
        self.bridge.close()

    def current_action_mask(self) -> np.ndarray:
        if self._last_mask is None or self._layout is None:
            raise RuntimeError("Environment must be reset before reading action mask")
        return self._encode_action_mask(self._last_mask, self._layout)

    @property
    def agent_ids(self) -> List[str]:
        if not self._controlled_client_ids:
            raise RuntimeError("Environment must be reset before reading agent IDs")
        return list(self._controlled_client_ids)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_by_client, _, info = self.reset_multi(seed=seed)
        if self.controlled_client_id is None:
            self.controlled_client_id = sorted(obs_by_client.keys())[0]
        client_id = self.controlled_client_id
        if client_id not in obs_by_client:
            raise RuntimeError(f"Controlled client {client_id} missing from reset output")
        info["client_id"] = client_id
        return obs_by_client[client_id], info

    def step(
        self, action_index: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._last_obs is None or self._last_mask is None or self._layout is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self.controlled_client_id is None:
            raise RuntimeError("No controlled client selected")

        next_obs, _, rewards, done, truncated, info = self.step_multi(
            {self.controlled_client_id: int(action_index)}
        )
        obs_vec = next_obs[self.controlled_client_id]
        reward = float(rewards[self.controlled_client_id])
        info["invalid_intent"] = bool(
            info.get("invalid_intents", {}).get(self.controlled_client_id, False)
        )
        if done:
            info["won"] = bool(info.get("wins", {}).get(self.controlled_client_id, False))
            info["placement"] = int(
                info.get("placements", {}).get(self.controlled_client_id, self.players)
            )
            info["survived"] = bool(
                info.get("survived", {}).get(self.controlled_client_id, False)
            )
        truncated = False
        return obs_vec, reward, done, truncated, info

    def reset_multi(
        self, *, seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        chosen_seed = self.default_seed if seed is None else int(seed)
        params: Dict[str, Any] = {
            "seed": chosen_seed,
            "observation_mode": self.observation_mode,
        }
        if self.controlled_client_id is not None:
            params["controlled_client_ids"] = [self.controlled_client_id]

        response = self.bridge.reset(params)
        observations = response["observations"]
        action_masks = response["action_masks"]
        if not observations:
            raise RuntimeError("Bridge reset returned no observations")

        self._controlled_client_ids = sorted(observations.keys())
        self._last_obs_by_client = {
            client_id: observations[client_id] for client_id in self._controlled_client_ids
        }
        self._last_mask_by_client = {
            client_id: action_masks[client_id] for client_id in self._controlled_client_ids
        }

        sample_client = self._controlled_client_ids[0]
        self._last_obs = self._last_obs_by_client[sample_client]
        self._last_mask = self._last_mask_by_client[sample_client]
        if self._layout is None:
            self._layout = ActionLayout.from_mask(self._last_mask)
        if self.max_players_features is None:
            self.max_players_features = len(self._last_obs.get("players_public", []))

        obs_by_client = {
            client_id: self._encode_observation(self._last_obs_by_client[client_id])
            for client_id in self._controlled_client_ids
        }
        mask_by_client = {
            client_id: self._encode_action_mask(
                self._last_mask_by_client[client_id], self._layout
            )
            for client_id in self._controlled_client_ids
        }
        self._obs_dim = int(next(iter(obs_by_client.values())).shape[0])
        info = {
            "tick": int(response["tick"]),
            "done": bool(response["done"]),
            "agent_ids": list(self._controlled_client_ids),
            "episode_stats": response.get("episode_stats"),
        }
        return obs_by_client, mask_by_client, info

    def step_multi(
        self, action_indices: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, float],
        bool,
        bool,
        Dict[str, Any],
    ]:
        if self._layout is None or not self._controlled_client_ids:
            raise RuntimeError("Environment must be reset before stepping")

        actions_payload: Dict[str, Dict[str, Any]] = {}
        for client_id in self._controlled_client_ids:
            mask = self._last_mask_by_client[client_id]
            action_index = int(action_indices.get(client_id, 0))
            actions_payload[client_id] = self._decode_action_index(
                action_index,
                mask,
                self._layout,
            )

        response = self.bridge.step({"actions": actions_payload})
        observations = response["observations"]
        action_masks = response["action_masks"]

        self._last_obs_by_client = {
            client_id: observations[client_id] for client_id in self._controlled_client_ids
        }
        self._last_mask_by_client = {
            client_id: action_masks[client_id] for client_id in self._controlled_client_ids
        }
        # Keep single-agent convenience pointers in sync.
        if self.controlled_client_id in self._last_obs_by_client:
            self._last_obs = self._last_obs_by_client[self.controlled_client_id]
            self._last_mask = self._last_mask_by_client[self.controlled_client_id]

        obs_by_client = {
            client_id: self._encode_observation(self._last_obs_by_client[client_id])
            for client_id in self._controlled_client_ids
        }
        mask_by_client = {
            client_id: self._encode_action_mask(
                self._last_mask_by_client[client_id], self._layout
            )
            for client_id in self._controlled_client_ids
        }
        rewards = {
            client_id: float(response["rewards"][client_id])
            for client_id in self._controlled_client_ids
        }
        invalids = {
            client_id: bool(response["invalid_intents"][client_id])
            for client_id in self._controlled_client_ids
        }
        done = bool(response["done"])
        info: Dict[str, Any] = {
            "tick": int(response["tick"]),
            "invalid_intents": invalids,
            "episode_stats": response.get("episode_stats"),
            "agent_ids": list(self._controlled_client_ids),
        }
        if done:
            stats_players = response.get("episode_stats", {}).get("players", {})
            placements: Dict[str, int] = {}
            wins: Dict[str, bool] = {}
            survived: Dict[str, bool] = {}
            for client_id in self._controlled_client_ids:
                p = stats_players.get(client_id, {})
                placements[client_id] = int(p.get("placement", self.players))
                wins[client_id] = bool(p.get("won", False))
                survived[client_id] = bool(p.get("survived", False))
            info["placements"] = placements
            info["wins"] = wins
            info["survived"] = survived

        truncated = False
        return obs_by_client, mask_by_client, rewards, done, truncated, info

    def _encode_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        players_public = sorted(
            obs.get("players_public", []),
            key=lambda p: int(p.get("small_id", 0)),
        )
        max_players = int(self.max_players_features or len(players_public))

        total_tiles = max(1.0, float(sum(p.get("tiles_owned", 0) for p in players_public)))
        self_tiles = float(obs.get("self_tiles_owned", 0))
        self_troops = float(obs.get("self_troops", 0))
        self_gold = float(obs.get("self_gold", 0))

        target_slots = obs.get("target_slots", [])
        attackable = sum(1 for t in target_slots if t.get("can_attack"))
        targetable = sum(1 for t in target_slots if t.get("can_target"))
        target_count = max(1, len(target_slots))

        # Global + self summary features.
        features: List[float] = [
            float(obs.get("tick", 0)) / max(1.0, float(self.max_ticks)),
            1.0 if obs.get("in_spawn_phase") else 0.0,
            1.0 if obs.get("self_is_alive") else 0.0,
            1.0 if obs.get("self_has_spawned") else 0.0,
            self_tiles / total_tiles,
            self_troops / max(1.0, self_tiles),
            self_gold / 1_000_000.0,
            float(obs.get("self_outgoing_attacks", 0)) / 20.0,
            float(obs.get("self_incoming_attacks", 0)) / 20.0,
            float(attackable) / float(target_count),
            float(targetable) / float(target_count),
        ]

        # Per-player public features (padded to fixed-size block).
        for i in range(max_players):
            if i < len(players_public):
                p = players_public[i]
                p_tiles = float(p.get("tiles_owned", 0))
                p_troops = float(p.get("troops", 0))
                p_gold = float(p.get("gold", 0))
                features.extend(
                    [
                        1.0 if p.get("is_self") else 0.0,
                        1.0 if p.get("is_alive") else 0.0,
                        1.0 if p.get("has_spawned") else 0.0,
                        p_tiles / total_tiles,
                        p_troops / max(1.0, p_tiles),
                        p_gold / 1_000_000.0,
                        float(p.get("outgoing_attacks", 0)) / 20.0,
                        float(p.get("incoming_attacks", 0)) / 20.0,
                    ]
                )
            else:
                features.extend([0.0] * 8)

        return np.asarray(features, dtype=np.float32)

    def _encode_action_mask(
        self, mask: Dict[str, Any], layout: ActionLayout
    ) -> np.ndarray:
        action_mask = np.zeros((layout.action_dim,), dtype=np.float32)
        action_mask[layout.no_op_offset] = 1.0

        # Attack actions are a Cartesian product of target slots and ratio bins.
        if bool(mask["attack"]["enabled"]):
            target_mask = mask["attack"]["target_slots"]
            ratio_mask = mask["attack"]["ratio_bins"]
            for target_slot, target_ok in enumerate(target_mask):
                if not target_ok:
                    continue
                for ratio_bin, ratio_ok in enumerate(ratio_mask):
                    if not ratio_ok:
                        continue
                    idx = layout.attack_offset + target_slot * layout.n_ratio_bins + ratio_bin
                    action_mask[idx] = 1.0

        if bool(mask["cancel_attack"]["enabled"]):
            for slot, ok in enumerate(mask["cancel_attack"]["attack_slots"]):
                if ok:
                    action_mask[layout.cancel_offset + slot] = 1.0

        if bool(mask["spawn"]["enabled"]):
            for slot, ok in enumerate(mask["spawn"]["spawn_slots"]):
                if ok:
                    action_mask[layout.spawn_offset + slot] = 1.0

        if bool(mask["target_player"]["enabled"]):
            for slot, ok in enumerate(mask["target_player"]["target_slots"]):
                if ok:
                    action_mask[layout.target_player_offset + slot] = 1.0

        return action_mask

    def _decode_action_index(
        self, action_index: int, mask: Dict[str, Any], layout: ActionLayout
    ) -> Dict[str, Any]:
        # Always keep no-op fallback valid to avoid invalid-intent spam.
        if action_index == layout.no_op_offset:
            return {"type": "no_op"}

        attack_end = layout.cancel_offset
        cancel_end = layout.spawn_offset
        spawn_end = layout.target_player_offset
        target_player_end = layout.action_dim

        if layout.attack_offset <= action_index < attack_end:
            rel = action_index - layout.attack_offset
            target_slot = rel // max(1, layout.n_ratio_bins)
            ratio_bin = rel % max(1, layout.n_ratio_bins)
            if (
                bool(mask["attack"]["enabled"])
                and mask["attack"]["target_slots"][target_slot]
                and mask["attack"]["ratio_bins"][ratio_bin]
            ):
                return {
                    "type": "attack",
                    "target_slot": int(target_slot),
                    "ratio_bin": int(ratio_bin),
                }
            return {"type": "no_op"}

        if layout.cancel_offset <= action_index < cancel_end:
            slot = action_index - layout.cancel_offset
            if bool(mask["cancel_attack"]["enabled"]) and mask["cancel_attack"]["attack_slots"][slot]:
                return {"type": "cancel_attack", "attack_slot": int(slot)}
            return {"type": "no_op"}

        if layout.spawn_offset <= action_index < spawn_end:
            slot = action_index - layout.spawn_offset
            if bool(mask["spawn"]["enabled"]) and mask["spawn"]["spawn_slots"][slot]:
                return {"type": "spawn", "spawn_slot": int(slot)}
            return {"type": "no_op"}

        if layout.target_player_offset <= action_index < target_player_end:
            slot = action_index - layout.target_player_offset
            if bool(mask["target_player"]["enabled"]) and mask["target_player"]["target_slots"][slot]:
                return {"type": "target_player", "target_slot": int(slot)}
            return {"type": "no_op"}

        return {"type": "no_op"}
