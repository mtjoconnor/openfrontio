import { ClientID, Intent } from "../Schemas";

export const RL_BRIDGE_SCHEMA_VERSION_V1 = "v1" as const;

export type ObservationModeV1 = "teacher" | "student";

// Attack-first macro action space used by the simulator and training bridge.
// Each action references a slot so policies stay map-size agnostic.
export type ActionV1 =
  | { type: "no_op" }
  | { type: "attack"; target_slot: number; ratio_bin: number }
  | { type: "cancel_attack"; attack_slot: number }
  | { type: "spawn"; spawn_slot: number }
  | { type: "target_player"; target_slot: number };

// Canonicalized attack target candidates generated per step.
// Slot 0 is reserved for terra nullius in the environment implementation.
export interface TargetSlotV1 {
  slot: number;
  kind: "terra_nullius" | "player";
  player_id: string | null;
  client_id: ClientID | null;
  small_id: number | null;
  tiles_owned: number;
  troops: number;
  can_attack: boolean;
  can_target: boolean;
  is_alive: boolean;
  is_friendly: boolean;
}

export interface AttackSlotV1 {
  slot: number;
  attack_id: string;
  target_kind: "terra_nullius" | "player";
  target_player_id: string | null;
  target_client_id: ClientID | null;
  target_small_id: number | null;
  troops: number;
  retreating: boolean;
}

export interface SpawnSlotV1 {
  slot: number;
  tile: number;
  x: number;
  y: number;
}

// Per-action legality mask. The policy should only sample actions where:
// 1) the action kind is enabled, and
// 2) the referenced slot/bin is true.
export interface ActionMaskV1 {
  no_op: true;
  attack: {
    enabled: boolean;
    target_slots: boolean[];
    ratio_bins: boolean[];
  };
  cancel_attack: {
    enabled: boolean;
    attack_slots: boolean[];
  };
  spawn: {
    enabled: boolean;
    spawn_slots: boolean[];
  };
  target_player: {
    enabled: boolean;
    target_slots: boolean[];
  };
}

export interface PublicPlayerObsV1 {
  player_id: string;
  client_id: ClientID | null;
  small_id: number;
  is_self: boolean;
  is_alive: boolean;
  is_disconnected: boolean;
  has_spawned: boolean;
  tiles_owned: number;
  troops: number;
  gold: number;
  outgoing_attacks: number;
  incoming_attacks: number;
}

export interface PrivilegedPlayerObsV1 {
  player_id: string;
  client_id: ClientID | null;
  small_id: number;
  spawn_tile: number | null;
  border_tiles: number;
  is_immune: boolean;
  relation_to_self: number;
  can_attack_self: boolean;
}

export interface BaseObservationV1 {
  schema_version: typeof RL_BRIDGE_SCHEMA_VERSION_V1;
  mode: ObservationModeV1;
  tick: number;
  in_spawn_phase: boolean;

  self_client_id: ClientID;
  self_player_id: string;
  self_small_id: number;
  self_is_alive: boolean;
  self_has_spawned: boolean;
  self_tiles_owned: number;
  self_troops: number;
  self_gold: number;
  self_outgoing_attacks: number;
  self_incoming_attacks: number;

  target_slots: TargetSlotV1[];
  attack_slots: AttackSlotV1[];
  spawn_slots: SpawnSlotV1[];
}

export interface StudentObsV1 extends BaseObservationV1 {
  mode: "student";
  // Strictly client-equivalent features.
  players_public: PublicPlayerObsV1[];
}

export interface TeacherObsV1 extends BaseObservationV1 {
  mode: "teacher";
  players_public: PublicPlayerObsV1[];
  // Training-only privileged features.
  players_privileged: PrivilegedPlayerObsV1[];
  global_privileged: {
    alive_players: number;
    num_land_tiles: number;
    num_tiles_with_fallout: number;
    winner_set: boolean;
  };
}

export type ObservationV1 = TeacherObsV1 | StudentObsV1;

export interface PlayerEpisodeStatsV1 {
  client_id: ClientID;
  won: boolean;
  placement: number;
  survived: boolean;
  survived_ticks: number;
  invalid_intents: number;
  rate_limit_pressure: number;
  degenerate_spam: number;
  total_reward: number;
}

export interface EpisodeStatsV1 {
  schema_version: typeof RL_BRIDGE_SCHEMA_VERSION_V1;
  tick: number;
  done: boolean;
  winner_client_id: ClientID | null;
  players: Record<ClientID, PlayerEpisodeStatsV1>;
}

export interface ResetRequestV1 {
  seed: number;
  observation_mode?: ObservationModeV1;
  controlled_client_ids?: ClientID[];
}

export interface ResetResponseV1 {
  schema_version: typeof RL_BRIDGE_SCHEMA_VERSION_V1;
  tick: number;
  done: boolean;
  observations: Record<ClientID, ObservationV1>;
  action_masks: Record<ClientID, ActionMaskV1>;
  episode_stats: EpisodeStatsV1;
}

export interface StepRequestV1 {
  actions?: Partial<Record<ClientID, ActionV1>>;
}

export interface StepResponseV1 {
  schema_version: typeof RL_BRIDGE_SCHEMA_VERSION_V1;
  tick: number;
  done: boolean;
  observations: Record<ClientID, ObservationV1>;
  action_masks: Record<ClientID, ActionMaskV1>;
  rewards: Record<ClientID, number>;
  invalid_intents: Record<ClientID, boolean>;
  episode_stats: EpisodeStatsV1;
}

export interface ActionEncodingContextV1 {
  player_troops: number;
  ratio_bins: readonly number[];
  target_slots: readonly TargetSlotV1[];
  attack_slots: readonly AttackSlotV1[];
  spawn_slots: readonly SpawnSlotV1[];
  action_mask: ActionMaskV1;
}

export interface EncodedActionV1 {
  legal: boolean;
  intent: Intent | null;
  reason?: string;
}

const hasIndex = (index: number, arr: readonly unknown[]): boolean =>
  Number.isInteger(index) && index >= 0 && index < arr.length;

export function isActionLegalV1(
  action: ActionV1,
  mask: ActionMaskV1,
): boolean {
  // Central legality check used by both inference and tests.
  switch (action.type) {
    case "no_op":
      return true;
    case "attack":
      return (
        hasIndex(action.target_slot, mask.attack.target_slots) &&
        hasIndex(action.ratio_bin, mask.attack.ratio_bins) &&
        mask.attack.enabled &&
        mask.attack.target_slots[action.target_slot] &&
        mask.attack.ratio_bins[action.ratio_bin]
      );
    case "cancel_attack":
      return (
        hasIndex(action.attack_slot, mask.cancel_attack.attack_slots) &&
        mask.cancel_attack.enabled &&
        mask.cancel_attack.attack_slots[action.attack_slot]
      );
    case "spawn":
      return (
        hasIndex(action.spawn_slot, mask.spawn.spawn_slots) &&
        mask.spawn.enabled &&
        mask.spawn.spawn_slots[action.spawn_slot]
      );
    case "target_player":
      return (
        hasIndex(action.target_slot, mask.target_player.target_slots) &&
        mask.target_player.enabled &&
        mask.target_player.target_slots[action.target_slot]
      );
  }
}

export function encodeActionV1(
  action: ActionV1,
  ctx: ActionEncodingContextV1,
): EncodedActionV1 {
  // Hard-fail masked actions to prevent invalid intents leaking into rollout.
  if (!isActionLegalV1(action, ctx.action_mask)) {
    return {
      legal: false,
      intent: null,
      reason: "Action is masked as illegal",
    };
  }

  switch (action.type) {
    case "no_op":
      return { legal: true, intent: null };

    case "attack": {
      const targetSlot = ctx.target_slots[action.target_slot];
      if (!targetSlot) {
        return {
          legal: false,
          intent: null,
          reason: "Missing target slot",
        };
      }
      const ratio = ctx.ratio_bins[action.ratio_bin];
      if (ratio <= 0) {
        return {
          legal: false,
          intent: null,
          reason: "Non-positive ratio bin",
        };
      }
      // Convert ratio bin to an absolute troop amount expected by core intent.
      const troops = Math.max(1, Math.floor(ctx.player_troops * ratio));
      return {
        legal: true,
        intent: {
          type: "attack",
          targetID:
            targetSlot.kind === "terra_nullius" ? null : targetSlot.player_id,
          troops,
        },
      };
    }

    case "cancel_attack": {
      const attackSlot = ctx.attack_slots[action.attack_slot];
      if (!attackSlot) {
        return {
          legal: false,
          intent: null,
          reason: "Missing attack slot",
        };
      }
      return {
        legal: true,
        intent: {
          type: "cancel_attack",
          attackID: attackSlot.attack_id,
        },
      };
    }

    case "spawn": {
      const spawnSlot = ctx.spawn_slots[action.spawn_slot];
      if (!spawnSlot) {
        return {
          legal: false,
          intent: null,
          reason: "Missing spawn slot",
        };
      }
      return {
        legal: true,
        intent: {
          type: "spawn",
          tile: spawnSlot.tile,
        },
      };
    }

    case "target_player": {
      const targetSlot = ctx.target_slots[action.target_slot];
      if (!targetSlot || targetSlot.kind !== "player" || !targetSlot.player_id) {
        return {
          legal: false,
          intent: null,
          reason: "Target slot is not a player",
        };
      }
      return {
        legal: true,
        intent: {
          type: "targetPlayer",
          target: targetSlot.player_id,
        },
      };
    }
  }
}

export function decodeIntentV1(
  intent: Intent,
  ctx: ActionEncodingContextV1,
): ActionV1 | null {
  // Best-effort inversion of macro encoding; returns null for unsupported intents.
  switch (intent.type) {
    case "attack": {
      const slot = ctx.target_slots.find((t) =>
        intent.targetID === null
          ? t.kind === "terra_nullius"
          : t.kind === "player" && t.player_id === intent.targetID,
      );
      if (!slot) {
        return null;
      }
      const ratioBin = closestRatioBin(
        ctx.ratio_bins,
        intent.troops ?? 0,
        ctx.player_troops,
      );
      return {
        type: "attack",
        target_slot: slot.slot,
        ratio_bin: ratioBin,
      };
    }
    case "cancel_attack": {
      const slot = ctx.attack_slots.find((a) => a.attack_id === intent.attackID);
      if (!slot) {
        return null;
      }
      return { type: "cancel_attack", attack_slot: slot.slot };
    }
    case "spawn": {
      const slot = ctx.spawn_slots.find((s) => s.tile === intent.tile);
      if (!slot) {
        return null;
      }
      return { type: "spawn", spawn_slot: slot.slot };
    }
    case "targetPlayer": {
      const slot = ctx.target_slots.find(
        (t) => t.kind === "player" && t.player_id === intent.target,
      );
      if (!slot) {
        return null;
      }
      return { type: "target_player", target_slot: slot.slot };
    }
    default:
      return null;
  }
}

function closestRatioBin(
  ratioBins: readonly number[],
  troops: number,
  playerTroops: number,
): number {
  if (ratioBins.length === 0) {
    return 0;
  }
  if (playerTroops <= 0) {
    return 0;
  }
  const ratio = troops / playerTroops;
  let bestIndex = 0;
  let bestDistance = Infinity;
  ratioBins.forEach((bin, index) => {
    const distance = Math.abs(bin - ratio);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

export function actionSignatureV1(action: ActionV1): string {
  switch (action.type) {
    case "no_op":
      return "no_op";
    case "attack":
      return `attack:${action.target_slot}:${action.ratio_bin}`;
    case "cancel_attack":
      return `cancel_attack:${action.attack_slot}`;
    case "spawn":
      return `spawn:${action.spawn_slot}`;
    case "target_player":
      return `target_player:${action.target_slot}`;
  }
}
