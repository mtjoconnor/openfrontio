import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GameMapLoader, MapData } from "../../../src/core/game/GameMapLoader";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
} from "../../../src/core/game/Game";
import { MapManifest } from "../../../src/core/game/TerrainMapLoader";
import { GameConfig } from "../../../src/core/Schemas";
import {
  HeadlessSelfPlaySimulatorV1,
  SelfPlayPlayerConfigV1,
} from "../../../src/core/rl/HeadlessSelfPlaySimulator";
import {
  ActionEncodingContextV1,
  ActionMaskV1,
  ActionV1,
  ResetResponseV1,
  StepResponseV1,
  decodeIntentV1,
  encodeActionV1,
} from "../../../src/core/rl/RLBridgeV1";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_MAP_DIR = path.resolve(__dirname, "../../testdata/maps/plains");

class TestGameMapLoader implements GameMapLoader {
  private readonly mapData: MapData = {
    mapBin: async () => new Uint8Array(await readFile(path.join(TEST_MAP_DIR, "map.bin"))),
    map4xBin: async () =>
      new Uint8Array(await readFile(path.join(TEST_MAP_DIR, "map4x.bin"))),
    map16xBin: async () =>
      new Uint8Array(await readFile(path.join(TEST_MAP_DIR, "map16x.bin"))),
    manifest: async () =>
      JSON.parse(await readFile(path.join(TEST_MAP_DIR, "manifest.json"), "utf8")) as MapManifest,
    webpPath: path.join(TEST_MAP_DIR, "thumbnail.webp"),
  };

  getMapData(_: GameMapType): MapData {
    return this.mapData;
  }
}

const testPlayers: SelfPlayPlayerConfigV1[] = [
  { client_id: "AAAA0001", username: "Alpha" },
  { client_id: "BBBB0002", username: "Bravo" },
];

const testGameConfig: GameConfig = {
  gameMap: GameMapType.World,
  gameMapSize: GameMapSize.Normal,
  difficulty: Difficulty.Medium,
  donateGold: false,
  donateTroops: false,
  gameType: GameType.Private,
  gameMode: GameMode.FFA,
  nations: "disabled",
  bots: 0,
  infiniteGold: false,
  infiniteTroops: false,
  instantBuild: true,
  randomSpawn: false,
  disableNavMesh: true,
  startingGold: 1_000_000,
};

function createSimulator(): HeadlessSelfPlaySimulatorV1 {
  return new HeadlessSelfPlaySimulatorV1({
    game_config: testGameConfig,
    map_loader: new TestGameMapLoader(),
    players: testPlayers,
    controlled_client_ids: testPlayers.map((player) => player.client_id),
    max_ticks_per_episode: 200,
    target_slot_count: 8,
    attack_slot_count: 8,
    spawn_slot_count: 8,
    ratio_bins: [0.25, 0.5, 1],
    rate_limit_window_ticks: 8,
    max_intents_per_window: 3,
  });
}

function buildDeterministicActions(
  response: ResetResponseV1 | StepResponseV1,
): Record<string, ActionV1> {
  const actions: Record<string, ActionV1> = {};
  for (const [clientID, observation] of Object.entries(response.observations)) {
    const mask = response.action_masks[clientID];

    if (mask.spawn.enabled && observation.spawn_slots.length > 0) {
      actions[clientID] = {
        type: "spawn",
        spawn_slot: observation.spawn_slots[0].slot,
      };
      continue;
    }

    if (mask.attack.enabled) {
      const targetSlot = mask.attack.target_slots.findIndex(Boolean);
      const ratioBin = mask.attack.ratio_bins.findIndex(Boolean);
      if (targetSlot >= 0 && ratioBin >= 0) {
        actions[clientID] = {
          type: "attack",
          target_slot: targetSlot,
          ratio_bin: ratioBin,
        };
        continue;
      }
    }

    if (mask.cancel_attack.enabled) {
      const attackSlot = mask.cancel_attack.attack_slots.findIndex(Boolean);
      if (attackSlot >= 0) {
        actions[clientID] = {
          type: "cancel_attack",
          attack_slot: attackSlot,
        };
        continue;
      }
    }

    if (mask.target_player.enabled) {
      const targetSlot = mask.target_player.target_slots.findIndex(Boolean);
      if (targetSlot >= 0) {
        actions[clientID] = {
          type: "target_player",
          target_slot: targetSlot,
        };
        continue;
      }
    }

    actions[clientID] = { type: "no_op" };
  }
  return actions;
}

function snapshotReset(response: ResetResponseV1) {
  const clients = Object.keys(response.observations).sort();
  return {
    tick: response.tick,
    done: response.done,
    clients: clients.map((clientID) => {
      const obs = response.observations[clientID];
      return {
        clientID,
        alive: obs.self_is_alive,
        spawned: obs.self_has_spawned,
        tiles: obs.self_tiles_owned,
        troops: obs.self_troops,
        targetSlots: obs.target_slots.map((slot) => [
          slot.slot,
          slot.kind,
          slot.player_id,
          slot.can_attack,
          slot.can_target,
        ]),
        spawnSlots: obs.spawn_slots.map((slot) => slot.tile),
      };
    }),
  };
}

function snapshotStep(response: StepResponseV1) {
  const clients = Object.keys(response.observations).sort();
  return {
    tick: response.tick,
    done: response.done,
    rewards: clients.map((clientID) => Number(response.rewards[clientID].toFixed(8))),
    invalid: clients.map((clientID) => response.invalid_intents[clientID]),
    players: clients.map((clientID) => {
      const obs = response.observations[clientID];
      return {
        clientID,
        alive: obs.self_is_alive,
        spawned: obs.self_has_spawned,
        tiles: obs.self_tiles_owned,
        troops: obs.self_troops,
        outgoing: obs.self_outgoing_attacks,
        incoming: obs.self_incoming_attacks,
      };
    }),
  };
}

describe("HeadlessSelfPlaySimulatorV1", () => {
  it("encodes and decodes ActionV1 macros to intents", () => {
    const actionMask: ActionMaskV1 = {
      no_op: true,
      attack: {
        enabled: true,
        target_slots: [true, true],
        ratio_bins: [true, true, true],
      },
      cancel_attack: {
        enabled: true,
        attack_slots: [true],
      },
      spawn: {
        enabled: true,
        spawn_slots: [true],
      },
      target_player: {
        enabled: true,
        target_slots: [false, true],
      },
    };
    const context: ActionEncodingContextV1 = {
      player_troops: 100,
      ratio_bins: [0.25, 0.5, 1],
      target_slots: [
        {
          slot: 0,
          kind: "terra_nullius",
          player_id: null,
          client_id: null,
          small_id: null,
          tiles_owned: 0,
          troops: 0,
          can_attack: true,
          can_target: false,
          is_alive: true,
          is_friendly: false,
        },
        {
          slot: 1,
          kind: "player",
          player_id: "ENEMY001",
          client_id: "ENEMY001",
          small_id: 2,
          tiles_owned: 12,
          troops: 140,
          can_attack: true,
          can_target: true,
          is_alive: true,
          is_friendly: false,
        },
      ],
      attack_slots: [
        {
          slot: 0,
          attack_id: "atk-1",
          target_kind: "player",
          target_player_id: "ENEMY001",
          target_client_id: "ENEMY001",
          target_small_id: 2,
          troops: 30,
          retreating: false,
        },
      ],
      spawn_slots: [{ slot: 0, tile: 42, x: 2, y: 6 }],
      action_mask: actionMask,
    };

    const attackAction: ActionV1 = {
      type: "attack",
      target_slot: 1,
      ratio_bin: 1,
    };
    const encodedAttack = encodeActionV1(attackAction, context);
    expect(encodedAttack.legal).toBe(true);
    expect(encodedAttack.intent).toEqual({
      type: "attack",
      targetID: "ENEMY001",
      troops: 50,
    });
    const decodedAttack = decodeIntentV1(encodedAttack.intent!, context);
    expect(decodedAttack).toEqual(attackAction);

    const cancelAction: ActionV1 = {
      type: "cancel_attack",
      attack_slot: 0,
    };
    const encodedCancel = encodeActionV1(cancelAction, context);
    expect(encodedCancel.intent).toEqual({
      type: "cancel_attack",
      attackID: "atk-1",
    });
    expect(decodeIntentV1(encodedCancel.intent!, context)).toEqual(cancelAction);

    const spawnAction: ActionV1 = { type: "spawn", spawn_slot: 0 };
    const encodedSpawn = encodeActionV1(spawnAction, context);
    expect(encodedSpawn.intent).toEqual({
      type: "spawn",
      tile: 42,
    });
    expect(decodeIntentV1(encodedSpawn.intent!, context)).toEqual(spawnAction);

    const targetAction: ActionV1 = {
      type: "target_player",
      target_slot: 1,
    };
    const encodedTarget = encodeActionV1(targetAction, context);
    expect(encodedTarget.intent).toEqual({
      type: "targetPlayer",
      target: "ENEMY001",
    });
    expect(decodeIntentV1(encodedTarget.intent!, context)).toEqual(targetAction);
  });

  it("builds legal masks that block invalid intents", async () => {
    const simulator = createSimulator();
    const reset = await simulator.reset({ seed: 7 });
    const clientID = "AAAA0001";

    expect(reset.action_masks[clientID].spawn.enabled).toBe(true);
    expect(reset.action_masks[clientID].cancel_attack.enabled).toBe(false);

    const invalidAttack = simulator.step({
      actions: {
        [clientID]: {
          type: "attack",
          target_slot: 0,
          ratio_bin: 0,
        },
      },
    });
    expect(invalidAttack.invalid_intents[clientID]).toBe(true);

    const spawnSlot = invalidAttack.observations[clientID].spawn_slots[0];
    expect(spawnSlot).toBeDefined();

    const validSpawn = simulator.step({
      actions: {
        [clientID]: {
          type: "spawn",
          spawn_slot: spawnSlot.slot,
        },
      },
    });
    expect(validSpawn.invalid_intents[clientID]).toBe(false);

  });

  it("is deterministic for seeded resets and rollouts", async () => {
    const simulatorA = createSimulator();
    const simulatorB = createSimulator();

    let responseA = await simulatorA.reset({ seed: 1337 });
    let responseB = await simulatorB.reset({ seed: 1337 });
    expect(snapshotReset(responseA)).toEqual(snapshotReset(responseB));

    for (let i = 0; i < 16; i++) {
      const actions = buildDeterministicActions(responseA);
      const stepA = simulatorA.step({ actions });
      const stepB = simulatorB.step({ actions });
      expect(snapshotStep(stepA)).toEqual(snapshotStep(stepB));
      responseA = stepA;
      responseB = stepB;
      if (stepA.done) {
        break;
      }
    }
  });
});
