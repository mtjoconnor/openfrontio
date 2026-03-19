import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
} from "../core/game/Game";
import { GameConfig } from "../core/Schemas";
import { FilesystemGameMapLoader } from "../core/rl/FilesystemGameMapLoader";
import {
  HeadlessSelfPlaySimulatorV1,
  SelfPlayPlayerConfigV1,
} from "../core/rl/HeadlessSelfPlaySimulator";
import { ActionMaskV1, ActionV1, ObservationV1 } from "../core/rl/RLBridgeV1";

type SmokeOptions = {
  episodes: number;
  maxSteps: number;
  players: number;
  seed: number;
  mode: "student" | "teacher";
  logEvery: number;
  actionRate: number;
  maxEpisodeSeconds: number;
  allowAttack: boolean;
  maxGlobalOutgoingAttacks: number;
  slowStepWarnMs: number;
  warmupPostSpawnTicks: number;
};

function parseArgs(argv: string[]): SmokeOptions {
  const get = (flag: string, fallback: string): string => {
    const index = argv.indexOf(flag);
    if (index < 0 || index + 1 >= argv.length) {
      return fallback;
    }
    return argv[index + 1];
  };

  const modeArg = get("--mode", "student");
  const mode = modeArg === "teacher" ? "teacher" : "student";
  const allowAttack = argv.includes("--allow-attack");

  return {
    episodes: clampInt(get("--episodes", "4"), 1, 10_000),
    // Allow long smoke/soak runs (e.g. 1,000,000 steps) without silently capping.
    maxSteps: clampInt(get("--steps", "300"), 1, 10_000_000),
    players: clampInt(get("--players", "4"), 2, 24),
    seed: clampInt(get("--seed", "1337"), 0, Number.MAX_SAFE_INTEGER),
    mode,
    logEvery: clampInt(get("--log-every", "50"), 1, 100_000),
    actionRate: clampNumber(get("--action-rate", "0.08"), 0.01, 1),
    // Keep a safety timeout, but default high enough for larger step counts.
    maxEpisodeSeconds: clampNumber(get("--max-episode-seconds", "600"), 1, 86_400),
    allowAttack,
    maxGlobalOutgoingAttacks: clampInt(
      get("--max-global-outgoing-attacks", "12"),
      1,
      10_000,
    ),
    slowStepWarnMs: clampInt(get("--slow-step-warn-ms", "100"), 1, 60_000),
    warmupPostSpawnTicks: clampInt(get("--warmup-post-spawn-ticks", "3"), 0, 1000),
  };
}

function clampInt(value: string, min: number, max: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.max(min, Math.min(max, parsed));
}

function clampNumber(value: string, min: number, max: number): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.max(min, Math.min(max, parsed));
}

function makePlayers(count: number): SelfPlayPlayerConfigV1[] {
  const players: SelfPlayPlayerConfigV1[] = [];
  for (let i = 0; i < count; i++) {
    players.push({
      // 8-char IDs satisfy shared ID schema.
      client_id: `P${String(i + 1).padStart(7, "0")}`,
      username: `SmokeBot${i + 1}`,
      is_lobby_creator: i === 0,
    });
  }
  return players;
}

function makeGameConfig(): GameConfig {
  return {
    // Compact Bosphorus is one of the lightest maps in this repository.
    gameMap: GameMapType.BosphorusStraits,
    gameMapSize: GameMapSize.Compact,
    difficulty: Difficulty.Medium,
    donateGold: false,
    donateTroops: false,
    gameType: GameType.Singleplayer,
    gameMode: GameMode.FFA,
    nations: "disabled",
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    // Auto-spawn keeps smoke runs lightweight and avoids repeated spawn retries.
    randomSpawn: true,
    disableNavMesh: true,
    disableAlliances: true,
    startingGold: 100_000,
  };
}

function pickRandom<T>(arr: T[]): T | null {
  if (arr.length === 0) {
    return null;
  }
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomTrueIndices(arr: boolean[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i]) out.push(i);
  }
  return out;
}

function chooseAction(
  obs: ObservationV1,
  mask: ActionMaskV1,
  actionRate: number,
  allowAttack: boolean,
  globalAttackBudgetReached: boolean,
): ActionV1 {
  // Keep spawn fairly eager to leave spawn phase quickly.
  if (mask.spawn.enabled && Math.random() < 0.9) {
    const spawns = randomTrueIndices(mask.spawn.spawn_slots);
    const spawn = pickRandom(spawns);
    if (spawn !== null) {
      return { type: "spawn", spawn_slot: spawn };
    }
  }

  // Conservative action policy to avoid runaway combat/execution explosion.
  if (Math.random() > actionRate) {
    return { type: "no_op" };
  }

  // Keep at most one outgoing attack per player to avoid runaway execution growth.
  if (obs.self_outgoing_attacks > 0) {
    if (mask.cancel_attack.enabled && Math.random() < 0.2) {
      const attacks = randomTrueIndices(mask.cancel_attack.attack_slots);
      const attack = pickRandom(attacks);
      if (attack !== null) {
        return { type: "cancel_attack", attack_slot: attack };
      }
    }
    return { type: "no_op" };
  }

  if (mask.cancel_attack.enabled && Math.random() < 0.1) {
    const attacks = randomTrueIndices(mask.cancel_attack.attack_slots);
    const attack = pickRandom(attacks);
    if (attack !== null) {
      return { type: "cancel_attack", attack_slot: attack };
    }
  }

  if (!globalAttackBudgetReached && allowAttack && mask.attack.enabled) {
    const targets = randomTrueIndices(mask.attack.target_slots);
    const ratios = randomTrueIndices(mask.attack.ratio_bins);
    const target = pickRandom(targets);
    const ratio = pickRandom(ratios);
    if (target !== null && ratio !== null) {
      return { type: "attack", target_slot: target, ratio_bin: ratio };
    }
  }

  if (mask.target_player.enabled && Math.random() < 0.1) {
    const targets = randomTrueIndices(mask.target_player.target_slots);
    const target = pickRandom(targets);
    if (target !== null) {
      return { type: "target_player", target_slot: target };
    }
  }

  void obs;
  return { type: "no_op" };
}

function totalOutgoingAttacks(observations: Record<string, ObservationV1>): number {
  let total = 0;
  for (const obs of Object.values(observations)) {
    total += obs.self_outgoing_attacks;
  }
  return total;
}

function inSpawnPhase(observations: Record<string, ObservationV1>): boolean {
  const first = Object.values(observations)[0];
  return first?.in_spawn_phase ?? false;
}

async function run(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const players = makePlayers(options.players);

  const simulator = new HeadlessSelfPlaySimulatorV1({
    game_config: makeGameConfig(),
    map_loader: new FilesystemGameMapLoader(),
    players,
    controlled_client_ids: players.map((p) => p.client_id),
    max_ticks_per_episode: options.maxSteps,
  });

  const aggregate = {
    wins: new Map<string, number>(),
    meanPlacement: new Map<string, number[]>(),
    meanReward: new Map<string, number[]>(),
    invalidIntents: 0,
    episodesDone: 0,
  };

  console.log(
    `[rl-smoke] starting: episodes=${options.episodes} players=${options.players} steps<=${options.maxSteps} seed=${options.seed} mode=${options.mode} logEvery=${options.logEvery} actionRate=${options.actionRate} maxEpisodeSeconds=${options.maxEpisodeSeconds} allowAttack=${options.allowAttack} maxGlobalOutgoingAttacks=${options.maxGlobalOutgoingAttacks} warmupPostSpawnTicks=${options.warmupPostSpawnTicks}`,
  );

  const startedAtMs = Date.now();
  for (let episode = 0; episode < options.episodes; episode++) {
    let response = await simulator.reset({
      seed: options.seed + episode,
      observation_mode: options.mode,
    });

    // Warm up through spawn + a few post-spawn ticks so one-time transition work
    // does not count against episode timeout or step budget.
    let warmupTicksRemaining = options.warmupPostSpawnTicks;
    let warmupSteps = 0;
    while (!response.done) {
      const currentlyInSpawn = inSpawnPhase(response.observations);
      if (!currentlyInSpawn && warmupTicksRemaining <= 0) {
        break;
      }
      const noOps: Record<string, ActionV1> = {};
      for (const clientID of Object.keys(response.observations)) {
        noOps[clientID] = { type: "no_op" };
      }
      const warmupStep = simulator.step({ actions: noOps });
      warmupSteps++;
      if (!currentlyInSpawn && !inSpawnPhase(warmupStep.observations)) {
        warmupTicksRemaining--;
      }
      response = warmupStep;
      if (warmupSteps % 100 === 0) {
        console.log(
          `[rl-smoke] episode ${episode + 1}/${options.episodes} warmup step=${warmupSteps} tick=${response.tick} inSpawn=${inSpawnPhase(response.observations)}`,
        );
      }
      if (warmupSteps > 20_000) {
        console.log(
          `[rl-smoke] episode ${episode + 1}/${options.episodes} warmup safety break`,
        );
        break;
      }
    }

    let steps = 0;
    const episodeStartMs = Date.now();
    let timedOut = false;
    while (!response.done && steps < options.maxSteps) {
      const actions: Record<string, ActionV1> = {};
      const outgoingNow = totalOutgoingAttacks(response.observations);
      const globalAttackBudgetReached =
        outgoingNow >= options.maxGlobalOutgoingAttacks;
      for (const [clientID, obs] of Object.entries(response.observations)) {
        actions[clientID] = chooseAction(
          obs,
          response.action_masks[clientID],
          options.actionRate,
          options.allowAttack,
          globalAttackBudgetReached,
        );
      }

      const stepStartMs = Date.now();
      const step = simulator.step({ actions });
      const stepDurationMs = Date.now() - stepStartMs;
      if (stepDurationMs >= options.slowStepWarnMs) {
        const outgoingAfter = totalOutgoingAttacks(step.observations);
        console.log(
          `[rl-smoke] slow-step step=${steps + 1} ms=${stepDurationMs} outgoingBefore=${outgoingNow} outgoingAfter=${outgoingAfter}`,
        );
      }
      for (const invalid of Object.values(step.invalid_intents)) {
        if (invalid) aggregate.invalidIntents++;
      }
      response = step;
      steps++;

      if ((Date.now() - episodeStartMs) / 1000 >= options.maxEpisodeSeconds) {
        timedOut = true;
        break;
      }

      if (steps % options.logEvery === 0) {
        console.log(
          `[rl-smoke] episode ${episode + 1}/${options.episodes} progress step=${steps}/${options.maxSteps} tick=${response.tick} done=${response.done}`,
        );
      }
    }

    aggregate.episodesDone++;
    const stats = response.episode_stats.players;
    const isTerminalEpisode = response.done;
    for (const [clientID, s] of Object.entries(stats)) {
      if (isTerminalEpisode && s.won) {
        aggregate.wins.set(clientID, (aggregate.wins.get(clientID) ?? 0) + 1);
      }
      const placements = aggregate.meanPlacement.get(clientID) ?? [];
      placements.push(s.placement);
      aggregate.meanPlacement.set(clientID, placements);

      const rewards = aggregate.meanReward.get(clientID) ?? [];
      rewards.push(s.total_reward);
      aggregate.meanReward.set(clientID, rewards);
    }

    const winner = response.episode_stats.winner_client_id ?? "none";
    console.log(
      `[rl-smoke] episode ${episode + 1}/${options.episodes} done=${response.done} tick=${response.tick} winner=${winner}${timedOut ? " timeout=true" : ""}`,
    );
  }

  console.log("");
  console.log("[rl-smoke] summary");
  for (const player of players) {
    const wins = aggregate.wins.get(player.client_id) ?? 0;
    const placements = aggregate.meanPlacement.get(player.client_id) ?? [];
    const rewards = aggregate.meanReward.get(player.client_id) ?? [];
    const meanPlacement =
      placements.length === 0
        ? 0
        : placements.reduce((a, b) => a + b, 0) / placements.length;
    const meanReward =
      rewards.length === 0
        ? 0
        : rewards.reduce((a, b) => a + b, 0) / rewards.length;
    console.log(
      `${player.client_id} wins=${wins} meanPlacement=${meanPlacement.toFixed(2)} meanReward=${meanReward.toFixed(3)}`,
    );
  }
  console.log(
    `[rl-smoke] invalidIntents=${aggregate.invalidIntents} episodes=${aggregate.episodesDone} terminalEpisodesOnlyForWins=true`,
  );
  console.log(
    `[rl-smoke] elapsedSeconds=${((Date.now() - startedAtMs) / 1000).toFixed(2)}`,
  );
}

run().catch((error: unknown) => {
  console.error("[rl-smoke] failed", error);
  process.exitCode = 1;
});
