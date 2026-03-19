import { JWK } from "jose";
import { DefaultConfig, DefaultServerConfig } from "../configuration/DefaultConfig";
import { GameEnv } from "../configuration/Config";
import { Executor } from "../execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../execution/RecomputeRailClusterExecution";
import { getSpawnTiles } from "../execution/Util";
import { WinCheckExecution } from "../execution/WinCheckExecution";
import { createNationsForGame } from "../game/NationCreation";
import {
  Game,
  GameMapSize,
  GameMapType,
  Player,
  PlayerInfo,
  PlayerType,
  TeamGameSpawnAreas,
  UnitType,
} from "../game/Game";
import { createGame } from "../game/GameImpl";
import { GameMapLoader } from "../game/GameMapLoader";
import { genTerrainFromBin, Nation as ManifestNation } from "../game/TerrainMapLoader";
import { UserSettings } from "../game/UserSettings";
import { PseudoRandom } from "../PseudoRandom";
import { ClientID, GameConfig, GameID, GameStartInfo, StampedIntent, Turn } from "../Schemas";
import { simpleHash } from "../Util";
import {
  ActionEncodingContextV1,
  ActionMaskV1,
  ActionV1,
  AttackSlotV1,
  EpisodeStatsV1,
  ObservationModeV1,
  ObservationV1,
  PublicPlayerObsV1,
  ResetRequestV1,
  ResetResponseV1,
  RL_BRIDGE_SCHEMA_VERSION_V1,
  SpawnSlotV1,
  StepRequestV1,
  StepResponseV1,
  TargetSlotV1,
  TeacherObsV1,
  encodeActionV1,
  actionSignatureV1,
} from "./RLBridgeV1";

const DEFAULT_RATIO_BINS = [0.1, 0.2, 0.33, 0.5, 0.75, 1];
const DEFAULT_MAX_TICKS_PER_EPISODE = 30 * 60 * 10;
const DEFAULT_TARGET_SLOT_COUNT = 32;
const DEFAULT_ATTACK_SLOT_COUNT = 32;
const DEFAULT_SPAWN_SLOT_COUNT = 32;
const NO_OP_ACTION: ActionV1 = { type: "no_op" };

export interface RewardWeightsV1 {
  win_reward: number;
  placement_reward: number;
  survival_reward: number;
  territory_delta_reward: number;
  troop_efficiency_reward: number;
  survival_pressure_reward: number;
  invalid_intent_penalty: number;
  rate_limit_penalty: number;
  degenerate_spam_penalty: number;
  elimination_penalty: number;
}

const DEFAULT_REWARD_WEIGHTS: RewardWeightsV1 = {
  win_reward: 1,
  placement_reward: 0.4,
  survival_reward: 0.15,
  territory_delta_reward: 2.5,
  troop_efficiency_reward: 0.35,
  survival_pressure_reward: 0.3,
  invalid_intent_penalty: -0.05,
  rate_limit_penalty: -0.01,
  degenerate_spam_penalty: -0.01,
  elimination_penalty: -0.25,
};

export interface SelfPlayPlayerConfigV1 {
  client_id: ClientID;
  username: string;
  clan_tag?: string | null;
  is_lobby_creator?: boolean;
}

export interface HeadlessSelfPlaySimulatorConfigV1 {
  game_config: GameConfig;
  map_loader: GameMapLoader;
  players: SelfPlayPlayerConfigV1[];
  controlled_client_ids?: ClientID[];
  max_ticks_per_episode?: number;
  target_slot_count?: number;
  attack_slot_count?: number;
  spawn_slot_count?: number;
  ratio_bins?: number[];
  reward_weights?: Partial<RewardWeightsV1>;
  rate_limit_window_ticks?: number;
  max_intents_per_window?: number;
  degenerate_repeat_threshold?: number;
}

interface PlayerMetrics {
  territoryShare: number;
  troopEfficiency: number;
  survivalPressure: number;
  alive: boolean;
}

interface PlayerRuntimeState {
  metrics: PlayerMetrics;
  wasAlive: boolean;
  eliminatedTick: number | null;
  invalidIntents: number;
  rateLimitPressure: number;
  degenerateSpam: number;
  totalReward: number;
  actionRepeatCount: number;
  lastActionSignature: string | null;
  recentIntentTicks: number[];
}

interface EnvironmentRuntimeState {
  seed: number;
  gameID: GameID;
  game: Game;
  executor: Executor;
  turnNumber: number;
  done: boolean;
  observationMode: ObservationModeV1;
  controlledClientIDs: ClientID[];
  playerRuntime: Map<ClientID, PlayerRuntimeState>;
}

interface PlayerContext {
  observation: ObservationV1;
  actionMask: ActionMaskV1;
  actionContext: ActionEncodingContextV1;
}

class HeadlessServerConfig extends DefaultServerConfig {
  // The simulator runs offline and does not need live server integrations.
  turnstileSiteKey(): string {
    return "";
  }
  jwtAudience(): string {
    return "localhost";
  }
  numWorkers(): number {
    return 1;
  }
  env(): GameEnv {
    return GameEnv.Dev;
  }
  async jwkPublicKey(): Promise<JWK> {
    return {} as JWK;
  }
}

export class HeadlessSelfPlaySimulatorV1 {
  private readonly ratioBins: number[];
  private readonly maxTicksPerEpisode: number;
  private readonly targetSlotCount: number;
  private readonly attackSlotCount: number;
  private readonly spawnSlotCount: number;
  private readonly rewardWeights: RewardWeightsV1;
  private readonly rateLimitWindowTicks: number;
  private readonly maxIntentsPerWindow: number;
  private readonly degenerateRepeatThreshold: number;
  private readonly serverConfig = new HeadlessServerConfig();
  private state: EnvironmentRuntimeState | null = null;

  constructor(private readonly config: HeadlessSelfPlaySimulatorConfigV1) {
    this.ratioBins = normalizeRatioBins(config.ratio_bins ?? DEFAULT_RATIO_BINS);
    this.maxTicksPerEpisode = Math.max(
      1,
      Math.floor(config.max_ticks_per_episode ?? DEFAULT_MAX_TICKS_PER_EPISODE),
    );
    this.targetSlotCount = Math.max(
      1,
      Math.floor(config.target_slot_count ?? DEFAULT_TARGET_SLOT_COUNT),
    );
    this.attackSlotCount = Math.max(
      1,
      Math.floor(config.attack_slot_count ?? DEFAULT_ATTACK_SLOT_COUNT),
    );
    this.spawnSlotCount = Math.max(
      1,
      Math.floor(config.spawn_slot_count ?? DEFAULT_SPAWN_SLOT_COUNT),
    );
    this.rewardWeights = {
      ...DEFAULT_REWARD_WEIGHTS,
      ...(config.reward_weights ?? {}),
    };
    this.rateLimitWindowTicks = Math.max(
      1,
      Math.floor(config.rate_limit_window_ticks ?? 10),
    );
    this.maxIntentsPerWindow = Math.max(
      1,
      Math.floor(config.max_intents_per_window ?? 5),
    );
    this.degenerateRepeatThreshold = Math.max(
      1,
      Math.floor(config.degenerate_repeat_threshold ?? 3),
    );
  }

  async reset(request: ResetRequestV1): Promise<ResetResponseV1> {
    // reset(seed) rebuilds a brand new game state for deterministic episodes.
    const observationMode = request.observation_mode ?? "student";
    const controlledClientIDs = this.resolveControlledClientIDs(
      request.controlled_client_ids,
    );
    const state = await this.createRuntimeState(
      request.seed,
      observationMode,
      controlledClientIDs,
    );
    this.state = state;

    const contexts = this.buildContexts(state);
    return {
      schema_version: RL_BRIDGE_SCHEMA_VERSION_V1,
      tick: state.game.ticks(),
      done: state.done,
      observations: toObservationRecord(contexts),
      action_masks: toActionMaskRecord(contexts),
      episode_stats: this.buildEpisodeStats(state),
    };
  }

  step(request: StepRequestV1): StepResponseV1 {
    if (this.state === null) {
      throw new Error("Simulator must be reset before step()");
    }
    const state = this.state;

    const zeroRewards = makeNumberRecord(state.controlledClientIDs, 0);
    const zeroInvalid = makeBooleanRecord(state.controlledClientIDs, false);
    if (state.done) {
      const doneContexts = this.buildContexts(state);
      return {
        schema_version: RL_BRIDGE_SCHEMA_VERSION_V1,
        tick: state.game.ticks(),
        done: true,
        observations: toObservationRecord(doneContexts),
        action_masks: toActionMaskRecord(doneContexts),
        rewards: zeroRewards,
        invalid_intents: zeroInvalid,
        episode_stats: this.buildEpisodeStats(state),
      };
    }

    const preStepContexts = this.buildContexts(state);
    const intents: StampedIntent[] = [];
    const rewards = makeNumberRecord(state.controlledClientIDs, 0);
    const invalidIntents = makeBooleanRecord(state.controlledClientIDs, false);
    const actions = request.actions ?? {};

    for (const clientID of state.controlledClientIDs) {
      const action = actions[clientID] ?? NO_OP_ACTION;
      const context = preStepContexts.get(clientID);
      if (!context) {
        continue;
      }
      const runtime = state.playerRuntime.get(clientID);
      if (!runtime) {
        continue;
      }

      const encoded = encodeActionV1(action, context.actionContext);
      if (!encoded.legal) {
        invalidIntents[clientID] = true;
        runtime.invalidIntents += 1;
        rewards[clientID] += this.rewardWeights.invalid_intent_penalty;
        this.resetSpam(runtime);
        continue;
      }

      if (action.type === "no_op") {
        this.resetSpam(runtime);
        continue;
      }

      if (encoded.intent !== null) {
        // Reuse the same intent pipeline as multiplayer to keep behavior parity.
        intents.push({
          ...encoded.intent,
          clientID,
        });
        rewards[clientID] += this.applyRateLimitPenalty(runtime, state.game.ticks());
        rewards[clientID] += this.applyDegeneratePenalty(runtime, action);
      }
    }

    const turn: Turn = {
      turnNumber: state.turnNumber,
      intents,
    };
    state.turnNumber += 1;

    state.game.addExecution(...state.executor.createExecs(turn));
    state.game.executeNextTick();

    // Dense shaping uses metric deltas from pre-step to post-step state.
    for (const clientID of state.controlledClientIDs) {
      const runtime = state.playerRuntime.get(clientID);
      const player = state.game.playerByClientID(clientID);
      if (!runtime || !player) {
        continue;
      }
      const nextMetrics = this.computePlayerMetrics(state.game, player);
      const denseReward =
        this.rewardWeights.territory_delta_reward *
          (nextMetrics.territoryShare - runtime.metrics.territoryShare) +
        this.rewardWeights.troop_efficiency_reward *
          (nextMetrics.troopEfficiency - runtime.metrics.troopEfficiency) +
        this.rewardWeights.survival_pressure_reward *
          (runtime.metrics.survivalPressure - nextMetrics.survivalPressure);
      rewards[clientID] += denseReward;

      if (runtime.wasAlive && !nextMetrics.alive) {
        rewards[clientID] += this.rewardWeights.elimination_penalty;
        runtime.eliminatedTick ??= state.game.ticks();
      }

      runtime.wasAlive = nextMetrics.alive;
      runtime.metrics = nextMetrics;
    }

    state.done = this.shouldTerminate(state);
    if (state.done) {
      // Terminal bonus/penalty is applied once when episode transitions to done.
      this.applyTerminalRewards(state, rewards);
    }

    for (const clientID of state.controlledClientIDs) {
      const runtime = state.playerRuntime.get(clientID);
      if (!runtime) {
        continue;
      }
      runtime.totalReward += rewards[clientID];
    }

    const postStepContexts = this.buildContexts(state);
    return {
      schema_version: RL_BRIDGE_SCHEMA_VERSION_V1,
      tick: state.game.ticks(),
      done: state.done,
      observations: toObservationRecord(postStepContexts),
      action_masks: toActionMaskRecord(postStepContexts),
      rewards,
      invalid_intents: invalidIntents,
      episode_stats: this.buildEpisodeStats(state),
    };
  }

  private resolveControlledClientIDs(
    requested?: ClientID[],
  ): ClientID[] {
    const available = new Set(this.config.players.map((p) => p.client_id));
    const defaults =
      requested ??
      this.config.controlled_client_ids ??
      this.config.players.map((p) => p.client_id);
    const deduped = [...new Set(defaults)];
    for (const clientID of deduped) {
      if (!available.has(clientID)) {
        throw new Error(`Unknown controlled client ID: ${clientID}`);
      }
    }
    return deduped;
  }

  private async createRuntimeState(
    seed: number,
    observationMode: ObservationModeV1,
    controlledClientIDs: ClientID[],
  ): Promise<EnvironmentRuntimeState> {
    const gameID = makeGameID(seed);
    // Use a fresh map instance every reset so mutable tile state never leaks.
    const terrain = await loadTerrainMapFresh(
      this.config.game_config.gameMap,
      this.config.game_config.gameMapSize,
      this.config.map_loader,
    );

    const random = new PseudoRandom(simpleHash(gameID));
    const players = this.config.players.map((p, index) => {
      return new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.client_id,
        random.nextID(),
        p.is_lobby_creator ?? index === 0,
        p.clan_tag ?? null,
      );
    });

    const gameStartInfo = this.makeGameStartInfo(gameID);
    const nations = createNationsForGame(
      gameStartInfo,
      terrain.nations,
      players.length,
      random,
    );

    const config = new DefaultConfig(
      this.serverConfig,
      this.config.game_config,
      new UserSettings(),
      false,
    );
    const game = createGame(
      players,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      config,
      terrain.teamGameSpawnAreas,
    );
    const executor = new Executor(game, gameID, undefined);
    this.initializeEnvironmentExecutions(game, executor);

    const playerRuntime = new Map<ClientID, PlayerRuntimeState>();
    for (const clientID of controlledClientIDs) {
      const player = game.playerByClientID(clientID);
      if (!player) {
        throw new Error(`Player not found for client ID: ${clientID}`);
      }
      const metrics = this.computePlayerMetrics(game, player);
      playerRuntime.set(clientID, {
        metrics,
        wasAlive: metrics.alive,
        eliminatedTick: null,
        invalidIntents: 0,
        rateLimitPressure: 0,
        degenerateSpam: 0,
        totalReward: 0,
        actionRepeatCount: 0,
        lastActionSignature: null,
        recentIntentTicks: [],
      });
    }

    return {
      seed,
      gameID,
      game,
      executor,
      turnNumber: 0,
      done: false,
      observationMode,
      controlledClientIDs,
      playerRuntime,
    };
  }

  private initializeEnvironmentExecutions(game: Game, executor: Executor): void {
    // Mirror GameRunner initialization so training uses production mechanics.
    if (game.config().isRandomSpawn()) {
      game.addExecution(...executor.spawnPlayers());
    }
    if (game.config().bots() > 0) {
      game.addExecution(...executor.spawnTribes(game.config().bots()));
    }
    if (game.config().spawnNations()) {
      game.addExecution(...executor.nationExecutions());
    }
    game.addExecution(new WinCheckExecution());
    if (!game.config().isUnitDisabled(UnitType.Factory)) {
      game.addExecution(new RecomputeRailClusterExecution(game.railNetwork()));
    }
  }

  private makeGameStartInfo(gameID: GameID): GameStartInfo {
    return {
      gameID,
      lobbyCreatedAt: 0,
      config: this.config.game_config,
      players: this.config.players.map((p) => ({
        clientID: p.client_id,
        username: p.username,
        clanTag: p.clan_tag ?? null,
      })),
    };
  }

  private buildContexts(state: EnvironmentRuntimeState): Map<ClientID, PlayerContext> {
    // Build observation + mask + encoding context from the same slot lists.
    const contexts = new Map<ClientID, PlayerContext>();
    for (const clientID of state.controlledClientIDs) {
      const player = state.game.playerByClientID(clientID);
      if (!player) {
        continue;
      }

      const targetSlots = this.buildTargetSlots(state.game, player);
      const attackSlots = this.buildAttackSlots(player);
      const spawnSlots = this.buildSpawnSlots(state, player);
      const actionMask = this.buildActionMask(
        state.game,
        player,
        targetSlots,
        attackSlots,
        spawnSlots,
      );

      const actionContext: ActionEncodingContextV1 = {
        player_troops: player.troops(),
        ratio_bins: this.ratioBins,
        target_slots: targetSlots,
        attack_slots: attackSlots,
        spawn_slots: spawnSlots,
        action_mask: actionMask,
      };

      contexts.set(clientID, {
        observation: this.buildObservation(
          state,
          player,
          targetSlots,
          attackSlots,
          spawnSlots,
        ),
        actionMask,
        actionContext,
      });
    }
    return contexts;
  }

  private buildTargetSlots(game: Game, self: Player): TargetSlotV1[] {
    const slots: TargetSlotV1[] = [];
    const canIssueCombat = self.isAlive() && self.hasSpawned() && !game.inSpawnPhase();
    slots.push({
      slot: 0,
      kind: "terra_nullius",
      player_id: null,
      client_id: null,
      small_id: null,
      tiles_owned: 0,
      troops: 0,
      can_attack: canIssueCombat && this.canAttackTarget(game, self, null),
      can_target: false,
      is_alive: true,
      is_friendly: false,
    });

    // Stable ordering is required for reproducible slot semantics.
    const others = game
      .allPlayers()
      .filter((player) => player.id() !== self.id() && player.isAlive())
      .sort((a, b) => a.smallID() - b.smallID());

    for (const other of others) {
      if (slots.length >= this.targetSlotCount) {
        break;
      }
      slots.push({
        slot: slots.length,
        kind: "player",
        player_id: other.id(),
        client_id: other.clientID(),
        small_id: other.smallID(),
        tiles_owned: other.numTilesOwned(),
        troops: other.troops(),
        can_attack: canIssueCombat && this.canAttackTarget(game, self, other),
        can_target: canIssueCombat && self.canTarget(other),
        is_alive: other.isAlive(),
        is_friendly: self.isFriendly(other, true),
      });
    }
    return slots;
  }

  private canAttackTarget(game: Game, self: Player, target: Player | null): boolean {
    // "Attackable" here means there exists at least one legal border tile attack.
    if (!self.isAlive() || !self.hasSpawned() || game.inSpawnPhase()) {
      return false;
    }

    for (const borderTile of self.borderTiles()) {
      for (const neighbor of game.neighbors(borderTile)) {
        if (!game.isLand(neighbor)) {
          continue;
        }
        if (!self.canAttack(neighbor)) {
          continue;
        }
        const owner = game.owner(neighbor);
        if (target === null) {
          if (!owner.isPlayer()) {
            return true;
          }
          continue;
        }
        if (owner.isPlayer() && owner.id() === target.id()) {
          return true;
        }
      }
    }
    return false;
  }

  private buildAttackSlots(self: Player): AttackSlotV1[] {
    const slots: AttackSlotV1[] = [];
    const attacks = self
      .outgoingAttacks()
      .filter((attack) => attack.isActive())
      .sort((a, b) => a.id().localeCompare(b.id()));

    for (const attack of attacks) {
      if (slots.length >= this.attackSlotCount) {
        break;
      }
      const target = attack.target();
      const targetIsPlayer = target.isPlayer();
      slots.push({
        slot: slots.length,
        attack_id: attack.id(),
        target_kind: targetIsPlayer ? "player" : "terra_nullius",
        target_player_id: targetIsPlayer ? target.id() : null,
        target_client_id: targetIsPlayer ? target.clientID() : null,
        target_small_id: targetIsPlayer ? target.smallID() : null,
        troops: attack.troops(),
        retreating: attack.retreating(),
      });
    }

    return slots;
  }

  private buildSpawnSlots(state: EnvironmentRuntimeState, self: Player): SpawnSlotV1[] {
    const game = state.game;
    if (!this.canSpawn(game, self)) {
      return [];
    }

    const slots: SpawnSlotV1[] = [];
    const seenTiles = new Set<number>();
    // Seeded pseudo-random sampling first, then deterministic grid fallback.
    const attempts = Math.max(this.spawnSlotCount * 40, this.spawnSlotCount);

    for (let i = 0; i < attempts && slots.length < this.spawnSlotCount; i++) {
      const x = seededInt(
        state.seed,
        self.smallID(),
        game.ticks(),
        i * 2 + 17,
        game.width(),
      );
      const y = seededInt(
        state.seed,
        self.smallID(),
        game.ticks(),
        i * 2 + 23,
        game.height(),
      );
      const tile = game.ref(x, y);
      if (seenTiles.has(tile)) {
        continue;
      }
      seenTiles.add(tile);
      if (!game.isLand(tile)) {
        continue;
      }
      if (!this.isSpawnTileCandidate(game, tile)) {
        continue;
      }
      slots.push({
        slot: slots.length,
        tile,
        x,
        y,
      });
    }

    if (slots.length < this.spawnSlotCount) {
      const stride = Math.max(
        1,
        Math.floor(
          Math.sqrt(
            (game.width() * game.height()) / Math.max(1, this.spawnSlotCount * 4),
          ),
        ),
      );
      for (let y = 0; y < game.height() && slots.length < this.spawnSlotCount; y += stride) {
        for (
          let x = 0;
          x < game.width() && slots.length < this.spawnSlotCount;
          x += stride
        ) {
          const tile = game.ref(x, y);
          if (seenTiles.has(tile)) {
            continue;
          }
          seenTiles.add(tile);
          if (!game.isLand(tile)) {
            continue;
          }
          if (!this.isSpawnTileCandidate(game, tile)) {
            continue;
          }
          slots.push({
            slot: slots.length,
            tile,
            x,
            y,
          });
        }
      }
    }

    return slots;
  }

  private canSpawn(game: Game, self: Player): boolean {
    return game.inSpawnPhase() && !self.hasSpawned();
  }

  private isSpawnTileCandidate(game: Game, tile: number): boolean {
    return getSpawnTiles(game, tile, false).length > 0;
  }

  private buildActionMask(
    game: Game,
    self: Player,
    targetSlots: TargetSlotV1[],
    attackSlots: AttackSlotV1[],
    spawnSlots: SpawnSlotV1[],
  ): ActionMaskV1 {
    const canIssueCombat = self.isAlive() && self.hasSpawned() && !game.inSpawnPhase();

    const attackTargetMask = Array.from(
      { length: this.targetSlotCount },
      () => false,
    );
    for (const slot of targetSlots) {
      attackTargetMask[slot.slot] = canIssueCombat && slot.can_attack;
    }

    // Ratio bins are globally static but still masked during non-combat states.
    const ratioMask = this.ratioBins.map(
      (ratio) => canIssueCombat && ratio > 0 && self.troops() > 0,
    );

    const cancelAttackMask = Array.from(
      { length: this.attackSlotCount },
      () => false,
    );
    for (const slot of attackSlots) {
      cancelAttackMask[slot.slot] = canIssueCombat && !slot.retreating;
    }

    const spawnMask = Array.from({ length: this.spawnSlotCount }, () => false);
    for (const slot of spawnSlots) {
      spawnMask[slot.slot] = this.canSpawn(game, self);
    }

    const targetPlayerMask = Array.from(
      { length: this.targetSlotCount },
      () => false,
    );
    for (const slot of targetSlots) {
      targetPlayerMask[slot.slot] =
        canIssueCombat && slot.kind === "player" && slot.can_target;
    }

    return {
      no_op: true,
      attack: {
        enabled: attackTargetMask.some(Boolean) && ratioMask.some(Boolean),
        target_slots: attackTargetMask,
        ratio_bins: ratioMask,
      },
      cancel_attack: {
        enabled: cancelAttackMask.some(Boolean),
        attack_slots: cancelAttackMask,
      },
      spawn: {
        enabled: spawnMask.some(Boolean),
        spawn_slots: spawnMask,
      },
      target_player: {
        enabled: targetPlayerMask.some(Boolean),
        target_slots: targetPlayerMask,
      },
    };
  }

  private buildObservation(
    state: EnvironmentRuntimeState,
    self: Player,
    targetSlots: TargetSlotV1[],
    attackSlots: AttackSlotV1[],
    spawnSlots: SpawnSlotV1[],
  ): ObservationV1 {
    const game = state.game;
    const players = game.allPlayers().sort((a, b) => a.smallID() - b.smallID());
    const playersPublic: PublicPlayerObsV1[] = players.map((player) => ({
      player_id: player.id(),
      client_id: player.clientID(),
      small_id: player.smallID(),
      is_self: player.id() === self.id(),
      is_alive: player.isAlive(),
      is_disconnected: player.isDisconnected(),
      has_spawned: player.hasSpawned(),
      tiles_owned: player.numTilesOwned(),
      troops: player.troops(),
      gold: bigintToNumber(player.gold()),
      outgoing_attacks: player.outgoingAttacks().length,
      incoming_attacks: player.incomingAttacks().length,
    }));

    // Shared base fields for both student and teacher observation modes.
    const base = {
      schema_version: RL_BRIDGE_SCHEMA_VERSION_V1,
      mode: state.observationMode,
      tick: game.ticks(),
      in_spawn_phase: game.inSpawnPhase(),
      self_client_id: self.clientID() ?? "",
      self_player_id: self.id(),
      self_small_id: self.smallID(),
      self_is_alive: self.isAlive(),
      self_has_spawned: self.hasSpawned(),
      self_tiles_owned: self.numTilesOwned(),
      self_troops: self.troops(),
      self_gold: bigintToNumber(self.gold()),
      self_outgoing_attacks: self.outgoingAttacks().length,
      self_incoming_attacks: self.incomingAttacks().length,
      target_slots: targetSlots,
      attack_slots: attackSlots,
      spawn_slots: spawnSlots,
    } as const;

    if (state.observationMode === "student") {
      return {
        ...base,
        mode: "student",
        players_public: playersPublic,
      };
    }

    // Teacher adds privileged-only fields for training acceleration.
    const teacherObs: TeacherObsV1 = {
      ...base,
      mode: "teacher",
      players_public: playersPublic,
      players_privileged: players.map((player) => ({
        player_id: player.id(),
        client_id: player.clientID(),
        small_id: player.smallID(),
        spawn_tile: player.spawnTile() ?? null,
        border_tiles: player.borderTiles().size,
        is_immune: player.isImmune(),
        relation_to_self:
          player.id() === self.id() ? 0 : Number(self.relation(player)),
        can_attack_self:
          player.id() === self.id() ? false : player.canAttackPlayer(self, true),
      })),
      global_privileged: {
        alive_players: game.players().length,
        num_land_tiles: game.numLandTiles(),
        num_tiles_with_fallout: game.numTilesWithFallout(),
        winner_set: game.getWinner() !== null,
      },
    };
    return teacherObs;
  }

  private computePlayerMetrics(game: Game, player: Player): PlayerMetrics {
    const alive = player.isAlive();
    if (!alive) {
      return {
        territoryShare: 0,
        troopEfficiency: 0,
        survivalPressure: 1,
        alive: false,
      };
    }

    const tilesOwned = Math.max(1, player.numTilesOwned());
    const territoryShare = player.numTilesOwned() / Math.max(1, game.numLandTiles());
    // Log scaling prevents very large troop counts from dominating reward.
    const troopEfficiency = Math.log1p(player.troops() / tilesOwned);
    const incomingTroops = player
      .incomingAttacks()
      .reduce((sum, attack) => sum + attack.troops(), 0);
    // Pressure rises when incoming attack volume is high relative to strength.
    const survivalPressure = Math.min(
      5,
      incomingTroops / Math.max(1, player.troops() + player.numTilesOwned()),
    );

    return {
      territoryShare,
      troopEfficiency,
      survivalPressure,
      alive: true,
    };
  }

  private shouldTerminate(state: EnvironmentRuntimeState): boolean {
    if (state.game.getWinner() !== null) {
      return true;
    }
    if (state.game.ticks() >= this.maxTicksPerEpisode) {
      return true;
    }
    // Early-stop once all controlled agents are dead.
    return state.controlledClientIDs.every((clientID) => {
      const player = state.game.playerByClientID(clientID);
      return !player || !player.isAlive();
    });
  }

  private applyTerminalRewards(
    state: EnvironmentRuntimeState,
    rewards: Record<ClientID, number>,
  ): void {
    const allPlayers = state.game.allPlayers();
    const placements = computePlacements(allPlayers);
    const denominator = Math.max(1, allPlayers.length - 1);

    for (const clientID of state.controlledClientIDs) {
      const player = state.game.playerByClientID(clientID);
      const runtime = state.playerRuntime.get(clientID);
      if (!runtime || !player) {
        continue;
      }
      const placement = placements.get(player.id()) ?? allPlayers.length;
      // Map placement into [0, 1], where first place is 1.
      const placementNormalized = (allPlayers.length - placement) / denominator;

      rewards[clientID] +=
        placementNormalized * this.rewardWeights.placement_reward;
      if (placement === 1) {
        rewards[clientID] += this.rewardWeights.win_reward;
      }
      if (player.isAlive()) {
        rewards[clientID] += this.rewardWeights.survival_reward;
      }
    }
  }

  private buildEpisodeStats(state: EnvironmentRuntimeState): EpisodeStatsV1 {
    const allPlayers = state.game.allPlayers();
    const placements = computePlacements(allPlayers);
    const winnerClientID = getWinnerClientID(state.game);
    const players: EpisodeStatsV1["players"] = {};

    for (const clientID of state.controlledClientIDs) {
      const player = state.game.playerByClientID(clientID);
      const runtime = state.playerRuntime.get(clientID);
      if (!player || !runtime) {
        continue;
      }

      const placement = placements.get(player.id()) ?? allPlayers.length;
      players[clientID] = {
        client_id: clientID,
        won: placement === 1,
        placement,
        survived: player.isAlive(),
        survived_ticks: runtime.eliminatedTick ?? state.game.ticks(),
        invalid_intents: runtime.invalidIntents,
        rate_limit_pressure: runtime.rateLimitPressure,
        degenerate_spam: runtime.degenerateSpam,
        total_reward: runtime.totalReward,
      };
    }

    return {
      schema_version: RL_BRIDGE_SCHEMA_VERSION_V1,
      tick: state.game.ticks(),
      done: state.done,
      winner_client_id: winnerClientID,
      players,
    };
  }

  private applyRateLimitPenalty(runtime: PlayerRuntimeState, tick: number): number {
    // Sliding-window intent budget approximation used for reward shaping.
    const minTick = tick - this.rateLimitWindowTicks + 1;
    runtime.recentIntentTicks = runtime.recentIntentTicks.filter((t) => t >= minTick);
    runtime.recentIntentTicks.push(tick);

    if (runtime.recentIntentTicks.length > this.maxIntentsPerWindow) {
      runtime.rateLimitPressure += 1;
      return this.rewardWeights.rate_limit_penalty;
    }
    return 0;
  }

  private applyDegeneratePenalty(
    runtime: PlayerRuntimeState,
    action: ActionV1,
  ): number {
    // Penalize repeated identical macro actions beyond a threshold.
    const signature = actionSignatureV1(action);
    if (runtime.lastActionSignature === signature) {
      runtime.actionRepeatCount += 1;
    } else {
      runtime.lastActionSignature = signature;
      runtime.actionRepeatCount = 1;
    }

    if (runtime.actionRepeatCount > this.degenerateRepeatThreshold) {
      runtime.degenerateSpam += 1;
      return this.rewardWeights.degenerate_spam_penalty;
    }
    return 0;
  }

  private resetSpam(runtime: PlayerRuntimeState): void {
    runtime.lastActionSignature = null;
    runtime.actionRepeatCount = 0;
  }
}

export class HeadlessSelfPlaySimulatorBatchV1 {
  // Thin vectorized wrapper; each simulator remains fully independent.
  constructor(private readonly simulators: HeadlessSelfPlaySimulatorV1[]) {}

  async resetBatch(requests: ResetRequestV1[]): Promise<ResetResponseV1[]> {
    if (requests.length !== this.simulators.length) {
      throw new Error(
        `Expected ${this.simulators.length} reset requests, got ${requests.length}`,
      );
    }
    return Promise.all(
      this.simulators.map((simulator, i) => simulator.reset(requests[i])),
    );
  }

  stepBatch(requests: StepRequestV1[]): StepResponseV1[] {
    if (requests.length !== this.simulators.length) {
      throw new Error(
        `Expected ${this.simulators.length} step requests, got ${requests.length}`,
      );
    }
    return this.simulators.map((simulator, i) => simulator.step(requests[i]));
  }
}

function makeGameID(seed: number): GameID {
  // Convert arbitrary numeric seed into deterministic 8-char game ID.
  const normalized = Math.abs(Math.trunc(seed)) >>> 0;
  return normalized
    .toString(36)
    .toUpperCase()
    .padStart(8, "0")
    .slice(-8) as GameID;
}

function normalizeRatioBins(ratios: number[]): number[] {
  const cleaned = [...new Set(ratios.map((ratio) => Number(ratio)))].filter(
    (ratio) => Number.isFinite(ratio) && ratio > 0 && ratio <= 1,
  );
  cleaned.sort((a, b) => a - b);
  if (cleaned.length === 0) {
    throw new Error("ratio_bins must contain at least one value in (0, 1]");
  }
  return cleaned;
}

function makeNumberRecord(
  clientIDs: readonly ClientID[],
  value: number,
): Record<ClientID, number> {
  return Object.fromEntries(clientIDs.map((clientID) => [clientID, value])) as Record<
    ClientID,
    number
  >;
}

function makeBooleanRecord(
  clientIDs: readonly ClientID[],
  value: boolean,
): Record<ClientID, boolean> {
  return Object.fromEntries(clientIDs.map((clientID) => [clientID, value])) as Record<
    ClientID,
    boolean
  >;
}

function toObservationRecord(
  contexts: Map<ClientID, PlayerContext>,
): Record<ClientID, ObservationV1> {
  return Object.fromEntries(
    Array.from(contexts.entries()).map(([clientID, context]) => [
      clientID,
      context.observation,
    ]),
  ) as Record<ClientID, ObservationV1>;
}

function toActionMaskRecord(
  contexts: Map<ClientID, PlayerContext>,
): Record<ClientID, ActionMaskV1> {
  return Object.fromEntries(
    Array.from(contexts.entries()).map(([clientID, context]) => [
      clientID,
      context.actionMask,
    ]),
  ) as Record<ClientID, ActionMaskV1>;
}

function computePlacements(players: Player[]): Map<string, number> {
  // Deterministic tie-breakers keep placement stable across runs.
  const sorted = [...players].sort((a, b) => {
    const tileDiff = b.numTilesOwned() - a.numTilesOwned();
    if (tileDiff !== 0) {
      return tileDiff;
    }
    const troopDiff = b.troops() - a.troops();
    if (troopDiff !== 0) {
      return troopDiff;
    }
    return a.smallID() - b.smallID();
  });

  const placements = new Map<string, number>();
  sorted.forEach((player, index) => {
    placements.set(player.id(), index + 1);
  });
  return placements;
}

function getWinnerClientID(game: Game): ClientID | null {
  const winner = game.getWinner();
  if (winner === null || typeof winner === "string" || !winner.isPlayer()) {
    return null;
  }
  return winner.clientID();
}

function bigintToNumber(value: bigint): number {
  // Clamp to avoid unsafe Number conversions.
  const maxSafe = BigInt(Number.MAX_SAFE_INTEGER);
  if (value > maxSafe) {
    return Number.MAX_SAFE_INTEGER;
  }
  if (value < -maxSafe) {
    return Number.MIN_SAFE_INTEGER;
  }
  return Number(value);
}

function seededInt(
  seed: number,
  saltA: number,
  saltB: number,
  saltC: number,
  modulo: number,
): number {
  // Stateless deterministic integer sampler from mixed seed inputs.
  const mixed = mix32(seed ^ (saltA * 73856093) ^ (saltB * 19349663) ^ (saltC * 83492791));
  return modulo <= 0 ? 0 : mixed % modulo;
}

function mix32(value: number): number {
  let x = value | 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

interface FreshTerrainMapData {
  nations: ManifestNation[];
  gameMap: Awaited<ReturnType<typeof genTerrainFromBin>>;
  miniGameMap: Awaited<ReturnType<typeof genTerrainFromBin>>;
  teamGameSpawnAreas?: TeamGameSpawnAreas;
}

async function loadTerrainMapFresh(
  map: GameMapType,
  mapSize: GameMapSize,
  mapLoader: GameMapLoader,
): Promise<FreshTerrainMapData> {
  // Similar to loadTerrainMap(), but always returns fresh mutable map objects.
  const mapFiles = mapLoader.getMapData(map);
  const manifest = await mapFiles.manifest();

  const gameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map, await mapFiles.mapBin())
      : await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin());

  const miniGameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin())
      : await genTerrainFromBin(manifest.map16x, await mapFiles.map16xBin());

  const manifestNations = manifest.nations ?? [];
  const nations = manifestNations.map((nation) => ({
    ...nation,
    coordinates: [nation.coordinates[0], nation.coordinates[1]] as [number, number],
  }));

  if (mapSize === GameMapSize.Compact) {
    nations.forEach((nation) => {
      nation.coordinates = [
        Math.floor(nation.coordinates[0] / 2),
        Math.floor(nation.coordinates[1] / 2),
      ];
    });
  }

  let teamGameSpawnAreas = cloneTeamGameSpawnAreas(manifest.teamGameSpawnAreas);
  if (mapSize === GameMapSize.Compact && teamGameSpawnAreas) {
    const scaled: TeamGameSpawnAreas = {};
    for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
      scaled[key] = areas.map((area) => ({
        x: Math.floor(area.x / 2),
        y: Math.floor(area.y / 2),
        width: Math.max(1, Math.floor(area.width / 2)),
        height: Math.max(1, Math.floor(area.height / 2)),
      }));
    }
    teamGameSpawnAreas = scaled;
  }

  return {
    nations,
    gameMap,
    miniGameMap,
    teamGameSpawnAreas,
  };
}

function cloneTeamGameSpawnAreas(
  areas: TeamGameSpawnAreas | undefined,
): TeamGameSpawnAreas | undefined {
  if (!areas) {
    return undefined;
  }
  const clone: TeamGameSpawnAreas = {};
  for (const [key, value] of Object.entries(areas)) {
    clone[key] = value.map((area) => ({ ...area }));
  }
  return clone;
}
