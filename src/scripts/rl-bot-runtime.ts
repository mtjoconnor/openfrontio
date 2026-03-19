import { randomUUID } from "node:crypto";
import process from "node:process";
import WebSocket from "ws";
import { z } from "zod";
import { createGameRunner, GameRunner } from "../core/GameRunner";
import { getSpawnTiles } from "../core/execution/Util";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
} from "../core/game/Game";
import { FilesystemGameMapLoader } from "../core/rl/FilesystemGameMapLoader";
import {
  ClientIntentMessage,
  ClientJoinMessage,
  ClientPingMessage,
  ClientRejoinMessage,
  GameConfig,
  GameStartInfo,
  Intent,
  ServerMessage,
  ServerMessageSchema,
  Turn,
} from "../core/Schemas";
import { PseudoRandom } from "../core/PseudoRandom";
import { simpleHash } from "../core/Util";

type QueuePolicy = "drop_oldest" | "drop_newest";
type ModelSource = "heuristic" | "checkpoint";
type FallbackPolicy = "heuristic";

interface BotOptions {
  gameID: string;
  host: string;
  numWorkers: number;
  baseWorkerPort: number;
  workerPort: number | null;
  username: string;
  clanTag: string | null;
  token: string;
  seed: number;
  logEveryTicks: number;
  createPrivate: boolean;
  autoStart: boolean;
  startDelayMs: number;
  // Policy/runtime knobs.
  modelSource: ModelSource;
  modelPath: string | null;
  fallbackPolicy: FallbackPolicy;
  temperature: number;
  actionRate: number;
  spawnChance: number;
  attackChance: number;
  cancelChance: number;
  targetChance: number;
  minAttackTroops: number;
  maxOutgoingAttacks: number;
  // Pacing and queue safety.
  intentTickGap: number;
  maxIntentsPerSecond: number;
  maxIntentsPerMinute: number;
  queueCap: number;
  queuePolicy: QueuePolicy;
}

interface PolicyContext {
  runner: GameRunner;
  self: Player;
  random: PseudoRandom;
  options: BotOptions;
}

interface PolicyEngine {
  chooseIntent(ctx: PolicyContext): Intent | null;
}

// `null` is a valid attack target (terra nullius). We use `undefined` to mean
// "no legal target found" so wilderness expansion isn't dropped accidentally.
type AttackTargetID = string | null;

class UnsupportedCheckpointPolicy implements PolicyEngine {
  constructor(private readonly modelPath: string | null) {}

  chooseIntent(_: PolicyContext): Intent | null {
    throw new Error(
      `checkpoint inference is not yet implemented in TS runtime (modelPath=${this.modelPath ?? "null"})`,
    );
  }
}

class HeuristicPolicy implements PolicyEngine {
  chooseIntent(ctx: PolicyContext): Intent | null {
    const { runner, self, random, options } = ctx;
    const game = runner.game;

    // Spawn phase behavior: try to secure a legal spawn as soon as possible.
    if (game.inSpawnPhase() && !self.hasSpawned()) {
      if (random.next() <= options.spawnChance) {
        const spawnTile = this.pickSpawnTile(runner, self, random);
        if (spawnTile !== null) {
          return {
            type: "spawn",
            tile: spawnTile,
          };
        }
      }
      return null;
    }

    // No gameplay actions after elimination.
    if (!self.isAlive() || !self.hasSpawned()) {
      return null;
    }

    // Temperature softly scales decision entropy.
    const temperatureScale = clamp(options.temperature, 0, 2);
    const actionRate = clamp(options.actionRate * (0.5 + temperatureScale / 2), 0, 1);
    if (random.next() > actionRate) {
      return null;
    }

    const activeOutgoing = self
      .outgoingAttacks()
      .filter((attack) => attack.isActive() && !attack.retreating());

    if (activeOutgoing.length > 0 && random.next() <= options.cancelChance) {
      const attack = random.randElement(activeOutgoing);
      return {
        type: "cancel_attack",
        attackID: attack.id(),
      };
    }

    // Respect outgoing-attack cap to keep behavior and pacing stable.
    if (
      activeOutgoing.length < options.maxOutgoingAttacks &&
      random.next() <= options.attackChance
    ) {
      const target = this.pickAttackTarget(runner, self, random);
      if (target !== undefined) {
        const ratio = random.randElement([0.1, 0.2, 0.33, 0.5, 0.75]);
        const troops = Math.floor(self.troops() * ratio);
        if (troops >= options.minAttackTroops) {
          return {
            type: "attack",
            targetID: target,
            troops,
          };
        }
      }
    }

    if (random.next() <= options.targetChance) {
      const targetPlayer = this.pickTargetPlayer(runner, self, random);
      if (targetPlayer !== null) {
        return {
          type: "targetPlayer",
          target: targetPlayer.id(),
        };
      }
    }

    return null;
  }

  private pickSpawnTile(
    runner: GameRunner,
    self: Player,
    random: PseudoRandom,
  ): number | null {
    const game = runner.game;
    const attempts = 200;
    for (let i = 0; i < attempts; i++) {
      const x = random.nextInt(0, game.width());
      const y = random.nextInt(0, game.height());
      const tile = game.ref(x, y);
      if (!game.isLand(tile)) {
        continue;
      }
      if (getSpawnTiles(game, tile, false).length > 0) {
        return tile;
      }
    }

    // Deterministic fallback sweep to guarantee progress if random misses.
    const stride = Math.max(1, Math.floor(Math.sqrt((game.width() * game.height()) / 2000)));
    for (let y = 0; y < game.height(); y += stride) {
      for (let x = 0; x < game.width(); x += stride) {
        const tile = game.ref(x, y);
        if (!game.isLand(tile)) {
          continue;
        }
        if (getSpawnTiles(game, tile, false).length > 0) {
          return tile;
        }
      }
    }
    return null;
  }

  private pickAttackTarget(
    runner: GameRunner,
    self: Player,
    random: PseudoRandom,
  ): AttackTargetID | undefined {
    const game = runner.game;
    const counts = new Map<AttackTargetID, number>();

    // Build attackability from direct border adjacency in O(border) time.
    for (const borderTile of self.borderTiles()) {
      for (const neighbor of game.neighbors(borderTile)) {
        if (!game.isLand(neighbor)) {
          continue;
        }
        const owner = game.owner(neighbor);
        if (!owner.isPlayer()) {
          counts.set(null, (counts.get(null) ?? 0) + 1);
          continue;
        }
        if (owner.id() === self.id()) {
          continue;
        }
        if (!self.canAttackPlayer(owner)) {
          continue;
        }
        counts.set(owner.id(), (counts.get(owner.id()) ?? 0) + 1);
      }
    }

    if (counts.size === 0) {
      return undefined;
    }

    // Prefer stronger adjacency pressure; break ties with randomness.
    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const topScore = entries[0][1];
    const topTargets = entries
      .filter((entry) => entry[1] === topScore)
      .map((entry) => entry[0]);
    return random.randElement(topTargets);
  }

  private pickTargetPlayer(
    runner: GameRunner,
    self: Player,
    random: PseudoRandom,
  ): Player | null {
    const candidates = runner
      .game
      .allPlayers()
      .filter((other) => other.id() !== self.id() && other.isAlive() && self.canTarget(other));
    if (candidates.length === 0) {
      return null;
    }
    return random.randElement(candidates);
  }
}

class IntentPacer {
  private lastIntentTick = Number.NEGATIVE_INFINITY;
  private sentAtMs: number[] = [];

  constructor(private readonly options: BotOptions) {}

  canSend(currentTick: number, nowMs: number): boolean {
    if (currentTick - this.lastIntentTick < this.options.intentTickGap) {
      return false;
    }
    this.sentAtMs = this.sentAtMs.filter((t) => nowMs - t < 60_000);
    const sentLastSecond = this.sentAtMs.filter((t) => nowMs - t < 1000).length;
    if (sentLastSecond >= this.options.maxIntentsPerSecond) {
      return false;
    }
    if (this.sentAtMs.length >= this.options.maxIntentsPerMinute) {
      return false;
    }
    return true;
  }

  markSent(currentTick: number, nowMs: number): void {
    this.lastIntentTick = currentTick;
    this.sentAtMs.push(nowMs);
  }

  reset(): void {
    this.lastIntentTick = Number.NEGATIVE_INFINITY;
    this.sentAtMs = [];
  }
}

class BotRuntime {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private runner: GameRunner | null = null;
  private turnBuffer: Turn[] = [];
  private myClientID: string | null = null;
  private lastReceivedTurn = -1;
  private turnsSeen = 0;
  private intentQueue: Intent[] = [];
  private readonly random: PseudoRandom;
  private readonly pacer: IntentPacer;
  private readonly heuristicPolicy = new HeuristicPolicy();
  private readonly primaryPolicy: PolicyEngine;
  private lastStatusTick = -1;
  private connectionEpoch = 0;

  constructor(private readonly options: BotOptions) {
    this.random = new PseudoRandom(simpleHash(options.seed + options.gameID));
    this.pacer = new IntentPacer(options);
    this.primaryPolicy =
      options.modelSource === "heuristic"
        ? this.heuristicPolicy
        : new UnsupportedCheckpointPolicy(options.modelPath);
  }

  async start(): Promise<void> {
    if (this.options.createPrivate) {
      await this.createPrivateGame();
    } else {
      await this.assertGameExists();
    }
    this.connectAndJoin(false);
  }

  stop(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close(1000, "bot runtime stopping");
      } catch {
        // Best-effort shutdown.
      }
      this.ws = null;
    }
    this.intentQueue = [];
    this.pacer.reset();
  }

  private connectAndJoin(isRejoin: boolean): void {
    const epoch = ++this.connectionEpoch;
    const wsUrl = this.getWsUrl();
    console.log(`[rl-bot] connecting: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    this.ws = ws;

    ws.on("open", () => {
      if (!this.isCurrentConnection(epoch, ws)) {
        return;
      }
      console.log(`[rl-bot] websocket open (rejoin=${isRejoin})`);
      this.startPing();
      if (isRejoin) {
        this.sendRejoin();
      } else {
        this.sendJoin();
      }
    });

    ws.on("message", (buffer) => {
      if (!this.isCurrentConnection(epoch, ws)) {
        return;
      }
      const text = buffer.toString();
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch (error) {
        console.error("[rl-bot] invalid JSON from server", error);
        return;
      }
      const result = ServerMessageSchema.safeParse(parsed);
      if (!result.success) {
        console.error("[rl-bot] invalid server message schema", z.prettifyError(result.error));
        return;
      }
      this.handleServerMessage(result.data).catch((error) => {
        console.error("[rl-bot] message handling failed", error);
      });
    });

    ws.on("close", (code, reason) => {
      if (!this.isCurrentConnection(epoch, ws)) {
        return;
      }
      this.stopPing();
      console.warn(`[rl-bot] websocket closed code=${code} reason=${reason.toString()}`);
      this.intentQueue = [];
      this.pacer.reset();
      if (code === 1000) {
        return;
      }
      // Auto-rejoin with same token/client identity for resiliency.
      this.scheduleReconnect();
    });

    ws.on("error", (error) => {
      if (!this.isCurrentConnection(epoch, ws)) {
        return;
      }
      console.error("[rl-bot] websocket error", error);
    });
  }

  private isCurrentConnection(epoch: number, ws: WebSocket): boolean {
    return this.connectionEpoch === epoch && this.ws === ws;
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return;
    }
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connectAndJoin(true);
    }, 1500);
  }

  private async handleServerMessage(message: ServerMessage): Promise<void> {
    switch (message.type) {
      case "ping":
        return;
      case "error":
        console.warn(`[rl-bot] server error: ${message.error} ${message.message ?? ""}`);
        return;
      case "prestart":
        console.log(
          `[rl-bot] prestart map=${message.gameMap} size=${message.gameMapSize}`,
        );
        return;
      case "lobby_info":
        this.myClientID = message.myClientID;
        console.log(
          `[rl-bot] lobby_info clients=${message.lobby.clients?.length ?? 0} myClientID=${message.myClientID}`,
        );
        if (this.options.autoStart) {
          void this.startGameWithDelay();
        }
        return;
      case "start":
        await this.onStartMessage(message.gameStartInfo, message.turns, message.myClientID);
        return;
      case "turn":
        this.onTurnMessage(message.turn);
        return;
      case "desync":
        console.warn(
          `[rl-bot] desync turn=${message.turn} yourHash=${message.yourHash ?? "n/a"} correctHash=${message.correctHash ?? "n/a"}`,
        );
        return;
    }
  }

  private async onStartMessage(
    gameStartInfo: GameStartInfo,
    turns: Turn[],
    myClientID?: string,
  ): Promise<void> {
    this.myClientID = myClientID ?? this.myClientID;
    if (!this.myClientID) {
      throw new Error("missing myClientID after start message");
    }
    console.log(`[rl-bot] start received myClientID=${this.myClientID}`);

    // Build the runner once. On reconnect starts, keep the existing runner and
    // only apply missed turns.
    if (this.runner === null) {
      // createGameRunner expects client-style config fetch; emulate /api/env in Node.
      const originalFetch = globalThis.fetch;
      try {
        globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
          if (typeof input === "string" && input === "/api/env") {
            return new Response(JSON.stringify({ game_env: "dev" }), {
              status: 200,
              headers: { "Content-Type": "application/json" },
            });
          }
          return originalFetch(input as any, init);
        };

        this.runner = await createGameRunner(
          gameStartInfo,
          this.myClientID,
          new FilesystemGameMapLoader(),
          () => {
            // No renderer callback needed for external bot runtime.
          },
        );
      } finally {
        globalThis.fetch = originalFetch;
      }
      this.turnsSeen = 0;
      this.lastStatusTick = -1;
    }

    // Clear stale commands from disconnected windows before applying fresh turns.
    this.intentQueue = [];
    this.pacer.reset();

    const replayTurns = [...turns, ...this.turnBuffer];
    this.turnBuffer = [];
    this.ingestTurns(replayTurns);
    this.drainTurnsAndAct();
  }

  private onTurnMessage(turn: Turn): void {
    if (!this.runner) {
      this.turnBuffer.push(turn);
      return;
    }
    this.ingestTurn(turn);
    this.drainTurnsAndAct();
  }

  private ingestTurns(turns: Turn[]): void {
    if (!this.runner || turns.length === 0) {
      return;
    }
    const ordered = [...turns].sort((a, b) => a.turnNumber - b.turnNumber);
    for (const turn of ordered) {
      this.ingestTurn(turn);
    }
  }

  private ingestTurn(turn: Turn): void {
    if (!this.runner) {
      return;
    }
    // Deduplicate old turns that can show up during reconnect/start replay.
    if (turn.turnNumber < this.turnsSeen) {
      this.lastReceivedTurn = Math.max(this.lastReceivedTurn, turn.turnNumber);
      return;
    }

    // Keep turn numbers contiguous to mirror the client runtime behavior.
    while (turn.turnNumber > this.turnsSeen) {
      this.runner.addTurn({
        turnNumber: this.turnsSeen,
        intents: [],
      });
      this.turnsSeen++;
    }

    this.runner.addTurn(turn);
    this.turnsSeen++;
    this.lastReceivedTurn = Math.max(this.lastReceivedTurn, turn.turnNumber);
  }

  private drainTurnsAndAct(): void {
    if (!this.runner) {
      return;
    }
    while (this.runner.executeNextTick(this.runner.pendingTurns())) {
      this.maybePlanIntent();
      this.flushIntentQueue();
      const tick = this.runner.game.ticks();
      if (tick % this.options.logEveryTicks === 0 && tick !== this.lastStatusTick) {
        this.lastStatusTick = tick;
        this.logStatus();
      }
    }

    // If turns arrived before start, replay them once runner exists.
    if (this.turnBuffer.length > 0) {
      const buffered = [...this.turnBuffer];
      this.turnBuffer = [];
      this.ingestTurns(buffered);
      this.drainTurnsAndAct();
    }
  }

  private maybePlanIntent(): void {
    if (!this.runner || !this.myClientID) {
      return;
    }
    const self = this.runner.game.playerByClientID(this.myClientID);
    if (!self || !self.isPlayer()) {
      return;
    }

    let intent: Intent | null = null;
    try {
      intent = this.primaryPolicy.chooseIntent({
        runner: this.runner,
        self,
        random: this.random,
        options: this.options,
      });
    } catch (error) {
      console.warn("[rl-bot] primary policy failed, falling back", error);
      intent =
        this.options.fallbackPolicy === "heuristic"
          ? this.heuristicPolicy.chooseIntent({
              runner: this.runner,
              self,
              random: this.random,
              options: this.options,
            })
          : null;
    }

    if (intent === null) {
      return;
    }
    if (!this.isIntentValidForCurrentState(self, intent)) {
      return;
    }
    this.enqueueIntent(intent);
  }

  private enqueueIntent(intent: Intent): void {
    if (this.intentQueue.length < this.options.queueCap) {
      this.intentQueue.push(intent);
      return;
    }
    if (this.options.queuePolicy === "drop_oldest") {
      this.intentQueue.shift();
      this.intentQueue.push(intent);
      return;
    }
    // drop_newest
  }

  private flushIntentQueue(): void {
    if (!this.runner || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    if (this.intentQueue.length === 0) {
      return;
    }
    const nowMs = Date.now();
    const tick = this.runner.game.ticks();
    if (!this.pacer.canSend(tick, nowMs)) {
      return;
    }
    const intent = this.intentQueue.shift();
    if (!intent) {
      return;
    }
    const self = this.runner.game.playerByClientID(this.myClientID ?? "");
    if (!self || !self.isPlayer()) {
      return;
    }
    if (!this.isIntentValidForCurrentState(self, intent)) {
      return;
    }
    const payload = {
      type: "intent",
      intent,
    } satisfies ClientIntentMessage;
    const serialized = JSON.stringify(payload);
    if (serialized.length > 500) {
      console.warn("[rl-bot] dropping oversized intent payload");
      return;
    }
    this.ws.send(serialized);
    this.pacer.markSent(tick, nowMs);
  }

  private isIntentValidForCurrentState(self: Player, intent: Intent): boolean {
    const game = this.runner?.game;
    if (!game) {
      return false;
    }
    switch (intent.type) {
      case "spawn": {
        if (!game.inSpawnPhase() || self.hasSpawned()) {
          return false;
        }
        if (!game.isLand(intent.tile)) {
          return false;
        }
        return getSpawnTiles(game, intent.tile, false).length > 0;
      }
      case "attack": {
        if (game.inSpawnPhase() || !self.isAlive() || !self.hasSpawned()) {
          return false;
        }
        if (intent.targetID === self.id()) {
          return false;
        }
        if (
          intent.troops === null ||
          intent.troops < this.options.minAttackTroops ||
          intent.troops > self.troops()
        ) {
          return false;
        }
        if (intent.targetID === null || intent.targetID === game.terraNullius().id()) {
          return true;
        }
        if (!game.hasPlayer(intent.targetID)) {
          return false;
        }
        const target = game.player(intent.targetID);
        return target.isPlayer() && self.canAttackPlayer(target);
      }
      case "cancel_attack":
        return self
          .outgoingAttacks()
          .some((attack) => attack.id() === intent.attackID && attack.isActive());
      case "targetPlayer":
        if (intent.target === self.id() || !game.hasPlayer(intent.target)) {
          return false;
        }
        return self.canTarget(game.player(intent.target));
      default:
        return true;
    }
  }

  private async createPrivateGame(): Promise<void> {
    const httpBase = this.getHttpBase();
    const gameConfig = defaultPrivateBotGameConfig();
    const response = await fetch(`${httpBase}/api/create_game/${this.options.gameID}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.options.token}`,
      },
      body: JSON.stringify(gameConfig),
    });
    if (!response.ok) {
      throw new Error(`create_game failed: ${response.status} ${response.statusText}`);
    }
    console.log(`[rl-bot] private game created id=${this.options.gameID}`);
    if (this.options.autoStart) {
      await this.startGameWithDelay();
    }
  }

  private async assertGameExists(): Promise<void> {
    const httpBase = this.getHttpBase();
    const response = await fetch(`${httpBase}/api/game/${this.options.gameID}/exists`);
    if (!response.ok) {
      throw new Error(`game existence check failed: ${response.status} ${response.statusText}`);
    }
    const body = (await response.json()) as { exists?: boolean };
    if (!body.exists) {
      throw new Error(
        `game ${this.options.gameID} does not exist on worker endpoint ${httpBase}. ` +
          `Use --create-private or provide an existing game ID.`,
      );
    }
  }

  private async startGameWithDelay(): Promise<void> {
    if (this.options.startDelayMs > 0) {
      await sleep(this.options.startDelayMs);
    }
    const httpBase = this.getHttpBase();
    const response = await fetch(`${httpBase}/api/start_game/${this.options.gameID}`, {
      method: "POST",
    });
    if (!response.ok) {
      console.warn(
        `[rl-bot] start_game failed status=${response.status} ${response.statusText}`,
      );
      return;
    }
    console.log(`[rl-bot] start_game sent`);
  }

  private getWorkerIndex(): number {
    return simpleHash(this.options.gameID) % this.options.numWorkers;
  }

  private getWorkerPort(): number {
    if (this.options.workerPort !== null) {
      return this.options.workerPort;
    }
    return this.options.baseWorkerPort + this.getWorkerIndex();
  }

  private getWsUrl(): string {
    const port = this.getWorkerPort();
    return `ws://${this.options.host}:${port}`;
  }

  private getHttpBase(): string {
    const port = this.getWorkerPort();
    return `http://${this.options.host}:${port}`;
  }

  private sendJoin(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const joinMessage = {
      type: "join",
      token: this.options.token,
      gameID: this.options.gameID,
      username: this.options.username,
      clanTag: this.options.clanTag,
      turnstileToken: null,
    } satisfies ClientJoinMessage;
    this.ws.send(JSON.stringify(joinMessage));
  }

  private sendRejoin(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const rejoinMessage = {
      type: "rejoin",
      token: this.options.token,
      gameID: this.options.gameID,
      lastTurn: Math.max(0, this.lastReceivedTurn),
    } satisfies ClientRejoinMessage;
    this.ws.send(JSON.stringify(rejoinMessage));
  }

  private startPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
    }
    this.pingTimer = setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }
      const ping = {
        type: "ping",
      } satisfies ClientPingMessage;
      this.ws.send(JSON.stringify(ping));
    }, 5000);
  }

  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private logStatus(): void {
    if (!this.runner || !this.myClientID) {
      return;
    }
    const self = this.runner.game.playerByClientID(this.myClientID);
    if (!self) {
      return;
    }
    console.log(
      `[rl-bot] tick=${this.runner.game.ticks()} inSpawn=${this.runner.game.inSpawnPhase()} alive=${self.isAlive()} spawned=${self.hasSpawned()} tiles=${self.numTilesOwned()} troops=${self.troops()} outgoing=${self.outgoingAttacks().length} queue=${this.intentQueue.length}`,
    );
  }
}

function parseArgs(argv: string[]): BotOptions {
  const get = (flag: string, fallback: string): string => {
    const index = argv.indexOf(flag);
    if (index < 0 || index + 1 >= argv.length) {
      return fallback;
    }
    return argv[index + 1];
  };

  const requiredGameID = get("--game-id", "");
  if (requiredGameID.trim() === "") {
    throw new Error("Missing required --game-id");
  }

  const modelSourceRaw = get("--model-source", "heuristic");
  const modelSource: ModelSource =
    modelSourceRaw === "checkpoint" ? "checkpoint" : "heuristic";
  const queuePolicyRaw = get("--queue-policy", "drop_oldest");
  const queuePolicy: QueuePolicy =
    queuePolicyRaw === "drop_newest" ? "drop_newest" : "drop_oldest";

  const token = get("--token", randomUUID());
  const clanTagRaw = get("--clan-tag", "");
  const clanTag = clanTagRaw.trim() === "" ? null : clanTagRaw;

  return {
    gameID: requiredGameID,
    host: get("--host", "localhost"),
    numWorkers: clampInt(get("--num-workers", "2"), 1, 64),
    baseWorkerPort: clampInt(get("--base-worker-port", "3001"), 1, 65535),
    workerPort: argv.includes("--worker-port")
      ? clampInt(get("--worker-port", "3001"), 1, 65535)
      : null,
    username: get("--username", "RLBot"),
    clanTag,
    token,
    seed: clampInt(get("--seed", "1337"), 0, Number.MAX_SAFE_INTEGER),
    logEveryTicks: clampInt(get("--log-every-ticks", "50"), 1, 10_000),
    createPrivate: argv.includes("--create-private"),
    autoStart: argv.includes("--auto-start"),
    startDelayMs: clampInt(get("--start-delay-ms", "0"), 0, 60_000),
    modelSource,
    modelPath: argv.includes("--model-path") ? get("--model-path", "") : null,
    fallbackPolicy: "heuristic",
    temperature: clampFloat(get("--temperature", "1.0"), 0, 2),
    actionRate: clampFloat(get("--action-rate", "0.45"), 0, 1),
    spawnChance: clampFloat(get("--spawn-chance", "0.9"), 0, 1),
    attackChance: clampFloat(get("--attack-chance", "0.55"), 0, 1),
    cancelChance: clampFloat(get("--cancel-chance", "0.12"), 0, 1),
    targetChance: clampFloat(get("--target-chance", "0.08"), 0, 1),
    minAttackTroops: clampInt(get("--min-attack-troops", "3"), 1, 1_000_000),
    maxOutgoingAttacks: clampInt(get("--max-outgoing-attacks", "4"), 0, 1_000),
    intentTickGap: clampInt(get("--intent-tick-gap", "4"), 1, 1_000),
    maxIntentsPerSecond: clampInt(get("--max-intents-per-second", "3"), 1, 10),
    maxIntentsPerMinute: clampInt(get("--max-intents-per-minute", "120"), 1, 150),
    queueCap: clampInt(get("--queue-cap", "64"), 1, 10_000),
    queuePolicy,
  };
}

function clampInt(value: string, min: number, max: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.max(min, Math.min(max, parsed));
}

function clampFloat(value: string, min: number, max: number): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.max(min, Math.min(max, parsed));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function defaultPrivateBotGameConfig(): GameConfig {
  return {
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
    randomSpawn: true,
    disableNavMesh: true,
    disableAlliances: true,
    startingGold: 100_000,
  } satisfies GameConfig;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const runtime = new BotRuntime(options);

  console.log(
    `[rl-bot] starting gameID=${options.gameID} host=${options.host} tokenPersistentID=${options.token.substring(0, 8)}... modelSource=${options.modelSource}`,
  );

  process.on("SIGINT", () => {
    runtime.stop();
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    runtime.stop();
    process.exit(0);
  });

  await runtime.start();
}

main().catch((error: unknown) => {
  console.error("[rl-bot] failed", error);
  process.exitCode = 1;
});
