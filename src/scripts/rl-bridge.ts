import readline from "node:readline";
import util from "node:util";
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
import { ResetRequestV1, StepRequestV1 } from "../core/rl/RLBridgeV1";

type BridgeOptions = {
  players: number;
  maxTicks: number;
  seed: number;
  suppressNoncriticalWarnings: boolean;
};

type RequestID = string | number | null;

type ResetBridgeRequest = {
  id: RequestID;
  method: "reset";
  params: ResetRequestV1;
};

type StepBridgeRequest = {
  id: RequestID;
  method: "step";
  params: StepRequestV1;
};

type CloseBridgeRequest = {
  id: RequestID;
  method: "close";
  params?: Record<string, never>;
};

type PingBridgeRequest = {
  id: RequestID;
  method: "ping";
  params?: Record<string, never>;
};

type BridgeRequest =
  | ResetBridgeRequest
  | StepBridgeRequest
  | CloseBridgeRequest
  | PingBridgeRequest;

type BridgeResponse =
  | {
      id: RequestID;
      ok: true;
      result: unknown;
    }
  | {
      id: RequestID;
      ok: false;
      error: {
        message: string;
        stack?: string;
      };
    };

function parseArgs(argv: string[]): BridgeOptions {
  const get = (flag: string, fallback: string): string => {
    const index = argv.indexOf(flag);
    if (index < 0 || index + 1 >= argv.length) {
      return fallback;
    }
    return argv[index + 1];
  };

  return {
    players: clampInt(get("--players", "4"), 2, 24),
    maxTicks: clampInt(get("--max-ticks", "3000"), 1, 10_000_000),
    seed: clampInt(get("--seed", "1337"), 0, Number.MAX_SAFE_INTEGER),
    suppressNoncriticalWarnings: !argv.includes("--show-noncritical-warnings"),
  };
}

function clampInt(value: string, min: number, max: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return min;
  }
  return Math.max(min, Math.min(max, parsed));
}

function makePlayers(count: number): SelfPlayPlayerConfigV1[] {
  const players: SelfPlayPlayerConfigV1[] = [];
  for (let i = 0; i < count; i++) {
    players.push({
      client_id: `P${String(i + 1).padStart(7, "0")}`,
      username: `BridgeBot${i + 1}`,
      is_lobby_creator: i === 0,
    });
  }
  return players;
}

function makeGameConfig(): GameConfig {
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
  };
}

function writeJsonLine(payload: BridgeResponse): void {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function sendError(id: RequestID, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  writeJsonLine({
    id,
    ok: false,
    error: {
      message,
      stack: error instanceof Error ? error.stack : undefined,
    },
  });
}

function installStderrConsoleRouting(suppressNoncriticalWarnings: boolean): void {
  // The Python bridge reads stdout as NDJSON protocol, so all human-readable logs
  // must be redirected to stderr to prevent protocol corruption.
  const toStderr = (...args: unknown[]): void => {
    const rendered = args
      .map((arg) =>
        typeof arg === "string"
          ? arg
          : util.inspect(arg, {
              depth: 3,
              breakLength: 120,
              compact: true,
            }),
      )
      .join(" ");
    // Non-fatal attack cancellation races can produce high-volume noise during
    // long training runs. Suppress only this known message family by default.
    if (
      suppressNoncriticalWarnings &&
      rendered.startsWith("Didn't find outgoing attack with id ")
    ) {
      return;
    }
    process.stderr.write(`${rendered}\n`);
  };
  console.log = toStderr;
  console.info = toStderr;
  console.debug = toStderr;
  console.warn = toStderr;
}

function isBridgeRequest(value: unknown): value is BridgeRequest {
  if (!value || typeof value !== "object") {
    return false;
  }
  const method = (value as { method?: unknown }).method;
  return (
    method === "reset" ||
    method === "step" ||
    method === "close" ||
    method === "ping"
  );
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  installStderrConsoleRouting(options.suppressNoncriticalWarnings);
  const players = makePlayers(options.players);
  const simulator = new HeadlessSelfPlaySimulatorV1({
    game_config: makeGameConfig(),
    map_loader: new FilesystemGameMapLoader(),
    players,
    controlled_client_ids: players.map((p) => p.client_id),
    max_ticks_per_episode: options.maxTicks,
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr,
    terminal: false,
  });

  let queue: Promise<void> = Promise.resolve();
  let closed = false;

  const handleRequest = async (request: BridgeRequest): Promise<void> => {
    if (closed && request.method !== "ping") {
      writeJsonLine({
        id: request.id,
        ok: false,
        error: { message: "Bridge is closed" },
      });
      return;
    }

    switch (request.method) {
      case "ping":
        writeJsonLine({
          id: request.id,
          ok: true,
          result: {
            status: "ok",
            players: options.players,
            max_ticks: options.maxTicks,
            seed: options.seed,
          },
        });
        return;
      case "reset": {
        const params: ResetRequestV1 = {
          seed: request.params?.seed ?? options.seed,
          observation_mode: request.params?.observation_mode,
          controlled_client_ids: request.params?.controlled_client_ids,
        };
        const result = await simulator.reset(params);
        writeJsonLine({
          id: request.id,
          ok: true,
          result,
        });
        return;
      }
      case "step": {
        const result = simulator.step(request.params ?? {});
        writeJsonLine({
          id: request.id,
          ok: true,
          result,
        });
        return;
      }
      case "close":
        closed = true;
        writeJsonLine({
          id: request.id,
          ok: true,
          result: { closed: true },
        });
        rl.close();
        return;
    }
  };

  rl.on("line", (line) => {
    queue = queue
      .then(async () => {
        if (line.trim() === "") {
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(line);
        } catch (error) {
          sendError(null, new Error("Invalid JSON line received"));
          return;
        }
        if (!isBridgeRequest(parsed)) {
          sendError((parsed as { id?: RequestID })?.id ?? null, new Error("Invalid request shape"));
          return;
        }
        await handleRequest(parsed);
      })
      .catch((error: unknown) => {
        sendError(null, error);
      });
  });

  rl.on("close", () => {
    process.exit(0);
  });

  process.on("SIGINT", () => {
    process.exit(0);
  });
}

main().catch((error: unknown) => {
  sendError(null, error);
  process.exit(1);
});
