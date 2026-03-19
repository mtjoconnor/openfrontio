import { readFile } from "node:fs/promises";
import path from "node:path";
import { GameMapType } from "../game/Game";
import { GameMapLoader, MapData } from "../game/GameMapLoader";
import { MapManifest } from "../game/TerrainMapLoader";

export class FilesystemGameMapLoader implements GameMapLoader {
  private readonly maps = new Map<GameMapType, MapData>();

  constructor(
    private readonly rootDir: string = path.resolve(process.cwd(), "resources/maps"),
  ) {}

  getMapData(map: GameMapType): MapData {
    const cached = this.maps.get(map);
    if (cached) {
      return cached;
    }

    const mapFolder = this.mapFolderName(map);
    const basePath = path.resolve(this.rootDir, mapFolder);

    // Lazy file readers keep startup fast and avoid duplicate IO across resets.
    const mapData = {
      mapBin: this.makeBinaryLoader(path.join(basePath, "map.bin")),
      map4xBin: this.makeBinaryLoader(path.join(basePath, "map4x.bin")),
      map16xBin: this.makeBinaryLoader(path.join(basePath, "map16x.bin")),
      manifest: async () => {
        const manifestPath = path.join(basePath, "manifest.json");
        const raw = await readFile(manifestPath, "utf8");
        return JSON.parse(raw) as MapManifest;
      },
      webpPath: path.join(basePath, "thumbnail.webp"),
    } satisfies MapData;

    this.maps.set(map, mapData);
    return mapData;
  }

  private mapFolderName(map: GameMapType): string {
    const key = Object.keys(GameMapType).find(
      (k) => GameMapType[k as keyof typeof GameMapType] === map,
    );
    if (!key) {
      throw new Error(`Unknown map type: ${String(map)}`);
    }
    return key.toLowerCase();
  }

  private makeBinaryLoader(filePath: string): () => Promise<Uint8Array> {
    let cache: Promise<Uint8Array> | null = null;
    return () => {
      // Cache per-file promise so repeated calls return the same bytes.
      cache ??= readFile(filePath).then((buf) => new Uint8Array(buf));
      return cache;
    };
  }
}
