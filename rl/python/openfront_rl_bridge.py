#!/usr/bin/env python3
"""
Thin Python client for the TypeScript RL bridge.

Protocol:
- stdin/stdout newline-delimited JSON (single response per request)
- request shape: {"id": <int>, "method": "...", "params": {...}}
- response shape:
  - success: {"id": <int>, "ok": true, "result": {...}}
  - error:   {"id": <int>, "ok": false, "error": {"message": "..."}}
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class BridgeProtocolError(RuntimeError):
    """Raised when the bridge returns malformed or mismatched protocol data."""


class BridgeProcessError(RuntimeError):
    """Raised when the bridge process is unavailable or exits unexpectedly."""


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for spawning the TypeScript bridge process."""

    command: List[str]
    cwd: Optional[str] = None


class OpenFrontRLBridge:
    """
    Synchronous request/response wrapper around `src/scripts/rl-bridge.ts`.

    The API is intentionally small and deterministic for PPO rollouts:
    - `ping()`
    - `reset(params)`
    - `step(params)`
    - `close()`
    """

    def __init__(self, config: BridgeConfig):
        self._config = config
        self._next_id = 1
        self._proc = self._spawn_process()

    def _spawn_process(self) -> subprocess.Popen[str]:
        proc = subprocess.Popen(
            self._config.command,
            cwd=self._config.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Keep bridge logs visible to user on stderr.
            text=True,
            bufsize=1,
        )
        if proc.stdin is None or proc.stdout is None:
            raise BridgeProcessError("Failed to attach stdio pipes to bridge process")
        return proc

    def _ensure_running(self) -> None:
        if self._proc.poll() is not None:
            raise BridgeProcessError(
                f"Bridge process exited with code {self._proc.returncode}"
            )

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._ensure_running()
        req_id = self._next_id
        self._next_id += 1
        payload = {
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        line = json.dumps(payload, separators=(",", ":"))
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

        assert self._proc.stdout is not None
        while True:
            raw = self._proc.stdout.readline()
            if raw == "":
                raise BridgeProcessError("Bridge stdout closed unexpectedly")
            raw = raw.strip()
            if not raw:
                continue
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise BridgeProtocolError(
                    f"Bridge returned non-JSON line: {raw!r}"
                ) from exc

            if response.get("id") != req_id:
                raise BridgeProtocolError(
                    f"Mismatched bridge response id: expected {req_id}, got {response.get('id')}"
                )
            if response.get("ok") is True:
                return response.get("result")
            error_obj = response.get("error") or {}
            message = error_obj.get("message", "Unknown bridge error")
            raise BridgeProtocolError(message)

    def ping(self) -> Dict[str, Any]:
        return self._request("ping", {})

    def reset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("reset", params)

    def step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("step", params)

    def close(self) -> None:
        # Best effort protocol shutdown, then hard-stop if needed.
        try:
            if self._proc.poll() is None:
                self._request("close", {})
        except Exception:
            pass
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)

    def __enter__(self) -> "OpenFrontRLBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
