# Python RL Smoke Training

This folder contains a minimal TypeScript-to-Python training path:

- `openfront_rl_bridge.py`: subprocess client for the TS bridge protocol.
- `openfront_env.py`: gym-style wrapper with single-agent and multi-agent APIs.
- `train_ppo_smoke.py`: masked PPO trainer with shared-policy multi-agent self-play.

## Quick start

From repo root:

```bash
python3 -m pip install -r rl/python/requirements.txt
python3 rl/python/train_ppo_smoke.py --updates 5 --rollout-steps 256
```

The default smoke config uses `--max-ticks 300` so episodes terminate quickly and
placement metrics appear early in logs.

By default the TS bridge suppresses known non-critical warning spam:

- `Didn't find outgoing attack with id ...`

To show that family of warnings again, pass:

```bash
python3 rl/python/train_ppo_smoke.py \
  --bridge-cmd "node --import tsx src/scripts/rl-bridge.ts --players 4 --max-ticks 300 --show-noncritical-warnings"
```

## Optional long run

```bash
python3 rl/python/train_ppo_smoke.py \
  --updates 30 \
  --rollout-steps 512 \
  --players 4 \
  --max-ticks 3000 \
  --save-every 5 \
  --eval-every 5 \
  --eval-episodes 3
```

Checkpoints and eval snapshots are written to:

- `rl/python/checkpoints/*.pt`
- `rl/python/checkpoints/eval_history.jsonl`
