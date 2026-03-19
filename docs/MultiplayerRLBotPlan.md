# RL Multiplayer Bot Plan (Self-Hosted, CPU-Only, External Client)

## Summary
- Build a phased RL system for public FFA placement, starting with an attack-first macro action space and deploying as an external websocket bot client.
- Use a hybrid fairness strategy:
  1. Train a privileged teacher policy for fast learning.
  2. Distill and fine-tune a constrained student policy that uses only client-derivable state and obeys multiplayer rate limits.
- Keep rollout controlled: offline self-play, then private multiplayer lobbies, then opt-in public lobbies on self-hosted infrastructure.

## Key Implementation Changes

### Headless self-play simulator (TypeScript)
- Add a training environment layer around the core game loop (no real-time websocket turn pacing), with seeded `reset/step` episode control and batched parallel env instances.
- Define `ActionV1` (attack-first macro): `no_op`, `attack(target_slot, ratio_bin)`, `cancel_attack(attack_slot)`, `spawn(spawn_slot)`, `target_player(target_slot)`.
- Add legal-action masks generated from current game state to prevent invalid intents.
- Define two observation modes:
  - `TeacherObsV1` (privileged/global features).
  - `StudentObsV1` (strict client-equivalent features only).
- Implement reward shaping:
  - Terminal: win/placement/survival outcomes.
  - Dense: territory share delta, troop efficiency, survival pressure.
  - Penalties: invalid intents, rate-limit pressure, degenerate spam.

### Python training stack (PyTorch + TS bridge)
- Use PPO with masked discrete actions and vectorized env rollouts optimized for CPU.
- Add league self-play: latest checkpoint, historical snapshots, and current heuristic bot baselines.
- Curriculum schedule:
  1. Small FFA matches and maps.
  2. Medium player-count mixed maps.
  3. Production-like FFA map rotation and population settings.
- Hybrid pipeline:
  1. Train privileged teacher.
  2. Distill to student on student observation space.
  3. RL fine-tune student under strict fair-play constraints only.

### External multiplayer bot runtime
- Build a bot client that joins games and sends `intent` messages like a normal player.
- Enforce server-compatible pacing and budgets (`intent` rate/size limits) and fail-safe fallback to heuristic behavior.
- Add bot runtime config for model path, action temperature, queue policy, and safety limits.

### Evaluation and rollout
- Offline evaluation harness with fixed seeds and league matchups.
- Private-lobby multiplayer A/B with humans and existing bots.
- Progressive release gates and rollback triggers based on KPI regressions or protocol violations.

## Public APIs / Interfaces / Types
- Add versioned RL bridge contracts:
  - `ResetRequest/ResetResponse`, `StepRequest/StepResponse`
  - `ObservationV1` (`TeacherObsV1`, `StudentObsV1`)
  - `ActionV1`, `ActionMaskV1`
  - `EpisodeStatsV1` (win/placement/survival/invalid-intent/rate-limit counters)
- Add model artifact manifest:
  - `policy_type` (`teacher`/`student`), obs/action schema versions, training step, ELO tier, checksum.
- Add bot runtime config schema:
  - auth mode, model source, intent pacing caps, fallback policy settings.

## Test Plan

### Unit tests
- Action encoder/decoder maps each `ActionV1` to valid intent payloads.
- Legal-action mask correctness against game constraints.
- Deterministic seeded resets and reproducible rollouts.

### Integration tests
- Env `step` parity against core game execution for identical turn sequences.
- Bridge stability under batched rollouts (long-run memory/leak checks).
- External bot join/start/intent lifecycle in self-hosted multiplayer.

### Acceptance scenarios
- Student beats current heuristic baseline on FFA evaluation set (win rate and placement improvement thresholds).
- Zero kick events from message rate limiter in soak tests.
- Stable behavior across map rotation and player-count buckets.

## Assumptions and Defaults
- Target environment is self-hosted OpenFront, not direct public `openfront.io` deployment.
- Compute is CPU-only, so design optimizes sample efficiency and rollout batching over heavyweight model complexity.
- V1 learned control stays attack-first macro; advanced systems (boats/construction/nukes/full diplomacy) remain scripted and are phase-2 expansion targets.
- Fair-play deployment policy is the student model only; teacher remains training-only.
