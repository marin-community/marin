# FP8 loss-curve validation — bf16 (main) vs FP8 (PR #7079)

Issue: https://github.com/marin-community/marin/issues/7298 (part of #6699)
PR under test: https://github.com/marin-community/marin/pull/7079
Branches: `research/mcwitt/7079-fp8-loss-val` (FP8 arm = PR merged with main @ `512eade96` + launcher) ·
`research/mcwitt/7079-fp8-loss-val-ctl` (control arm = main @ `4a37b09f5` + launcher)
W&B: project `marin_moe`, group `fp8-loss-val-7079`, tags `fp8-loss-val` / `pr7079`
Experiment IDs: `FP8VAL-NNN`

## TL;DR

- (pending) Two-arm A/B at row-13 scale (d2560/26L/64E top-4, ~18B params) on
  4×8 H100 per arm, SlimPajama-6B, full MuonH schedule to zero LR,
  ~5.8B tokens/arm. Acceptance: train-loss gap ≤0.01 at matched steps through
  the cooldown tail (the #6486 threshold).

## Hypothesis queue

| id | hypothesis | status |
|---|---|---|
| H-parity | FP8 (grouped+dense+wire, delayed per-tensor scaling) tracks bf16 within 0.01 loss through the full schedule incl. cooldown | queued |
| H-tail | if a gap appears, it concentrates in the last 10% of the schedule (cooldown numerics, cf. #6486) | queued |

## Entries

### FP8VAL-001 — design + setup (2026-07-16)

Design constraints from the request: ≤8 h wall, ≤64 H100, parallel arms,
CoreWeave-cached data, full loss trajectories.

- **Config:** row-13 shapes exactly as the MFU line of work (`grug_moe_row13`),
  `ring` MoE backend (default on both refs), EP8 intra-node × FSDP 4 nodes,
  `gpu_fa4_cute`, `recompute_all`, capacity factor 1.0 (grug default on both
  refs), batch 128 × seq 4096, seed 0 both arms. Optimizer = production MuonH
  (`lr 1e-3 / adam_lr 1e-4 / warmup 0.1 / min_lr_ratio 0.0`) over the full step
  budget so the cooldown tail is exercised (#6486: bf16-master gaps only
  surfaced in the last 10% of a decay-to-zero schedule).
- **Data:** `slimpajama_6b_dataset()` llama3-tokenized cache (version
  2026.06.28) under `s3://marin-us-east-02a/marin` (in-cluster LOTA endpoint;
  no egress), feistel block shuffle. Eval suites (paloma etc.) skipped: their
  caches are not materialized under the CW prefix and would trigger
  cross-region tokenization; per-step train loss on an identical data stream
  is the comparison instrument.
- **Sizing:** measured single-node H100 throughput for this exact config
  (B200MFU H100 bench, 2026-07-16): bf16 1.501 s/step at B32/8GPU ≈ 87k tok/s;
  FP8 1.372 s/step ≈ 96k tok/s. At 32 GPUs (same per-device batch at B128) with
  cross-node FSDP overhead the bf16 arm should land in 1.6–2.0 s/step →
  11000 steps ≈ 5.8B tokens in ≤ ~6.5 h + compile. Smoke run calibrates before
  the full launch. 5.8B tokens ≈ 4× the budget #6486 needed to resolve
  0.01-level deltas.
- **Control purity:** `fp8-moe-mlp-comms` was 134 commits behind main; merged
  main into the PR (`512eade96`, conflict-free — the branch delta is 18
  FP8-scoped commits) so the arms differ by exactly the PR content. Same
  launcher file (`experiments/grug/moe/launch_fp8_loss_val.py`) on both
  branches; FP8 enabled only via `FP8VAL_FP8=1` (guarded import so the file
  runs unmodified on main).
- **Launch command** (per arm, from the respective worktree):
  `iris --config lib/iris/config/cw-us-east-02a.yaml job run --no-wait --cpu=1 --memory=4G --extra=cpu -e WANDB_API_KEY -e FP8VAL_* -e RUN_ID -- python -m experiments.grug.moe.launch_fp8_loss_val --version dev --run`
  (launcher is a CPU step; `run_grug` dispatches the 4×8-H100 gang via Fray).

Next: FP8VAL-002 smoke (30 steps/arm), then FP8VAL-003 full A/B.
