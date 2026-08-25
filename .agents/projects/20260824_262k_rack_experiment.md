# Plan: 262K hero-shape probe on one NVL72 rack (integration branch)

**Branch:** `long-context/262k-rack-experiment` (worktree `~/projects/marin.long-context-262k-experiment`)
**Goal:** measure MFU, drop rate, memory, and loss of the `moe_hero_ep` template at seq 262144
on one rack (16 nodes x 4 GB200), restored from the hero's step-6000 checkpoint. This is gates
1–2 of the early-cooldown list in `.agents/projects/20260824_535b_long_context_cooldown.md`
plus a first hero-scale CP datapoint. **No PRs/issues/GitHub posts. Local branch only.**

## Composition

1. Merge, in order: `long-context/mesh-context-axis` (final), `long-context/fa4-sequence-sharding`,
   `long-context/moe-context-sharding`, `long-context/te-cp-backend` (TE inert here; merged for a
   single integration point). Resolve `experiments/grug/moe_hero_ep/model.py` overlaps (FA4 owns
   attention-boundary hunks, MoE owns token-axis hunks — they were written to compose).
2. Port the bench-side launcher delta (`git diff origin/main...bench/8549-capacity-sweep`) for:
   `experiments/grug/moe_hero_ep/launch_mfu_test.py`, `experiments/grug/moe_hero_ep/train.py`,
   `experiments/grug/moe_hero_ep/model.py` (ragged splits field), `experiments/grug/dispatch.py`,
   `experiments/grug/pjrt_wheel.py`, `experiments/grug/checkpointing.py` if the launcher needs it.
   Verify every hunk lands on the intended symbol (backport hazard). Skip the autoresearch/ arm
   harness files.
3. New launcher flags: `--context-axis-size` (→ `GrugTrainerConfig.context_axis_size`),
   `--expert-axis-size` (override `HERO_EP_EXPERT_AXIS_SIZE=64`), `--qk-mult` (model override).
   The seq-len grid logic from `4c49b011fe` already holds tokens/step fixed (batch = tokens/seq).

## The EP+CP parameter-memory change (required, new work)

At EP16xCP4 each expert-rank holds 4x the hero's per-rank parameters: fp32 pinned-host master
+ Muon momentum ~= 1072 GB/node > ~850 GB available → restore would OOM the host. Recorded CP
design (grug-context-parallel-attention logbook): FSDP parameters over the composite
(data, context) so CP does not multiply parameter/optimizer memory.

Implementation: store EP-mode expert weights (and their master/optimizer state) sharded
`P(("expert", "context"), None, None)` instead of `P("expert", None, None)`; `moe_mlp` already
reshards weights to the EP spec before its shard_map, which becomes a per-layer all-gather over
`context` (~1.4 GB per rank per layer for routed weights — cheap on NVLink). Attention/dense
weights similarly extend `_FSDP_AXES` with `context`. Restore is unaffected structurally
(checkpoints record no mesh; the template dictates placement). Gate: at `context_axis_size == 1`
the composite spec is identical to today's.

## Run plan

- Mesh: replica 1 x data 1 x context 4 x expert 16 x model 1 (64 GPUs). Batch 16 x seq 262144
  = 4,194,304 tokens/step (hero-matched). B=16 over batch axes (1*1*16) = 1 sequence per
  expert-coordinate; 65,536 tokens per device.
- Restore: `--restore-from s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-6000`
  `--restore-master-params fp32_pinned_host --master-params fp32_pinned_host` (disabled would put
  268 GB fp32 on device at EP16 — does not fit).
- `--qk-mult 1.84` (= 1.3 x (0.1*ln(262144/4096) + 1), the YaRN-mscale recipe from #6811).
- `--num-steps 6100` (absolute stop step; ~100 measured steps), mixture data (never synthetic —
  a trained router routes by content).
- MoE: pooled transport (hero default), 1 process/node, capacity factors unchanged (1.15/1.15,
  3 waves) — drop rate at 262K vs the 4K baseline (MFU 22.13, drops 1.7e-4 from
  autoresearch/loop-260824-seqlen/results.tsv) is the headline measurement.
- Launch: iris coordinator pattern from `experiments/grug/moe_hero_ep/README.md` on
  `cw-us-east-08a`, `--priority interactive`, unique `IRIS_PORT_JAX`, per-run
  `JAX_COMPILATION_CACHE_DIR`, `IRIS_USER mwittmann`, `MARIN_PREFIX s3://marin-us-east-02a/marin`.

## Pre-flight gates (before the rack run)

1. GPU tests: the FA4 CP kernel changes are GPU-only-testable — run the new GPU-marked tests in
   `lib/levanter/tests/grug/test_fa4_cute_attention.py` (and `tests/test_moe_hero_ep.py -k
   context_parallel`) on a small GB200 allocation (2–4 GPUs) from this integration branch.
2. CPU: full `uv run --no-project infra/ci/run_tests.py` green on the merged branch.
3. A short small-shape rack-free smoke (single node, 4 GPUs, CP4 over a tiny model config) to
   catch integration-level sharding errors before burning a rack slot.

## Measurements to record

`tokens/s`, s/step (raw elapsed stamps, not smoothed tqdm), MFU, `moe/drop_fraction` +
sender/receiver split, loss trajectory vs the step-6000 baseline, peak HBM + host RSS, compile
time, and any XLA collective anomalies. Compare drop rate against the 4K one-rack baseline.
