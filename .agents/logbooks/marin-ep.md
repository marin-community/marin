---
topic: marin-ep
description: Clean-room fused expert-parallel MoE kernel for the grug MoE hero on GB200 NVL72
author: mcwitt
---

# Marin EP: Task Logbook

Experiment ID prefix: `MEP`.

## Scope
- Goal: from-scratch fused EP MoE kernel (dispatch + expert GEMM + combine)
  for `moe_hero_ep` at EP64 on GB200 NVL72; native JAX/XLA stack only.
- Primary metric(s): drop-adjusted tokens/s and MFU at hero shape; drop rate
  < 2% at cf <= 1.33 (trained router).
- Constraints: no vendored code from DeepEP/MoonEP/MoK/Megatron; simulator-first
  development; oracle parity at every phase. Plan and gates:
  `.agents/projects/20260814_marin_ep_kernel.md`. Behavior spec:
  `experiments/marin_ep/SPEC.md`.
- Coordinating issue/PR: none yet (file experiment issue when hardware runs
  begin / first PR opens).

## Baseline
- Date: 2026-08-14
- Code refs: `fixed_all_to_all` backend
  (`lib/levanter/src/levanter/grug/_moe/ep_fixed_all_to_all.py`) on the hero
  (`experiments/grug/moe_hero_ep/`).
- Baseline numbers: hero EP64 @ cf 1.30: 262,683 tok/s last-50 mean, ~3.96%
  drops, 252,271 drop-adjusted tok/s (`.agents/logbooks/7279-moe-hero-ep.md`).
  Per-layer routed-MoE segment baseline (from an existing profile): TODO —
  needed to evaluate gate G1b.

## Current TL;DR
- M0-M2 skeleton done in one pass ([MEP-001]): spec, dense oracle,
  message-passing correctness simulator with explicit backward, L0 roofline.
  23 CPU tests green (value+grad parity, drop accounting, EP invariance,
  determinism, trace accounting).
- L0 says hero fwd layer is GEMM-bound (17.3 ms GEMM vs 4.4 ms transport per
  device at 70%/80% efficiency): transport fully hideable under overlap.

## Hypothesis Queue

### Active
- `MEP-H1`: pooled per-expert capacity (SPEC S2) at cf 1.33 gives < 2% drops
  on trained-router distributions where per-cell fixed_a2a gives ~4%.
  Next test: replay recorded hero routing distributions through
  `pooled_keep_mask`.
- `MEP-H2`: a count-then-write NVLink-store transport + persistent grouped
  GEMM can hide >= 90% of transport time at hero shape. Next test: L1
  discrete-event sim with tile dependencies.
- `MEP-H3`: symmetric-memory remote stores are reachable from JAX via one of
  (a) Mosaic-GPU distributed, (b) CuTe DSL nvshmem via cutlass_call, (c)
  custom CUDA FFI. Next test: M5 hardware spike, 1-2 nodes.

## Entry Log

### 2026-08-14 17:00 - MEP-001: plan + spec + oracle + simulator + L0 roofline
- Hypothesis: n/a (scaffolding milestone M0-M3 partial).
- Commit Hash: 365a9d1b81 (branch `marin-ep`).
- Command: `uv run pytest experiments/marin_ep/tests -q` -> 23 passed.
  `uv run python -c "from experiments.marin_ep.perfmodel.roofline import hero_report; print(hero_report())"`
- Config: hero shape T=65536/dev, K=4, H=3072 (latent), I=6272, E=192, EP64,
  cf 1.33 -> per-expert capacity C=116,218 rows, one-way remote 1.48 GiB/dev/layer.
- Result: L0 roofline (GB200 bf16 2.5 PF/s @ 70% GEMM eff, NVLink5 900 GB/s
  @ 80% link eff), per device per layer:
  | dir | transport | gemm | serial | overlapped |
  |---|---|---|---|---|
  | fwd | 4.40 ms | 17.32 ms | 21.72 ms | 17.32 ms |
  | bwd | 4.40 ms | 34.63 ms | 39.04 ms | 34.63 ms |
- Interpretation: layer is GEMM-bound at hero shape; the win over the current
  backend comes from removing serialized a2a rounds + capacity-padded cells
  and overlapping transport, not from shrinking payload. Spec's pooled
  capacity rule is EP-degree invariant (tested exactly at EP 1/2/4/8).
- Next action: L1 discrete-event simulator over simcore traces (MEP-H2);
  drop-rate replay on realistic routing (MEP-H1); measure baseline routed-MoE
  segment time from an existing hero profile for G1b.
