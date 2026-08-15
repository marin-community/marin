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
- M0-M3 done ([MEP-001], [MEP-002]): spec, dense oracle, message-passing
  correctness simulator with explicit backward, L0 roofline, L1
  discrete-event simulator. 39 CPU tests green.
- Drop rule adopted (SPEC S2): group-pooled capacity with per-expert floor,
  G=3 → 0.31% drops at cf 1.33 vs ~4% for per-cell/per-expert (MEP-H1
  promoted).
- L1 at hero shape: fwd 19.3 ms pipelined vs 25.7 ms bulk per layer
  (GEMM floor 17.3 ms) — transport ~89% hidden; kernel must rotate send
  order and interleave GEMM tiles across local experts.

## Hypothesis Queue

### Active
- `MEP-H2`: a count-then-write NVLink-store transport + persistent grouped
  GEMM can hide >= 90% of transport time at hero shape. Status: L1 sim says
  ~89% fwd (1.9 ms exposed of 8.8 ms transport+count on a 17.3 ms GEMM
  floor); the residual is the last expert's combine tail — interleaving
  GEMM tiles across local experts should shrink it. Evidence: [MEP-002].
- `MEP-H3`: symmetric-memory remote stores are reachable from JAX via one of
  (a) Mosaic-GPU distributed, (b) CuTe DSL nvshmem via cutlass_call, (c)
  custom CUDA FFI. Next test: M5 hardware spike, 1-2 nodes.
- `MEP-H4`: rotated send order and cross-expert GEMM-tile interleave are
  required in the real kernel (L1 predicts convoy/tail costs otherwise).
  Verify on hardware in M6.

### Promoted
- `MEP-H1`: group-pooled capacity (G=3, per-owner) gives < 2% drops at
  cf <= 1.33 where per-expert pooling and per-cell both give ~4%. Decision:
  adopted into SPEC S2 as waterfilling with per-expert floor ([MEP-002]);
  residual risk: synthetic Dirichlet skew, not recorded router
  distributions — revisit with real hero routing histograms before M6.

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

### 2026-08-14 18:30 - MEP-002: group-pooled drop rule + L1 event simulator
- Hypothesis: MEP-H1 (pooling beats per-cell capacity) and MEP-H2
  (transport hideable under pipelining).
- Commit Hash: 7bc59228ef.
- Command: `uv run python experiments/marin_ep/bench/drop_rate_study.py`;
  `uv run python -c "from experiments.marin_ep.perfmodel.eventsim import hero_l1_report; print(hero_l1_report())"`;
  `uv run pytest experiments/marin_ep/tests -q` -> 39 passed.
- Config: skew alpha=9.08 calibrated so per-cell reproduces 3.96% drops at
  cf 1.30 (hero measurement); hero shape as MEP-001.
- Result (drop rates, homogeneous routing):
  | cf | per_cell | per_expert | per_owner G=3 |
  |---|---|---|---|
  | 1.30 | 4.67% | 4.66% | 0.45% |
  | 1.33 | 4.10% | 4.09% | **0.31%** |
  | 1.50 | 1.77% | 1.73% | 0.00% |
  With device heterogeneity 0.3: per_cell 7.8%, per_expert 3.6%, per_owner
  0.34% at cf 1.33. `exploratory` (synthetic Dirichlet skew).
- Result (L1 makespans, hero, balanced): fwd 19.26 ms pipelined vs 25.65 ms
  bulk-synchronous; bwd 35.90 vs 42.97 (L0 overlapped bounds 17.32/34.63).
  Pipelining alone buys ~20% per layer over a bulk schedule with the same
  payloads. Design facts surfaced by the sim: (1) naive in-order sends
  convoy on owner-0 ingress (rotate send order); (2) strict-FIFO link
  modeling is unfaithful (backfill scheduler adopted); (3) the last
  expert's combine tail is the main exposed time.
- Interpretation: SPEC v1.1 adopts group-pooled waterfilling (G=3). R2
  looks comfortably achievable at cf 1.33; even cf ~1.2 (0.9-1.5% drops)
  may be preferable — buys back GEMM time and memory. Worth an arm when
  hardware A/Bs start.
- Next action: measure baseline routed-MoE segment from an existing hero
  profile (G1b denominator); cross-expert GEMM-tile interleave experiment
  in L1; then M4 autoresearch loop over (tile_rows, send order, G, cf).
