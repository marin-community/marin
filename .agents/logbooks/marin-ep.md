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
- M0-M5 done ([MEP-001]..[MEP-006]): spec, dense oracle, message-passing
  correctness simulator with explicit backward, L0/L1 perf models, and a
  hardware-validated XLA stepping-stone backend (oracle-conformant values
  + grads + drops on GB200 with production ragged transport). 43 CPU
  tests green + 3 GPU-marked.
- Transport substrate proven on GB200 ([MEP-005]): Pallas Mosaic-GPU
  remote puts + semaphores, 584 GB/s/device egress untuned (65% of
  NVLink5 peak), collective-metadata path, zero custom CUDA.
- M6 in flight: generic put_segments kernel drafted; its
  dispatch/combine metadata builders are CPU-proven against the
  simulator ([MEP-006]).
- Watch out: haliax ragged_dot's GPU Triton path silently miscomputes at
  small dims and pins tf32 ([MEP-006]).
- Drop rule adopted (SPEC S2): group-pooled capacity with per-expert floor,
  G=3 → 0.31% drops at cf 1.33 vs ~4% for per-cell/per-expert (MEP-H1
  promoted).
- L1 at hero shape ([MEP-003]): fwd 17.9 ms pipelined vs 23.4 ms bulk per
  layer (GEMM floor 17.4 ms) — transport ~97% hidden with rotated
  simultaneous sends; tiles 2-8k rows. Under hero-calibrated skew the
  layer is compute-imbalance-bound (23.3 ms fwd at cf 1.33) — cf trades
  drops vs balance (MEP-H5: cf 1.10 matches baseline drops, ~17% faster).

## Hypothesis Queue

### Active
- `MEP-H2`: a count-then-write NVLink-store transport + persistent grouped
  GEMM can hide >= 90% of transport time at hero shape. Status: L1 sim says
  ~89% fwd (1.9 ms exposed of 8.8 ms transport+count on a 17.3 ms GEMM
  floor); the residual is the last expert's combine tail — interleaving
  GEMM tiles across local experts should shrink it. Evidence: [MEP-002].
- `MEP-H4`: rotated, simultaneous per-source sends are required in the real
  kernel (L1 predicts ingress convoys and a serialized combine tail
  otherwise; cross-expert GEMM-tile interleave turned out not to matter).
  Evidence: [MEP-003]. Verify on hardware in M6.
- `MEP-H5`: with G=3 pooling, lowering cf toward ~1.1 beats cf 1.33 on
  drop-adjusted throughput under realistic skew (compute balance dominates).
  Evidence: [MEP-003] sim table. Next test: hardware A/B arm with loss
  tracking (drops are not throughput-equivalent).

### Promoted
- `MEP-H3`: symmetric-memory remote stores ARE reachable from JAX —
  Pallas Mosaic-GPU distributed has the full surface in installed jax.
  Decision: Mosaic-GPU is the transport substrate ([MEP-004]); hardware
  validation of the spike is the M5 gate.
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

### 2026-08-14 19:10 - MEP-003: L1 tail root-caused; tile + cf sweeps
- Hypothesis: MEP-H2 (transport hideable), MEP-H4 (schedule shape matters).
- Commit Hash: 13696051e9.
- Command: inline sweeps via `estimate_layer_makespan` (see commit message);
  `uv run pytest experiments/marin_ep/tests -q` -> 39 passed.
- Result:
  - The 1.9 ms fwd tail was an emission-order artifact (device-major
    dispatch made the last device's rows arrive last everywhere -> its
    combine ingress serialized at the end). Step-major rotated emission —
    the sim's proxy for simultaneous per-device sends — fixes it: hero fwd
    17.88 ms (GEMM busy floor 17.4), bwd 35.21 ms. Transport ~97% hidden.
    Pipelined vs bulk: fwd 17.88 vs 23.36, bwd 35.21 vs 40.68 (~19%/13%).
  - GEMM-tile interleave and arrival-order processing knobs: no effect
    under readiness-ordered scheduling (removed). MEP-H4 narrows to
    "rotated + simultaneous sends"; per-expert tile order is free.
  - tile_rows: 2048-4096 best (17.55 ms fwd), 32768 costs ~10%. Real
    per-tile overheads will push the optimum up; calibrate on hardware.
  - Skewed routing (alpha=9.08, hero-calibrated): layer becomes
    compute-imbalance-bound — fwd 23.3 ms at cf 1.33 (hottest owner's
    capacity-clipped GEMM). cf now trades drops vs balance:
    | cf | drops | fwd+bwd ms | drop-adj rel throughput |
    |---|---|---|---|
    | 1.10 | 4.5% | 57.6 | 1.16 |
    | 1.25 | 1.5% | 65.6 | 1.05 |
    | 1.33 | 0.46% | 69.6 | 1.00 |
    G=3 pooling lets us either keep cf 1.33 with ~9x fewer drops than
    today's baseline, or match today's ~4.5% drops at cf 1.10 and take a
    ~17% faster layer. Proper metric is loss-per-wallclock -> hardware A/B
    arm. `exploratory` (synthetic skew; sim-only). New hypothesis MEP-H5.
- Interpretation: v1 kernel design facts frozen so far: count-then-write
  ragged pools (G=3 waterfilling), rotated simultaneous sends, tile
  ~2-8k rows, GEMM consumes arrival-flagged tiles, combine streams per
  tile. Skew-driven compute imbalance is the dominant remaining cost at
  hero shape — expert placement and cf are the levers.
- Next action: G1b baseline segment measurement; M5 transport-substrate
  survey (Mosaic-GPU distributed vs CuTe nvshmem vs custom FFI) from local
  sources; draft kernels/ API skeleton against SPEC.

### 2026-08-14 20:30 - MEP-004: XLA stepping-stone backend + substrate decision
- Hypothesis: MEP-H3 (a JAX-reachable symmetric-memory substrate exists).
- Commit Hash: 0a2bc790ed (+ this commit).
- Command: `uv run pytest experiments/marin_ep/tests -q` -> 41 passed.
- Result 1 (implementation): `kernels/xla_backend.py` — the SPEC contract
  (group-pooled waterfilling, ragged pools) inside shard_map with stock
  collectives (`ragged_all_to_all` + `ragged_dot`), signature-compatible
  with the levanter `shard_local_fn` dispatch. Values, drop counts, and
  all four grads match the oracle on a real 8-device CPU mesh (via an
  all_gather transport emulation, since ragged-a2a is unimplemented on
  XLA:CPU); the production ragged path shape-checks at EP64 topology.
  This is both the first hardware-runnable Marin EP artifact and the
  fallback if the fused kernel slips.
- Result 2 (M5 substrate survey, from installed-source inspection):
  **Pallas Mosaic-GPU distributed is the substrate** — complete
  device-initiated remote-memory surface in installed jax 0.10.1/0.11.0
  (`plgpu.remote_ref`, peer TMA both directions, remote ld/st,
  `pl.semaphore_signal(device_id=)`, `semaphore_signal_parallel`,
  multimem), with the in-tree `collective_matmul_mgpu.py` example doing
  exactly our put+signal pattern under shard_map. CuTe DSL: multimem PTX
  only, no nvshmem bindings, no symmetric allocator (stays the GEMM
  engine candidate). DeepEP-style FFI: single-process intranode only
  (template for plumbing, not a substrate). Constraints: nvshmem path
  needs 1 process/GPU (hero runs 1/node today; ncclep bench has per-GPU
  supervisor precedent), `--xla_gpu_experimental_enable_nvshmem`, Lane
  semantics for peer TMA, `nvidia-nvshmem-cu13` ships with the gpu extra.
  MEP-H3 resolved -> promoted.
- Result 3: M5 spike harness written (`bench/spike_transport_mgpu.py`,
  rotated all-to-all put + arrival semaphores, correctness vs all_gather +
  egress-bandwidth calibration for the L1 model). Import-checked only;
  needs GB200 to run.
- Next action: M5 GPU session on cw-us-east-08a (reserve-gpu, 1-2 nodes):
  run jax's collective_matmul example as litmus, then the spike; calibrate
  L1 link_efficiency/message_latency; then M6 transport kernel
  (count exchange + dispatch/combine puts) with XLA GEMMs in the middle.
  Also: bench xla_backend vs fixed_all_to_all on GPU (cheap first win).

### 2026-08-14 21:30 - MEP-005: M5 spike GREEN on GB200 (single node)
- Hypothesis: MEP-H3 hardware validation (remote puts + semaphores work
  from Pallas Mosaic-GPU on GB200 via the collective-metadata path).
- Commit Hash: 49a3f2a90c (+ pod-side fixes folded into next commit).
- Command: dev_gpu 1x GB200 tray on cw-us-east-08a
  (`scripts/iris/dev_gpu.py ... allocate --gpu-variant gb200 --cpu 32
  --memory 200GB`), pod `uv sync --all-packages --extra=gpu`, then
  `uv run python experiments/marin_ep/bench/spike_transport_mgpu.py`
  (single process, 4 local CudaDevices, jax 0.10.1 GPU).
- Result:
  - Litmus (minimal remote-put + remote-semaphore kernel): exact
    correctness on 4 devices. No nvshmem needed at single-node scope; XLA
    auto-inserts a MultiGpuBarrierWithNcclKernel (~7 us) around the launch.
  - Rotated all-to-all put spike, correctness exact vs all_gather:
    | case | kernel time | egress/device |
    |---|---|---|
    | 3x50 MB blocks | 0.252 ms | **584 GB/s** (65% of 900 peak, untuned single-buffered) |
    | 3x131 KB blocks | 0.023 ms | latency floor ~= kernel launch + 3 put/signal rounds |
  - Porting gotchas recorded in the spike script: jax 0.10.1 uses
    `out_shape` (0.11: `out_type`); `pl.run_scoped` needs
    `collective_axes="wg"` in multithreaded kernels; async copies cap at
    256 elements/dim (view [rows, H] as [rows, H/256, 256]); SMEM budget
    ~228 KB/SM. Dev-pod holder released itself after ~7 min
    ("holder job terminated unexpectedly", exit 0) — reallocate rather
    than debug; keep GPU sessions short and scripted.
- Interpretation: the substrate works end-to-end on GB200 with zero custom
  CUDA. 584 GB/s untuned (no double buffering, 1 warpgroup/SM) already
  matches the L1 model's 0.8 link efficiency assumption territory; the
  ~20 us launch overhead argues for one persistent kernel per layer
  direction rather than per-phase launches. MEP-H3 CONFIRMED at
  single-node scope; multi-node nvshmem path (1 proc/GPU) still untested.
- Next action: ragged-transport GPU conformance of xla_backend (values +
  grads at EP4); then M6: fused dispatch/combine transport kernel
  (count exchange + puts per SPEC F1-F3/F5) with XLA GEMMs between.

### 2026-08-14 22:15 - MEP-006: xla_backend GPU-conformant; M6 metadata done
- Hypothesis: the production (ragged_all_to_all) path of xla_backend is
  oracle-conformant on real GPUs.
- Commit Hash: (this commit). Second GB200 tray session (~15 min; first
  holder pod self-released mid-session, second pod sat SchedulingGated
  ~6 min behind another user's gated fleet).
- Command: pod `uv run python /tmp/gpu_conformance.py` (now in-repo as
  `test_real_gpu_ragged_transport_matches_oracle`).
- Result:
  - CONFORMANT at EP4 x E16, group sizes {1, 2, 4}: values, drop counts,
    and dx/dweights/dw13/dw2 all match the oracle. Live pooling
    confirmation: drops on identical routing fall 430 -> 320 -> 134 as
    G goes 1 -> 2 -> 4.
  - Debug chain worth remembering: initial 68.7% value mismatch was NOT
    the transport (a ragged-vs-gathered probe matched exactly) but
    **haliax ragged_dot's GPU Triton path silently miscomputing at
    small dims** (H=32/I=48: max|err| ~29 even for zero-size groups;
    H=I=256: correct up to tf32). It also ignores
    jax_default_matmul_precision (stays tf32). Conformance dims must be
    Triton-tile safe; tolerance 1e-2. Candidate upstream issue to file.
  - M6 transport kernel drafted (`kernels/mgpu_transport.py`): one generic
    `put_segments` kernel (rotated destinations, per-destination arrival
    signals, static TMA boxes with row-wise ragged tails) driven by pure
    metadata builders `dispatch_segments`/`combine_segments`. The builders
    are CPU-tested: the dispatch plan reproduces the correctness
    simulator's receive pools bit-exactly and dispatch->combine
    round-trips (`test_mgpu_transport_metadata.py`).
  - Env note: pod resolves jax/jaxlib 0.10.1 with a non-functional 0.11
    cuda plugin warning yet CUDA devices work; the CI/local 0.10/0.11
    split memory applies to pods too. Mosaic distributed + ragged a2a both
    fine on 0.10.1.
- Interpretation: the stepping-stone backend is hardware-validated and
  drop-in shaped for levanter integration; the fused-transport kernel's
  routing math is proven; only the Mosaic kernel body needs GPU debugging.
- Next action (next session): validate `put_segments` on GB200 vs the
  gathered transport at EP4; integrate as `transport="mgpu"` in
  xla_backend; measure vs ragged; then EP4 microbench vs fixed_all_to_all
  and the levanter backend registration.
