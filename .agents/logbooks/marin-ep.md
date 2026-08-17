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
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/8311

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
- M0-M6 done ([MEP-001]..[MEP-007]): spec, dense oracle, message-passing
  correctness simulator with explicit backward, L0/L1 perf models, a
  hardware-validated XLA stepping-stone backend, and the fused Mosaic-GPU
  transport (`transport="mgpu"`) — oracle-conformant values + grads +
  drops on GB200 for both transports at G {1,2,4}. 43 CPU tests green +
  8 GPU tests green on GB200.
- M6 numbers ([MEP-007]): transport 1.6-1.7 ms/direction for ~800 MB
  (~500 GB/s/device); layer A/B at EP4 hero widths: 6.37x fwd / 5.07x
  fwd+bwd over the ragged stepping-stone, bit-identical outputs. Kernel
  lessons: mesh-dict device ids on multi-axis meshes; dynamic tile-loop
  bounds (static-bound predication cost 30-41 ms/direction).
- GEMM headroom for M7: Triton ragged_dot at true hero per-device shape
  = 920-930 TF/s vs 1747 TF/s dense XLA on GB200 (CuTe grouped GEMM
  reached 2.2 PF/s in the MXFP8 work).
- Transport substrate proven on GB200 ([MEP-005]): Pallas Mosaic-GPU
  remote puts + semaphores, 584 GB/s/device egress untuned (65% of
  NVLink5 peak), collective-metadata path, zero custom CUDA. Multi-node
  nvshmem path still unvalidated.
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
  Evidence: [MEP-003]. Status: the M6 kernel implements rotated sends and
  hits ~500 GB/s/device at EP4 ([MEP-007]); the A/B against non-rotated
  order is still untested on hardware (matters more at EP64).
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

### 2026-08-15 00:10 - MEP-007: M6 GREEN — fused transport validated, 6.4x layer A/B at EP4
- Hypothesis: `put_segments` is correct on hardware and beats the ragged
  stepping-stone transport at layer level.
- Commit Hash: 0c2ad0d210 (branch pushed to origin/marin-ep)
- Command: `pytest experiments/marin_ep/tests/test_mgpu_transport_gpu.py
  test_xla_backend.py -k 'put_segments or real_gpu' -n0`;
  `python experiments/marin_ep/bench/bench_backend_ep4.py` on one GB200
  tray (dev pod, cw-us-east-08a, jax 0.10.1).
- Result:
  - All 8 GPU tests pass: standalone kernel vs NumPy plan interpreter
    (mixed sizes + zero-row segments + relaunch/semaphore-reset check;
    full-tile path), and the conformance matrix {ragged, mgpu} x
    G {1,2,4} — values, drops, and all four grads (the `put_with_transpose`
    custom_vjp runs the reverse-plan kernel in bwd).
  - Two hardware fixes: (1) `remote_ref`/`semaphore_signal` device ids
    must be the mesh-dict form `{axis: dest}` on multi-axis meshes
    (scalar = "1 id for a 4D mesh" error); (2) tile loops must have
    dynamic bounds — the static `max_tiles` loop from the draft burned
    30-41 ms/direction in predicated no-op iterations. With
    `pl.loop(sm_id, num_full, step=num_sms)` + shifted-last-tile tails:
    1.6-1.7 ms/direction for ~800 MB payload ~= 500 GB/s/device (spike
    was 584 untuned).
  - Layer A/B at EP4, 32k tokens/device, hero widths (H=3072, I=6272,
    E=192, cf 1.33, G=3): fwd 219.3 -> 34.4 ms (6.37x), fwd+bwd
    348.2 -> 68.7 ms (5.07x), outputs bit-identical (same GEMM path).
    Stage bisection (mgpu fwd): permute+compact 0.85, dispatch 1.67,
    GEMMs 27.4, combine 1.58 ms.
  - The 27.4 ms GEMM stage is a 48-group EP4 artifact of Triton
    `ragged_dot`; at the true hero per-device shape (El=3, ~29k rows per
    expert) it runs 920-930 TF/s (7.2 + 3.7 ms) vs 1747 TF/s dense XLA —
    so ~1.9x GEMM headroom remains for M7 (CuTeDSL grouped GEMM hit
    2.2 PF/s in the MXFP8 work).
  - GPU tolerances rescaled to tf32-intermediate magnitudes (values
    atol 0.2, grads atol 0.5 x depth/256): the old result-relative 1e-2
    bounds failed on benign 1-in-64k cancellation outliers, ragged arm
    included.
- Interpretation: M6 done — fused transport is correct, differentiable,
  and near link speed; the mgpu path also deletes both local permutes.
  Transport is no longer the EP4 bottleneck; GEMM quality and e2e
  integration are.
- Next action: M8-first (levanter `moe_mlp` backend registration + EP4
  smoke on the held tray + hero EP64 rack benchmark vs the 24.04% MFU /
  262,683 tok/s baseline), M7 fusion/GEMM-upgrade as the follow-on
  autoresearch lever. Multi-node nvshmem path still unvalidated.

### 2026-08-15 07:25 - MEP-008: multi-process mosaic transport is DEAD upstream; M8 pivots to ragged internode
- Hypothesis: the nvshmem path (1 proc/GPU + `--xla_gpu_experimental_enable_nvshmem`)
  extends `put_segments` across nodes, per the jax 0.10/0.11-era docs.
- Command: 8-process (2 nodes x 4 GPUs, 1 proc/GPU) smoke of `put_segments`
  on cw-us-east-08a, on locked jax 0.10.1, stock 0.11.0, and nightly
  0.11.1.dev20260814.
- Result: FALSIFIED at every version, each failing one layer deeper:
  - 0.10.1 and 0.11.0 host jaxlib: `Unknown flag in XLA_FLAGS` FATAL — the
    flag never shipped in a released host jaxlib's parser (the cu13 PJRT
    plugin knows it; the host env parser wins). `compiler_options` is
    validated host-side too: `No such compile option`.
  - Nightly: flag still rejected because upstream REMOVED it —
    `xla_gpu_experimental_enable_nvshmem` is `reserved` in current
    `xla/xla.proto`. Working around jax's stale substring gate (it checks
    for the literal in XLA_FLAGS; satisfied via
    `--xla_dump_hlo_pipeline_re=--xla_gpu_experimental_enable_nvshmem`)
    plus `MOSAIC_GPU_NVSHMEM_BC_PATH` got all the way to the runtime, which
    returns `UNIMPLEMENTED: NVSHMEM is not supported in XLA.` — jax commit
    2026-07-20 "[NFC] Remove leftover NVSHMEM support from Mosaic GPU
    custom calls" deleted the path; current direction is XLA-managed
    symmetric memory, explicitly single-process ("Use symmetric memory
    peer address API ... in a single-process mode", 2026-07-13).
- Interpretation: `plgpu.remote_ref` is SINGLE-PROCESS ONLY in every
  installable jax today; the M5-era multi-node recipe described a feature
  that existed only briefly at jax HEAD. Cross-node fused transport needs
  either upstream's multi-process symmetric memory to land, or a
  hierarchical scheme (mosaic puts intranode within each 4-GPU process +
  NCCL collective internode) — the DeepEP NVL/RDMA factorization. Hero
  jobs run 1 process/node, which is exactly the topology hierarchy wants.
- Decision: M8 lands `implementation="marin_ep"` with transport
  auto-selection (`mgpu` iff `jax.process_count() == 1`, else
  `ragged_all_to_all`); the EP64 hero benchmark measures the drop-rule win
  on ragged transport. Hierarchical mgpu-intranode transport becomes the
  first autoresearch lever (with M7 GEMM work).
- Env note: dev-pod `uv sync --all-packages --extra=gpu` resolves jax
  0.10.1 + cu13 plugin 0.11.0 (a fray TPU test group's marker collides
  with the gpu extra); the lock intends jax 0.11.0 for gpu. The mgpu
  single-process path works on both.
- Next action: EP8 2-node/2-process conformance smoke of the ragged path
  (hero topology in miniature), then the EP64 hero benchmark run.

### 2026-08-15 00:30 - MEP-009: EP8 multiproc CONFORMANT; EP64 hero benchmark launched
- Hypothesis: the marin_ep ragged path is correct multi-controller and the
  EP64 hero run beats the fixed_all_to_all baseline on drop-adjusted
  throughput via the ~10x drop reduction.
- Commit Hash: branch tip after "multi-process fallback to ragged
  transport; ep-marin hero flavor" (pushed).
- Command: EP8 smoke: `bash run_ep8_smoke.sh {0,1} 10.186.210.61:9876`
  (2 nodes x 4 GPUs, 1 proc/node). Hero:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait
  --enable-extra-resources --target-cluster cw-us-east-08a --priority
  production --cpu 2 --memory 8GB --disk 32GB --timeout 21600
  --max-retries 50 --job-name mep-m8-marin-25-20260815-coord
  -e WANDB_MODE offline -- python -m experiments.grug.moe_hero_ep.launch
  --run-id mep-m8-marin-25-20260815 --dp-racks 1 --num-steps 25
  --flavor ep-marin --version 2026.08.15 --run`
- Result: EP8 2-process smoke CONFORMANT (values, drops=1019 bit-exact vs
  oracle, grads) on the ragged path in hero topology. Traps: (1) loss fn
  must not close over global arrays multi-process; (2) backward NCCL
  window registration fails at ~148 GB without
  XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async (hero env sets it). Experiment
  issue filed: #8311. Hero coordinator + 16-task training gang running
  (building) as of 00:30.
- Next action: monitor to completion; compare last-50 tok/s + drops vs
  262,683 / 3.96% baseline; then M8 PR, then hierarchical transport + M7
  GEMM levers via autoresearch.

### 2026-08-15 01:25 - MEP-010: EP64 hero OOMs — backend exonerated, ragged-class regression
- Hypothesis: the marin_ep EP64 crash (CUDA_ERROR_ILLEGAL_ADDRESS ~4 min in,
  both with and without the latency-hiding scheduler) is a memory problem,
  and ours specifically.
- Command: hero runs mep-m8-marin-25 / -25b (killed); layer-level compile
  memdiff `experiments/marin_ep/bench/memdiff_backends.py` on one tray at
  hero per-device shapes (T=65536, El=3, A_global/E=87,381); control run
  mep-ctl-ragged-25 (levanter's ep-ragged flavor, untouched by this
  branch, same flags).
- Result:
  - Both marin_ep runs failed at the same point with
    `hlo_rematerialization: Can't reduce memory use below 171.87GiB; only
    reduced to 183.17GiB` against a 138.22 GiB cudaMallocAsync limit,
    followed by an alloc-failure -> illegal-address cascade. The LHS flag
    change made no difference (it was a red herring for THIS failure; it
    remains required for ragged correctness per wiki:101 / PR #8081).
  - Layer-level fwd+bwd temp at cf 1.33: marin_ep(ragged) 27.36 GiB ==
    ep_ragged 27.36 GiB, ep_fixed 25.69 GiB. At cf 1.1 marin_ep is 23.18
    GiB — BELOW fixed at 1.33. Our backend adds nothing at layer level.
  - Control CONFIRMED: mep-ctl-ragged (stock levanter ep-ragged flavor)
    hits the byte-identical remat warning (196,678,221,837 vs ...845
    bytes) and the same crash. The ragged class OOMs at hero cf 1.33 on
    today's tree; MHEP-001 ran the same flavor fine on 2026-08-01
    (snapshot b0d20062a, 14.96% MFU) — a ~45 GiB regression landed in the
    two weeks since.
  - A 1.7 GiB/layer flavor delta cannot explain 45 GiB; candidate
    mechanism: under recompute_all XLA cannot rematerialize the a2a
    collectives, so per-layer transport buffers stay live across the
    48-layer scan (48 x ~3.75 GiB ~= 180 GiB matches the reported peak) —
    the fixed flavor's structured custom VJP exists to avoid exactly this.
    Unverified; the fixed control (mep-ctl-fixed-25, stock flags) is
    running to decide flavor-specific vs global regression.
- Interpretation: M8's benchmark is blocked on a pre-existing tree
  regression, not on marin_ep. Next decision forks on the fixed control:
  fixed fits -> ragged-specific regression (bisect or borrow PR #8081's
  memory work); fixed OOMs too -> global regression (bisect the hero
  stack, or benchmark from the last-known-good base).
- Next action: read fixed-control verdict; then either bisect or launch
  ep-marin at --capacity-factor 1.1 on the surviving configuration.

### 2026-08-15 03:20 - MEP-011: illegal-address root cause narrowed to the NCCL 2.30.7 bump; fixed baseline refreshed
- Hypothesis chain (each falsified in turn): OOM (remat estimates 183-195
  GiB) -> allocator limit -> process topology. Killers: (1) the fixed
  flavor's estimate (195.09 GiB) is LARGER than ragged's (183.17) yet
  fixed trains — the remat warning is benign; (2) raising
  XLA_PYTHON_CLIENT_MEM_FRACTION to 0.80 left the peak at 139.74 GiB,
  7.7 GiB under the limit, crash unchanged — not memory at all; (3)
  1 process/GPU (wiki:101 topology) crashed identically. Teardown showed
  `ncclCommWindowDeregister ... unhandled cuda error` — the ragged a2a
  runs on NCCL windows/symmetric memory.
- Repro ladder on one tray (`bench_backend_ep4.py 65536`,
  `repro_ragged_multiproc.py`): EP4 single-process at full hero
  per-device shapes: clean (and mgpu is 6.75x fwd / 4.55x fwd+bwd over
  ragged at this size, bit-identical); EP4 with 4 processes over NCCL
  P2P: 3 fwd+bwd steps clean. The trigger requires the internode
  MNNVL/NCCL-window path.
- Delta hunt: my branch base is fb91b3dc94 (2026-08-10); MHEP-001
  (ep-ragged, EP64, PASSED 2026-08-01) ran nvidia-nccl-cu13 2.28.9; the
  2026-08 native-package advance moved the lock to 2.30.7 (plus the known
  cu12/cu13 libnccl.so.2 path collision). jax/jaxlib 0.11.0 predates
  MHEP-001 and is exonerated.
- Action: pinned `nvidia-nccl-cu13==2.28.9` via override-dependencies
  (commit on marin-ep), filed #8313 (affects main: both ragged flavors),
  launched confirmation run mep-m8-marin-nccl-25 (ep-marin, cf 1.1,
  LHS off + overlap 1, 1 proc/GPU) — Kueue-gated behind cluster capacity
  as of 03:17.
- Baseline refreshed on this exact base: mep-ctl-fixed-25 completed 25/25
  at 16.3 s/it (~257k tok/s at 4.19M tokens/step), drop_fraction 3.575%,
  loss 6.x falling — consistent with the 24.04% MFU-era baseline.
- Next action: confirmation-run verdict; on success, record the ep-marin
  EP64 numbers vs the fixed baseline and proceed to the M8 PR.

### 2026-08-15 07:15 - MEP-012: NCCL downgrade falsified; MNNVL-off probe launched
- Result: the 2.28.9 pin SEGFAULTS host-side in the first train step (all
  64 ranks, inside compiled-executable __call__), in both 1 proc/GPU and
  1 proc/node topologies — worse than the illegal address it replaced.
  Candidate mechanisms (undistinguished): cu12/cu13 libnccl.so.2 wheel
  collision leaving the cu12 build loaded under CUDA 13, or the jax 0.11
  cu13 plugin requiring 2.30-era symbols. Pin reverted (a7bb70c13d);
  #8313 updated. Also: 1 proc/GPU on this branch base segfaults even
  before that (the per-GPU dispatch machinery postdates the branch
  point) — launcher reverted to 1 proc/node (4bb600c3a1).
- Now running: mep-m8-marin-nomnnvl-25 — stock 2.30.7 with
  NCCL_MNNVL_ENABLE=0 (cross-node traffic to IB instead of MNNVL
  windows), ep-marin cf 1.1, LHS off, overlap 1. If it trains, the fault
  is scoped to the MNNVL/NCCL-window transport and we have a benchmark
  configuration; if not, next lever is measuring on top of PR #8081's
  branch (its EP16 4-node proxy passes on newer machinery).
- Also this session: #8311 retitled/reworded (experimental framing, per
  request).

### 2026-08-15 09:00 - MEP-013: current main still crashes; measurement branch on PR #8081
- Result chain: (a) rebased marin-ep onto origin/main (2ed6233bf5, 28
  commits clean; main now defaults 1 process/GPU with attempt-scoped
  coordinator and PGLE off — the machinery whose absence segfaulted the
  old-base per-GPU attempt, and whose per-node PGLE failure modes explain
  the earlier silent wedge); (b) ep-marin on that base STILL dies with
  the #8313 illegal-address teardown at step 1 — main alone does not
  contain the ragged fix; (c) MNNVL-off probe on the old base wedged
  silently (killed).
- Pivot: measurement branch `marin-ep-8081` = PR #8081 head (bd5a7b84d1,
  the branch whose exact EP16 runs pass and hit 18-22% MFU) + merge of
  marin-ep. Merge notes: launch.py switched to the Flavor registry
  (added `ep-marin`); `_local_permute_from_counts` now returns
  physical+active group sizes (ours coincide — no pool padding);
  marin_ep joined _RAGGED_ALL_TO_ALL_IMPLEMENTATIONS so
  `--ragged-all-to-all-splits-per-peer` threads through
  `_shard_a2a_params` into our backend for free; 8081-only launchers
  needed the storage_path import fix from main's rigging refactor.
  82 tests pass. Pushed as marin-ep-8081.
- Launched: mep-m8-marin-8081-25 — ep-marin, cf 1.1, split-32, EP64,
  the launcher's own runtime recipe (per-GPU, LHS off, overlap 1,
  command buffers off, PGLE off, NCCL_BUFFSIZE=1M).
- Next action: verdict; on success score steps 5-24 vs the S28/S33
  baselines (18.08% split-32 ragged, 19.63% cute, 21.66% cudnn-cute at
  EP16; fixed EP64 ~16.3 s/it from mep-ctl-fixed) and vs drops.

### 2026-08-15 10:40 - MEP-014: FIRST PASSING multi-node marin_ep training — EP16 exact proxy, 0.044% drops
- Run: mep-m8-marin-ep16d-25 on branch marin-ep-8081 — ep-marin, EP16
  (4 nodes x 4 GPUs, 1 proc/GPU), E48 (El=3), batch 256 (= 1,048,576
  tokens/step, 65,536 tokens/device — the weaver S-series exact-proxy
  load and the hero per-device load), cf 1.1, split-32 ragged transport,
  LHS off, overlap 1, PGLE/command buffers off.
- Result: ALL 25 STEPS COMPLETE, zero faults. rate 19.3 s/it (whole-run
  mean incl. compile), loss 6.18 falling, **drop_fraction 0.00044
  (0.044%)** — vs 3.575% for fixed_all_to_all at cf 1.33 on the EP64
  hero (mep-ctl-fixed) and vs sim prediction 0.9-1.5% at cf 1.1 under
  trained-router skew (near-uniform init routing gives even fewer).
  ~80x fewer dropped assignments at 17% smaller buffers. R2 satisfied
  with two orders of magnitude of margin on this geometry.
- Path here (context for the run-count): EP64 ragged-class is broken on
  every tree incl. PR #8081's own branch (its exact runs are all 4-node;
  full-rack is its open gate) -> #8313. EP16 at batch 1024/512 OOMs
  (261k/131k tokens/device is 4x/2x the hero load — my batch arithmetic
  error, weaver's exact proxy is batch 256).
- Now running: mep-ctl2-ragged-ep16-25 — stock ep-ragged at the identical
  b256/cf1.1/split-32 config for the same-bytes drop + speed A/B.
- Next: control comparison table (also vs S28 18.08% MFU at cf 1.33);
  logbook + issue; then M8 PR.

### 2026-08-15 10:55 - MEP-015: EP16 A/B complete — parity speed, fewer drops, cf-1.1 win confirmed
- Result (EP16 exact proxy, 4 nodes x 4 GPUs, E48/El=3, batch 256 =
  1,048,576 tokens/step, split-32, LHS off, overlap 1; steady-state
  per-step from tqdm elapsed deltas over the scored window):
  | arm | cf | steady s/step | drop_fraction |
  |---|---|---|---|
  | ep-marin (mep-m8-marin-ep16d) | 1.1 | 19.25-19.5 | 0.00044 |
  | ep-ragged (mep-ctl2-ragged-ep16) | 1.1 | 19.5-19.7 | 0.00067 |
  | ep-ragged S28 (weaver, 8081 branch) | 1.33 | 21.11 | 0 |
  | ep-ragged-cute S33 (weaver) | 1.33 | 19.11 (19.63% MFU) | 0 |
- Interpretation: at the same configuration marin_ep is speed-parity-to-
  slightly-faster than stock ragged with ~1.5x fewer drops (init-uniform
  routing understates the drop advantage; MEP-H1 says ~10x under
  trained-router skew, where per-receiver pooling saturates). Both cf-1.1
  arms beat the cf-1.33 baseline ~8%, in line with MEP-H5. Equivalent
  MFU ~19.7% — ties the CuTe-GEMM variant while still on Triton
  ragged_dot (M7 GEMM headroom is additive from here).
- Caveats: single placement draw each (memory says ±2pp variance);
  init-routing drops only; EP64 remains blocked by #8313, so the R4/goal
  MFU targets are unmeasurable at rack scale until that fix lands.
- Next: M8 PR from the rebased marin-ep branch (measurements cite the
  marin-ep-8081 measurement branch); then M7 GEMM + hierarchical
  transport as autoresearch levers.

### 2026-08-15 12:30 - MEP-016: cuDNN/QuACK GEMM lever lands — 18.0 s/step (~21.2% MFU-equivalent)
- Run: mep-m8-cute-ep16-25 — new `marin_ep_cudnn_cute` implementation
  (marin_ep drop rule + `_cudnn_cute_expert_mlp` grouped GEMMs via the
  `expert_mlp` seam factored into `marin_ep_moe_local`), EP16 exact proxy,
  b256, cf 1.1, split-32. All 25 steps, zero faults.
- EP16 exact-proxy ladder (1,048,576 tokens/step; steady s/step from
  tqdm deltas; MFU-equivalents from the S-series 18.077% <-> 21.109 s
  calibration):
  | arm | cf | s/step | ~MFU | drops |
  |---|---|---|---|---|
  | ep-ragged (S28, weaver) | 1.33 | 21.11 | 18.08% | 0 (init) |
  | ep-marin (ragged_dot) | 1.1 | 19.4 | 19.7% | 0.044% |
  | ep-ragged-cute (S33) | 1.33 | 19.11 | 19.63% | 0 (init) |
  | ep-marin-cudnn-cute | 1.1 | 18.05 | 21.2% | 0.066% |
  | ep-ragged-cudnn-cute (weaver) | 1.33 | 17.62 | 21.66% | 0 (init) |
  | ep-ragged-cudnn-cute (weaver) | 1.00 | 17.09 | 22.32% | 0.486% |
- Interpretation: the GEMM seam transfers cleanly (+8.8% for marin_ep,
  same as the ragged->cudnn-cute delta on weaver's arms). marin cf1.1 vs
  weaver cf1.33 cudnn-cute differ by 2.4% — inside the ±2pp placement
  variance band, needs repeat draws to separate. The drop rule remains
  the structural advantage: their cf1.00 speed point costs 0.486% drops
  at init-uniform routing (and per-cell rules degrade ~10x under trained
  skew), ours holds 0.04-0.07% at cf1.1.
- Toward the 25%/350k EP64 goal: remaining levers are EP64 unblocking
  (#8313 — upstream), fused-mgpu/hierarchical transport (single-tray
  6.7x fwd over ragged suggests several pp at rack scale), M7 GEMM+
  transport fusion, and weight-gradient kernels (weaver bounded packaged
  gains at ~19.74% on their arms; our transport savings stack on top).
- Next: monitor PR #8320 review; EP64 rerun on #8313 fix; consider
  repeat draws for variance bounds before finer claims.

### 2026-08-15 14:50 - MEP-017: FIRST FULL EP64 HERO RUN — 227k tok/s (~20.7% MFU) at 0.66% drops
- Run: mep-ep64c-cute-25 — ep-marin-cudnn-cute, TRUE hero (d6144/L48/E192/
  top-4, batch 1024 = 4,194,304 tokens/step, EP64, 1 rack), cf 1.1,
  split-32, LHS off, overlap 1, `--xla_gpu_ragged_all_to_all_mode=symmetric`,
  and the #8077 kMaxPeers-128 patched PJRT wheel via the newly ported
  `--pjrt-wheel` (prefix-glob fetch + forced single NCCL). 25/25 steps,
  16/16 tasks succeeded, loss 6.02 falling, drop_fraction 0.00662.
- Scored: steady 18.5 s/step (20it@7:14 -> 24it@8:28) = 226.7k tok/s/rack
  = ~20.75% MFU on the 24.04% <-> 262,683 tok/s hero calibration.
  Consistent with EP16's 21.2% minus rack-scale transport overhead, and
  with the sibling campaign's 20.20% EP64 patched-wheel run (E384/top-8).
- Unblock credit: the #8077 campaign root-caused #8313 as the
  MultiGpuBarrierWithNcclKernel kMaxPeers=32 overflow at 64 ranks and
  built the patched wheel. Ported here: --pjrt-wheel + setup-script
  composition (fray setup_scripts REPLACES the default uv-sync — custom
  scripts must be appended after an explicit default_setup_script; and
  the wheel fetch must glob the S3 prefix).
- Goal gap (>=25% MFU / >=350k tok/s): need <=13.2 s/step, -29% from
  here. Levers: EP64 same-config stock control (launched:
  mep-ctl3-ragcute-ep64) for the rack-scale A/B; cf sweep toward 1.0
  (drop rule holds 0.66% at 1.1 — room below R2's 2%); splits-per-peer
  retune at 64 ranks; wgrad kernels; hierarchical mgpu-intranode
  transport (EP4: fused path 6.7x over ragged); M7 fusion.
- Note: fixed_all_to_all still leads wall-clock (16.3 s/step / 24.04%)
  but at 3.58% drops; drop-adjusted the gap is 253k vs 225k effective
  assignments/s.

### 2026-08-15 15:10 - MEP-018: EP64 same-config A/B — marin_ep leads stock ragged by ~3% at equal drops
- Run: mep-ctl3-ragcute-ep64-25 — ep-ragged-cudnn-cute at the identical
  EP64 hero config (cf 1.1, split-32, patched wheel, symmetric mode).
  25/25 steps, 16/16 succeeded, drop_fraction 0.684%. Steady state from
  step marks (14it@18:52 -> 22it@21:24): 19.0 s/step (long early-window
  data-loader stall inflates the cumulative average; scored window only).
- EP64 hero ladder (4,194,304 tokens/step, one rack):
  | arm | cf | s/step | tok/s | ~MFU | drops |
  |---|---|---|---|---|---|
  | fixed_all_to_all (mep-ctl-fixed) | 1.33 | 16.3 | 257k | 24.04% | 3.58% |
  | ep-marin-cudnn-cute | 1.1 | 18.5 | 227k | 20.75% | 0.66% |
  | ep-ragged-cudnn-cute | 1.1 | 19.0-19.2 | 220k | 20.1% | 0.68% |
- Interpretation: the drop-rule backend is now the fastest ragged-class
  arm at both EP16 and EP64 (~3% over stock at rack scale, single draws).
  fixed_all_to_all keeps a 13% wall-clock lead bought with 5.4x more
  drops; on effective (non-dropped) assignment throughput the gap is
  253k vs 225k. The 25%/350k goal needs -29% step time from here — the
  ragged transport segment and wgrad kernels are the dominant remaining
  costs (weaver's XProf on the EP16 proxy).
- Next: cf 1.0 marin arm (MEP-H5 curve, expect ~-4% step at ~1.5-2%
  drops); splits-per-peer retune at 64 ranks; then the deep levers
  (hierarchical fused transport, wgrad kernels, M7).

### 2026-08-15 16:10 - MEP-019: cf 1.0 EP64 arm — drop-adjusted wash vs cf 1.1; keep cf 1.1
- Run: mep-ep64-cf10-25-20260815 — ep-marin-cudnn-cute at EP64 hero,
  cf 1.0, split-32, patched wheel, symmetric mode. 25/25 steps, 16/16
  tasks succeeded, loss 6.01 falling, final-step drop_fraction 3.27%
  (still declining from the early routing-warmup peak at step 24).
- Scored: steady 18.0 s/step (14it@5:04 -> 25it@8:22 = 198 s / 11
  steps) = 233.0k tok/s = ~21.3% MFU. Cumulative rate 17.9 s/it agrees.
- vs cf 1.1 (MEP-017: 18.5 s, 226.7k, 0.66% drops): cf 1.0 buys -2.7%
  step time for 5x the drops. Drop-adjusted effective assignments/s:
  225.4k (cf 1.0) vs 225.2k (cf 1.1) — a wash, exactly the MEP-H5
  plateau the simulator predicted. Single draws; ±2pp placement band.
- Decision: keep cf 1.1 as the marin_ep operating point — same
  effective throughput, 5x fewer drops (loss-side risk strictly lower).
  cf sweep concluded (1.33 -> 1.1 -> 1.0 measured on hardware).
- Next: splits-per-peer retune at 64 ranks (s8/s16 arms vs the s32
  baseline 18.5 s; 63 peers make per-peer slices ~4x smaller than at
  EP16 where s32 won, so fewer splits should now be optimal); then
  wgrad kernels / hierarchical fused transport / M7.

### 2026-08-15 17:20 - MEP-020: splits-per-peer retune at 64 ranks — flat; keep 32
- Runs: mep-ep64-s16-25-20260815 (steady ~18.2-18.9 s/step, drops
  0.58%), mep-ep64-s64-25-20260815 (steady ~18.7 s/step from marks
  10it@3:59 -> 25it@8:39 = 280 s / 15, drops 0.64%). Both
  ep-marin-cudnn-cute cf 1.1 EP64 hero, 25/25 steps. s8 arm was
  preempted mid-run by the s64 production gang (one rack actually free;
  the interactive-priority second rack did not materialize) and
  requeued; its result lands later but cannot change the decision.
- Ladder vs the s32 baseline (MEP-017: 18.5 s/step): s16 and s64 are
  both within the ±2pp placement band. The splits knob is FLAT at EP64
  in the 16-64 range — with 63 peers the per-peer slices are already
  small enough that block occupancy is not the constraint (unlike EP16,
  where s32 vs s1 was 1.4x). Decision: keep splits-per-peer=32.
- Tuning knobs (cf, splits) are now exhausted at EP64: best ragged-class
  step time holds at 18.5 s (~20.75% MFU, 227k tok/s). The 25%/350k
  goal needs 13.2 s/step — only the deep levers can close it: wgrad
  kernels (Triton ragged_dot ~920 TF/s vs 1747 dense XLA on GB200),
  hierarchical fused transport (mgpu intranode 6.7x at EP4 + NCCL
  internode), M7 persistent fusion (sim: ~97% of transport hidable).
- Next: scope the wgrad lever — find what the cudnn_cute expert MLP
  backward actually executes and whether the CuTeDSL grouped GEMM
  (2.2 PF/s in the MXFP8 work) can take dwgrad/dgrad.

### 2026-08-15 17:35 - MEP-021: first EP64 step profile — barrier+metadata overhead named, LHS-off is suspect
- Run: mep-ep64-prof-25-20260815 — ep-marin-cudnn-cute cf 1.1 s32 with
  `--profile-steps 3 --profile-start-step 12`; run itself matched
  baseline (25/25, 18.7 s/it cumulative). Rank-0 XProf uploaded to
  s3://marin-us-east-02a/tmp/ttl=30d/xprof/mep-ep64-prof-25-20260815;
  analyzed via the hosted xprof service HTTP API (IAP headers from
  rigging `credentials_for`; kernel_stats + op_profile tools).
- Per-step attribution (19.5 s rawTime; 18.5 s wall):
  | bucket | s/step | note |
  |---|---|---|
  | custom-call (QuACK/cuDNN/FA4) | 8.92 | compute |
  | XLA fusions (loop+input+fmt) | 4.49 | 41k launches/step |
  | ragged-all-to-all (incl. barrier) | 2.30 | EP transport |
  | all-gather + all-reduce + reduce-scatter | 2.33 | FSDP/DP collectives |
  | plain all-to-all (routing counts) | 0.96 | metadata exchange! |
  | copy | 0.50 | |
- Kernel-level: MultiGpuBarrierWithNcclKernelImpl = 1.39 s/step busy
  (576/step, avg 2.4 ms, max 48 ms) — the symmetric-mode barrier
  absorbing cross-rank skew; RaggedAllToAllWithSymmetricMemoryKernel
  0.91 s/step; ncclDevKernel_SendRecv 0.78 s/step.
- Two immediate levers surfaced:
  1. LHS is OFF (pre-patch stability recipe). The #8313 root cause
     (kMaxPeers barrier overflow) plausibly explains the original
     "LHS corrupts ragged-a2a" incident too — with the patched wheel,
     LHS may be safe again, unlocking overlap of the 2.3 s DP
     collectives. A/B launched: mep-ep64-lhs-25-20260815 (symmetric
     mode + patched wheel, LHS/overlap flags dropped).
  2. The 0.96 s/step routing-counts all-to-all is pure metadata sync
     (~10 ms per call at 2/layer) — the M7 fused design folds this
     into the put kernel; also reachable sooner by caching/overlapping
     the counts exchange.
- Exploratory (single trace, rank 0 only).

### 2026-08-15 17:58 - MEP-022: LHS scheduler is SAFE and ~-2% with the patched wheel
- Run: mep-ep64-lhs2-25-20260815 — ep-marin-cudnn-cute cf 1.1 s32 with
  EXPLICIT `--xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_experimental_parallel_collective_overlap_limit=4` (+
  symmetric mode + kMaxPeers-128 wheel). 25/25 steps, no illegal
  address, drops 0.635%. Steady ~18.1-18.2 s/step (12it@4:37 ->
  24it@8:14 = 217 s / 12; cumulative 18.1) vs baseline 18.5 and the
  same-config replicate 18.8 — ~-2%, and the fragile LHS-off recipe is
  no longer needed. Supports the hypothesis that the historical "LHS
  corrupts ragged-a2a" incident (wiki:101, PR #8081 flags) was really
  the #8313 kMaxPeers barrier overflow.
- Gotcha logged: train.py `_apply_hero_ep_runtime_defaults` re-applies
  LHS=false/overlap=1 for ragged implementations whenever the flags are
  absent from XLA_FLAGS — a flag-dropping A/B silently replicates
  baseline (run mep-ep64-lhs-25-20260815 was this accident: 18.8
  s/step, a useful draw-noise replicate). A/Bs must pass explicit
  values.
- New best marin arm: ~18.1 s/step = ~232k tok/s = ~21.2% MFU at 0.64%
  drops. Exploratory (single draw).
- Next: command buffers arm (mep-ep64-cb-25, +`command_buffer=FUSION`
  on top of LHS-on; #5675 crash was bisected to COLLECTIVES capture so
  FUSION-only should be safe); then counts-a2a overlap and M7.

### 2026-08-15 18:25 - MEP-023: splits-1 under symmetric mode is 1.4x WORSE; cb neutral; recipe settles
- Runs: mep-ep64-cb-25-20260815 (LHS-on + `command_buffer=FUSION`):
  steady ~18.2 s/step — command buffers neutral on top of LHS.
  mep-ep64-s1sym-25-20260815 (LHS-on, splits-per-peer=1): 25.4 s/it
  cumulative — 1.4x worse; the symmetric-memory kernel still needs the
  split fan-out for block occupancy. s32 confirmed necessary.
- Settled EP64 recipe for ep-marin-cudnn-cute: cf 1.1, splits 32,
  patched kMaxPeers wheel, `--xla_gpu_ragged_all_to_all_mode=symmetric
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_experimental_parallel_collective_overlap_limit=4`.
  Best: ~18.1-18.2 s/step, ~232k tok/s, ~21.2% MFU, 0.64% drops.
- Knob space is exhausted (cf, splits, LHS, overlap, command buffers).
  Remaining -4.9 s to the 13.2 s goal is structural: counts-a2a
  fold/overlap (0.96 s), barrier rounds (1.39 s), fusion-launch mass
  (4.5 s), M7.

### 2026-08-15 19:45 - MEP-024: HIERARCHICAL GATE PASSED — fused intranode puts work in a multi-process mesh (nightly stack)
- Spike: experiments/marin_ep/bench/spike_hier_intranode.py on 2 GB200
  nodes (dev pods, 1 process/node x 4 GPUs). put_segments over a
  process-local "gpu" sub-axis while the mesh's "node" axis spans
  processes; all 8 pool shards asserted against the NumPy plan
  reference, cross-talk-tagged sends.
- Result ladder (stock jax 0.11.0 -> dev20260809 nightly):
  | probe | stock 0.11.0 | nightly |
  |---|---|---|
  | psum over local "gpu" sub-axis | OK | OK |
  | psum over global 8-rank clique | NCCL bootstrap "connection refused to own IP" (leader deadlocked in clique-init rendezvous callback) | OK |
  | fused spike (put_segments intranode) | wedged in global clique init | CORRECT on both procs, twice (semaphore reuse fine) |
- Conclusion: the 1-process-per-node topology (required by the two-hop
  hierarchical transport) is BROKEN on stock jax 0.11.0 (multi-local-GPU
  cross-process clique init deadlock — same family as the fast-restart
  clique hang, but cache rotation does NOT fix it) and WORKS on the
  production nightly (dev20260809). The 148 GB cuMemAllocAsync errors in
  the logs are allocator pool-probing noise, not the failure.
- Unblocked design: EP64 as 16 processes x 4 GPUs, two-hop dispatch —
  ragged/NCCL internode hop (63->15 peer fan-out, splits retune needed)
  + fused mgpu intranode hop (replaces local permutes; EP4 measured
  6.7x over ragged). Combine transposes both hops.
- Next: implement the two-hop plans (split `accepted` by destination
  node; stage-A ragged a2a to same-local-rank peers, stage-B
  put_segments within the node), CPU-conformance first, then EP16 proxy
  at 4 nodes x 1 proc/node.

### 2026-08-15 20:30 - MEP-025: hier transport GPU-validated single-process (values + grads)
- Change: `transport="hier"` implemented end to end — hop A =
  `ragged_all_to_all` on the flat expert axis driven by
  `_shard_a2a_params(hier_flat_counts(...))` (nonzero only for
  same-local-rank peers, node-order staging receive layout); hop B =
  `put_segments` with `num_devices=intranode_size` and flat intranode
  dest ids (`hier_dispatch_segments`/`hier_combine_segments` in
  marin_ep_transport); combine transposes both hops; vjp reuses
  `put_with_transpose` + jax's ragged-a2a transpose. Registered as
  `marin_ep_hier_cudnn_cute` / flavor `ep-marin-hier-cudnn-cute`
  (launcher switches that flavor to processes_per_task=1). Host plan
  tests in experiments/marin_ep/tests/test_hier_plans.py.
- GPU (1 GB200 tray, single process, nodes=2 x gpus=2 factorization):
  conformance matrix `pytest experiments/marin_ep/tests/test_xla_backend.py
  -n0` = 11 passed — ragged/mgpu/hier x group sizes, values AND grads vs
  the dense oracle; hier result identical across intra=4 and intra=2
  factorizations.
- Two traps burned an hour:
  1. Fresh `uv sync --extra=gpu` left nvidia-nccl-cu12 2.28.9 owning
     libnccl.so.2 — every ragged-class program segfaulted with EMPTY
     logs (even hier at nodes=1). Fix as always: uninstall cu12,
     reinstall nvidia-nccl-cu13==2.30.7. mgpu (no NCCL in the seam)
     passed throughout, which is what fingered the collision.
  2. pytest-xdist runs worker PROCESSES that share the tray's 4 GPUs —
     mixed hier+mgpu sessions failed a different mgpu case each run.
     Sequential `-n0` is required for multi-GPU collective tests.
- Next: 2-node smoke of hier (1 proc/node, nightly stack — stock 0.11
  cannot run the topology per MEP-024), then the EP16 proxy hero run
  with `--flavor ep-marin-hier-cudnn-cute`.

### 2026-08-15 20:45 - MEP-026: first multi-node hier run — EP16 proxy 19.2 s/step at 0.068% drops
- Run: mep-hier-ep16-25-20260815 — ep-marin-hier-cudnn-cute at the EP16
  exact proxy (4 nodes x 1 process/node, batch 256, E48), settled recipe
  (cf 1.1, split-32, patched wheel, symmetric, LHS on) + JAX_ENABLE_PGLE
  false (harness defaults PGLE ON for processes_per_task=1 — the
  per-node auto-PGLE wedge class). 25/25 steps, 4/4 tasks, loss 6.19
  falling.
- Scored: steady 19.2 s/step (12it@5:02 -> 24it@8:52 = 230 s / 12);
  drop_fraction 0.068% — matches the flat path's EP16 proxy drops
  (0.066%), confirming transport-invariance of the drop rule on real
  hardware.
- vs flat single-hop ep-marin-cudnn-cute at the same proxy: 18.05 s
  (MEP-016) — hier v1 is ~6% slower HERE, but EP16 has only 3 internode
  peers so the 63->15 peer reduction that motivates hier barely
  registers; the unoptimized costs (4x-pool staging zeros, two extra
  kernel launches per direction, no inter-hop overlap, hop-A splits
  untuned) dominate. The informative A/B is EP64 (launched:
  mep-hier-ep64-25-20260815).

### 2026-08-15 21:10 - MEP-027: hier v1 at EP64 = 19.0 s/step; profile shows the sync scope never shrank
- Runs: mep-hier-ep64-25-20260815 (25/25, steady 19.0 s/step from
  12it@4:51 -> 24it@8:39 = 228 s / 12, drops 0.538%, loss 6.02) and
  mep-hier-prof-25-20260815 (25/25, 19.3 cumulative, XProf steps 12-15
  uploaded; analyzed via the hosted xprof API — note kernel_stats now
  aggregates 4 devices/process, divide by 4).
- A/B: hier v1 19.0 vs flat LHS-on 18.1-18.2 — ~5% SLOWER. Per-device
  attribution vs the flat profile (MEP-021):
  | bucket | flat | hier v1 |
  |---|---|---|
  | ragged-a2a + barrier (op bucket) | 2.30 | 1.46 |
  | barrier kernel busy | 1.39 | 1.83 (288/step, fewer but longer) |
  | ragged-a2a kernel busy | 0.91 | 0.88 |
  | mosaic puts | — | 0.60 |
  | custom-call total | 8.92 | 10.48 |
  | fusions | 4.05 | 4.39 |
- ROOT CAUSE of the miss: hop A ran on the FLAT 64-rank axis with
  zero-sized sends to the 48 different-rank peers — the NCCL
  communicator and symmetric-mode barriers stayed 64-rank, so the
  63->15 peer reduction reduced traffic but not synchronization scope.
  The staging pass + extra launches then netted ~+0.9 s.
- Fix implemented (hier v2): `jax.lax.ragged_all_to_all` accepts
  `axis_index_groups` — hop A now runs in 4 disjoint node-ordered
  same-rank groups of 16 ranks (group-relative `_shard_a2a_params` on
  the [Nodes, Nodes] restricted count matrix; group order = node order
  = the staging layout, so hop-B plans are unchanged). 16-rank comms
  also sit below the kMaxPeers=32 overflow for hop A itself. A/B
  launched: mep-hier2-ep64-25-20260815.

### 2026-08-15 21:25 - MEP-028: scoped ragged groups are a dead end in current XLA — v2/v2b OOM
- Runs: mep-hier2-ep64-25 (symmetric mode, groups) and mep-hier3-ep64-25
  (no symmetric, mem fraction 0.72) both died in
  `ncclCuMemAlloc ... 'out of memory'` inside a DENSE `ncclAlltoAll`
  (`send_contiguous`/`recv_contiguous`) during jit_train_step.
- Diagnosis: with `axis_index_groups`, XLA does NOT use the
  symmetric-memory ragged-a2a kernel; it falls back to the
  dense-alltoall decomposition with contiguous packed buffers, whose
  allocation at hero shape exceeds HBM regardless of the symmetric flag
  or memory fraction. The group-comms commit is REVERTED — the branch is
  back on working hier v1 (flat 64-rank hop A, 19.0 s/step).
- Remaining routes to a scoped hop A:
  1. Real ("node","gpu") mesh axes in the trainer so the symmetric
     ragged kernel runs on a named 16-rank sub-axis (invasive mesh
     surgery through grug sharding).
  2. Fixed-internode hybrid: replace hop A with a dense
     `jax.lax.all_to_all` over node-capacity-padded chunks (the
     fixed_all_to_all trick at node granularity; waterfill still governs
     drops, padding costs bytes not assignments) + fused intranode puts.
     Predictable memory, one NCCL op, no barrier kernels; plain
     all_to_all with axis_index_groups is a long-standing supported
     path, unlike ragged.
- Next: option 2 is the cheaper experiment and composes with the
  existing hop-B plans (fixed chunk offsets replace ragged offsets).

### 2026-08-15 21:55 - MEP-029: no cheap wins remain — XLA ragged a2a is synchronous, so chunked pipelining is out
- Observation from both step profiles: `all-gather-start` / `all-reduce-start`
  appear as async pairs, but `ragged-all-to-all` has no `-start` form — XLA
  executes it synchronously. Chunked dispatch/GEMM pipelining (split experts
  into chunks, overlap a2a[c+1] with GEMM[c] under the LHS scheduler) therefore
  cannot overlap anything: chunks would serialize with extra launch overhead.
  The idea is dead at the XLA level; it needs either upstream async ragged
  a2a or the M7 custom persistent kernel.
- Session-close ladder (EP64 hero, one rack, single draws, ±2pp):
  | arm | s/step | tok/s | ~MFU | drops |
  |---|---|---|---|---|
  | fixed_all_to_all cf1.33 | 16.3 | 257k | 24.0% | 3.58% |
  | ep-marin-cudnn-cute cf1.1 s32 LHS-on | 18.1-18.2 | 232k | 21.2% | 0.64% |
  | ep-marin-cudnn-cute cf1.1 s32 (LHS off) | 18.5 | 227k | 20.8% | 0.66% |
  | ep-marin-hier-cudnn-cute v1 | 19.0 | 221k | 20.2% | 0.54% |
  | ep-ragged-cudnn-cute cf1.1 s32 | 19.0-19.2 | 220k | 20.1% | 0.68% |
- Goal gate: 350k tok/s = 12.0 s/step. Ordered remaining levers by size
  (from the profiles): grouped/attention GEMM efficiency (8.9 s bucket),
  XLA fusion mass (4.5 s, 41k launches), FSDP collective overlap (2.3 s),
  M7 fused metadata+transport+GEMM (sync 1.4 s + counts 1.0 s + hides the
  0.9 s a2a). None is a knob; all are engineering projects.
- Rack experiments paused; PR #8320 monitored for human review.

### 2026-08-16 09:10 - MEP-030: multi-process fused transport is BACK — collective-metadata spike PASSES cross-node
- Context: MEP-008 concluded multi-process mosaic transport was dead
  upstream (NVSHMEM removed). That conclusion is now obsolete: upstream
  REPLACED the backend rather than dropping the capability — "XLA is
  deprecating NVSHMEM. Instead of NVSHMEM customers should use NCCL
  Device API through collective metadata" (jax 181661c3a5, 2026-06-18),
  and jax c98d8fa09c migrated tests/mosaic/gpu_distributed_test.py to a
  multi-process (1 device/process) collective-metadata test. In current
  jax, `to_remote`/`async_copy(gmem_peer_id=...)` read per-peer parameter
  pointers from a collective-metadata buffer whenever the mesh spans
  processes and `is_nvshmem_available()` is false (no XLA flags needed);
  Pallas `remote_ref`/semaphores lower through the same path with no
  process-count gate. All machinery predates our dev20260809 nightly.
- Spike: `smoke_transport_multinode.py` with the nvshmem hack DELETED
  (commit f49d1df629), 2 GB200 nodes x 4 GPUs, 1 process/GPU (8 procs),
  stock nightly wheels + nccl-cu13 2.30.7.
- Result: ALL 8 processes CORRECT (pool 28182x1024, both attempts),
  per-rank egress 153-209 GB/s through the fused puts. Confirmed again
  on the dev20260816 nightly. The `transport="mgpu" iff process_count()==1`
  gate and its docstring describe the July state and are now wrong.
- Interpretation: the direct flat fused transport at EP64 (production
  1 proc/GPU topology, no hierarchy, no FFI) is viable in-framework —
  standalone. Full-layer integration hits an XLA bug (next entry).

### 2026-08-16 09:40 - MEP-031: full-layer mgpu multi-process blocked by an XLA buffer-alignment bug; patched-wheel fix in progress
- Smoke: new `smoke_mgpu_train_multiproc.py` (b246e3e59c) — the EP8-style
  oracle conformance smoke with `transport="mgpu"` forced, 1 proc/GPU.
- Result: FAILS at first execution on both nightlies:
  `ncclDevrWindowRegisterInGroup ... Window address must be suitably
  aligned.` -> `INTERNAL: NCCL operation ... invalid argument` in
  jit_marin_ep_moe_local. Reproduces with 2 procs on one node.
- Bisection ladder (repro_align.py on put_segments): single kernel call
  passes with entry-param, temp, computed-plan, and consumed-output
  variants; TWO put_segments calls in one executable FAIL. Instrumented
  registration (TF_CPP_VMODULE=nccl_symmetric_memory=3): first window =
  2 MiB arena base (aligned, OK); second window = ptr 0x...cf300
  (offset 0x300 = 768 = 3x256) size 876544 (= send_rows x H x 4 — the
  second put's src temp) in the DEFAULT BFC arena, not the collective
  arena. Diagnosis: XLA's GpuCollectiveBufferAnalysis misses some mosaic
  collective-metadata params; the runtime then window-registers a plain
  temp at XLA's 256-byte packing alignment
  (kXlaAllocatedBufferAlignBytes), and NCCL requires
  NCCL_WIN_REQUIRED_ALIGNMENT = 4096. A minimal pure-Pallas two-kernel
  repro does NOT trigger the analysis miss (its temps get copied into
  collective space correctly) — the miss needs put_segments-like
  structure; upstream issue to follow once the fix is validated.
- Fix in progress: rebuild `jax-cuda13-pjrt` at the #8077 pins
  (jax 8d1be7d / xla 60f8069) with kMaxPeers 32->128 (as in the existing
  patched wheel) PLUS buffer-assignment color_alignment 256->4096 for all
  colors (compile_module_to_llvm_ir.cc) so every registered window offset
  is 4KB-aligned. Collective-color-only alignment would NOT cover the
  default-arena escapee. Build gotcha: aarch64 XNNPACK fp8 kernels need
  `--define=ynn_enable_arm64_neonfp8=false` (jax ci_linux_aarch64_base
  does the same); hermetic llvm18 otherwise fine.
- Pod traps: the task image has no ps/pgrep/pkill — poll scripts must use
  /proc or log staleness; `git checkout <sha>` keeps prior patch edits
  (reset --hard before repatching).
- Next: BUILD_OK -> 2-proc layer smoke on the patched wheel -> 8-proc
  cross-node -> `ep-marin-mgpu-cudnn-cute` hero flavor -> EP16 proxy ->
  EP64. Then file the openxla/xla issue with the validated patch.

### 2026-08-16 09:52 - MEP-032: fused transport CONFORMANT multi-process — three-layer runtime fix validated at 8 ranks cross-node
- Fix stack (all validated on 2 GB200 nodes, 8 procs, 1/GPU):
  1. Patched PJRT wheel v3 at the #8077 pins: kMaxPeers 32->128 + buffer-
     assignment color_alignment 256->4096 + tsl BFC RoundedBytes 4096
     granularity (chunk offsets must be 4KB for NCCL windows;
     kMinAllocationBits bump trips the BinForSize invariant — round sizes
     instead). Wheel:
     s3://marin-us-east-02a/marin/research/mcwitt-ra2a/pjrt-kmax128-align4096-dev0811-20260816/
  2. `--xla_gpu_enable_dynamic_slice_fusion=false`: the pass wraps the
     combine put + its pool slice into a %dynamic-slice-fusion computation,
     hiding the custom call from GpuCollectiveBufferAnalysis (matches on
     the raw instruction) -> params escape collective coloring.
  3. `--xla_gpu_enable_allocator_spatial_partitioning=false` (BFC-allocator
     only): with preallocate + spatial partitioning ON, ONE shared 138 GiB
     pool serves both default and collective memory; every mosaic window
     then reserves the whole pool's VA in NCCL's window space
     (limit 0x2f00000000 = 188 GiB) and the second executable's window
     fails with `ncclSpaceAlloc ... No suitable space`. Disabling forces
     the separate CollectiveBFCAllocator (small grow-on-demand regions).
     Hero runs default to XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async, which
     already creates the separate collective allocator — flag 3 matters
     for BFC-allocator contexts (tests/smokes) only.
- Diagnosis ladder for the record: registration VLOG
  (TF_CPP_VMODULE=nccl_symmetric_memory=3) gives ptr/size per window; the
  failing sizes identified each layer (0x...300 offset = 256B BFC chunk
  packing; 0x228e600000 = the whole 138 GiB shared pool).
- Result: smoke_mgpu_train_multiproc.py 8/8 ranks CONFORMANT — forward
  pools, drop count (1019, bit-identical to the ragged-path EP8 smoke's
  historic value), and x/w13/w2 gradients vs the oracle, cross-node.
- Launched: mep-mgpu-ep16-25-20260816 — EP16 proxy (4 nodes, b256, E48,
  cf 1.1) on ep-marin-mgpu-cudnn-cute (2bb65aa8ae) + wheel v3 + settled
  recipe flags + dynamic-slice-fusion off. Basis: flat ragged 18.05 s,
  hier 19.2 s at this proxy.
- Upstream: three findings to file on openxla/xla once EP64 evidence is in
  (BFC chunk alignment vs NCCL_WIN_REQUIRED_ALIGNMENT; dynamic-slice
  fusion vs collective coloring; shared-pool window VA exhaustion).

### 2026-08-16 10:02 - MEP-033: EP16 proxy NEW BEST — fused transport 17.0 s/step, 6% ahead of flat ragged
- Run: mep-mgpu-ep16-25-20260816 (ep-marin-mgpu-cudnn-cute, wheel v3,
  settled flags + --xla_gpu_enable_dynamic_slice_fusion=false, EP16 exact
  proxy: 4 nodes x 4 GPUs 1 proc/GPU, b256, E48, cf 1.1). 25/25 SUCCEEDED,
  loss 6.73->falling.
- Scored: steady 17.0 s/step (12it@4:23 -> 24it@7:47 = 204 s / 12);
  drop_fraction 0.00044 — IDENTICAL to the flat path's proxy value
  (MEP-014), third transport to reproduce it exactly.
- Proxy ladder now: mgpu 17.0 < flat ragged 18.05 < hier 19.2 (s/step).
  First transport-level win of the campaign; EP16 has only 3 internode
  peers, so EP64 (63 peers — where symmetric-mode barriers and splits cost
  the ragged path ~2.3 s) should widen the gap if put_segments scales.
- Launched: mep-mgpu-ep64-25-20260816 (same recipe, full hero EP64).
  Caveat for the A/B: the 18.1 s flat basis ran the OLD kmax128 wheel; if
  mgpu wins at EP64, rerun the flat control on wheel v3 to isolate
  transport from the wheel's alignment patches.

### 2026-08-16 10:13 - MEP-034: EP64 hero on fused transport — 17.75 s/step, new rack best
- Run: mep-mgpu-ep64-25-20260816 (ep-marin-mgpu-cudnn-cute, full hero
  EP64, b1024, cf 1.1, wheel v3, settled flags + dynamic-slice-fusion
  off). 25/25 SUCCEEDED, loss 6.38 falling.
- Scored: steady 17.75 s/step (10it@3:50 -> 22it@7:23 = 213 s / 12);
  drop_fraction 0.00631 (matches flat 0.00662). ~236k tok/s/rack,
  ~21.6% MFU on the hero calibration.
- Ladder: mgpu 17.75 < flat ragged 18.1-18.2 (old wheel) < hier 19.0 <
  ragged-stock 19.0-19.2. ~2% ahead of flat — smaller than EP16's 6%,
  suggesting put_segments' 63-peer fan-out has scaling costs of its own
  (worth a profile: did the 1.4 s barrier + 0.9 s a2a buckets shrink and
  what replaced them?).
- In flight: mep-ctl4-cute-ep64-25-20260816 — flat ragged control on
  wheel v3 + same flags, to isolate transport delta from the wheel's
  alignment patches (placement variance is ±2pp; single draws).
- Goal gate: 12.0 s/step. Transport-lever ceiling nearly reached; the
  remaining mass is GEMMs/fusions/FSDP per MEP-021's attribution.

### 2026-08-16 10:28 - MEP-035: upstream report filed — openxla/xla#47406
- Filed the three collective-metadata findings (BFC/buffer-assignment
  256B offsets vs NCCL_WIN_REQUIRED_ALIGNMENT; dynamic-slice fusion hiding
  mosaic calls from GpuCollectiveBufferAnalysis; shared spatial-partitioned
  pool exhausting NCCL window VA) with validated one-line fixes and the
  public repro pointers. Cross-referenced #47283.
- Profile run mep-mgpu-prof-25-20260816 in flight (steps 12-15).

### 2026-08-16 11:00 - MEP-036: fused-EP64 profile — transport is no longer the story
- Trace: mep-mgpu-prof-25-20260816 steps 12-15 (17.6-18.1 s/step whole-run
  rate), s3://marin-us-east-02a/tmp/ttl=30d/xprof/mep-mgpu-prof-25-20260816
  (30d TTL). Kernel-stats busy-time attribution, rank-0 device, per step
  (streams overlap — busy != exposed; treat as relative):
  | bucket | ms/step | share |
  |---|---|---|
  | GEMMs (quack/cudnn/nvjet/fa4) | 4832 | 27% |
  | "other" (ncclSymk one-shots incl. FSDP AG/RS/AR) | 3856 | 22% |
  | XLA fusions | 3758 | 21% |
  | nccl misc | 2618 | 15% |
  | MultiGpuBarrierWithNcclKernelImpl | 1531 | 9% |
  | mosaic put_segments (all four puts + vjp) | 1163 | 6.5% |
- Key finding: the barrier kernels (3 sites x 48/step, 1.53 s busy) ride
  the FSDP symmetric one-shot collectives (AllGather_STMC 790 ms,
  ReduceScatter_LDMC 625 ms, AllReduce_AGxLLMC 314 ms), NOT the MoE
  transport. The fused transport itself is 1.16 s busy vs the ragged
  path's 2.30 s ragged+barrier — the win the wall-clock delta (0.25 s)
  only partially realizes because much of both was already overlapped.
- Interpretation: transport-lever ceiling reached, as MEP-029 predicted.
  Remaining mass to the 12.0 s goal: GEMM efficiency (4.8 s busy),
  fusion mass (3.8 s), FSDP one-shot collectives + their barriers
  (~3 s), routing metadata. M7 (persistent GEMM consuming pool segments
  behind arrival flags) attacks transport+sync+part of the fusion mass
  around dispatch; GEMM/fusion work is independent of transport.
- Next: M7 groundwork begins (task #10) — the multi-process substrate it
  needs is now validated.

### 2026-08-16 11:55 - MEP-037: PGLE arm FAILS (OOM in dense ncclAlltoAll during profiling pass) — negative result
- Run: mep-mgpu-pgle-25-20260816 (fused flavor + JAX_ENABLE_PGLE=true,
  otherwise the MEP-034 recipe). All 16 tasks crashed at the PGLE
  profiling step: `ncclAlltoAll(send_contiguous...) ... Cuda failure 2`
  (out of memory) — the profiling compile falls back to the DENSE
  all-to-all decomposition with contiguous packed buffers (same failure
  class as hier v2's axis_index_groups OOM, MEP-028), which does not fit
  at hero shape. Killed; PGLE stays OFF for the fused flavor. Third
  independent PGLE dead end on this stack (B200 null, 1-proc/node wedge,
  now the mgpu OOM).
- Launched: mep-mgpu-cb-25-20260816 — command buffers FUSION arm
  (--xla_gpu_enable_command_buffer=FUSION) on the fused flavor; historic
  FUSION result on ragged was neutral, retrying because the fused step
  has 41k launches and no ragged a2a in the capture set.
- Ops note: killed my wedged mep-mgpu-prof coordinator (all ranks exited
  0 at 17:35Z but the job held its rack >1h; teardown wedge).

### 2026-08-16 12:10 - MEP-038: command buffers neutral on the fused flavor — knob space exhausted again
- Run: mep-mgpu-cb-25-20260816 (--xla_gpu_enable_command_buffer=FUSION on
  the MEP-034 recipe): steady 17.83 s/step (12it@4:34 -> 24it@8:08 =
  214 s / 12), drops 0.58%. Wash vs 17.75 baseline (within the ±2pp
  placement band), same verdict as the ragged-path FUSION test.
- Fused-flavor knob ledger: PGLE OOMs (MEP-037), command buffers neutral,
  LHS/overlap/symmetric already in the recipe, splits N/A. Remaining
  levers are all engineering: M7 fused consume-behind-flags, grouped-GEMM
  efficiency, XLA fusion mass, FSDP one-shot collective cost.
- M7a starts: warp-specialized put+consume prototype (plan in
  .agents/projects/marin-ep-m7.md).

### 2026-08-16 12:55 - MEP-039: M7a gate — flag-gated consumption works; naive consumer hides only 7%
- Spike: experiments/marin_ep/bench/spike_fused_consume.py (7730ae74e0 +
  docstring commit) on 1 GB200 tray, 4 devices, pool 14594x2560 f32.
  Warp-specialized kernel: transport wg (put_segments loop, tile-strided)
  + consumer wg gated on per-expert arrival semaphores.
- Mechanism findings (all validated, exact checksums, no deadlocks):
  1. Mosaic semaphore_signal requires a CONSTANT increment
     (`_ir_constant` in the lowering) — row-count arrivals are
     inexpressible; use unit signals with a host-computed expected count
     per expert (one per participating SM per entry + one per tail).
     semaphore_wait DOES take dynamic values.
  2. Traced values cannot cross two pl.loop nesting levels into a
     pl.when body ("Unsupported constant: OpResult") — hoist signals to
     the entry scope.
  3. Multithreaded kernels need kernel-level scratch (scratch_types), not
     per-copy run_scoped SMEM.
  4. SEND ORDER IS THE OVERLAP LEVER: dest-major order gives every expert
     ~zero head start (hidden fraction -4%); expert-major order (all
     dests' expert-j before expert-j+1) creates real head starts, the M4
     simulator's interleave result reproduced on hardware.
- Perf: fused 0.953 ms vs serial put+consume 0.981 ms vs put-only
  0.578 ms — the GEMM-weight stand-in consumer (REPEAT=8 rotated reads,
  ~0.40 ms) hides only ~7% because it is MEMORY-bound and contends with
  transport TMA for HBM/NVLink bandwidth. A real tensor-core GEMM
  consumer is compute-bound and does not fight the copy path — that is
  the M7b experiment (tcgen05 tile loop as the consumer wg).
- Next: M7b per .agents/projects/marin-ep-m7.md — bf16 grouped-GEMM tile
  loop in the consumer warpgroup, target >= 0.7x cuDNN-cute on the hero
  shard shape before any further integration.

### 2026-08-16 13:10 - MEP-040: upstream Pallas Blackwell ragged dot beats the production expert GEMM 2.17x at hero shard shape
- Discovery while scouting the M7b consumer: jax nightly ships
  jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py — a
  warp-specialized tcgen05 grouped GEMM with group-aware tile scheduling
  (GroupInfo) designed for reuse (do_matmul takes external grid indices).
- Isolated GEMM (m=2560, k=6144, n=3072, G=3, bf16, 1 GB200 GPU): best
  config (tile 128x128x64, collective 2-CTA, gtw 12, mcs 6) = 66.3 us =
  1457 TF/s (65% of bf16 peak), 2.49x XLA ragged_dot. collective=False
  configs fail ("unbound axis name: x" — kernel assumes a cluster axis);
  128x256 tiles exceed TMEM/SMEM.
- Full fwd MLP A/B at the hero shard shape (M=2560,H=6144,I=1536,G=3):
  | arm | us | TF/s |
  |---|---|---|
  | production _cudnn_cute_expert_mlp (QuACK SwiGLU + cuDNN) | 252.1 | 575 |
  | pallas brd (2x ragged_dot_kernel + unfused f32 SwiGLU) | 116.3 | 1247 |
  | XLA lax.ragged_dot MLP | 162.9 | 890 |
  The pallas path wins 2.17x over production DESPITE an unfused
  activation; XLA also beats the cute path at this shape. Microbench
  caveat applies (bf16-mosaic-vs-triton lesson: e2e can differ), but the
  margin is large and it attacks the profile's biggest bucket
  (GEMMs 4.8 s busy/step).
- Plan: expert_mlp variant with custom_vjp — brd for fwd + dgrad
  (transposed-weight feeds), keep the cuDNN grouped wgrad (different
  primitive); tray conformance; hero flavor A/B. M7b fusion then gates
  THIS kernel's group loop on arrival semaphores (its GroupInfo scheduler
  is the natural wait point).

### 2026-08-16 12:45 - MEP-041: cuDNN grouped Wgrad correctness bug found + fixed; brd expert MLP conformant; hero A/B launched
- While validating the brd expert MLP's backward, isolated a correctness
  bug in `cudnn_grouped_wgrad`: the kernel silently miscomputes when a
  group's row offset is not 64-aligned. Stock 8-row padding -> max err
  0.17-0.55 rel-to-peak (numpy reference, GB200, incl. production dims
  k=6144/1536 n=3072/6144 G=3); 64/128/256-aligned -> 0.003 (bf16
  floor); 32 fails. Grows with G (0.17 at G=2 -> 0.55 at G=12).
- Scope: NOT on main — introduced by PR #8081 and carried by its derived
  branches (all our cudnn-cute measurement runs trained with corrupted
  dw13/dw2; loss still fell, atol in old grad checks absorbed it). Filed
  #8339 (initially overclaimed "main", corrected), flagged PR #8081 with
  the fix pointer.
- Fix: _GROUP_ALIGNMENT 8 -> 64 (f2c9f702f3) + pad test update. Layer
  conformance after fix: gx 0.006, gw13 0.004, gw2 0.0045 rel-to-peak.
- brd expert MLP (be00be74fa): custom_vjp with pallas ragged_dot fwd +
  dgrad (transposed-weight feeds) + cudnn wgrad; rows padded to 256;
  EP4 single-process conformance PASSED at the bf16 floor. Note the
  earlier defensive rowmask in _bwd proved unnecessary
  (pad_grouped_rows excludes tail rows by construction) and is not in
  the committed version, which is what conformance ran.
- Launched: mep-brd-ep64-25-20260816 (ep-marin-mgpu-brd, MEP-034 recipe)
  vs the 17.75 s mgpu-cute basis. GEMM microbench predicts a large win;
  e2e verdict pending (microbench-overstates-e2e caveat).

### 2026-08-16 12:58 - MEP-042: EP64 hero on pallas ragged-dot GEMMs — 17.08 s/step, third consecutive best
- Run: mep-brd2-ep64-25-20260816 (ep-marin-mgpu-brd: fused mosaic
  transport + pallas blackwell ragged-dot fwd/dgrad + 64-aligned cudnn
  wgrad, N padded to the collective tile for hero 2I=6272/I=3136).
  25/25 SUCCEEDED, loss 6.09 falling.
- Scored: steady 17.08 s/step (12it@4:33 -> 24it@7:58 = 205 s / 12),
  drop_fraction 0.00492. ~245.5k tok/s/rack, ~22.5% MFU.
- EP64 ladder (wheel v3, same flags): brd 17.08 < mgpu-cute 17.75 <
  flat ragged cute 18.0 < old-wheel flat 18.1-18.2. Delta vs mgpu-cute
  (-0.67 s) matches the microbench prediction (fwd MLP 305 vs 482 us at
  true hero shape = 1.58x on the swapped legs).
- First fix of the true-shape trap: hero 2I=6272 and I=3136 are not
  256-multiples — the first launch failed on the collective tile; the
  padded `_grouped` (597d5bcb4e) resolves it at ~2-6% extra N FLOPs on
  the padded legs.
- Goal gap: 12.0 s needs another 5.1 s. Next: brd TuningConfig sweep at
  true hero shapes (971 TF/s leaves headroom vs the 1457 seen at the
  aligned shape), then a profile of the brd step, then M7b fusion.

### 2026-08-16 13:12 - MEP-043: per-leg configs marginal — EP64 17.0 s/step; GEMM-config knob exhausted
- Run: mep-brd3-ep64-25-20260816 (per-leg TuningConfigs, 6b60a0edb7):
  steady 17.0 s/step (12it@4:32 -> 24it@7:56 = 204 s / 12), drops
  0.582%. ~+0.08 s over the single-config brd2 (within noise).
- Sweep at true hero leg shapes found per-leg bests 1195-1531 TF/s
  (fwd13 tn128/gtw8/gmd0, fwd2 tn128/gtw12/gmd0, dact tn64/gtw16/gmd1,
  dx tn128/gtw8/gmd1, all mcs6 collective) — but the e2e step barely
  moves; the GEMM legs are no longer the exposed mass.
- Day ladder (all EP64, 25 steps): 18.1 (start) -> 17.75 (fused
  transport) -> 17.08 (pallas GEMMs) -> 17.0 (tuned configs).
  ~247k tok/s, ~22.6% MFU. Goal 12.0 s.
- In flight: mep-brd-prof-25-20260816 (xprof steps 12-15) — next-lever
  attribution: expect fusion mass (was 3.8 s), FSDP one-shots + barriers
  (~4 s), wgrad legs (cudnn, was 0.97 s busy for both) to dominate now.
- M7a tray released; M7b (arrival-gated brd group scheduler) to be
  planned against the new profile.

### 2026-08-16 13:25 - MEP-044: brd-step profile — MoE side largely optimized; goal gap now dominated by non-MoE mass
- Trace: mep-brd-prof-25-20260816 steps 12-15,
  s3://marin-us-east-02a/tmp/ttl=30d/xprof/mep-brd-prof-25-20260816
  (30d TTL). Busy-time per step (rank-0 device; streams overlap):
  | bucket | ms/step | note |
  |---|---|---|
  | attention + dense GEMMs (fa4/nvjet) | 4948 | fa4 bwd single kernel 499 |
  | pallas bucket (brd ragged dots + puts) | 3574 | brd legs ~2.4 s, puts ~1.15 s |
  | XLA fusions | 3545 | |
  | FSDP one-shot collectives (ncclSymk) | 2847 | RS_LDMC alone 1034 |
  | MultiGpuBarrier | 1055 | down from 1531 on the cute arm |
  | cudnn wgrad | 589 | |
- Interpretation: after today's ladder (18.1 -> 17.0 s/step) the
  MoE-specific exposed window is small; even a perfect M7b (zero exposed
  transport + GEMM launch structure) cannot close the remaining 5.0 s to
  the 12.0 s / 350k gate. The binding buckets are attention/dense GEMM
  efficiency, XLA fusion mass, and FSDP one-shot collective cost — all
  outside the EP transport scope this campaign has been optimizing.
  Candidate next projects, by expected size: fp8 expert+dispatch wire
  (#7665 machinery composes with brd), fa4 attention tuning, fusion-mass
  reduction, FSDP collective scheduling.
- M7b remains worthwhile as MoE polish (~up to 1 s of launch/sync
  structure + enabling fp8-fused variants) but is no longer the
  goal-critical path by itself.

### 2026-08-16 17:45 - MEP-045: M7b prototype WORKS — arrival-gated fused dispatch+GEMM correct, GEMM fully hidden
- Spike: experiments/marin_ep/bench/spike_fused_gemm.py (20e8b03152) on
  1 GB200 tray, 4 devices, pool 16640x2560 bf16 -> 1536. One persistent
  kernel per device: wg0/wg1 = upstream blackwell_ragged_dot do_matmul
  (imported, unmodified) iterating the tile grid with a per-tile
  `semaphore_wait(arrivals[group], expected[group])` gate; wg2 = the
  put_segments transport loop (expert-major, unit signals) writing peers'
  pools. Bit-correct vs the plan-reference x numpy GEMM.
- Timing: fused 0.626 ms vs put_segments+ragged_dot two-launch 0.620 ms
  (0.99x). Attribution: baseline = put 0.35 + gemm 0.27 serial; fused =
  transport-bound 0.626 with the GEMM fully hidden inside it — the
  overlap works, but the in-kernel transport (1 wg x 72 clusters,
  12-row stages) is ~1.8x slower than the dedicated 148-SM Lane put,
  cancelling the win at this toy GEMM:transport ratio. At hero ratios
  (gemm 305 us >> transport ~150 us/layer-direction) fused =
  max(transport, gemm) beats serial if the in-kernel transport stays
  under the GEMM time — favorable.
- Framework findings this arc (logbook-only until upstreamable):
  1. plgpu GMEM ref `.reshape` is not TMA-lowerable ("non-indexing
     transforms") — keep refs 2D and copy in LANE-wide column chunks
     (SMEM stage per-lane blocks for contiguity).
  2. Warpgroup-semantics TMA descriptors are built on the HOST per
     (ref, peer): remote peer ids must be host-recomputable —
     `device_id()+const` works, GMEM loads and loop induction vars do
     not; unroll the dest loop statically.
  3. UPSTREAM FOOTGUN: blackwell_ragged_dot/matmul hardcode cluster axis
     name "x"; running them under a shard_map mesh whose axis is also
     named "x" DEADLOCKS silently (axis-name collision). Our hero mesh
     dodges it by using replica_dcn/data/expert/model.
  4. In 2-CTA clusters, gate single-copy work on cluster_idx==0 or both
     CTAs duplicate sends/signals; host-side expected counts must use
     core_count//2 (the "sm" grid axis counts clusters).
- Next tuning levers for the fused transport: double-buffered stages,
  both CTAs sending disjoint lane halves, bigger stages (SMEM budget
  trade vs max_concurrent_steps), Lane-semantics transport warp inside
  the WG kernel if mixing is possible. Then a bwd (combine) variant and
  hero-shape measurement.

### 2026-08-16 17:55 - MEP-046: both-CTA lane-split transport is a wash — fused spike stays at the single-CTA variant
- Variant: both cluster CTAs sending disjoint lane halves (2x participants,
  5-lane batches, 24-row stages): 0.653 ms vs 0.626 ms single-CTA — the
  in-kernel transport is issue/latency-bound, not SM-count-bound. Reverted
  to 20e8b03152.
- M7b tuning state: fused = 0.626 ms vs 0.620 two-launch at the toy EP4
  ratio (GEMM fully hidden; transport-bound). Remaining levers if resumed:
  double-buffered stages (cross-tile pipelining), and hero-ratio
  measurement where GEMM >> transport makes the same structure a
  projected net win. Integration into marin_ep (fwd fused + combine
  separate, custom_vjp) is the next full milestone.
- Tray released; M7b prototype milestone met (correct + overlap proven).

### 2026-08-16 19:45 - MEP-047: gate reframed to 25% MFU; parallel-session import; b2048 arm
- User reframe: 350k tok/s/rack likely not theoretically reachable; the gate
  is now >=25% MFU at hero EP64 (~15.4 s/step at b1024 calibration), tok/s
  best-effort. Gap from 17.0 s best: ~1.6 s.
- Imported findings from parallel sessions:
  - ragged-a2a (#8317): closed at 21.82% MFU; flag/compute levers exhausted,
    defers to marin-ep for structural transport. Reference: fixed transport
    iso-compute 23.6% MFU at ~3.9% drops -- our 22.6% at <1% drops is at
    parity with the fixed baseline.
  - mixture-of-kittens (#8108, 2026-08-16 comment): megakernel 25.28% MFU
    dropless but at 16 layers / 8 experts / expert-axis 4 (in-process
    peers); EP64+E192 blocked on their fabric deadlock. Two measured
    transferable levers: (1) 128 seq/node (global batch 2048, the sealed
    two-rack posture) gave +1.1pp because the DP reduction is per-step,
    not per-token -- attacks exactly our 2.8 s FSDP + 1.05 s barrier
    buckets; (2) cf 1.1 costs nothing in drops (independently confirms
    our setting).
- Launched mep-b2048-brd-25-20260816: MEP-043 recipe (ep-marin-mgpu-brd,
  wheel v3, cf 1.1, splits 32) with --batch-size 2048. Risk: 2x activation
  memory; hero keeps MuonH state host-offloaded so HBM verdict empirical.
- fp8 wire (#7665, 1.144x fwd+bwd at EP64) flagged as candidate but may
  violate the fidelity constraint cited in #8317 (no quantization beyond
  main's moe_hero_ep, per the Slack fidelity discussion) -- needs a user
  ruling before any arm.
- Next action: score b2048; if positive adopt as the new posture; then M7b
  integration (fwd fused dispatch+GEMM) as the remaining structural lever.

### 2026-08-16 20:35 - MEP-048: batch lever memory-blocked; mgpu_fused lands both conformance gates
- b2048 arm (mep-b2048-brd-25-20260816): FAILED, single 188.66 GiB
  jit_train_step allocation (over HBM). b1280 arm: FAILED, ncclAlltoAll
  CUDA OOM inside the step. The MoK-derived batch lever is
  activation-memory-blocked at 48 layers without remat surgery -- matches
  #8108's "48 layers at the hero routed width does not fit". Negative
  result; both jobs killed.
- M7b integration (fee6ffd54f + 2a1824da69): marin_ep_mgpu_fused --
  fused_dispatch_brd single-kernel dispatch + arrival-gated gate/up GEMM,
  pool dual-write, custom_vjp backward = brd _bwd + transpose put.
  Lane-chunked transport staging (chunk 6 x 8 rows) fits the hero 24-lane
  SMEM budget at mcs4.
- Gate 1 (tray, in-process EP4, pytest): bitwise equal to mgpu_brd on
  values + dx/dw13/dw2 at tile-straddling dims (hidden 512, 2I 640).
- Gate 2 (tray, 4 processes x 1 GPU): MGPU_FUSED CONFORMANT bitwise vs
  the mgpu+brd reference; needs the v3 wheel + dynamic-slice-fusion off +
  cuda_async (same 3-part fix as mgpu). First multi-process validation of
  Warpgroup-semantics remote TMA.
- A/B trap: the smoke's default expert_mlp is XLA ragged_dot; comparing
  fused (Pallas GEMMs) against it shows 63% ULP-level "mismatches" (max
  abs 0.5 at |y|~20). Reference leg must pin expert_mlp=brd.
- Launched mep-fused-ep64-25-20260817 (hero EP64, MEP-043 recipe, flavor
  ep-marin-mgpu-fused). Tray released.
- Next action: score fused vs 17.0 s brd baseline; then logbook/issue
  milestone; remaining lever ladder: fused combine leg, fp8 wire (user
  ruling pending), XLA fusion mass.

### 2026-08-16 22:35 - MEP-049: L1 three-mode split — combine fusion carries the whole fused win
- While the hero fused arm queues (rack contended), replaced the event
  sim's `pipelined` bool with `PipelineMode` {bulk, dispatch_fused, full};
  dispatch_fused matches the landed mgpu_fused flavor (arrival-gated GEMM,
  combine behind a GEMM barrier).
- Hero EP64, cf 1.1, balanced counts, transport 1.8x-penalized (MEP-045
  in-kernel measurement) for the gated modes, 48-layer fwd+bwd MoE mass:
  bulk 3.074 s | dispatch_fused 3.118 s | full 2.583 s.
- PREREGISTERED PREDICTION for mep-fused-ep64-25-20260817: the
  dispatch-only fusion is ~neutral vs the 17.0 s brd baseline (-0.04 s,
  within placement noise). The entire modeled win (+0.49 s/step, ~+0.7pp
  MFU) requires the combine leg: per-tile combine puts streaming from the
  GEMM epilogue. If the arm confirms neutral, build combine fusion next.
- Caveat: sim "bulk" has no XLA cross-layer overlap, so absolute deltas
  are upper bounds; the mode RANKING is the claim (exploratory).
- Reusability note (user question): perfmodel/roofline + the
  WorkItem/run_schedule scheduler are marin-ep-agnostic (other branches
  supply their own program builder; bulk mode already approximates the
  stock XLA shape). simcore/oracle stay SPEC-coupled by design. Extraction
  to a neutral home is ~1h if #8317/#8108 want it.

### 2026-08-16 22:45 - MEP-050: mgpu_fused2 conformant after an empty-group deadlock fix
- Built during the rack-capacity hold: fused_gemm_combine (66dc77e2b5) runs
  the down GEMM and the combine put in one kernel -- the store warpgroup
  publishes per-128-block semaphores (write-visible wait; do_matmul's own
  wait is SMEM-reuse only) and the transport streams each segment once its
  covering blocks land. fused_full_moe fuses the whole local pipeline; the
  backward reuses the kernel to fuse dx GEMM + return put. Flavor
  ep-marin-mgpu-fused2. Sim (MEP-049) attributes the entire 0.49 s/step
  modeled win to this leg.
- DEADLOCK found by the in-process pytest (skewed dirichlet routing):
  GroupInfo.create gives an empty group one grid slot when its start is
  mid-block (floor-division final_block == start_block) and that slot
  still signals; expected_tile_signals excluded all empty groups, so the
  transport waited on unreachable totals and wedged the device. The CPU
  replay test had encoded the same wrong assumption -- fixed both, forced
  empty groups into the replay (7e699995ed). Smoke's flatter routing had
  no empty kept groups, which is why it passed first.
- Validation after fix, GB200 tray: pytest fused+fused2 vs mgpu_brd
  bitwise (2 passed); 4-process smoke MGPU / MGPU_FUSED / MGPU_FUSED2 all
  CONFORMANT bitwise. Tray released.
- Queued mep-fused2-ep64-25-20260817 behind mep-fused-ep64-25-20260817
  (both Kueue-gated on a free NVLink domain). Prediction stands: fused
  ~neutral vs 17.0 s, fused2 carries ~-0.5 s.
