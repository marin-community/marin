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
