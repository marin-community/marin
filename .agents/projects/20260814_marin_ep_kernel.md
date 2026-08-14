# Marin EP: a clean-room fused expert-parallel MoE kernel

Status: plan (2026-08-14). Branch: `marin-ep`. Dev home: `experiments/marin_ep/`.

## Goal

Design and build an efficient expert-parallel (EP) MoE kernel for the grug MoE
hero architecture, from first principles. No code vendored from existing
projects (small, generally useful utilities excepted); heavy conceptual
borrowing from prior art is expected and encouraged:

- [DeepEP](https://github.com/deepseek-ai/DeepEP) — count-then-write dispatch,
  NVLink/RDMA direct stores, SM-budgeted comm kernels
- [MoE ECHO (Megatron-LM #2368)](https://github.com/NVIDIA/Megatron-LM/pull/2368)
  — comm/GEMM overlap scheduling inside one layer
- [MoonEP](https://github.com/moonshotAI/moonep) — EP transport specialization
- [Mixture-of-Kittens](https://cursor.com/blog/mixture-of-kittens) — full-layer
  megakernel: warp-specialized persistent kernel fusing dispatch, grouped GEMM,
  and combine with tile-granular pipelining

Several of these are megakernels — a single fused kernel spanning the whole
MoE layer, including dispatch/combine comms and expert GEMMs. That is the
design direction here too, reached incrementally (fused transport first,
GEMM fusion second).

## Requirements

- **R1 — scale**: EP up to one full NVL72 rack (64–72 GPUs). Hero mesh is
  EP64 (16 nodes x 4 GB200).
- **R2 — drops**: dropless, or controllable drop rate < 2% in the
  trained-router regime at `moe_hero_ep` shape. (Baseline: cf 1.30 measures
  ~3.96% drops; cf 1.33 is the configured hero value; cf 1.34 OOMs.)
- **R3 — stack**: native JAX/XLA only (Mosaic-GPU, CuTe DSL via
  `cutlass_call`, CUDA FFI are all fine). No PyTorch dependencies.
- **R4 — perf**: SoTA MFU/throughput on GB200 NVL72, comparable to
  Mixture-of-Kittens-class implementations. Concrete bar in "Gates" below.

## Target workload (fixed reference shape)

From `experiments/grug/moe_hero_ep/heuristic.py` (`HERO_MODEL`) and
`launch.py`. All per-layer, per-device numbers assume the hero step:
global batch 1024 seqs x 4096 tokens on one rack (64 GPUs).

| quantity | value |
|---|---|
| d_model / routed (latent) width H | 6144 / **3072** (LatentMoE) |
| expert intermediate I | 6272 (SwiGLU: w13 `[3072, 2*6272]`, w2 `[6272, 3072]`) |
| experts E / top-k | 192 / 4 (+2 shared experts outside the routed path) |
| layers | 48, all MoE |
| EP degree | 64 → **3 local experts/device** |
| tokens/device/step | 65,536 → **262,144 assignments/device** |
| capacity (cf 1.33) | ceil(1.33 * 262144 / 192) = **1816 rows per (expert, src-shard) cell** |
| dispatch payload/device/layer | 262,144 x 3072 x 2 B ≈ **1.5 GiB** each way (bf16, pre-drop) |
| routed GEMM FLOPs/device/layer (fwd) | 6 * 262,144 * 3072 * 6272 ≈ **3.0e13** |
| params / active | 546.3 B / 24.7 B per token |

First-pass roofline at GB200 (bf16 dense 2.5 PF/s, NVLink5 ~900 GB/s per
direction per GPU): fwd routed GEMM ≈ 12 ms/layer ideal (~17 ms at 70%
GEMM efficiency); dispatch ≈ 1.8 ms/layer each way. Comms is ~10–20% of
GEMM time — the win comes from (a) not serializing 3 sequential
`all_to_all` round-trips per direction (current `fixed_all_to_all` path),
(b) not materializing capacity-padded send/recv cells (~2 GiB send buffer
plus a 6 GiB fp32 `grad_rows` in backward), and (c) overlapping transport
with expert GEMMs at tile granularity. The simulator refines these numbers.

**Baseline to beat**: hero EP64 run at cf 1.30 → 262,683 tok/s (last-50
mean), ~24–28% MFU depending on segment (see
`.agents/logbooks/7279-moe-hero-ep.md`). We will additionally measure a
per-layer routed-MoE segment baseline (dispatch + expert GEMM + combine,
fwd+bwd) from an existing profile before writing kernel code, since that —
not end-to-end MFU — is what the kernel directly moves.

## Design sketch (to be refined into SPEC.md)

The NVL72 rack is a single NVLink domain (MNNVL): every GPU can load/store
every other GPU's mapped memory. This permits a DeepEP/MoK-style design with
no NCCL collectives in the MoE layer:

1. **Routing (per device)**: router logits → top-k → per-expert counts.
   Histogram + exclusive prefix-sum over (expert, src-shard) establishes
   write offsets. Counts exchanged via direct remote writes (a few KB), not
   `all_gather`.
2. **Dispatch**: each device writes its tokens directly into the owning
   device's per-expert receive buffer at the computed offsets (remote
   stores / TMA over NVLink, batched into tiles), then sets a per-tile
   arrival flag. Ragged, not capacity-padded: capacity becomes a receiver
   pool bound, not a cell shape.
3. **Expert GEMM**: persistent warp-specialized kernel; MMA warpgroups
   consume receive-buffer tiles as arrival flags are set (grouped GEMM over
   ragged rows, SwiGLU fused in the epilogue).
4. **Combine**: expert outputs stream back to source devices as tiles
   complete; source device accumulates k weighted contributions per token
   (fp32 accumulate).
5. **Backward**: same transport reversed (combine-grad dispatch → dgrad/wgrad
   grouped GEMMs → dispatch-grad combine). Wgrad consumes the saved permuted
   activations; avoid the fp32 `grad_rows` materialization of the current
   path.

Later compositions (explicitly out of scope for v1, but the design must not
preclude them): fp8 dispatch wire (#7665), fp8/MXFP8 expert GEMMs, overlap
of MoE with attention of the adjacent layer.

**Transport substrate** is the main open question and gets an early
hardware spike (M5): candidate mechanisms are (a) Mosaic-GPU distributed
(NVSHMEM-backed remote refs/semaphores), (b) CuTe DSL device-side nvshmem
via `cutlass_call`, (c) a thin custom CUDA FFI extension in the shape of
`lib/levanter/src/levanter/kernels/deepep/` (built from scratch, not
vendored). All three satisfy R3. The choice is driven by: symmetric-memory
allocation from JAX, XLA buffer aliasing/donation constraints, and
compatibility with `shard_map` + `custom_vjp`.

## Approach

### Oracle

A simple, obviously-correct reference implementation is maintained at every
phase and the WIP implementation is continuously tested against it on varied
(random and adversarially skewed) inputs:

- **Value/grad oracle**: dense per-token expert compute (the pattern of
  `lib/levanter/tests/grug/test_grugformer_moe.py::test_fixed_all_to_all_matches_dense_cross_shard_value_and_gradients`),
  plus the existing `fixed_all_to_all` backend as a second reference for
  drop semantics.
- **Fuzz harness**: randomized routing distributions including hot-expert
  skew, empty experts, all-tokens-one-expert, capacity overflow, and
  degenerate shapes; fwd value + gradients checked every run.

### Simulator

Cluster time is expensive and the compile/queue/spinup cycle is long, so the
bulk of development runs against a simulator. Two distinct artifacts:

- **Correctness simulator**: a pure-JAX/NumPy executable model of the
  kernel's *algorithm* — per-device programs exchanging messages
  (remote-write + flag primitives), runnable on CPU with
  `--xla_force_host_platform_device_count`. This is the executable spec: it
  must match the oracle exactly (up to documented dtype casts) and is the
  artifact the fuzz harness drives. Its API converges toward the real
  kernel's launch API so the final translation is mechanical.
- **Performance simulator**, layered coarse→fine, adding fidelity only when
  a design decision demands it:
  - **L0 — analytic roofline**: per-phase bytes/FLOPs with an overlap
    model; fast enough for design-space search. Seeds from
    `lib/fray/src/fray/device_flops.py` and NVLink5 link specs.
  - **L1 — discrete-event tile simulator**: SMs, per-link bandwidth/latency,
    tile-dependency graph (dispatch tile → GEMM tile → combine tile),
    warp-role occupancy. Estimates per-layer time and step MFU.
  - **L2+ — fidelity upgrades on demand**: NVLink switch contention,
    HBM bandwidth contention with GEMM epilogues, launch/flag-poll
    latencies — each added only when validation against hardware shows it
    matters.

Simulator fidelity is verified up a ladder: (1) specs/docs (NVLink5, CUDA,
NCCL, PTX multimem), (2) open-source implementations (XLA, DeepEP public
code — read for behavior, not copied), (3) probing the real cluster (up to
2 NVL72 racks on `cw-us-east-08a` via iris). Calibration anchors: measured
segment times from existing hero profiles and microbenches we run once and
cache.

A behavior spec (`SPEC.md`) is maintained separately from implementation
code from day one: message formats, buffer layouts, flag protocols, drop
semantics, dtype/rounding rules, and the invariants the fuzz harness
enforces (e.g. the FP8-scaling causal invariant: no scale shared across
tokens along the sequence axis, if/when fp8 wire composes in).

Once a working simulated implementation exists, an autoresearch loop
optimizes simulated MFU/throughput over the design space (tile sizes,
dispatch batching, flag granularity, warp-role allocation, SM split between
transport and MMA, schedule shape).

**Simulator-phase gates**
- G1a: correctness — oracle parity (value + grad + drop semantics) on the
  fuzz corpus, all EP degrees in {1, 4, 8, 64}.
- G1b: performance — simulated per-layer routed-MoE time consistent with
  plausible SoTA on real hardware; concretely, simulated segment time
  ≤ 0.7x the measured baseline segment at hero shape, with L0/L1 estimates
  agreeing within 20%.

### Real-world testing and perf tuning

First goal on hardware: end-to-end run and green tests on GB200 NVL72
(start at 1 node / EP4, scale to rack). Then an autoresearch loop that
simultaneously:

1. optimizes real-world MFU/throughput, and
2. improves simulator fidelity (every hardware measurement becomes a
   calibration point; disagreements > 20% get root-caused).

Once the simulator is trusted as a rough proxy for a class of changes,
iterate in the simulator and confirm winners on hardware, shortening cycle
time dramatically.

**Hardware-phase gates**
- G2: kernel runs e2e and passes the oracle test suite on GB200 (EP4, then
  EP64).
- G3: hero-shape A/B — drop-adjusted tokens/s ≥ +15% over the
  `fixed_all_to_all` baseline at equal loss curves, drop rate < 2% at
  cf ≤ 1.33, no HBM regression that forces a smaller batch.
- G4 (landing): integrated as a `moe_implementation` backend, variant
  contract test lowers, `commit` + PR flow, experiment issue closed with
  results.

## Code layout

Development home (self-contained, `experiments/ncclep/`-style):

```
experiments/marin_ep/
  README.md          # orientation, how to run tests/benches
  SPEC.md            # behavior spec, maintained apart from implementation
  oracle.py          # dense reference + fixed_a2a cross-check helpers
  simcore.py         # correctness simulator: device programs + message rail
  perfmodel/         # L0 roofline, L1 discrete-event simulator
  kernels/           # real Mosaic/CuTe/FFI kernels as they materialize
  tests/             # collected by root pytest (testpaths includes experiments)
  bench/             # microbenches + hardware calibration probes
```

Graduation path when the kernel works: reusable kernel + transport code
moves to `lib/levanter/src/levanter/kernels/marin_ep/` (DeepEP-wrapper
shape: `availability.py`, `preflight.py`, build cache) and registers as a
backend in `lib/levanter/src/levanter/grug/_moe/` (`MoeImplementation`
literal in `common.py`, dispatch in `grug_moe.py`) — a ~5-line integration.
Respect dependency direction: {iris, haliax} → {levanter, zephyr} → marin;
nothing CUDA-specific goes in haliax.

Remember the `lib/` global-gitignore trap: `git add -f` for new files under
`lib/*`.

## Milestones

- **M0** — this plan committed on `marin-ep`; `experiments/marin_ep/`
  scaffold with SPEC.md skeleton. Experiment issue filed per the
  run-research convention when the first PR opens; logbook at
  `.agents/logbooks/marin-ep.md`.
- **M1** — oracle + fuzz harness, CPU-runnable, green.
- **M2** — correctness simulator (message-passing device programs) passes
  G1a at EP {1,4,8}; EP64 in CI-sized shapes.
- **M3** — perf simulator L0+L1; calibrated against existing measured hero
  segment times and one cheap hardware microbench session; G1b evaluated.
- **M4** — autoresearch loop in the simulator over the design space;
  design frozen for v1 kernel.
- **M5** — transport substrate spike on real GPUs (1–2 nodes): symmetric
  memory + remote store/flag microbench from JAX for candidates (a)–(c);
  pick one.
- **M6** — fused dispatch/combine transport kernel (XLA grouped GEMM still
  in the middle) passing oracle tests on GB200 EP4; measure.
- **M7** — GEMM fusion into the persistent kernel; EP64 rack scale; G2 →
  hardware autoresearch loop → G3.
- **M8** — integration, A/B at hero shape, land (G4).

M6 is deliberately staged before M7: fused transport with XLA GEMMs is a
shippable intermediate that already removes the serialized a2a round-trips
and capacity-padded buffers, and de-risks the megakernel step.

## Risks / open questions

- **Symmetric memory from JAX**: XLA owns the allocator; NVSHMEM-style
  registration of peer-mapped buffers under `cuda_async` is the biggest
  unknown → early spike (M5). Fallback: FFI-owned side allocations outside
  the XLA pool (DeepEP-wrapper precedent), sized once at init.
- **custom_vjp x shard_map x scan pinning** (see B200 row13 logbook):
  remat interactions must be designed in, not bolted on.
- **Command buffers are disabled** on the hero (#5675) and PGLE is on;
  a persistent kernel changes both calculus — revisit runtime flags in M7.
- **Simulator drift**: guard with the calibration-point discipline (every
  hardware run logs measured-vs-predicted; >20% disagreement is a bug to
  root-cause, not a constant to fudge).
- **Trained-router drop target (R2)**: <2% at cf ≤ 1.33 likely requires the
  ragged receiver-pool design (drop only on pool exhaustion, not per-cell
  overflow); validate drop statistics against recorded routing
  distributions from hero checkpoints, not just synthetic skew.
- **NCCL coexistence**: attention/FSDP collectives still run NCCL around
  the MoE kernel; SM budget and NVLink bandwidth are shared. L1 simulator
  models the layer in isolation first; step-level interaction measured in
  M7.
