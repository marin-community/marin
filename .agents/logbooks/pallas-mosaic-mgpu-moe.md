# Pallas MoE Dispatch-Up Mosaic GPU: Research Logbook

## Scope

- Goal: Build a DeepEP-free single-node EP8 Grug MoE backend using Pallas
  Mosaic GPU remote refs. The first subkernel target is dispatch + W13/SiLU
  (`moe_dispatch_up`), before W2/combine and backward integration.
- Primary metrics: fused dispatch + W13/SiLU time, MoE forward time,
  MoE forward+backward time, end-to-end fwd+bwd+SGD MFU.
- Constraints: single-node 8xH100 first, `model_axis=1`, no DeepEP/NCCL
  EP/FFI transport/NIC/RDMA/NVSHMEM in this path.
- Experiment issue: https://github.com/marin-community/marin/issues/6597
- Spec: `.agents/projects/pallas_mosaic_mgpu_moe_spec.md`

## Baseline

- Date: 2026-06-23
- Code refs:
  - `lib/levanter/src/levanter/grug/_moe/ep_ring.py`
  - `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`
  - `lib/levanter/src/levanter/grug/_moe/ep_padded_all_to_all.py`
  - `lib/levanter/src/levanter/grug/_moe/sonic.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
- Baseline numbers:
  - May387 DeepEP EP16/N2 B64 `save_moe`: about 17 MFU, but DeepEP remains
    fragile.
  - May363/May378 EP8/B16 readable profiles: MoE around 37-40% of step time.
  - Ring/ragged A2A comparison: `ragged_all_to_all` was slower than ring in the
    tested shape.

## Experiment Log

### 2026-06-23 - Kickoff spec handoff

- Hypothesis: A Pallas Mosaic GPU dispatch-up subkernel can route token blocks directly into
  destination expert-major tiles and consume them with W13/SiLU, avoiding
  DeepEP's opaque FFI/runtime fragility and avoiding high-level collective
  permutation overhead.
- Command: No benchmark command yet; this entry creates the research branch,
  spec, and issue.
- Config: Target is single-node EP8 8xH100, top-k 4, bf16 activations/weights,
  May D2560 Grug MoE shape.
- Result: Created issue #6597 and checked in the initial implementation spec.
- Interpretation: The spec deliberately distinguishes a remote-dispatch
  validation slice from the product target. The first product target is fused
  dispatch + W13/SiLU; training integration also requires backward/custom VJP.
- Next action: Assign an implementation agent to start with the routing/layout
  JAX reference and Mosaic remote-dispatch validation slice.

### 2026-06-23 - Milestones 1 and 2 validation slice

- Hypothesis: A shard-local Mosaic GPU remote-ref dispatch can reproduce the
  prepacked expert-major layout exactly, and a local W13/SiLU Pallas kernel can
  consume that layout for top-k 1 and top-k 4 routing.
- Commands:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/kernels/test_pallas_moe_dispatch_up.py -q`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name 6597-mosaic-mgpu-smoke-20 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 1 --top-k 1 --hidden 64 --intermediate 64 --dtype bf16 --run-pallas`
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --cpu 16 --memory 128GB --disk 50GB --extra gpu --timeout 1800 --job-name 6597-mosaic-mgpu-smoke-21 -- ... bench_moe_dispatch_up_mosaic_gpu.py --ep-size 8 --tokens-per-rank 8 --experts-per-rank 4 --top-k 4 --hidden 64 --intermediate 64 --dtype bf16 --run-pallas`
- Config: Single-node CoreWeave `cw-us-east-02a`, H100x8, explicit `expert`
  mesh, bf16 activations/weights, `H=64`, `I=64`, `T/rank=8`.
- Result:
  - Local tests: 8 passed.
  - Top-k 1: dispatch payload max error 0, dispatch metadata errors
    `valid=0 local_expert=0 src_rank=0`, W13/SiLU max bf16 error 2.
  - Top-k 4: dispatch payload max error 0, dispatch metadata errors
    `valid=0 local_expert=0 src_rank=0`, W13/SiLU max bf16 error 2.
- Interpretation: Milestone 1 is validated for the prepacked remote-ref layout.
  Milestone 2 is validated for local dispatch-up W13/SiLU on the dispatched layout at
  the requested top-k 1 and top-k 4 slices. The dispatch kernel is still a
  deliberately serial validation primitive, not a performance implementation.
- Next action: Replace serial scalar remote writes with tiled/staged copies and
  add backward/custom-VJP coverage before Grug integration.
