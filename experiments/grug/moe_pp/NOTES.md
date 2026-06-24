# moe_pp — pipeline parallelism for the production grug-MoE

Goal: pipeline parallelism (toward zero-bubble) for the **production** grug-MoE,
composing with expert parallelism (EP) and FSDP, on both TPU and GPU. PP is the
final parallelism axis added on top of the existing EP/FSDP grug stack.

## What's here (the TPU baseline)

`pipeline.py` runs the production `Transformer` inside a single `shard_map` that
manualizes all mesh axes, with a per-stage manual `jax.vjp` GPipe backward,
inline ring-EP, the megablox GMM, and a manual FSDP weight all-gather. It is
gradient-exact vs the non-pipelined oracle and trains on real hardware
(v6e-8 ~0.78× FSDP single-host; v6e-32 ~1.22× FSDP — PP wins once the param
all-gather crosses DCN). `benchmark.py` drives it; `oracle.py` is the
non-pipelined reference.

## What we tried and dropped

- **`stage`-axis `shard_map` + GSPMD for data/expert** (the `moe_zb` toy). The
  schedule and the pairwise PP×FSDP / PP×EP compositions worked, but the full
  PP×FSDP×EP does not lower on TPU: XLA's SPMD partitioner can't factor two
  GSPMD axes' device groups under a third manual axis. Fix was to manualize all
  axes (above), so no GSPMD axis touches any operand.
- **Autodiff through the forward pipeline** as the backward. It materializes a
  weight-grad buffer stacked across all stages (~48 GiB at 40B) that OOMs the
  XLA partitioner. Replaced by the per-stage manual `jax.vjp` backward, whose
  cotangents never carry a `[num_stages, …]` axis.
- **TPU Pallas remote-DMA ring-shift** (a drop-in for `ppermute`). De-risked
  and bit-exact vs `ppermute`, but unnecessary: XLA sharding on TPU is fine, so
  TPU needs no custom kernel. Dropped.

## Direction

- **PP is threaded manually** (explicit per-round communication / separate
  launches), not via a `stage` mesh axis.
- **EP/FSDP** stay XLA / jax map operations on **TPU**, but must be expressed as
  **Pallas kernels on GPU** — XLA cannot lower the combined PP×EP×FSDP sharding
  on GPU and OOMs. The grug refactor makes EP/FSDP expressible either way:
  Pallas on GPU, traditional jax map operations on TPU.
