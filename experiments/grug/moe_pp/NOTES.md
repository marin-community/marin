# moe_pp — pipeline parallelism for the production grug-MoE

Goal: pipeline parallelism (toward zero-bubble) for the **production** grug-MoE,
composing with expert parallelism (EP) and FSDP, on both TPU and GPU. PP is the
final parallelism axis added on top of the existing EP/FSDP grug stack.

## What's here (the TPU baseline)

`pipeline.py` runs the production `Transformer` inside a single `shard_map` that
manualizes every mesh axis (`{stage, expert, data}` are load-bearing),
differentiated with whole-program `jax.value_and_grad`. A GPipe microbatch
schedule ripples activations stage→stage+1 via `ppermute`; the MoE runs the real
sparse ring-EP inline (`all_gather` dispatch + megablox `ragged_dot` GMM +
`psum_scatter` collect over `expert`); FSDP shards weights over `data`, and a
custom-vjp `all_gather` keeps the weight grad `/data`-sharded (its backward is
pinned to `psum_scatter`, so the full weight cotangent is never built). It is
gradient-exact vs the non-pipelined oracle (`oracle.py`) and trains on real
hardware (v6e-8 ~0.78× FSDP single-host; v6e-32 ~1.22× FSDP — PP wins once the
param all-gather crosses DCN). `benchmark.py` drives it.

Manualizing *every* axis (not just `stage`) is required on TPU: the megablox
Mosaic GMM can't be auto-partitioned, and the SPMD partitioner touches every
GSPMD axis on every op, so any residual GSPMD axis trips it.

## What we tried and dropped

- **`stage`-axis `shard_map` + GSPMD for data/expert** (the `moe_zb` toy). The
  schedule and the pairwise PP×FSDP / PP×EP compositions worked, but the full
  PP×FSDP×EP does not lower on TPU: XLA's SPMD partitioner can't factor two
  GSPMD axes' device groups under a third manual axis. Fixed by manualizing all
  axes (above).
- **The same all-manual pipeline on GPU.** Whole-program autodiff there
  materializes a weight-grad buffer stacked across all stages (~48 GiB at 40B)
  that OOMs the GPU partitioner — a GPU-partitioner failure a CPU HLO scan can't
  predict, and which does not happen on TPU. A per-stage manual `jax.vjp`
  backward plus the custom-vjp FSDP all-gather dodged that specific buffer, but
  XLA still can't lower the full PP×EP×FSDP sharding on GPU — so the whole
  XLA-on-GPU line is dropped in favor of Pallas kernels (below).
- **TPU Pallas remote-DMA ring-shift** (a drop-in for `ppermute`). De-risked and
  bit-exact vs `ppermute`, but unnecessary: XLA sharding on TPU is fine, so TPU
  needs no custom kernel. Dropped.

## Direction

- **PP is threaded manually** (explicit per-round communication / separate
  launches), not via a `stage` mesh axis.
- **EP/FSDP** stay XLA / jax map operations on **TPU**, but must be expressed as
  **Pallas kernels on GPU** — XLA cannot lower the combined PP×EP×FSDP sharding
  on GPU and OOMs. The grug refactor makes EP/FSDP expressible either way:
  Pallas on GPU, traditional jax map operations on TPU.
