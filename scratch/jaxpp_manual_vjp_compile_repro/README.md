# JaxPP manual-VJP compile-growth reproducer

This directory isolates the compile-graph shape present at Marin commit
`f52c1bbf9be226252a5327b5097dd65a6f6e7572`. It does not import Marin or JaxPP
and does not require a GPU.

The production symptom was a one-layer-per-stage L4 configuration compiling and
executing while the L8 `2,2,2,2` split remained in XLA compilation for more than
23 minutes in `grug_1f1b_mb0_stage3_loss_backward_accumulating`. At that commit,
`mapped_bwd` in `moe_mlp_accumulating_weight_gradient` called `jax.vjp` over the
complete local Ring graph from inside a second `shard_map`.

## What is retained

`repro.py` keeps the parts needed to expose that expansion:

- a 2x2 CPU mesh with `data` and `expert` axes;
- activations sharded over both axes;
- expert weights sharded over `expert` and replicated over `data`;
- exact integer top-k routing with capacity counts and nonzero drops;
- Ring-shaped `all_gather` and `psum_scatter` traffic;
- two accumulating grouped matmuls with explicit custom VJPs;
- the outer custom VJP whose backward takes a VJP of the complete local stack
  from inside `shard_map`;
- one or two distinct MoE blocks.

The grouped matmul is a small dense CPU implementation rather than the Triton
kernel. The reproducer targets tracing and StableHLO growth, not GPU kernel
performance or the 23-minute wall-clock stall.

## Run

From the repository root:

```bash
uv run python scratch/jaxpp_manual_vjp_compile_repro/repro.py
```

The script forces four CPU devices before importing JAX, lowers and compiles
every graph, executes each graph once, checks parity, and overwrites
`results.json`. It exits nonzero if any floating output, loss, or gradient leaf
exceeds relative-L2 `0.002`, or if routing counts and drops differ.

The recorded run used Python 3.12.11, JAX 0.11.0, and four CPU devices.

## Recorded result

Times are single-process CPU observations and are included for reproducibility,
not as stable performance benchmarks.

| Graph | Blocks | Lower (s) | Compile (s) | StableHLO bytes | Lines | Recursive JAXPR equations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward only | 1 | 0.0346 | 0.0973 | 23,306 | 264 | 137 |
| forward only | 2 | 0.0392 | 0.1744 | 41,104 | 456 | 255 |
| nested full VJP | 1 | 0.0369 | 0.1816 | 59,319 | 641 | 351 |
| nested full VJP | 2 | 0.0501 | 0.2207 | 108,812 | 1,161 | 664 |
| `lax.scan` full VJP | 1 | 0.0521 | 0.2009 | 83,609 | 872 | 363 |
| `lax.scan` full VJP | 2 | 0.0502 | 0.2979 | 83,609 | 872 | 363 |
| outlined full VJP | 1 | 0.0324 | 0.1776 | 63,620 | 683 | 359 |
| outlined full VJP | 2 | 0.0536 | 0.2284 | 114,655 | 1,203 | 663 |

The unrolled nested full-VJP graph grows from 59,319 to 108,812 StableHLO bytes
and from 351 to 664 recursive JAXPR equations when the second block is added.
The added forward block alone costs 17,798 StableHLO bytes; the corresponding
training-graph delta is 49,493 bytes, a 2.78x amplification.

The operation deltas identify replay and transpose construction as the dominant
expansion:

| Operation | Forward delta | Nested full-VJP train delta | Amplification |
| --- | ---: | ---: | ---: |
| `dot_general` | 2 | 8 | 4.00x |
| `all_gather` | 3 | 7 | 2.33x |
| `reduce_scatter` | 1 | 4 | 4.00x |
| `gather` | 8 | 22 | 2.75x |
| `scatter` | 4 | 16 | 4.00x |

The largest raw per-block deltas are also plumbing generated around those
operations: 120 `broadcast_in_dim`, 91 constants, 57 adds, 30 compares, 22
gathers, and 16 scatters. This is consistent with `mapped_bwd` retracing the
whole local forward to construct its pullback, then adding the contraction and
collective transposes, rather than with one isolated grouped-matmul lowering.

## Reduction formulations

`outlined_vjp` adds `jax.jit(..., inline=False)` around the local function before
`jax.vjp`. It does not contain the graph: the two-block StableHLO is 114,655
bytes, larger than the original 108,812 bytes.

`scan_vjp` expresses repeated blocks with `jax.lax.scan`. Its one- and two-block
modules are both 83,609 bytes with 363 recursive JAXPR equations and identical
operation counts. At two blocks it is 23.2% smaller than the unrolled full-VJP
module. The CPU compile time is not lower in this small case because compiling
the loop machinery dominates; the result is evidence for bounded graph size,
not a production performance claim.

All alternatives produced zero relative-L2 difference for output, loss, and
every gradient leaf. The routing fixture has per-data-replica counts
`[16, 6, 5, 5]`, accepted counts `[16, 4, 5, 5]`, and two dropped assignments.
The global drop count is exactly four per block and eight for two blocks in
every formulation.

The actionable production direction is either to stage homogeneous repeated
blocks through a scan-compatible representation or to provide an explicit
local Ring pullback whose forward residuals are saved once. Merely outlining
the nested VJP does not remove the expansion.
