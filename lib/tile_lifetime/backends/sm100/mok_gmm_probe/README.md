# MoK grouped-GEMM primitive probe

This directory contains a standalone BF16 routed-expert GEMM probe from
Mixture-of-Kittens (MoK) plus generated CUDA operations for its local relation
boundaries. It is not a wrapper around MoK's complete
dispatch–MLP–combine megakernel.

The CUDA extension includes `csrc/mok_megakernel.cuh` from an external checkout
at the pinned MoK revision and invokes
`expert_grouped_gemm_kernel<false>()` directly. The grouped-GEMM wrapper:

- accepts an expert-contiguous activation matrix, expert weights, and
  256-padded per-expert row counts;
- launches one two-CTA cluster per 256-by-256 output tile;
- initializes only the primitive's TMA, tensor-memory, and semaphore pipeline;
- passes null for all dispatch, SwiGLU, combine, and buffer-reuse events; and
- stores BF16 output after FP32 accumulation.

W2 is one invocation. W13 is measured as two invocations, one each for the gate
and up projections, matching the two primitive calls made by MoK. The extension
also contains generated CUDA boundary operations used to compose this mainloop
with compiler-produced relation metadata:

- `padded_pack_bf16_out` expands coalesced receive rows into 256-padded,
  expert-contiguous rows;
- `fixed_capacity_relation_plan_out` constructs destination-group row maps,
  per-group counts, and ordered edge weights from runtime relation edges. One
  CTA owns each equal-capacity group and scans edges in source/slot order; no
  semantic value is atomically accumulated;
- `adjacent_pair_map_bf16_out` and `row_halves_pair_map_bf16_out` apply the
  compiler-generated adjacent-pair `Map` to either separate or concatenated
  inputs;
- `indexed_weighted_ordered_fold_bf16_out` applies the compiler-generated
  contribution and ordered-update expressions over an indexed edge list;
- `indirect_weighted_fold_base_map_out` is an owner-local diagnostic that
  performs the same generated `Fold` and post-fold base `Map`; and
- `partitioned_ordered_fold_base_map_bf16_out` executes that generated ordered
  fold over returned partition partials plus a source-local base value. It
  consumes a compiler-built `[partition, source_item]` row map and does not use
  atomics.

The scalar expressions are emitted from `MapFoldSemantics` into
`generated_map_fold.inc`. The extension exports the selected program's SHA-256
digest, and the natural StableHLO benchmark rejects an extension whose digest
does not match the recovered plan. Changing the activation or merge expression
therefore regenerates CUDA without changing these indexing and loop skeletons.

None of these entrypoints implements communication, CLC work redistribution,
or a full MoE forward pass. `indirect_weighted_fold_base_map_out` cannot replace
a cross-rank semantic combine by itself.

`partitioned_ordered_fold_base_map_bf16_out` can replace the reduction portion
of that combine only when a separate payload-only reverse transport has already
returned every rank partial. The distributed benchmark uses
`all_to_all_single` for that clean boundary; DeepEP remains the forward payload
transport.

Build and run it through the benchmark driver:

```bash
python lib/tile_lifetime/benchmarks/gb200_mok_gmm_probe.py \
  --mok-root /path/to/mixture-of-kittens \
  --component w2 \
  --json-output /tmp/mok-w2.json
```

The checkout must be at MoK commit
`3e1cf43ab93ad040afed52a45ab03cb490ffe4be`, with ThunderKittens submodule
commit `1c3920d993404dd49a6d4c7267ea11d583bd5c68`. The build needs the same
Blackwell CUDA 13 and PyTorch 2.10 environment as MoK. The verified isolated
toolchain pins NVIDIA CUDA NVCC 13.0.88, CCCL 13.0.85, CUDA CRT 13.0.88, and
NVVM 13.0.88 together. Mixing the 13.0 compiler and ptxas with 13.2 headers or
13.3 CRT/NVVM packages fails before compiling the probe. The pip CUDA toolkit's
library directory must also be on `LIBRARY_PATH` so the host linker can find
`libcudadevrt.a` and `libcudart_static.a`.

Input contract:

- `activations`: contiguous BF16 `[sum(padded_counts), K]`;
- `weights`: contiguous BF16 `[experts, N, K]`;
- `output`: contiguous BF16 `[sum(padded_counts), N]`;
- `padded_counts`: contiguous CUDA int32 `[experts]`;
- every count, `sum(padded_counts)`, and each segment boundary is divisible by
  256;
- `K` is divisible by 64 and `N` is divisible by 256.

The benchmark validates the count contract on the host before timing. The CUDA
entrypoint intentionally avoids copying counts back to the CPU on every launch.

The already-dispatched composition benchmark uses the generic `RelationPlan`
to project one receiver rank from the exact MoK route fixture. It independently
checks receiver order, route slots, local expert grouping, padding, and inverse
mappings before launching generated CUDA:

```bash
PYTHONPATH=lib/tile_lifetime/src:lib/tile_lifetime/benchmarks \
python lib/tile_lifetime/benchmarks/backends/gb200_deepep_mok_local.py \
  --mok-root /path/to/mixture-of-kittens \
  --deepep-root /path/to/DeepEP \
  --route-fixture /path/to/mok-route-fixture.npz \
  --json-output /tmp/deepep-mok-local.json
```

Its primary timing begins at a simulated coalesced receive boundary and ends
with one BF16 contribution per received token. Official DeepEP dispatch and
combine and shared-expert compute are outside that timing.
