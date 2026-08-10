# H100 Contract/Map backend evidence staging plan

TL;DR: the first H100 sweep will compare ordinary XLA, Shuttle
`SOURCE_ORDERED`, and Shuttle `FAST` on anonymous dense Contract/Map programs.
It will measure kernel-only and full forward/backward boundaries separately.
The checked-in harness only emits an immutable plan. GPU execution remains
disabled until both Shuttle backends and the resource collectors use the
ordinary-JAX transform seam and pass review.

## Status and architecture boundary

This checkpoint is `architecture_nonconforming`. It contains no GPU results.
It does not import JAX, query a device, reserve an H100, or call a retained
workload-specific rewriter.

The current native path under `lib/shuttle/mlir` forms generic f32 Contract and
Map algebra and lowers it back to StableHLO inside XLA. It does not yet lower
that algebra to an H100 kernel. `contract_map_backend.py` now forms the
anonymous BF16 graph `x @ w0 -> Map -> @ w1` and derives its reverse from the
same `TensorProgram`. `cuda_contract_map_backend_codegen.py` emits multi-CTA
direct FFI source for both policies and all four reviewed shapes. This direct
FFI source is not connected to the native ordinary-JAX transform.

CUDA reverse lowering accepts exactly five differentiated operations in order:
hidden-adjoint Contract, second-weight-adjoint Contract, pointwise-adjoint Map,
input-adjoint Contract, and first-weight-adjoint Contract. Kernel operands,
output axes, reduction axes, and the hidden-adjoint-to-Map fusion come from
those operations. Missing, reordered, or rewired operations reject before
source generation.

The direct ABI uses signed 32-bit flattened indices. Every positive rank-two
buffer product, kernel work-item product, grid numerator, grid block count, and
BF16 byte size is checked before source generation. Each value must be at most
`2,147,483,647`; the two-byte BF16 boundary limits a buffer to
`1,073,741,823` elements. The grid-x limit is `2,147,483,647` blocks. These
bounds apply to both `SOURCE_ORDERED` and `FAST`; exceeding any bound rejects
the candidate.

Each generated variant is rendered twice. The first render uses a fixed
placeholder in both FFI targets, all host symbols, the reported backend
fingerprint, and all six CUDA kernel entry names. SHA-256 over that complete
source, a NUL separator, and the canonical JSON encoding of the closed physical
record defines the physical digest. The physical record includes the semantic
digest, target prefix, threads, launch-check or command-buffer traits, complete
FFI ABI, kernel topology, five reverse operations, and declared fusion. The
second render substitutes the full digest for every placeholder, including
every CUDA artifact stem. This fixed-placeholder scheme avoids a circular
source hash while ensuring that an emitted kernel-body change rotates every
linkable identity. This is source-only evidence; no CUDA compile or GPU result
is part of this checkpoint.

`contract_map_chain.py`, `cuda_contract_map_chain_codegen.py`, and
`h100_generated_contract_map_chain_training.py` remain historical. They use a
one-CTA shared-memory body and reconstruct a different residual-gated graph
from an HLO artifact. Current H100 backend code must not import them.

The launch gate therefore requires all of the following before it can be
enabled:

1. A reviewed generated `SOURCE_ORDERED` backend reached through the native
   ordinary-JAX Shuttle transform.
2. A reviewed generated `FAST` backend reached through the same transform.
3. Resource collectors for generated and ordinary-XLA executables.
4. A review that removes the `architecture_nonconforming` status.

The benchmark CLI's `--execute-gpu` preflight fails before importing
`tile_lifetime` or JAX, creating an output file, or inspecting an accelerator.

## Reused prototype evidence

The staging schema reuses behavior that already has focused tests:

- `benchmark_boundary.py` checks physical layouts, numerical acceptance, repeat
  hashes, and pairwise drift.
- `command_buffer_capture.py` records raw counterbalanced samples and handler
  count checkpoints.
- `benchmark_metadata.py` records the command, toolchain, and device metadata.
- `contract_map_backend.py` defines the anonymous forward graph, mechanical
  reverse, semantic fingerprint, and source-ordered CPU reference.
- `cuda_contract_map_backend_codegen.py` emits six global-intermediate kernels.
  `SOURCE_ORDERED` assigns one output to one thread and folds each reduction
  from index zero upward with explicit round-to-nearest operations. `FAST`
  assigns one output to one warp and combines lane partials with a fixed shuffle
  tree under the rounding-reorder policy.
- `jax_contract_map_backend_ffi.py` defines the typed forward and reverse ABI.
  The reverse exposes its preactivation-adjoint scratch result so XLA owns the
  buffer; the handler does not allocate device memory.
- `contract_map_backend_resources.py` defines retained CUDA artifact commands,
  ptxas resource parsing, and the expected logical boundary. Profiler,
  unexpected-copy, and ordinary-XLA collectors are still missing.
- `h100_contract_map_backend_training.py` maps the reviewed structural cases to
  scalar ASTs and creates both generated policies without reading HLO fixtures.
- `contract_map_chain.py`, `cuda_contract_map_chain_codegen.py`,
  `jax_contract_map_chain_ffi.py`, and
  `h100_generated_contract_map_chain_training.py` are historical evidence.
- `linear_pair_map.py`, `cute_pair_map_codegen.py`, and
  `h100_generated_linear_pair_map_training.py` provide realistic generated
  CuTe/QuACK Contract/Map components and saved-versus-recomputed boundaries.
- `h100_event_tensor_split_fold.py` shows the current ptxas and CUDA function
  resource capture pattern.

These files remain in `tile_lifetime`. The new staging code does not modify or
promote their workload-oriented CLI, CODA denominator, StableHLO recovery, or
`DenseDebugConfig` wiring.

## Structural sweep

Each case contains only `(rows, reduction, features, scalar_map)`. Its ID is a
digest of those fields. The schema has no model, layer, attention-head,
sequence-length, fixture, or workload key.

The four initial row extents are odd (`43`, `131`, `269`, and `521`). Reduction
and feature extents remain multiples of eight for tensor-core-compatible
physical candidates. The scalar Maps cover sigmoid-product, tanh-product, and
cubic-mix ASTs. This combination varies algebra and shape without reproducing a
recognizable attention signature.

Every case runs these primary backends in fixed schema order:

1. `ordinary_xla`
2. `shuttle_source_ordered`
3. `shuttle_fast`

The timing protocol enumerates all six backend permutations. Twenty-four raw
steady-state samples cover each permutation four times. Aggregate medians are
derived from the retained rows; they never replace them.

## Boundaries and costs

Each backend reports two boundaries:

- `kernel_only`: the generated or XLA kernel sequence after input and output
  buffers already satisfy the physical ABI.
- `logical_training_step`: forward and backward from logical inputs through
  layout adapters, copies, save-or-recompute policy, output adapters, and final
  synchronization.

Input and output layouts are nonempty lists of canonical strings. The remaining
logical-boundary arrays use closed records with these exact fields and types:

- layout adapter: `value`, `input_layout`, and `output_layout` strings plus a
  `materialized` boolean;
- materialized copy: `source` and `destination` strings plus nonnegative integer
  `bytes`;
- transpose: `input` and `output` strings, a complete integer `permutation`, and
  a `materialized` boolean;
- bitcast: `input` and `output` strings plus nonempty positive-integer
  `input_shape` and `output_shape` lists with equal element counts;
- recompute operation: `output` and `operation` strings plus a nonnegative
  integer `launch_count`.

Saved state is a mapping from canonical value names to nonnegative integer byte
counts. Null values, boolean substitutes for integers, missing fields, and
unknown fields reject the record.

The full boundary records every saved tensor and its byte size. A recompute
candidate records the removed save and all extra Contract/Map launches. The
comparison rejects a row if one backend includes an adapter, copy, or saved
state that another backend silently excludes.

Timing uses isolated processes and records four costs independently:

1. process start through executable compilation;
2. first execution and all ten warmup samples;
3. counterbalanced steady-state samples;
4. persistent-cache cold and hit processes under isolated cache roots.

Compile and persistent-cache samples are not folded into steady-state latency.

## Numerical gates

Numerical floors are immutable fields of schema version 1. They are serialized
before a run and cannot be supplied by the result producer. The schema records
the canonical SHA-256 digest of the complete floor tuple. Plan construction and
result validation reject a different tuple or digest even when every replacement
value is finite and internally consistent.

`SOURCE_ORDERED` uses an explicit source-ordered FP32 reference, a maximum
absolute error of `0.0078125`, a mean absolute error of `0.0005`, a maximum ULP
distance of `1`, a mean ULP distance of `0.05`, and bitwise repeatability.

Ordinary XLA and `FAST` use an FP64 real-algebra reference. Their maximum
absolute error is `0.03125`, mean absolute error is `0.002`, maximum ULP
distance is `4`, and mean ULP distance is `0.25`. Repeat drift is bounded at
`0.0078125` maximum and `0.0005` mean. All policies permit zero nonfinite
values.

ULP distance is computed after converting each physical BF16 result to its
ordered unsigned BF16 bit representation. Absolute and ULP metrics are emitted
per output for the forward result, input adjoint, and both weight adjoints.
Untimed repeat evidence retains output hashes and every pairwise drift record.
Timing starts only after all floors pass.

The result validator recomputes this decision from every emitted output metric;
the producer's `floors_passed_before_timing` flag is not accepted on its own.
NaN, infinity, nonfinite outputs, out-of-floor absolute or ULP error, and
out-of-floor pairwise repeat drift reject the record. Bitwise-repeatable records
also require identical SHA-256 output identities for every repeat.

## Physical evidence

Every backend and boundary must retain:

- final optimized HLO and custom-call or fusion manifest;
- PTX and SASS hashes plus the disassembly artifacts;
- registers per thread, spill-load bytes, spill-store bytes, and static and
  dynamic shared memory per kernel;
- block size, active blocks per SM, limiting occupancy resource, and achieved
  occupancy when the profiler exposes it;
- kernel launch count and ordered kernel names;
- device-to-device and host-device copy counts and bytes;
- logical adapters, transposes, bitcasts, materialized copies, and saved-state
  bytes;
- raw counterbalanced timing rows, command, environment, compiler flags,
  canonical source SHA, and persistent-cache identity.

A missing resource field, missing raw sample, unexpected copy, malformed or
empty artifact identity, empty provenance field, empty compile/warmup/cache
sample list, or launch-count mismatch rejects the headline row. PTXAS text
alone is insufficient when SASS or profiler evidence is missing.

The machine result schema requires 24 records: one for each of the four reviewed
structural cases, three backends, and two measurement boundaries. Kernel
records bind PTX and SASS paths and hashes to registers, spills, static and
dynamic shared memory, block size, active blocks per SM, limiting occupancy
resource, and achieved occupancy. Numerical records contain separate forward,
`dx`, `dw0`, and `dw1` metrics, repeat hashes, and pairwise drift. Timing records
embed the exact 24-row schedule, four rows for each backend permutation, and
reject raw rows that omit a backend, boundary, or scheduled order.

## External comparator admission

FA4 requires attention-score, normalized-exponential, and Fold structure. Grug
requires segmented Contract, Relation, and Transport structure. The initial
dense Contract/Map sweep has only Contract and Map, so both comparators are
recorded as inapplicable.

A later region may admit one of these comparators only from its structural
feature set. Model names, symbol names, fixture names, and shape lookups cannot
participate in admission or backend selection.

## Staging command

This command writes the plan and exits without accelerator access:

```bash
uv run --frozen --package marin-tile-lifetime python \
  lib/tile_lifetime/benchmarks/h100_contract_map_backend_evidence.py \
  --shuttle-revision ca2091a4b27a366c4f3625cd339b21e139886450 \
  --json-output /tmp/shuttle-h100-contract-map-plan.json
```

Passing `--execute-gpu` fails before importing `tile_lifetime` or JAX and before
creating the output file. It will remain disabled until the ordinary-JAX
SOURCE_ORDERED and FAST backends, resource collectors, and architecture review
are present.
