# H100 Contract/Map backend evidence staging plan

TL;DR: the first H100 sweep will compare ordinary XLA, Shuttle
`SOURCE_ORDERED`, and Shuttle `FAST` on anonymous dense Contract/Map programs.
It measures kernel-only and full forward/backward boundaries separately. The
checked-in runner is executable but has not been launched. GPU execution
remains gated on review, and the direct FFI backends remain outside the native
ordinary-JAX Shuttle transform seam.

## Status and architecture boundary

This checkpoint is `architecture_nonconforming`. It contains no GPU results and
has not reserved an H100. The staging-manifest CLI remains package-independent
and refuses `--execute-gpu` before importing JAX. The separate executable
runner performs its source, tool, device, and fresh-directory preflight before
importing JAX in isolated workers.

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

The launch gate requires all of the following:

1. A reviewed generated `SOURCE_ORDERED` backend reached through the native
   ordinary-JAX Shuttle transform.
2. A reviewed generated `FAST` backend reached through the same transform.
3. Review of the checked-in generated and ordinary-XLA resource collectors.
4. A review that removes the `architecture_nonconforming` status.

The source-only runner is
`lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py`. Direct local
execution uses an unset `XLA_FLAGS`, a clean exact source SHA, a fresh artifact
directory, and explicit JAX and NVCC identities:

```bash
python lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py \
  --execute \
  --source-root "$PWD" \
  --source-sha "$(git rev-parse HEAD)" \
  --artifact-directory /fresh/path/contract-map-h100 \
  --require-jax-version <exact-version> \
  --nvcc /absolute/cuda/bin/nvcc
```

Preflight requires one visible H100 with compute capability 9.0, `sm_90a`, and
absolute executable paths for Git, `nvidia-smi`, NVCC, sibling `ptxas` and
`cuobjdump`, `ncu`, and `nsys`. It records each tool's content hash and version.
An existing artifact directory, tracked source modification, version mismatch,
missing profiler, or different GPU rejects before worker execution.

The reviewed Iris image target is `task-h100-evidence`. It inherits the current
`task` image, including Nsight Systems 2026.1.3, and adds the hash-pinned Debian
12 amd64 CUDA 13.2.86 compiler closure, `cuobjdump`, and Nsight Compute
2026.1.1. The dedicated manual-only `ops-h100-evidence-image.yaml` workflow
checks out an explicitly selected ref, resolves it to one full commit SHA,
publishes only `iris-task-h100-evidence:<full-git-sha>`, and reports the
resulting OCI digest. Before that workflow exists on the default branch, the
existing `ops-docker-images.yaml` dispatcher can select `image_set=h100-evidence`
to call the dedicated workflow from the dispatch commit. Its default
`image_set=all` retains the existing Iris, finelog, and TPU image builds.
A launch must use
`ghcr.io/marin-community/iris-task-h100-evidence:<full-git-sha>@sha256:<digest>`;
the tag alone, `latest`, and a date tag are not accepted. The launch overrides
the task image and requests `--gpu H100x1` explicitly against
`cw-us-west-04a`; it does not modify that cluster's shared default image.

This image is necessary but not sufficient for launch. The repository's full
tracked tree exceeds Iris's 25 MiB bundle limit and includes historical evidence
that the runner does not import. The checked-in source-capsule wrapper therefore
does not claim to transport a clean Git worktree. It accepts only a globally
clean checkout at the requested full commit, records that commit and its tree,
and expands the checked-in closed allowlist: root and package configuration,
the evidence plan, runner, training adapter, wrapper, and every tracked Python
file in `lib/tile_lifetime/src/tile_lifetime`. Every capsule member has a closed
path/type/mode/size/SHA-256 record in a canonical manifest. New package Python
files are included mechanically; changes to the exact runtime/config list
require an allowlist review.

Preparation is local and does not query a device or submit a job:

```bash
python lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py prepare \
  --source-root "$PWD" \
  --source-sha "$(git rev-parse HEAD)" \
  --output-directory /fresh/path/contract-map-source-capsule
```

The output contains only the deterministic capsule ZIP, canonical manifest,
and stdlib-only launcher, and reports the SHA-256 of both manifest and launcher.
The eventual submitted command must use trusted image tooling to verify both
identities before invoking `/opt/h100-evidence-runtime/bin/python` by absolute
path. The `run` command requires that same absolute path through
`--runtime-python`; it never resolves or falls back to an unqualified `python`
from `PATH`. Launcher self-verification cannot establish its own trust. As
defense in depth, `run` also requires the expected launcher SHA-256 explicitly
before it verifies the trusted manifest hash,
commit and tree, archive hash, exact member set, bounds, paths, modes, symlinks,
and member hashes before starting the runner. Runner preflight repeats the
manifest and extracted-file checks without asserting Git-clean status. After
each local import boundary and again before acceptance, the coordinator and
isolated workers reject any `tile_lifetime` or benchmark module loaded outside
the capsule or with a hash not present in the manifest. This detects provenance
drift but is not a sandbox: imported code has already executed. Accepted
evidence records the commit, tree, and capsule-manifest digest.

An image built before this interpreter contract is not accepted for another
launch. A rebuilt image must be published under its full-Git-SHA tag, resolved
to an OCI digest, and independently reviewed together with the source capsule.
The task command must invoke both the launcher and runner with the exact frozen
interpreter:

```text
/opt/h100-evidence-runtime/bin/python h100_contract_map_source_payload.py run \
  --runtime-python /opt/h100-evidence-runtime/bin/python ...
```

A future launch must use the default container profile, one explicit
`H100x1`, no UV sync against the partial capsule, a fresh artifact path outside
the capsule, and `max_retries=0`. Nsight Compute permission failure remains a
fail-closed runtime result; this plan does not enable the privileged profile.

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
- `contract_map_backend_resources.py` defines executable retained CUDA artifact
  commands, ptxas resource parsing, and the expected logical boundary.
- `h100_contract_map_backend_training.py` maps the reviewed structural cases to
  scalar ASTs and creates both generated policies without reading HLO fixtures.
- `h100_contract_map_backend_runner.py` compiles and registers both policies,
  compiles ordinary JAX forward plus JAX VJP, runs numerical and repeat gates
  before timing, and coordinates isolated compile, cache, Nsight Compute, and
  Nsight Systems workers. It publishes `accepted_bundle.json` only after the
  existing validator accepts all 24 records in fixed order. Ordinary-XLA
  boundary evidence comes from the same final optimized HLO identity observed
  by its compile, cache, timing, and profiler workers; it never reuses a
  generated candidate ABI.
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

The ordinary-XLA combined executable has the reviewed entry contract
`x,w0,w1,do -> y,dx,dw0,dw1`. The runner maps its four parameters by exact
`parameter(0..3)` attributes and its four outputs by the tuple root, then
records every parameter and root shape/layout. Reachable entry copies,
transposes, or bitcasts reject until separate evidence can prove their
materialization. Fusion and custom-call facts remain in the HLO manifest;
launch count and order come only from nsys and ncu. The exact reviewed entry
proves no saved state crosses the executable boundary and does not prove
semantic recomputation, so both records are empty. An additional parameter or
root output rejects. Kernel-only and logical records describe the same
executable and can differ only through proven host or layout adapters.

Timing uses isolated processes and records four costs independently:

1. coordinator subprocess spawn through executable compilation, including
   interpreter startup, imports, device initialization, FFI registration, and
   deterministic input setup;
2. first execution and all ten warmup samples;
3. counterbalanced steady-state samples;
4. persistent-cache cold and hit processes under isolated cache roots.

Compile and persistent-cache samples are not folded into steady-state latency.
The logical sample is the host interval for 100 compiled forward-plus-VJP
executions followed by device synchronization. Nsight Systems captures the same
NVTX range through CUPTI. The kernel-only sample is the sum of contained CUDA
kernel durations divided by 100 and rounded to the nearest nanosecond. Every
iteration must have the same ordered kernel sequence.

## Numerical gates

Numerical floors are immutable fields of schema version 2. They are serialized
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

Generated variants retain the emitted source, loaded shared library, PTX,
cubin, cubin disassembly, and loaded-image disassembly. Kernel records use the
SASS extracted from the shared object that JAX loaded. The separately compiled
cubin remains an available cubin artifact but cannot supply the loaded-image
SASS identity. Loaded-image admission requires exact unique coverage of every
generated function with valid addressed instructions. Missing, extra,
duplicate, lookalike, or malformed function sections reject the run.
`cuobjdump` section order is not treated as launch order; launch order remains
owned by the Nsight Systems trace and must match the generated topology.
Generated ptxas registers, spills, and static shared memory must agree with
Nsight Compute launch evidence.

Ordinary XLA must expose exactly one nonempty PTX from its first isolated
compile worker through the pinned public jaxlib dump flags. Cubin evidence is a
closed tagged record. Generated backends require `available` with a concrete
path and SHA-256. Ordinary XLA may report `unavailable` only with
`public_xla_dump_omits_cubin`; mixing unavailable and available fields or using
another reason rejects the record. Ordinary-XLA SASS comes from the retained
Nsight Compute report's public `--page source --print-source sass` export, not a
private executable extractor. Admission requires one exact `Kernel Name:`
section for every profiled kernel, no duplicate or lookalike sections, and at
least one address-bearing recognized SASS instruction in every section.
Warnings, errors, unavailable-source notices, unknown mnemonics, malformed
instruction rows, and unstructured text reject the run. The run aborts if a
validated instruction record has an `LDL` or `STL` opcode because the public
profiler path does not provide spill-byte evidence; incidental text cannot be
mistaken for a spill instruction.

Nsight Compute must report the closed launch, register, shared-memory,
occupancy-limit, and achieved-occupancy metric set for every launch. Nsight
Systems profiles with the exact `cuda,nvtx` trace set and exports SQLite with
`--lazy=true`. [NVIDIA's 2026.1 exporter contract](https://archive.docs.nvidia.com/nsight-systems/2026.1/UserGuide/index.html#cli-export-command-switch-options)
creates a lazy table only when it contains data, so an omitted
`CUPTI_ACTIVITY_KIND_MEMCPY` table means zero
copies only when the same database contains nonempty
`CUPTI_ACTIVITY_KIND_KERNEL` activity, exact scheduled `NVTX_EVENTS`, valid
`StringIds`, and nonempty `TARGET_INFO_GPU` identities covering every kernel's
device. An absent kernel table, empty kernel activity, invalid GPU identity,
schedule mismatch, malformed schema, ambiguous kernel identity, launch-order
drift, substring-only kernel match, or any steady-state CUDA copy aborts the
run. Result-evidence schema v4 retains closed H2D, D2H, and D2D count and byte
accounting before the no-copy gate. Three independent compile workers and three
paired cold/hit cache roots retain compile, first-execution, and cache samples:
nine records over six physical roots. The runner disables JAX's auxiliary
root-relative XLA caches and pins the unbounded flat-file cache mode. It
identifies exactly one `jit_step-<SHA256>-cache` target entry in each record.
The exact target cache key and SHA-256 of the serialized executable must match
across all nine records. The executable digest removes only JAX 0.10.1's
documented four-byte big-endian cached compile-time prefix after bounded zlib
decompression. The compressed entry and whole-root identities remain
diagnostics, while each cold/hit pair must preserve its complete path-and-byte
root identity. Public JAX monitoring must report one request and one write
event for each compile or cold record, and one request and one hit for each hit
record. Their final optimized HLO must still equal the timing and profile
worker HLO before their evidence can be merged.

Each compile sample begins on the coordinator's monotonic clock immediately
before spawning its isolated worker. The worker records the same host monotonic
clock immediately after `.compile()` returns, before first execution,
synchronization, cache scanning, HLO serialization, and result publication.
The coordinator accepts the timestamp only when it falls between spawn and
worker exit, computes compile time from spawn to that timestamp, and records
the remaining worker lifetime separately as post-compile time.

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
