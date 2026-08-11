# H100 Contract/Map v10 Nsight attribution

## Observed boundary

The single reviewed v10 job used source
`e9c050131c301b9bf98e93784f41fd31b8e5ef47` and immutable image
`ghcr.io/marin-community/iris-task-h100-evidence:3247e17d6c0f0fbf7263b5aee7891d209c978ac9@sha256:1543d62f4537773d09a1d7968139665d4f01b4b193412ea71f9f98a9a7f21e45`.
Source authentication, H100 and tool preflight, generated candidate compilation,
loaded-image topology checks, every numerical gate, the first case's complete
steady schedule, NVTX capture, and lazy SQLite export passed. The parser then
found no CUDA kernel fully contained in the first required ordinary-XLA range.
The task failed once, with failure retries disabled, and was not relaunched.

The sealed negative artifact is
`lib/tile_lifetime/benchmarks/artifacts/h100_contract_map_evidence_tenth_launch_failure_e9c050_v0/`.
The failed container's report and SQLite database were not exported from Iris,
so the artifact does not establish its table inventory, timestamps, kernel
names, or whether the lazy export included an empty memcpy table.

## Attribution audit

The v10 runner left `--cuda-graph-trace` implicit. The [Nsight Systems 2026.1
user guide](https://docs.nvidia.com/nsight-systems/2026.1/UserGuide/index.html)
documents `graph` as the default on supported systems. Whole-graph mode does
not collect individual CUDA Graph node activities; `node` mode collects the
node activities but not the whole-graph activity. The pinned [2026.1 SQLite
schema](https://docs.nvidia.com/nsight-systems/2026.1/nsys-exporter/exported_data.html)
likewise separates `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` from
`CUPTI_ACTIVITY_KIND_KERNEL`.

This is a plausible explanation for v10, not a fact recovered from its missing
database. Graph rows cannot satisfy the evidence contract because they do not
provide the exact ordered kernel identities and durations checked against the
generated and ordinary executables. The runner therefore selects
`--cuda-graph-trace=node` explicitly and continues to accept only kernel rows.
The image's bounded `nsys profile --help` validator now requires exactly one
closed `graph,node` option enumeration in addition to the existing capture-end
contract. A later authorized image build must execute that validator against
the pinned 2026.1.3 binary before another evidence launch.

Kernel attribution is stricter than interval overlap. Each kernel must be
fully contained in the NVTX push/pop range, resolve exactly once through
`StringIds`, and have exactly one runtime activity with the same correlation
ID. That runtime launch must be fully contained in the same range and run on
the range's thread. Graph rows remain diagnostic-only. Equal kernel start
timestamps fail because they cannot support the claimed exact launch order.

## Failure diagnostics

A no-kernel failure now emits one bounded canonical JSON diagnostic. It records
the exact range name, timestamps, duration, event type, domain, and threads;
kernel row count and timestamp envelope; before, overlap, contained, and after
counts; nearest offsets; hashed ordered and unique names; correlated runtime
counts; graph-activity interval counts; SQLite and report byte counts and
SHA-256 identities; database version fields; hashed table inventory and the
closed relevant-table schema/count summary; and the exact profile/export
arguments. Raw kernel names, arbitrary table names, paths, environment values,
arrays, and model data are excluded. The complete exception is capped at 4,096
characters and fails rather than truncating if the reviewed schema exceeds the
bound.

The report and SQLite files remain in the case artifact directory before
parsing. If that directory is not exported after a failed task, their bounded
identities and structural summary still reach the task log.

## Validation

Behavior tests cover the production profile command, explicit node mode,
strict full containment, exact runtime correlation and thread association,
NVTX type/domain/thread/order, ambiguous equal-start kernels, diagnostic
interval boundaries, graph rows that cannot substitute for kernels, malformed
or ambiguous schemas, retained file identities, and the 4,096-character bound.
The pinned-help tests cover missing, duplicated, incomplete, repeated, and
unknown CUDA Graph option values. No image build, workflow dispatch, GPU query,
evidence launch, retry, or relaunch was performed for this source repair.

The focused runner, pinned-help, and image-policy suites passed 161 tests. The
broader H100 numerical, benchmark, capsule, and image matrix passed 360 tests.
The full tile-lifetime suite passed 1,035 of 1,036 tests. Its sole failure is
the pre-existing ignored raw snapshot
`stateful_scan_generated_h100/raw/mutation_per_key_r1_b1_t64_h32_k128_v128_bv32.stdout.log`,
which is named by that artifact's `SHA256SUMS` but absent from a normal Git
checkout. Changed-files Ruff, Black, Pyrefly, license, syntax, structured-file,
whitespace, and Markdown checks passed.
