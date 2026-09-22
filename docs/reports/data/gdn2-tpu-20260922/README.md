# GDN-2 TPU measurements, 2026-09-22

These row-level artifacts preserve representative-shape sweeps and confirmation
runs before Iris's seven-day task-output retention expires. Each row identifies
the measured source hash.

| File | Iris job | Seed |
| --- | --- | --- |
| `v5e-full2.jsonl` | `/dlwh/gdn2-kernels-v5e-0922-full2` | 1 |
| `v5p-full1.jsonl` | `/dlwh/gdn2-kernels-v5p-0922-full1` | 0 |
| `v5e-confirm.jsonl` | `/dlwh/gdn2-kernels-v5e-0922-confirm` | 0 |
| `v5p-confirm.jsonl` | `/dlwh/gdn2-kernels-v5p-0922-confirm` | 1 |
| `v5e-forgetting.jsonl` | `/dlwh/gdn2-kernels-v5e-0922-forgetting` | 0 |
| `v5p-forgetting.jsonl` | `/dlwh/gdn2-kernels-v5p-0922-forgetting` | 0 |
| `v4-full1.jsonl` | `/dlwh/gdn2-kernels-v4-0922-full1` | 0, 1 |
| `v5e-cachecheck.jsonl` | `/dlwh/gdn2-kernels-v5e-0922-cachecheck` | 0 |

The full sweeps contain 64 rows each: upstream/candidate, batches 4/8, length 4096,
six heads, dimension 128, FP32/BF16, BT128/256, MB16/32, and
forward/forward+backward. Each timing has 20 synchronized samples on one
TPU device. Forward outputs, final state, and gradients for six token
inputs plus the initial state
passed against the CPU oracle before timing was enabled. Confirmation files
contain 32 rows each, restricted to BT128 and MB16/32, for 192 preserved rows
overall. Confirmation runs execute the candidate first and use the other seed.
The v5 forgetting checks below add 64 rows. The v4 small-shape sweep adds
16 rows, including eight rejected rows. The v4 forgetting checks add 16
accepted rows. The v4 representative-shape check adds 32 accepted rows, and
the cache check adds six, bringing the preserved total to 326.

The full-sweep source SHA256 is
`15b57d022c5c5ba614e40b0697a16c6e81d4e4d025f609afcb7c64df7febfe22`.
The digest covers sorted relative paths and contents of every Python file
in the GDN-2 package, followed by the benchmark script contents. The base
Git revision alone does not recover these uncommitted kernel changes.
[source-15b57d.json](source-15b57d.json) preserves the exact kernel Python
files, benchmark, LICENSE, and NOTICE as a mapping from repository-relative
filenames to UTF-8 contents. Its manifest records the base Git revision,
upstream revision, bundle URI, generation, and SHA256.

The snapshot was recovered from the bundle identified by the launch request
for `/dlwh/gdn2-kernels-v5e-0922-full2`:
`gs://marin-us-central2/iris/marin/state/bundles/69aaa408e107984f60c1d0ed3b3879ead133a473437c482422ea6c407e21283b`.
The 12,534,481-byte download passed the bundle SHA256 check. Only the listed
source and license files were preserved. Recomputing the source digest from
the JSON contents reproduces the hash above. Restore these files at their
recorded relative paths on the recorded base revision to recover the measured
kernel and harness sources; environment versions remain recorded in the rows.

Confirmation source SHA256 is
`0146da672e15e7ca0ccdadfd41e8b6ffb7303c61e28c0dd75354a441874b506e`.
[source-0146da.json](source-0146da.json) preserves the same scoped files and
manifest fields for the confirmation runs. The exact task bundle is
`gs://marin-us-central2/iris/marin/state/bundles/ff0b2c9e1d41b6b7d190b6108c7ece627fc56679efef4405fd045dfe416d2672`
(12,537,598 bytes, generation `1790106122917583`). Both the downloaded bundle
SHA256 and the reconstructed source SHA256 were verified.

Rows retain environment flags, hardware, compilation latency, timing
samples, and numerical error summaries. Repeated TPU-versus-CPU reference
diagnostics and worst-error element coordinates/values are omitted from
this compact export. No timing samples were filtered.

Candidate FP32 BT256/MB32 on v5p has repeated timing spikes (three of
20 forward+backward samples at batch 4, five of 20 at batch 8). Do not use
its median alone as a reliable latency claim. BT128 sample coefficients
of variation are at most 0.236% on v5e and 0.691% on v5p.
By median latency, candidate BT128/MB16 is fastest on both hardware types
for both timing modes in every measured shape/dtype bucket; upstream
BT128/MB32 is its fastest configuration.

Confirmation retains the feature-last score layout on v5e/v5p and reproduces
candidate BT128/MB16 as the fastest configuration in every measured bucket.
Its median latencies differ from the earlier sweep by -0.222% to +0.228% on
v5e and -0.338% to +0.120% on v5p. Source, seed, and execution order changed,
so these comparisons establish repeatability rather than isolate a code change.
Candidate BT128/MB16 within-row coefficients of variation are at most 0.110%
on v5e and 0.196% on v5p.

Confirmation forward+backward medians in milliseconds:

| TPU | Batch | Dtype | Candidate BT128/MB16 | Upstream BT128/MB32 | Best-versus-best speedup |
| --- | --- | --- | ---: | ---: | ---: |
| v5e | 4 | FP32 | 19.505 | 59.565 | 3.054x |
| v5e | 4 | BF16 | 19.162 | 58.673 | 3.062x |
| v5e | 8 | FP32 | 38.372 | 120.187 | 3.132x |
| v5e | 8 | BF16 | 37.757 | 118.755 | 3.145x |
| v5p | 4 | FP32 | 15.487 | 49.764 | 3.213x |
| v5p | 4 | BF16 | 15.448 | 49.622 | 3.212x |
| v5p | 8 | FP32 | 30.153 | 98.584 | 3.269x |
| v5p | 8 | BF16 | 30.079 | 98.346 | 3.270x |

The confirmation candidate v5e batch-4 BF16 MB32 configuration has one forward
spike near 104 ms and four forward+backward spikes at 109-115 ms among 20
samples; ordinary samples are approximately 8.33 and 20.14 ms. This does not
affect the selected MB16 configuration. Retain all samples when comparing
tail latency. The evidence supports BT128/MB16 at the measured batch endpoints
4 and 8, length 4096, six heads, and dimension 128. Batches 5-7 are unmeasured.

Strong-forgetting validation uses batch 1, length 512, one head, dimension 128,
seed 0, FP32/BF16, and all four BT128/256 and MB16/32 combinations. Log-decay
inputs are sampled uniformly from [-0.5, 0] or [-5, 0]. All 32 rows on each
hardware type pass the unchanged forward and all-input-gradient gates against
the CPU oracle. These candidate-only checks establish numerical behavior;
five timing samples per v5e row and three per v5p row do not support comparative
performance claims.

| Stress artifact | Exact source snapshot | Source SHA256 |
| --- | --- | --- |
| `v5e-forgetting.jsonl` | [source-7d1ecb.json](source-7d1ecb.json) | `7d1ecbaf3778030631c862c2ba7817a55c79fbeab42d290eda2dbe8f21da9187` |
| `v5p-forgetting.jsonl` | [source-567acb.json](source-567acb.json) | `567acb3014690ba8bc56318b341ef744cb4d4d4fcef60c69c02cfd157f0a2313` |

Both stress snapshots were retrieved from the respective task's exact launch
bundle. Each manifest records the object URI, generation, bundle SHA256,
upstream revision, and base Git revision. Downloaded bundle hashes and source
hashes reconstructed from the JSON mappings were verified. The stress runs
use different source revisions and are reported separately.

`v4-forgetting.jsonl` adds the same strong-forgetting checks on v4, limited
to the compilable BT128 configurations (MB16/32). All 16 rows pass forward
and all seven input-gradient gates against the CPU oracle. The job is
`/dlwh/gdn2-kernels-v4-0922-forgetting`; its source hash matches the preserved
[source-567acb.json](source-567acb.json). Each row has three timing samples;
these are numerical stress checks, not comparative performance evidence.

The stress output archives were fully decompressed with `zstandard`, parsed
with Python `tarfile`, and checked byte-for-byte against the extracted JSONL.
Both contain one complete JSONL member and no data discrepancy. macOS
`bsdtar --zstd` returned a child-process error for the v5p archive; independent
`zstd -t` and full Python archive validation passed.

See the [kernel harness reference](../../../references/gdn2-kernels.md)
for arithmetic, tolerances, provenance, and reproduction commands.

`v4-bwdtiles.jsonl` preserves 16 rows from
`/dlwh/gdn2-kernels-v4-0922-bwdtiles`, using the already preserved
[source-567acb.json](source-567acb.json). At `[1,256,1,128]`, seed 0,
decay 0.01, BT128 passes forward and all seven input-gradient gates in
FP32/BF16 for MB16/32 (eight accepted timing rows, ten samples each).
BT256 backward compilation exceeds v4's 16 MiB VMEM: 40.97 MiB for FP32
and 40.88 MiB for BF16, independent of MB16/32. Its eight timing rows are
rejected, including forward rows whose paired backward failed. These
small-shape results do not establish representative-shape performance.
Compiler error strings in this compact export are limited to 1,000
characters, with original length and truncation recorded; full tracebacks
and duplicate reference diagnostics are omitted. Timing samples and
numerical summaries are retained without filtering.

`v4-full1.jsonl` validates candidate BT128 at batches 4/8, length 4096,
six heads, dimension 128, FP32/BF16, MB16/32, and seeds 0/1. All 32 rows
pass forward and all-input-gradient gates against the CPU oracle, using
the preserved [source-567acb.json](source-567acb.json). The medians below
aggregate each seed's median of 20 synchronized samples.

| Batch | Dtype | MB16 forward ms | MB16 forward+backward ms | MB32 forward+backward ms |
| --- | --- | ---: | ---: | ---: |
| 4 | FP32 | 28.267 | 48.319 | 49.780 |
| 4 | BF16 | 24.913 | 45.007 | 46.404 |
| 8 | FP32 | 55.912 | 95.651 | 98.499 |
| 8 | BF16 | 49.362 | 89.098 | 91.952 |

MB16 wins forward+backward in every seed and bucket, with within-row CV at
most 0.238% and seed-median spread at most 0.40%. This supports BT128/MB16
for the measured v4 endpoints. Upstream does not compile on v4, so these
are absolute candidate measurements with no upstream speedup claim.
Batch-8 FP32 MB16 forward at seed 0 has five samples at 87-89 ms versus
approximately 55.9 ms otherwise; seed 1 is stable. The selected tile has
repeatable forward+backward latency, but forward tail latency is not uniformly
stable. No samples were removed.

`v5e-cachecheck.jsonl` preserves a cache miss (four rows, MB16/32) followed
by fresh validation of the cached MB32 tile (two rows). All gates pass at
`[1,256,1,128]`, FP32, seed 0. The hit recompiles forward and forward+backward
in 1.44 and 2.71 seconds and produces new timing samples. Its objective
median is 0.336 ms, replacing the earlier 0.311 ms measurement. Each row
has only three samples; this checks cache behavior and supplies no tile-tuning
claim. `source_result_file` preserves the two original JSONL identities.
[v5e-cachecheck-cache.json](v5e-cachecheck-cache.json) retains the final cache
entry, including its exact context key and original results path.

The cache-check source is [source-6e915d.json](source-6e915d.json), SHA256
`6e915d62c411aebb5dd658dddc893ec6edff4118c41aed7d66b6f8361bfc4c9e`.
Its task bundle was retrieved by exact launch-request bundle ID, checked
against the bundle SHA256, and reduced to the kernel Python sources,
benchmark, and license notices. The reconstructed source hash matches all
six cache-check rows; the manifest records the bundle URI and generation.

To match the v5 full sweeps, pass both `--shape 4,4096,6,128` and
`--shape 8,4096,6,128`, the single hardware-specific seed in the table,
both dtypes, BT128/256, MB16/32, and `--repetitions 20`.
For confirmation, restrict to `--bt 128` and specify
`--implementation candidate --implementation upstream`.
For the v4 representative sweep, use both shapes and dtypes,
`--implementation candidate --bt 128 --mb 16 --mb 32 --seed 0 --seed 1`,
and `--repetitions 20`, without a scoped-VMEM override.
