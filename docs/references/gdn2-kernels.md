# GDN-2 TPU kernel harness

The experimental GDN-2 implementation lives in
`lib/levanter/src/levanter/kernels/pallas/gdn2/`. It contains compute kernels,
an independent token-serial reference, and a correctness-gated benchmark.
It is not integrated into a model or distributed training API.

## Sources and arithmetic

`upstream/` vendors four compute files from
[Atomic Ops](https://github.com/Akseleu-J/atomic-ops/tree/a9bd7c5315907d23ccf1ce6b033d46a196566888),
revision `a9bd7c5315907d23ccf1ce6b033d46a196566888`. The package's `LICENSE`
and `NOTICE` retain the MIT attribution and describe local changes.
The baseline retains upstream clipping and non-finite replacement.
`wy_eps` controls solve damping. The benchmark disables damping in both
implementations by setting `wy_eps=0`, so damping is not an experimental
difference in these comparisons.

`candidate/` removes clipping, non-finite replacement, and solve damping.
Off-diagonal score blocks and their gradients use boundary-centered matrix
products centered on the cumulative log-decay at the last token of the
earlier block. Diagonal score blocks mask before exponentiation and feature
reduction. These score constructions keep exponent arguments nonpositive
for finite log-decays `g <= 0`, without clipping valid decay values.
The benchmark chooses a feature-last score layout on v5e/v5p/v6e and a
feature-first layout on v4. The v4 compiler rejects trailing-feature
reductions of rank-three tiles with an unsupported sublane gather.
Transposing the full intermediate avoids that error but exceeds v4 VMEM;
the v4 candidate instead constructs the diagonal tile feature-first.
On v4, the candidate passes the output, final-state, and all-seven-gradient
hardware gates at `[1,256,1,128]`, seed 0, decay 0.01, for BT128, MB16/32,
and FP32/BF16 ([small-shape rows](../reports/data/gdn2-tpu-20260922/v4-bwdtiles.jsonl)).
BT256 backward requires 40.88–40.97 MiB of VMEM, exceeding v4's 16 MiB limit.
The same candidate BT128 tile/dtype combinations pass at `[1,512,1,128]`,
seed 0, with `--decay 0.5 --decay 5`, covering stronger forgetting
([stress rows](../reports/data/gdn2-tpu-20260922/v4-forgetting.jsonl)).
The v4 representative sweep also passes all gates at `[B,4096,6,128]`
for batches 4/8, FP32/BF16, BT128, MB16/32, and seeds 0/1
(32 accepted rows; see the
[measurement report](../reports/data/gdn2-tpu-20260922/README.md)).

For each token, the reference computes the following recurrence. `g` is
per-feature log-decay, and `*` denotes elementwise multiplication.

```text
S_decay = diag(exp(g)) @ S
residual = w * v - (b * k)^T @ S_decay
S = S_decay + k @ residual^T
output = scale * q^T @ S
```

`q`, `k`, `v`, `w`, `b`, and `g` have shape `[B, L, H, D]`;
the initial state has shape `[B, H, D, D]` and must be FP32. Log-decays
must be finite and nonpositive. Kernel inputs must be local
to one device, `D=128`, and `L` must be divisible by `BT`.
`BT` is the token chunk length, `BC=BT/2` is its score/solve subblock
length, and `MB` is the triangular solver's microblock size. The current
solve requires `BC` divisible by `MB`.
The harness covers FP32 and BF16 inputs, FP32 outputs and state, and
gradients with respect to all six token inputs and the initial state.

## Correctness and timing

From the repository root on an allocated TPU host, use the Levanter
workspace environment with TPU dependencies and its CPU backend enabled:

```bash
JAX_PLATFORMS=tpu,cpu uv run --package marin-levanter --extra tpu \
  python lib/levanter/scripts/bench/bench_atomic_gdn2.py \
  --reference-device cpu --shape 4,4096,6,128 \
  --dtype float32 --dtype bfloat16 \
  --bt 128 --bt 256 --mb 16 --mb 32 --seed 0 --seed 1 \
  --repetitions 20
```

For v5e and v5p, set
`LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000` before starting
the process. For v6e, use
`LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304`.
Do not set this override on v4.
On v4, replace the two `--bt` options above with
`--implementation candidate --bt 128`; the upstream path and candidate
BT256 backward do not compile.

The CPU oracle receives the exact TPU-generated inputs and output/state
cotangents. The harness also records the TPU reference's deviations from
the CPU oracle. The token-serial TPU reference has exceeded the FP32
parity threshold on otherwise passing kernel cases; use the CPU oracle
for acceptance. FP32 checks use `atol=rtol=1e-4`; BF16 checks use
`atol=rtol=1e-2`. No tolerance override is provided.

Each implementation's output, final state, and all seven input gradients
are compared directly against the CPU oracle. Both value and gradient
checks must pass before either
forward or forward+backward receives timing samples. Compilation and
first-execution latency are reported separately. Timing uses synchronized
single-device calls and includes dispatch overhead. Inspect the recorded
sample distribution and repeat comparisons before choosing tiles.

JSONL results include the source SHA256, upstream/candidate identity,
hardware, JAX version, environment flags, tile sizes, input seed, error
metrics, and individual timing samples. Files default to `IRIS_OUTPUT_DIR`
when running on Iris, or `/tmp` otherwise, with name `gdn2-<timestamp>.jsonl`.
Use `--output <path>` to choose a new file explicitly. Iris task-output archives have
limited retention; preserve accepted measurement artifacts before expiry.
The [2026-09-22 measurement artifacts](../reports/data/gdn2-tpu-20260922/README.md)
retain representative-shape samples and numerical error summaries.

`gdn2.tuned_block_sizes.select_kernel_config` selects candidate
BT128/BC64/MB16 for JAX device kinds `TPU v4`, `TPU v5 lite`, and
`TPU v5`, FP32/BF16, and local shapes `[B,4096,6,128]` with
`4 <= B <= 8`. For `TPU v6 lite` (v6e), it selects BT128/BC64/MB32
in the same dtype/shape bucket. The returned configuration uses
`ScoreLayout.FEATURE_FIRST` on v4 and `ScoreLayout.FEATURE_LAST` on v5e/v5p/v6e.
Batches 4 and 8 passed hardware correctness gates and
repeated timing comparisons; batches 5–7 interpolate between those
endpoints and remain unmeasured. Pass an explicit
`fallback=KernelConfig(...)` for unmatched devices, dtypes, or shapes.
This experimental lookup does not validate inputs, compile kernels,
change backends, or configure the runtime. The measured environment used
JAX 0.11.1. The v5e/v5p runs set both `--xla_tpu_scoped_vmem_limit_kib=50000` and
`--xla_tpu_use_enhanced_launch_barrier=true` in `LIBTPU_INIT_ARGS`.
The v4 runs did not use the VMEM override.
The v6e runs used a 98304 KiB scoped-VMEM limit and the enhanced launch barrier.

On v6e, BT128/MB32 wins both timing modes at both measured batch endpoints
and dtypes in the full sweep and reversed-order, second-seed confirmation.
Confirmation forward+backward medians are 11.652/11.649 ms at batch 4
(FP32/BF16) and 22.510/22.444 ms at batch 8, or 3.12–3.22x faster than
upstream's fastest tested tile (also BT128/MB32). Selected-tile medians
differ by at most 0.22% between runs. All forward, final-state, and
seven-input-gradient CPU gates pass, including separate strong-forgetting
checks at `[1,512,1,128]`, both dtypes, all four tiles, and decays 0.5/5.
The [row-level report](../reports/data/gdn2-tpu-20260922/README.md) retains
outliers in other tiles; the static selection does not guarantee tail latency.

In the v4 representative sweep, MB16 has approximately 3% lower median
forward+backward latency than MB32 for every batch-4/8, FP32/BF16,
seed-0/1 combination at `[B,4096,6,128]`. The median of the two per-seed
MB16 medians is:

| Batch | FP32 forward+backward (ms) | BF16 forward+backward (ms) |
| --- | ---: | ---: |
| 4 | 48.319 | 45.007 |
| 8 | 95.651 | 89.098 |

The batch-8 FP32 seed-0 MB16 forward row has five of 20 samples at
87–89 ms, versus approximately 55.9 ms for ordinary samples; seed 1
does not show these spikes. The selection does not establish stable tail
latency. The upstream baseline does not compile on v4, so these results
provide no v4 candidate-versus-upstream speedup.

Add `--tuning-cache /path/to/gdn2-tiles.json` to opt into a persistent,
single-writer benchmark tile cache. Without this option, the harness runs
the full requested tile grid. A cache miss measures that bounded grid and
stores the accepted tile with the lowest median forward+backward latency.
A hit reruns the complete value/gradient gate and both timing modes for
the cached tile. Only accepted forward and forward+backward measurements
for that implementation skip the remaining tiles.
If the hit fails, its failure rows remain in the new JSONL and the harness
tries every remaining tile. The run still exits with failure, even if an
alternative passes. A context with no accepted tile is removed from the
cache. Cached timings are never copied into a new measurement.

Cache keys include the source digest, JAX version, device kind, shape,
dtype, implementation, recorded backend/XLA flags, requested tile set,
seed, decay, oracle backend, tolerances, warmup/repetition counts, and
forward+backward objective. Changing any of these produces a miss.
The [hardware cache check](../reports/data/gdn2-tpu-20260922/README.md) has
six accepted rows covering a miss sweep and a freshly validated hit.
This validates successful miss/hit behavior; it does not establish a
performance improvement. This opt-in benchmark cache is separate from the static
`select_kernel_config` lookup.

`lib/levanter/scripts/bench/probe_atomic_gdn2.py` separately compiles forward stages and small
layout primitives. Its successful execution does not establish numerical
correctness or performance. CPU interpretation tests are in
`lib/levanter/tests/kernels/test_atomic_gdn2.py`.

Invoke the diagnostic in the same TPU environment with
`python lib/levanter/scripts/bench/probe_atomic_gdn2.py`.

## Analytical performance limits

The candidate's diagonal score construction and its feature-last backward
form rank-three decay tensors and reduce over features or token pairs.
For `BT=128`, `BC=64`, and `[B,4096,6,128]`, the source-level counts are:

```text
chunk_tiles = B * H * L / BT
diagonal_exp_elements_forward = chunk_tiles * 2 * BC * BC * D
token_elements = B * L * H * D
state_elements = B * H * D * D
minimum_forward_io_bytes = (6 * input_bytes + 4) * token_elements
                           + 8 * state_elements
```

Here `input_bytes` is 4 for FP32 and 2 for BF16. The I/O formula counts six
token inputs, one FP32 output, and FP32 initial and final states, each once.
It excludes intermediates, copies, and spills.
At batch 4, forward has 805,306,368 diagonal exponent tensor elements and
minimum I/O of 355 MB for FP32 inputs or 204 MB for BF16 inputs (decimal MB).
Batch 8 doubles those counts. Feature-last backward constructs the same
number of diagonal exponent elements again. These are logical source
counts including masked positions, before compiler optimization; they do
not measure instructions, HBM traffic, or stage latency.

Each FP32 `[64,64,128]` diagonal tensor occupies 2 MiB. BT256 increases
that tensor to 8 MiB and doubles total diagonal pair work at fixed sequence
length. The off-diagonal score block instead uses two `[64,128]` exponent
tensors followed by matrix products. Kernel-local tensor volume does not
imply an equivalent HBM transfer.

The source also casts token inputs to FP32, requests `HIGHEST` matmul
precision, and retains 32 dependent inter-chunk scan steps at BT128.
Consequently, BF16 input storage alone does not give native-BF16 matmul
throughput. Diagonal exp/reduction work, small high-precision matrix
products, and scan dependencies are candidates for profiling. The accepted
end-to-end measurements do not establish which stage dominates or whether
HBM bandwidth is saturated.
