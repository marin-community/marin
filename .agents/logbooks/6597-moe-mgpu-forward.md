# 6597 MoE MGPU Forward Logbook

## 2026-07-02 source-push inbox diagnostic decomposition

Patch added a non-production diagnostic runner for the source-push inbox kernel:

- `lib/levanter/scripts/bench/bench_source_push_inbox_diagnostics.py`
- Diagnostic variants in `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`:
  - `full`
  - `semaphore_only`
  - `copy_release_only`
  - `compute_only_local`
  - `store_zero`
  - `wgmma_tiny_output`

Production profile/API remains unchanged; diagnostics are exposed only through the separate bench script and
`run_source_push_inbox_diagnostic`.

H100 runs on `cw-us-east-02a`, stable profile `hopper_source_push_inbox_rough_balanced_216`, synthetic-block input,
`warmup=1`, `steps=3`, `repeat_runs=12`, `--separate-compile`, `--no-progress-events`.

First full diagnostic job:

- Job: `/dlwh/source-push-inbox-diagnostics-6a3a13f53-20260702-165511`
- Result: succeeded, but `wgmma_tiny_output` failed to lower because scalar accumulator store sliced a `(1, 1)`
  fragment, while Mosaic required slice shapes to be multiples of the accumulator tile shape.

Second full diagnostic job:

- Job: `/dlwh/source-push-inbox-diagnostics-6a3a13f53-20260702-170006`
- Result: succeeded. Five variants ran; `wgmma_tiny_output` still failed because accumulator slicing lowered to
  unsupported `dynamic_slice` under `LoweringSemantics.Lane`.

Medians from `/dlwh/source-push-inbox-diagnostics-6a3a13f53-20260702-170006`:

| variant | median steady state | median W13 TFLOP/s/rank | median send GB/s/rank | error rows |
| --- | ---: | ---: | ---: | ---: |
| `full` | 8.2915 ms | 219.82 | 85.87 | 0 |
| `semaphore_only` | 0.1967 ms | 9268.23 | 3620.40 | 0 |
| `copy_release_only` | 5.4869 ms | 332.18 | 129.76 | 0 |
| `compute_only_local` | 6.5068 ms | 280.11 | 109.42 | 0 |
| `store_zero` | 5.5957 ms | 325.72 | 127.24 | 0 |

Narrow fixed tiny-output job:

- Job: `/dlwh/source-push-inbox-diag-tiny-6a3a13f53-20260702-170402`
- Fix: store a static `block_m x 8` accumulator slice per N tile instead of a scalar.
- Result: succeeded.

Medians from `/dlwh/source-push-inbox-diag-tiny-6a3a13f53-20260702-170402`:

| variant | median steady state | median W13 TFLOP/s/rank | median send GB/s/rank | error rows |
| --- | ---: | ---: | ---: | ---: |
| `wgmma_tiny_output` | 7.9744 ms | 228.56 | 89.28 | 0 |

Initial tax reading:

- Semaphore/queue handoff alone is small: `0.1967 ms`.
- Copy plus release path dominates the stripped non-compute kernel: `copy_release_only - semaphore_only = 5.2903 ms`.
- Full zero-store over copy-release adds only `0.1087 ms`, but this is a zero-fill diagnostic and may not equal real bf16
  hidden stores.
- WGMMA plus activation with tiny output adds `2.4875 ms` over copy-release.
- Full hidden store tax relative to tiny output is about `0.3171 ms`.
- The full path median is `8.2915 ms`, comparable to recent stable source-push medians and faster than the
  `9.692 ms` ring-prologue + Pallas-W13 hybrid estimate.
