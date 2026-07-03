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

## 2026-07-03 invertible source-push planner phase 1

Added the first host-side planner for the revised invertible source-push contract:

- Commit Hash: `cf5df315b`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_plan.py`
  - `lib/levanter/tests/grug/test_source_push_plan.py`
- Scope:
  - Builds source-major queue metadata from real `selected_experts` and `combine_weights`.
  - Preserves source-owned inverse metadata:
    `assignment_id -> token_id, route_slot, combine_weight`.
  - Uses transport order `[src, dst_ordinal, entry, row_in_block]`.
  - Stores compact kernel metadata fields:
    `src_rank`, `local_expert`, `local_row_start_within_src_expert`, `valid_rows`.
  - Derives destination-side expert-major offsets from accepted counts:
    `expert_base[dst, expert] + src_base_by_expert[dst, src, expert] + local_row_start + row`.
  - Uses existing `_clip_receiver_group_sizes` for deterministic receiver-capacity clipping, so accepted/dropped
    assignment sets match the current EP reference policy.
  - Adds reference helpers for packing source tokens into queue order and scattering returned rows into a
    deterministic `[source, token, route_slot, D]` route buffer.
  - Adds row accounting for useful rows, rounded rows, live entries, row efficiency, masked row fraction, and drops.
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - `20 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed
- Interpretation:
  - This is Phase 1 only: planner/inverse-map correctness and CPU/JAX checks.
  - The existing source-push inbox kernel is intentionally unchanged because its current `send_meta` field 2 is a
    destination row start, while the new invertible plan makes field 2 the source-local row start and computes the
    destination expert-major offset from compact count-derived bases.
- Next action:
  - Adapt the source-push W13 kernel to consume `SourcePushPlan` metadata and store W13/SwiGLU directly into
    expert-major rows.

## 2026-07-03 metadata-row-start W13 store

Changed the current source-push inbox W13 store to use metadata row starts instead of queue-order rows:

- Commit Hash: `f158cc2c6`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - `_store_hidden` and `_store_zero_hidden` now use `recv_meta[..., 2]` for the hidden row start.
  - CPU validation/reference helpers now use `send_meta[..., 2]`.
  - Added a regression test where metadata row start and queue-order row start intentionally differ.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - `21 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed

H100 validation/perf run:

- Job: `/dlwh/source-push-inbox-metadata-offset-f158cc2c6-20260703-0026`
- Cluster: `cw-us-east-02a`
- Command:

  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-inbox-metadata-offset-f158cc2c6-20260703-0026 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 3600s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_inbox.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --warmup 1 --steps 3 --repeat-runs 12 --check \
    --jsonl scratch/source_push_inbox_metadata_offset_f158cc2c6.jsonl
  ```

- Result: `JOB_STATE_SUCCEEDED`, exit code 0.
- Median rows from 12 repeats:

| metric | median | min | max |
| --- | ---: | ---: | ---: |
| `steady_state_time` | `0.008677s` | `0.008287s` | `0.014664s` |
| `w13_tflops_per_rank` | `210.05` | `124.29` | `219.93` |
| `send_gbps_per_rank` | `82.05` | `48.55` | `85.91` |
| `max_abs_diff` | `0.250900` | `0.250900` | `0.250900` |
| `hidden_mean_abs_diff` | `0.021131` | `0.021131` | `0.021131` |
| `hidden_unwritten_max_abs` | `0.001564` | `0.001564` | `0.001564` |
| `metadata_mismatches` | `0` | `0` | `0` |

Queue stats:

- `dropped_rows_total=0`
- `dropped_entries_total=0`
- `valid_rows_per_rank_mean=131072`
- `rounded_rows_per_rank_mean=139056`
- `masked_row_fraction=0.0574157`

Interpretation:

- The metadata-row-start W13 store lowers and runs on H100.
- This run is slower than the earlier stable source-push inbox median (`8.677 ms` vs `8.2915 ms`) and should not be
  treated as a performance win.
- The benchmark still uses the old padded destination row-start metadata, not the new exact `SourcePushPlan`
  local-row-start plus `expert_base/src_base` contract.
- Exact `SourcePushPlan` expert-major stores still need masked tail-row stores; writing a full `block_m` at exact
  unrounded row starts would corrupt the next contiguous source/expert slice.
- The nonzero synthetic-reference diff needs a like-for-like baseline check before using it as a correctness gate; this
  benchmark reports it but does not fail on it.
- Next action: implement exact expert-major stores behind a separate path that passes count-derived bases and masks tail
  rows, then validate on a small H100 shape before re-running the target profile.

## 2026-07-03 source-plan W13 storage path

Adapted the source-push inbox benchmark to accept real `SourcePushPlan`-packed tokens and source-owned inverse metadata.

- Commits:
  - `d3fbb4184`: added source-plan input mode and destination-side `expert_base + src_base + local_row_start` W13 store path.
  - `cf3905547`: tried explicit WGMMA-layout row masks for exact tail stores.
  - `095781e54`: switched to source-padded expert-major slices because masked GMEM stores are unsupported in Lane lowering.
  - `d612a2af1`: added `--git-sha` metadata to the source-push benchmark rows.
  - `b3bdb1f76`: precomputed source-padded expert-major row starts on the host to remove hot-kernel base lookups.
- Local verification after final patch:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - `24 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed

H100 lowering/correctness sequence:

| job | commit | result |
| --- | --- | --- |
| `/dlwh/source-push-plan-exact-smoke-d3fbb4184-20260703-0038` | `d3fbb4184` | Failed bad smoke config: WGMMA `m=8` unsupported; WGMMA requires `m` multiple of 64. |
| `/dlwh/source-push-plan-exact-smoke-d3fbb4184-20260703-0044` | `d3fbb4184` | Reached exact masked-store path; failed iota layout inference for row mask. |
| `/dlwh/source-push-plan-exact-smoke-cf3905547-20260703-0110` | `cf3905547` | Row-mask iota fixed, but failed on `masked_swap`: masked GMEM stores are not implemented for Lane lowering in warpgroup code. |
| `/dlwh/source-push-plan-padded-smoke-095781e54-20260703-0101` | `095781e54` | Succeeded with source-padded expert-major slices; no drops, `metadata_mismatches=0`, `max_abs_diff=2.40065e-4`, `hidden_unwritten_max_abs=0`. |
| `/dlwh/source-push-plan-precomputed-smoke-b3bdb1f76-20260703-0117` | `b3bdb1f76` | Succeeded with precomputed source-padded row starts; no drops, `metadata_mismatches=0`, `max_abs_diff=2.40065e-4`, `hidden_unwritten_max_abs=0`. |

Target rough-balanced H100 comparison, stable profile
`hopper_source_push_inbox_rough_balanced_216` (`T/rank=32768`, `K=4`, `D=2560`, `I=1280`, `EP=8`,
`block_m=64`, `block_n=128`, `block_k=128`, `entries_per_rank=288`, `inbox_slots=12`,
`send_pipeline_depth=1`, `n_groups_per_job=2`, `capacity_factor=1.25`, 48 repeats):

| input mode / row-start mode | job | median time | median rounded TFLOP/s/rank | median useful TFLOP/s/rank | drops |
| --- | --- | ---: | ---: | ---: | ---: |
| `source_push_plan` / `local_row_start_source_padded` | `/dlwh/source-push-plan-target-rough-d612a2af1-20260703-0108` | `8.762 ms` | `208.02` | `196.07` | 0 |
| `compact_routing` / precomputed row start | `/dlwh/source-push-compact-target-rough-d612a2af1-20260703-0112` | `8.582 ms` | `212.37` | `200.18` | 0 |
| `source_push_plan` / `source_padded_row_start` | `/dlwh/source-push-plan-target-precomputed-b3bdb1f76-20260703-0120` | `8.441 ms` | `215.94` | `203.54` | 0 |

Common row accounting for target runs:

- `plan_useful_rows_total=1048576`
- `plan_padded_rows_total=1112448`
- `plan_padded_rows_per_rank_mean=139056`
- `plan_row_efficiency=0.9425843`
- `plan_masked_row_fraction=0.0574157`
- `live_entries_total=17382`
- `tail_entries_total=2032`
- `zero_entries_skipped=1050`

Interpretation:

- Exact contiguous per-source slices would require tail masking, but Mosaic currently cannot lower masked GMEM stores
  from this Lane/WGMMA path (`masked_swap` unsupported). A separate tail kernel or structural split is needed for exact
  contiguous hidden rows.
- Source-padded expert-major slices preserve the source-owned plan/inverse contract and avoid overlapping tail writes:
  invalid packed rows are zero, full-tile stores are unique, and W2 can ignore source padding using the same plan metadata.
- Computing `expert_base + src_base + local_row_start` in the hot W13 store path costs about `0.18 ms` on the target
  profile. Precomputing the source-padded row start before launch recovers the compact fast path and meets the useful
  throughput bar (`203.54 TFLOP/s/rank`, median).
- Current best production-relevant W13 source-plan result is commit `b3bdb1f76`, job
  `/dlwh/source-push-plan-target-precomputed-b3bdb1f76-20260703-0120`.

Next action:

- Use `source_padded_row_start` as the Phase 2 W13 layout for integration.
- Carry the richer source-owned inverse metadata forward into W2 return/combine; exact contiguous source slices can be
  revisited only if a supported tail-store path or separate compaction kernel is added.

## 2026-07-03 W2 return/combine reference contract

Added the Phase 3/4 CPU/JAX reference contract for source-push W2 return and deterministic combine.

- Commit Hash: `62ff3394b`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_plan.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`
  - `lib/levanter/tests/grug/test_source_push_plan.py`
- Change:
  - Added `source_push_w2_return(hidden_expert_major, w_down, plan, ...)`, which reads destination expert-major hidden
    rows and writes W2 output back into the source-visible queue layout `[src, dst_ordinal, entry, row, D]`.
  - Added `source_push_combine(return_y, plan)`, which scatters returned route rows through the source-owned inverse
    metadata and sums fixed route slots in `[T, K, D]` order.
  - Promoted the source-padded expert-major row-base derivation into `source_push_plan.py` so W13 and W2 references use
    the same padded row layout.
  - Kept the exact contiguous plan bases supported as the default reference, while allowing explicit source-padded bases
    for the current production-relevant W13 layout.
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - `26 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed
- Interpretation:
  - The source-owned inverse metadata now proves the full identity round trip at the reference level:
    queue row -> expert-major hidden row -> W2 return queue row -> `(token_id, route_slot)` combine.
  - This is not yet a Pallas W2/return kernel and does not include a new H100 benchmark row.
- Next action:
  - Implement Kernel B against this reference: destination-side W2 over source-padded expert-major hidden and return
    writes into source queue slots, then validate on a small H100 shape.
