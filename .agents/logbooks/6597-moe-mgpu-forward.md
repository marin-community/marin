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

## 2026-07-03 Destination-local W2 return kernel harness

Added a package-private Phase 3 W2-return harness that consumes source-padded expert-major hidden rows and compact
`recv_meta`, computes W2 with a Lane-lowered Mosaic WGMMA kernel, and writes destination-local return blocks indexed by
`(dst_rank, recv_src_ordinal, entry, row, d)`.

- Commit Hashes:
  - `8c7f40290`: W2 return kernel harness and bench wrapper.
  - `f3a9dc5e7`: synthetic expert-major hidden mode for target W2-only timing.
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py`
  - `lib/levanter/scripts/bench/bench_source_push_w2_return.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - `32 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed
- H100 smoke:
  - Job: `/dlwh/source-push-w2-return-smoke-8c7f40290-20260703-0900`
  - Config: `EP=8`, `T/rank=16`, `K=2`, `D=128`, `I=128`, `E_local=2`, `block_m=64`,
    `block_k=64`, `block_n=64`, `entries_per_rank=2`, `hidden_input_mode=w13_reference`, `check=true`.
  - Result: succeeded, `steady_state_time=0.000379963s`, `w2_tflops_per_rank=0.0883`,
    `max_abs_diff=0.00746334`, `source_queue_max_abs_diff=0.00746334`, no drops/errors.
- Target W2-only timing:
  - Job: `/dlwh/source-push-w2-return-target-f3a9dc5e7-20260703-0908`
  - Command: `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_w2_return.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --hidden-input-mode synthetic --no-check --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events --git-sha f3a9dc5e7 --jsonl scratch/source_push_w2_return_target_f3a9dc5e7.jsonl`
  - Config: target rough-balanced profile, `capacity_factor=1.25`, source-padded row starts, synthetic
    expert-major hidden to isolate W2 kernel timing.
  - Rows: `rounded_rows_per_rank=139056`, `useful_rows_per_rank=131072`, `row_efficiency=0.942584`,
    `masked_row_fraction=0.057416`, `dropped_routes=0`.
  - Repeat times: `[2.898, 2.899, 4.681, 2.917, 2.899] ms`.
  - Median: `steady_state_time=2.899 ms`, `w2_tflops_per_rank=314.35`, `return_gbps_per_rank=368.38`.
- Interpretation:
  - The W2 math path itself is fast enough at the target layout: ~`314 TFLOP/s/rank` rounded-row W2 throughput is close
    to the prior isolated Pallas W13 scale and much faster than the current source-push+W13 stage time.
  - This is not yet full Kernel B from the spec: the kernel writes destination-local return blocks, and the
    source-visible reorder is currently a host-side validation adapter. The missing structural step is remote
    destination-to-source return writes, then full source-side deterministic combine in the actual forward path.
  - The target timing used synthetic hidden and `--no-check`; correctness was validated on the small H100 smoke with
    W13-reference hidden and on CPU/JAX reference tests.
- Next action:
  - Add the remote/source-visible return stage for W2 output, either as a separate return-copy kernel or by extending
    the W2 kernel if Mosaic supports remote stores from this Lane/WGMMA path.

## 2026-07-03 Source-visible W2 return-copy kernel

Added a separate MGPU return-copy kernel after destination-local W2. The new kernel reads each destination's
`[src_ordinal, entry, block_m, D]` W2 output and writes it back to the owning source rank's return queue at
`[dst_ordinal, entry, block_m, D]` using remote GMEM stores through SMEM. This keeps W2 math and return transport
separate for now; it does not fuse remote stores into the WGMMA kernel.

- Commit Hash: `971ac17c2`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py`
  - `lib/levanter/scripts/bench/bench_source_push_w2_return.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - `33 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed
- H100 correctness smoke:
  - Job: `/dlwh/source-push-w2-return-copy-smoke-971ac17c2-20260703-0918`
  - Config: `EP=8`, `T/rank=16`, `K=2`, `D=128`, `I=128`, `E_local=2`, `block_m=64`,
    `block_k=64`, `block_n=64`, `hidden_input_mode=w13_reference`, `copy_to_source=true`, `check=true`.
  - Result: succeeded, `steady_state_time=0.00231898s`, `max_abs_diff=0.00789702`,
    `source_queue_max_abs_diff=0.00789702`, no drops/errors.
- Target W2 + return-copy timing:
  - Job: `/dlwh/source-push-w2-return-copy-target-971ac17c2-20260703-0921`
  - Command: `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_w2_return.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --hidden-input-mode synthetic --copy-to-source --no-check --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events --git-sha 971ac17c2 --jsonl scratch/source_push_w2_return_copy_target_971ac17c2.jsonl`
  - Rows: `rounded_rows_per_rank=139056`, `useful_rows_per_rank=131072`, `row_efficiency=0.942584`,
    `masked_row_fraction=0.057416`, `dropped_routes=0`.
  - Repeat times: `[6.027, 6.021, 5.935, 5.890, 5.911] ms`.
  - Median: `steady_state_time=5.935 ms`, `w2_tflops_per_rank=153.55`, `return_gbps_per_rank=179.94`.
- Comparison:

| path | target median | rounded TFLOP/s/rank | note |
| --- | ---: | ---: | --- |
| W2 destination-local only (`f3a9dc5e7`) | `2.899 ms` | `314.35` | no source-visible return |
| W2 + separate return-copy (`971ac17c2`) | `5.935 ms` | `153.55` | writes source-owned return queue |

- Interpretation:
  - Separate remote return-copy costs about `3.036 ms` at the target rough-balanced layout.
  - The source-visible return path is now correct on a small H100 smoke, but a separate full-buffer copy roughly halves
    effective W2 throughput. Fusing return writes into the W2 kernel or pipelining copy with downstream combine is likely
    necessary if this tax shows up in full forward.
  - Estimated source-push W13 (`8.441 ms`) + W2+return-copy (`5.935 ms`) is about `14.376 ms` before source combine.
- Next action:
  - Wire the source-visible return queue into deterministic source combine, then benchmark W13 + W2 + return + combine
    end to end. In parallel, investigate whether W2 can directly remote-store output tiles without the separate
    GMEM-to-SMEM-to-remote-GMEM return-copy kernel.

## 2026-07-03 Source-side deterministic combine harness

Added a package-private Phase 4 source-combine harness. The kernel consumes the source-visible return queue
`[src, dst_ordinal, entry, row, D]` plus source-owned inverse metadata, writes a dense route buffer `[T, K, D]`,
and sums route slots in fixed order to produce source-local `[T, D]` outputs. The Pallas kernel uses a route-inverse
gather instead of scatter into an uninitialized output buffer, so dropped route slots write deterministic zero rows.

- Commit Hash: `22144d46b`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_combine.py`
  - `lib/levanter/scripts/bench/bench_source_push_combine.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q`
  - `29 passed, 11 warnings`
  - `./infra/pre-commit.py --changed-files --fix`
  - all checks passed
- H100 smoke, invalid config:
  - Job: `/dlwh/source-push-combine-smoke-22144d46b-20260703-0231`
  - Config: `EP=8`, `T/rank=64`, `K=2`, `D=128`, `I=128`, `E_local=2`, `block_m=64`,
    `block_k=64`, `block_n=64`, `entries_per_rank=2`, `check=true`.
  - Result: failed during Mosaic lowering with
    `memref<64xbf16, strided<[1], offset: ?>> must have a number of elements that is a multiple of 128`.
  - Interpretation: bad smoke config for Lane-lowered bf16 vector loads; target profile uses `block_n=128`.
- H100 correctness smoke:
  - Job: `/dlwh/source-push-combine-smoke-22144d46b-20260703-0234`
  - Config: `EP=8`, `T/rank=64`, `K=2`, `D=128`, `I=128`, `E_local=2`, `block_m=64`,
    `block_k=64`, `block_n=128`, `entries_per_rank=2`, `check=true`.
  - Result: succeeded, `steady_state_time=0.000630316s`, `combine_gbps_per_rank=0.181953`,
    `max_abs_diff=0.0009765625`, `mean_abs_diff=0.00012967`, `dropped_routes=0`.
- Target rough-balanced combine timing:
  - Job: `/dlwh/source-push-combine-target-22144d46b-20260703-0237`
  - Command: `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_combine.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --no-check --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events --git-sha 22144d46b --jsonl scratch/source_push_combine_target_22144d46b.jsonl`
  - Rows: `rounded_rows_per_rank=139056`, `useful_rows_per_rank=131072`, `row_efficiency=0.942584`,
    `masked_row_fraction=0.057416`, `dropped_routes=0`.
  - Repeat times: `[1.335, 1.333, 1.331, 1.333, 1.336] ms`.
  - Median: `steady_state_time=1.333 ms`, `combine_gbps_per_rank=1635.95`.
- Current forward-stage estimate:

| path segment | target median |
| --- | ---: |
| source-push W13 stable profile (`216.949 TFLOP/s/rank`) | `~8.4 ms` |
| W2 + separate return-copy (`971ac17c2`) | `5.935 ms` |
| source combine (`22144d46b`) | `1.333 ms` |
| estimated W13 + W2 + return + combine | `~15.7 ms` |

- Interpretation:
  - Deterministic source combine is not the dominant remaining tax: the dense route-buffer gather and fixed-order sum
    costs about `1.33 ms` at the target rough-balanced shape.
  - The route buffer is large (`335,544,320` bf16 elements/rank), so this is mostly a memory-bandwidth tax. The measured
    model bandwidth is about `1.64 TB/s/rank` using the harness byte model.
  - The separate W2 return-copy tax (`~3.0 ms`) remains the bigger post-W13 structural cost.
- Next action:
  - Wire the three package-private pieces into a full source-push forward comparison path, then compare full forward
    against `ring`/`ragged_all_to_all` on a small H100 mesh before optimizing return-copy or fusing combine.

## 2026-07-03 03:49 - Full source-push forward harness and reference fix

Added the package-private full-forward harness and corrected a bad CPU/JAX oracle that made the first H100 smoke look
wrong.

- Commit Hashes:
  - `6438ad4f0`: full source-push forward harness chaining W13 -> W2 return-copy -> source combine.
  - `ac2a35f02`: single-JIT return-copy completion barrier experiment.
  - `bb457edfc`: diagnostic `staged_host_sync` execution mode.
  - `bc0c377c1`: fixed `source_push_w2_return` destination ordinal inverse and fp32 reference accumulation.
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_plan.py`
  - `lib/levanter/scripts/bench/bench_source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - Result: `42 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
  - Result: all checks passed.
- Debug sequence:
  - Initial full-forward smoke `/dlwh/source-push-forward-smoke-6438ad4f0-20260703-0304` completed but reported
    `max_abs_diff=0.732421875`, `mean_abs_diff=0.0743179`.
  - Initial target run `/dlwh/source-push-forward-target-6438ad4f0-20260703-0307` returned a structured
    `JaxRuntimeError: CUDA_ERROR_ILLEGAL_ADDRESS` row and no timing rows.
  - Barrier smoke `/dlwh/source-push-forward-barrier-smoke-ac2a35f02-20260703-1021` still reported
    `max_abs_diff=0.732421875`; staged smoke `/dlwh/source-push-forward-staged-smoke-bb457edfc-20260703-1025`
    also reported `max_abs_diff=0.732421875`.
  - Isolated same-shape W13 `/dlwh/source-push-w13-same-shape-bb457edfc-20260703-1028` was correct:
    `metadata_mismatches=0`, `hidden_max_abs_diff=0.000949293`, `hidden_unwritten_max_abs=0`.
  - Isolated same-shape W2+return `/dlwh/source-push-w2-return-same-shape-bb457edfc-20260703-1029` was correct:
    `max_abs_diff=0.00907660`, `source_queue_max_abs_diff=0.00907660`.
  - A W2+combine probe with reference hidden reproduced the full diff, which exposed that the plan-level W2 reference
    used `dst_ordinal(src, dst_ord, ep_size)` as the inverse of a source-local destination ordinal. That only works for
    `EP=2`; the correct inverse is `dst = (src + dst_ord) % ep_size`.
  - After `bc0c377c1`, full-forward smoke
    `/dlwh/source-push-forward-fixed-ref-smoke-bc0c377c1-20260703-1038` succeeded with
    `max_abs_diff=0.0078125`, `mean_abs_diff=0.000339662`, `dropped_routes=0`.
- Target full-forward benchmark commands:
  - Rough-balanced cf1.25:
    `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --no-check --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events --git-sha bc0c377c1 --jsonl scratch/source_push_forward_target_bc0c377c1.jsonl`
  - Balanced cf1.0:
    `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --routing balanced --capacity-factor 1.0 --no-check --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events --git-sha bc0c377c1 --jsonl scratch/source_push_forward_balanced_cf100_bc0c377c1.jsonl`
  - Balanced cf1.25:
    `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --routing balanced --capacity-factor 1.25 --no-check --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events --git-sha bc0c377c1 --jsonl scratch/source_push_forward_balanced_cf125_bc0c377c1.jsonl`
- Target full-forward results:

| routing | capacity | job | repeat times (ms) | median (ms) | rounded TFLOP/s/rank | useful TFLOP/s/rank | row efficiency | drops | CUDA illegal address logs |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | `1.0` | `/dlwh/source-push-forward-balanced-cf100-bc0c377c1-20260703-1043` | `[17.794, 17.422, 28.973, 17.313, 17.248]` | `17.422` | `147.91` | `147.91` | `1.000000` | 0 | 0 |
| balanced | `1.25` | `/dlwh/source-push-forward-balanced-cf125-bc0c377c1-20260703-1045` | `[17.596, 19.085, 25.202, 17.394, 17.345]` | `17.596` | `146.45` | `146.45` | `1.000000` | 0 | 0 |
| roughly_balanced | `1.25` | `/dlwh/source-push-forward-target-bc0c377c1-20260703-1039` | `[18.091, 17.918, 17.801, 17.961, 17.973]` | `17.961` | `152.22` | `143.48` | `0.942584` | 0 | 0 |

- Interpretation:
  - The previous large full-forward diff was a bad oracle, not a kernel error. `EP=3` now has a regression test that
    compares the plan-level W2 return reference against destination-local W2 plus source-queue reorder.
  - The target full-forward harness now runs at the three benchmark-gate routing/capacity rows without drops or
    `CUDA_ERROR_ILLEGAL_ADDRESS`.
  - Median full forward is about `17.4-18.0 ms`, worse than the prior segment estimate of `~15.7 ms`; the remaining
    gap is likely from composing all stages in one large graph plus the separate return-copy/source-combine memory tax.
  - This is still a package-private harness, not the final integrated `grug_moe` implementation.
- Next action:
  - Compare single-JIT full forward against the staged diagnostic and stage-sum timings on target to quantify graph
    composition overhead, then decide whether to keep full forward as three explicit stages or introduce a real
    producer/consumer return path.

## 2026-07-03 04:10 - Full-forward staged decomposition and return-barrier check

Added per-stage timing rows to the package-private full-forward `staged_host_sync` diagnostic and used it to split the
target rough-balanced cf1.25 path into W13, W2-return, and source-combine costs.

- Commit Hashes:
  - `8b988579d`: `staged_host_sync` now emits `stage_repeat` rows for `w13`, `w2_return`, and `combine`.
  - `1346a3f90`: removed the single-JIT return barrier as a diagnostic experiment.
  - `582406ee8`: reverted the no-barrier experiment after it failed at the target shape.
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_forward or bench_cli_imports or repro_wrapper'`
  - Result: `7 passed, 11 warnings`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_forward'`
  - Result: `4 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
  - Result: all checks passed.
- H100 staged decomposition:
  - Job: `/dlwh/source-push-forward-staged-breakdown-8b988579d-20260703-105631`
  - Command:
    `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --routing roughly_balanced --capacity-factor 1.25 --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 5 --no-check --progress-events --git-sha 8b988579d --jsonl scratch/source_push_forward_staged_breakdown_8b988579d_target.jsonl`
  - Result: succeeded, no drops.

| row | repeat times (ms) | median (ms) | median throughput |
| --- | ---: | ---: | ---: |
| total staged forward | `[16.137, 16.012, 16.152, 16.462, 16.086]` | `16.137` | `169.42` rounded / `159.69` useful TFLOP/s/rank |
| W13 source-push | `[8.615, 8.567, 8.660, 8.505, 8.549]` | `8.567` | `212.75` W13 TFLOP/s/rank |
| W2 return | `[6.083, 6.014, 6.062, 6.526, 6.097]` | `6.083` | `149.81` W2 TFLOP/s/rank |
| source combine | `[1.340, 1.334, 1.338, 1.342, 1.343]` | `1.340` | `1627.88` GB/s/rank |

- Same-SHA single-JIT comparison:
  - Job: `/dlwh/source-push-forward-singlejit-8b988579d-20260703-105955`
  - Result: succeeded, no drops.
  - Repeat times: `[18.126, 24.283, 26.983, 17.833, 17.902] ms`.
  - Median: `18.126 ms`, `150.83` rounded / `142.17` useful TFLOP/s/rank.
- No-barrier diagnostic:
  - Small checked smoke: `/dlwh/source-push-forward-nobarrier-smoke-1346a3f90-20260703-110343`
    - Result: succeeded, `max_abs_diff=0.0078125`, `mean_abs_diff=0.000339662`, `dropped_routes=0`.
  - Target no-barrier: `/dlwh/source-push-forward-nobarrier-target-1346a3f90-20260703-110642`
    - Result: structured `JaxRuntimeError` row during steady-state:
      `CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered [executable_name='jit_fn']`.
  - Decision: keep the single-JIT return barrier or an equivalent synchronization boundary for target-shape safety.
- Interpretation:
  - The source-push W13 stage is still close to the stable source-push inbox path (`8.57 ms` vs prior `~8.3-8.4 ms`).
  - W2-return costs about `6.08 ms`; source combine costs about `1.34 ms`.
  - The staged three-kernel path (`16.14 ms`) is about `2.0 ms` faster than the same-SHA single-JIT path median and
    avoids the single-JIT slow-tail repeats in this run.
  - Same-JIT no-barrier is unsafe at target shape even though a small smoke passes, so the one-graph path needs the
    explicit return barrier. The currently better production-relevant structure is staged/multi-kernel, not one giant
    JIT graph.
- Next action:
  - Treat `staged_host_sync` as the current full-forward performance baseline for the source-push chain.
  - If integrating behind an opt-in backend, prefer explicit W13 -> W2-return -> combine stage boundaries first, then
    optimize W2-return or replace the barrier with a real producer/consumer return synchronization only after profiling.

## 2026-07-03 04:21 - Real-input full-forward adapter

Moved the package-private full-forward harness one step closer to production integration by adding an adapter that accepts
real source-local MoE arrays instead of only synthetic benchmark inputs.

- Commit Hash: `dafbf3337`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - Added `make_source_push_forward_inputs(config, x, selected_experts, combine_weights, w_gate_up, w_down)`.
  - The adapter validates real array shapes, builds `SourcePushPlan`, packs source tokens into queue order, derives
    source-padded expert-major row bases, and preserves the provided expert weights.
  - The synthetic `make_source_push_forward_source_plan_inputs` benchmark builder now calls through this real-input
    adapter with `input_mode="source_push_plan"`.
  - Added a small no-drop CPU/JAX test that feeds real `[S,T,D]` tokens, real routing/combine weights, and real sharded
    W13/W2 weights, then compares the full source-push reference round trip against an independent naive MoE oracle.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
  - Result: `43 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
  - Result: all checks passed.
- H100 smoke:
  - Job: `/dlwh/source-push-forward-real-adapter-smoke-dafbf3337-20260703-1120`
  - Command:
    `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --ep-size 8 --tokens-per-rank 64 --hidden-dim 128 --intermediate-dim 128 --experts-per-rank 2 --topk 2 --capacity-factor 1.25 --entries-per-rank 2 --inbox-slots 1 --block-m 64 --block-k 64 --block-n 128 --n-group 1 --n-groups-per-job 1 --send-worker-programs-per-peer 1 --worker-programs-per-peer 8 --send-pipeline-depth 1 --routing balanced --execution-mode single_jit --warmup 0 --steps 1 --repeat-runs 1 --check --progress-events --git-sha dafbf3337 --jsonl scratch/source_push_forward_real_adapter_smoke_dafbf3337.jsonl`
  - Result: succeeded, `steady_state_time=0.00642532s`, `max_abs_diff=0.0078125`,
    `mean_abs_diff=0.000339662`, `dropped_routes=0`.
- Interpretation:
  - The full-forward source-push path no longer requires synthetic token/weight generators at its package-private API
    boundary. This is still not a public `grug_moe` backend, because `SourcePushPlan` is currently host-built and cannot
    be constructed inside the existing `shard_map` EP local function.
  - The next integration step is to decide where host-side plan construction lives for an opt-in backend, or to replace
    it with a device/JAX-compatible planner before wiring `MoeImplementation`.

## 2026-07-03 04:33 - Package-private callable forward API

Added a package-private callable full-forward API so callers can invoke the source-push path directly instead of going
through benchmark row runners.

- Commit Hash: `ee790d1de`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - Added `source_push_forward(config, x, selected_experts, combine_weights, w_gate_up, w_down, implementation=...)`.
  - Supported implementations:
    - `reference`: host/JAX reference using the same real-input adapter and `SourcePushPlan`.
    - `pallas_mgpu`: staged or single-JIT package-private Pallas path.
  - Shared the benchmark runner's sharded `device_put` path with the callable implementation to avoid drift.
  - Extended the real-input CPU/JAX test to call `implementation="reference"` and compare against the independent naive
    MoE oracle.
- Local verification:
  - Focused:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'real_inputs or source_push_forward'`
    - Result: `5 passed, 11 warnings`.
  - Full source-push subset:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q`
    - Result: `43 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 callable smoke:
  - First attempt: `/dlwh/source-push-forward-callable-smoke-ee790d1de-20260703-1133`
    - Result: Iris success but no JSON row because `python -` stdin was not part of the remote command.
  - Rerun: `/dlwh/source-push-forward-callable-smoke-ee790d1de-20260703-1139`
    - Result: succeeded.
    - Config: `EP=8`, `T/rank=64`, `K=2`, `D=128`, `I=128`, `E_local=2`, `block_m=64`,
      `block_k=64`, `block_n=128`, `entries_per_rank=2`, `execution_mode=staged_host_sync`.
    - Metrics: `max_abs_diff=0.0078125`, `mean_abs_diff=0.000339662`, `expected_dropped=0`,
      `observed_dropped=0`, output shape `[8, 64, 128]`.
- Interpretation:
  - The source-push full-forward path now has a real package-private API boundary that returns outputs and dropped-route
    count from real arrays. This is the callable primitive needed before a public opt-in MoE backend can be wired.
  - Public `grug_moe` integration is still not complete: the current callable expects explicit source-major arrays and
    builds `SourcePushPlan` on the host, while `moe_mlp` dispatch currently enters an EP shard-local function through
    `shard_map`.

## 2026-07-03 04:56 - Public EP forward comparison smoke

Added a small integration comparison harness that feeds the same production-like source-plan raw arrays into both the
package-private source-push forward callable and public `moe_mlp` EP backends.

- Commit Hashes:
  - `814624a71`: add raw source-plan arrays plus
    `lib/levanter/scripts/bench/bench_source_push_forward_public_compare.py`.
  - `13be43c8d`: use an explicit-axis mesh for the public `moe_mlp` baseline.
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/scripts/bench/bench_source_push_forward_public_compare.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - Added `SourcePushForwardRawInputs` and `make_source_push_forward_source_plan_raw_inputs(config)`.
  - The existing source-plan forward input builder now goes through the same real-array adapter used by the package-private
    callable API.
  - Added a bench script that compares `source_push_forward(..., implementation="pallas_mgpu")` against public
    `moe_mlp(..., implementation="ragged_all_to_all"|"ring")` on the same flattened source-major tokens, routing,
    combine weights, and expert weights.
  - Public `moe_mlp` comparison uses an `AxisType.Explicit` mesh plus `jax.set_mesh(mesh)` because named
    `PartitionSpec` resharding rejects Auto-axis meshes.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q`
    - Result: `37 passed, 11 warnings`.
  - Focused after explicit-mesh fix:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'public_compare or source_push_forward_public_compare or bench_cli_imports'`
    - Result: `5 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 smoke attempt 1:
  - Job: `/dlwh/source-push-forward-public-compare-814624a71-20260703-114751`
  - Result: Iris succeeded, but the script emitted a structured `ValueError` row before numeric comparison:
    `PartitionSpec passed to reshard cannot contain axis names that are of type Auto or Manual`.
  - Cause: the compare script reused source-push `_make_mesh`, whose default JAX axis type is Auto. Public `moe_mlp`
    now requires an explicit-axis mesh for named `PartitionSpec` resharding.
- H100 smoke rerun:
  - Job: `/dlwh/source-push-forward-public-compare-13be43c8d-20260703-115228`
  - Command:
    `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward_public_compare.py --ep-size 8 --tokens-per-rank 64 --hidden-dim 128 --intermediate-dim 128 --experts-per-rank 2 --topk 2 --capacity-factor 1.25 --entries-per-rank 2 --inbox-slots 1 --block-m 64 --block-k 64 --block-n 128 --n-group 1 --n-groups-per-job 1 --send-worker-programs-per-peer 1 --worker-programs-per-peer 8 --send-pipeline-depth 1 --routing balanced --source-push-implementation pallas_mgpu --source-push-execution-mode staged_host_sync --public-implementations ragged_all_to_all,ring --git-sha 13be43c8d5753b6e3f5cfb3269218d310ad58f69 --jsonl scratch/source-push-forward-public-compare-13be43c8d-20260703-115228.jsonl`
  - Iris summary: succeeded, one task, `duration_ms=73514`, `exit_code=0`.

| public baseline | max abs diff | mean abs diff | source-push dropped | public dropped | dropped delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ragged_all_to_all` | `0.0078125` | `0.0006504738703370094` | `0` | `0` | `0` |
| `ring` | `0.0078125` | `0.0005966257303953171` | `0` | `0` | `0` |

- Interpretation:
  - The package-private source-push forward callable now matches both public EP baselines on a real H100x8 sharded
    forward smoke using real packed/source-plan inputs.
  - This is a correctness/integration smoke only; it is not a target performance row.
  - Remaining production gap: wiring behind public `MoeImplementation` still needs a decision about host-side
    `SourcePushPlan` construction versus a device/JAX-compatible planner inside the existing public EP dispatch path.
