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
