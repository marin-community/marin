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

## 2026-07-03 05:04 - H100 pytest for public EP forward comparison

Promoted the package-private source-push vs public EP full-forward comparison into a checked-in H100-only pytest.

- Commit Hash: `59792faf5`
- Code:
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Added `test_source_push_forward_matches_public_ep_backends_on_h100`.
  - The test is selected by the objective command's `-k 'source_push'`, skips unless 8 visible H100 devices are present,
    and asserts:
    - no structured compare errors;
    - source-push and public dropped-route counts are both zero;
    - dropped-route delta is zero;
    - output shape is `[8, 64, 128]`;
    - `max_abs_diff <= 0.03125`;
    - `mean_abs_diff <= 0.002`.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'source_push' -n 0`
  - Result: `1 skipped, 19 deselected` on non-H100 local environment.
  - `./infra/pre-commit.py --changed-files --fix`
  - Result: all checks passed.
- H100 verification:
  - Job: `/dlwh/source-push-forward-h100-pytest-59792faf5-20260703-120023`
  - Command:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k 'source_push'`
  - Result: succeeded, `1 passed, 19 deselected, 1 warning in 53.48s`.
  - Iris summary: succeeded, one task, `exit_code=0`.
- Interpretation:
  - The integration acceptance smoke is now a durable pytest gate for H100x8, not only an ad hoc benchmark script.
  - This still validates the package-private callable path; public `MoeImplementation` wiring remains future work.

## 2026-07-03 05:27 - Public `pallas_mgpu_source_push` opt-in backend

Wired the source-push full-forward callable behind an explicit public `moe_mlp` implementation name.

- Commit Hashes:
  - `cbcd29ab1`: add `implementation="pallas_mgpu_source_push"` dispatch.
  - `32847fc23`: pass the validated explicit public mesh through to the package-private source-push path.
- Code:
  - `lib/levanter/src/levanter/grug/_moe/common.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_public.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Added an H100-only, SiLU-only public adapter for flattened public EP `moe_mlp` inputs.
  - The adapter reshapes public `[S*T, D]`, `[S*T, K]`, and `[S*E, ...]` layouts into source-major arrays,
    builds the host-side `SourcePushPlan`, runs the staged source-push W13/W2/return/combine path, then reshapes back
    to the original public output layout and sharding.
  - Added fail-fast behavior when the backend is requested without a concrete H100 expert mesh.
  - Updated the H100 pytest to exercise `moe_mlp(..., implementation="pallas_mgpu_source_push")` directly against both
    public `ragged_all_to_all` and `ring` baselines on the existing small H100 smoke shape.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'source_push' -n 0`
    - Result after final mesh fix: `1 passed, 1 skipped, 19 deselected, 1 warning in 3.07s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'public_compare or source_push_forward_public_compare or bench_cli_imports or real_inputs'`
    - Result: `6 passed, 11 warnings`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0`
    - Result before final mesh-only fix: `14 passed, 7 skipped, 1 warning in 11.01s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
  - `./infra/pre-commit.py --review --agent-command='codex exec'`
    - Result: aborted after several minutes with no stdout; no review findings were produced.
- H100 attempt 1:
  - Job: `/dlwh/source-push-public-backend-h100-pytest-cbcd29ab1-20260703-122117`
  - Command:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
  - Result: failed, `1 failed, 1 passed, 19 deselected, 1 warning in 13.54s`.
  - Iris summary: failed, one task, `exit_code=1`, `duration_ms=35445`.
  - Failure:
    `ValueError: The context mesh AbstractMesh('expert': 8, axis_types=(Explicit,), ...) should match the mesh passed to shard_map Mesh('expert': 8, axis_types=(Auto,))`.
  - Cause:
    the new public adapter called `source_push_forward(..., mesh=_make_mesh(ep_size))` while the public `moe_mlp`
    comparison was inside an explicit-axis mesh context.
- H100 rerun:
  - Job: `/dlwh/source-push-public-backend-h100-pytest-32847fc23-20260703-122507`
  - Command:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
  - Result: succeeded, `2 passed, 19 deselected, 1 warning in 53.96s`.
  - Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=76078`.
- Interpretation:
  - The public opt-in backend now reaches the same H100 full-forward comparison gate as the package-private callable.
  - The path remains intentionally narrow and host-plan/staged; it is not yet a fully jittable production training backend.

## 2026-07-03 05:50 - Public source-push deterministic repeat smoke

Added an H100 integration assertion for deterministic repeated public source-push runs under fixed inputs.

- Commit Hash: `ffa3ea279`
- Code:
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - `test_source_push_forward_matches_public_ep_backends_on_h100` now calls
    `moe_mlp(..., implementation="pallas_mgpu_source_push")` twice with the same sharded inputs and explicit H100 mesh.
  - The test asserts exact output equality between the two source-push calls and identical dropped-route counts before
    comparing against `ragged_all_to_all` and `ring`.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'source_push' -n 0`
    - Result: `1 passed, 1 skipped, 19 deselected, 1 warning in 1.99s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 verification:
  - Job: `/dlwh/source-push-public-determinism-h100-pytest-ffa3ea279-20260703-124725`
  - Command:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
  - Result: succeeded, `2 passed, 19 deselected, 1 warning in 55.97s`.
  - Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=75734`.
- Interpretation:
  - The public opt-in source-push backend now has direct H100 coverage for the integration acceptance requirement that
    repeated fixed-seed/fixed-input runs are deterministic.

## 2026-07-03 06:15 - Public source-push tail/empty/top-k-4 edge smoke

Added an H100 public-backend edge-case smoke for non-full queue blocks, empty local experts, and `topk=4`.

- Commit Hashes:
  - `e76764efd`: add the edge-case public H100 smoke.
  - `8c71fd694`: scale random test values down so the smoke checks routing/layout rather than large-magnitude bf16
    drift.
- Code:
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Added `test_source_push_forward_handles_tail_blocks_empty_experts_topk4_on_h100`.
  - Shape: `EP=8`, `T/rank=65`, `K=4`, `D=128`, `I=128`, `E_local=2`.
  - Routing uses only local expert `0` on each destination, leaving local expert `1` empty everywhere.
  - The test asserts the route counts include live blocks with `valid_rows % 64 != 0`, then compares public
    `implementation="pallas_mgpu_source_push"` against public `ragged_all_to_all` and `ring` with zero drops.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'source_push' -n 0`
    - Result after scaling fix: `1 passed, 2 skipped, 19 deselected, 1 warning in 1.81s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 attempt 1:
  - Job: `/dlwh/source-push-public-edge-h100-pytest-e76764efd-20260703-130908`
  - Command:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
  - Result: failed, `1 failed, 2 passed, 19 deselected, 1 warning in 102.90s`.
  - Iris summary: failed, one task, `exit_code=1`, `duration_ms=132351`.
  - Failure: edge test `max_abs_diff=32.0` against a public baseline with random-normal unscaled inputs/weights.
  - Interpretation: the routing/layout executed but the random-normal data made the existing bf16 absolute tolerance a
    magnitude stress test rather than a targeted edge-case smoke.
- H100 rerun:
  - Job: `/dlwh/source-push-public-edge-h100-pytest-8c71fd694-20260703-131305`
  - Command:
    `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
  - Result: succeeded, `3 passed, 19 deselected, 1 warning in 97.72s`.
  - Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=121470`.
- Interpretation:
  - The public opt-in backend now has H100 integration coverage for tail-block transport, empty local experts, and
    `topk=4` source combine through the public `moe_mlp` API.

## 2026-07-03 06:36 - Source-push W13 useful-throughput reporting

Added explicit useful-row and rounded-row W13 throughput fields to source-push benchmark rows.

- Commit Hash: `23dccd061`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/scripts/bench/bench_source_push_inbox_diagnostics.py`
  - `lib/levanter/scripts/bench/bench_source_push_inbox_consolidation.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
  - `lib/levanter/tests/grug/test_source_push_plan.py`
- Change:
  - Kept `w13_tflops_per_rank` as the existing rounded-row alias.
  - Added `rounded_w13_tflops_per_rank` and `useful_w13_tflops_per_rank` to W13 inbox rows and full-forward rows.
  - Added the new fields to diagnostic/consolidation summary medians.
  - Added CPU planner coverage for balanced `capacity_factor=1.0` routing with no dropped routes and no masked rows.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_inbox.py -q -n 0`
    - Result: `46 passed, 1 warning in 13.66s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- PR state:
  - PR #6841 CI monitor fired with `conclusion=success`, `observed_checks=32`, no pending or failing checks before this new commit.
- Interpretation:
  - Future W13 benchmark rows can report the denominator required by the invertible source-push spec: useful routed rows
    separately from rounded WGMMA rows.

## 2026-07-03 07:16 - Current W13 target gate rows with useful denominator

Ran the source-push W13-only target benchmark gate on current PR #6841 head with explicit useful-vs-rounded throughput
fields.

- Commit Hash: `ddab3fc37`
- Job: `/dlwh/source-push-w13-gate-ddab3fc37-20260703`
- Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=260035`.
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-w13-gate-ddab3fc37-20260703 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s bash -lc 'set -euo pipefail
  JSONL=scratch/source_push_w13_gate_ddab3fc37_20260703.jsonl
  rm -f "$JSONL"
  COMMON="uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_inbox.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --input-mode source_push_plan --no-check --warmup 2 --steps 5 --repeat-runs 9 --separate-compile --no-progress-events --git-sha ddab3fc37 --jsonl $JSONL"
  $COMMON --routing balanced --capacity-factor 1.0
  $COMMON --routing balanced --capacity-factor 1.25
  $COMMON --routing roughly_balanced --capacity-factor 1.25
  '
  ```
- Result rows:

  | routing | capacity | repeats | median time | rounded W13 TFLOP/s/rank | useful W13 TFLOP/s/rank | row efficiency | masked rows | drops |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | balanced | `1.0` | 9 | `7.997 ms` | `214.84` | `214.84` | `1.000000` | `0.000000` | 0 |
  | balanced | `1.25` | 9 | `7.959 ms` | `215.85` | `215.85` | `1.000000` | `0.000000` | 0 |
  | roughly_balanced | `1.25` | 9 | `8.508 ms` | `214.23` | `201.93` | `0.942584` | `0.057416` | 0 |

- Repeat times:
  - balanced cf1.0: `[8.172, 8.064, 8.064, 8.046, 7.997, 7.948, 7.986, 7.992, 7.966] ms`
  - balanced cf1.25: `[8.207, 8.170, 7.914, 8.007, 7.936, 7.959, 7.968, 7.926, 7.929] ms`
  - roughly-balanced cf1.25: `[8.384, 8.381, 8.424, 8.356, 9.152, 14.492, 10.025, 8.550, 8.508] ms`
- Interpretation:
  - The current source-push W13 path meets the target rough-balanced acceptance bar on the median:
    useful throughput `201.93 TFLOP/s/rank` and median time `8.508 ms`.
  - There is a slow-tail signal in rough-balanced repeats (`14.492 ms`, `10.025 ms`, one `9.152 ms`), so the
    slow-tail investigation item remains open even though the median gate passes.

## 2026-07-03 07:36 - W13 slow-tail distribution check

Investigated the source-push W13 slow-tail condition from the benchmark gate by extracting repeat distributions from
the previous 48-repeat precomputed source-plan run and the current 9-repeat gate run. No new H100 job was launched for
this check.

- Current gate evidence:
  - Job: `/dlwh/source-push-w13-gate-ddab3fc37-20260703`
  - Rows: rough-balanced cf1.25, `repeat_runs=9`, `row_efficiency=0.942584`, drops `0`.
  - Useful-throughput slow threshold: `<160 TFLOP/s/rank`.
  - Distribution: median `8.508 ms`, useful median `201.93 TFLOP/s/rank`, slow repeats `1/9 = 11.1%`.
  - Slow repeat: `[14.492] ms`.
- Prior 48-repeat evidence:
  - Job: `/dlwh/source-push-plan-target-precomputed-b3bdb1f76-20260703-0120`
  - Rows: rough-balanced cf1.25, `repeat_runs=48`, `row_efficiency=0.942584`, drops `0`.
  - Distribution: median `8.441 ms`, min `8.293 ms`, max `14.855 ms`, p90 `11.395 ms`, p95 `12.488 ms`.
  - Rounded/useful medians: `215.94` / `203.54 TFLOP/s/rank`.
  - Slow repeats: `6/48 = 12.5%`.
  - Slow repeat times: `[14.855, 11.753, 13.069, 12.488, 11.395, 13.187] ms`.
- Interpretation:
  - The slow-tail condition is replicated: both the 9-repeat current gate and the earlier 48-repeat target run exceed
    the spec's `>10%` slow-repeat trigger.
  - This does not look like route drops, row accounting, or a median-performance failure: both runs have zero drops and
    the same row efficiency, and the median gate still passes.
  - Balanced cf1.0/cf1.25 current gate rows showed no comparable slow repeats, so the tail correlates with the
    rough-balanced/source-padded queue shape (`tail_entries_total=2032`, `max_slot_reuse_depth=23`) rather than the
    fixed balanced queue (`tail_entries_total=0`, `max_slot_reuse_depth=22`).
- Next action:
  - If we continue performance work before landing PR #6841, isolate whether rough-balanced slow repeats come from
    queue-slot reuse/semaphore skew or WGMMA scheduling variance. The most direct follow-up is a bounded diagnostic
    run on rough-balanced cf1.25 comparing full W13 against `copy_release_only` and `compute_only_local` with the same
    queue metadata, reporting p90/p95 and slow-repeat count rather than only medians.

## 2026-07-03 08:06 - W13 slow-tail decomposition with source-plan diagnostics

Ran the bounded rough-balanced source-plan diagnostic proposed above on H100s. This uses the PR #6841 source-push
profile and identical source-plan queue metadata across variants.

- Commit Hash: `eb27dbc86`
- Job: `/dlwh/source-push-w13-tail-diag-eb27dbc86-20260703`
- Cluster: `cw-us-east-02a`
- Iris summary: succeeded, one task, `exit_code=0`.
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-w13-tail-diag-eb27dbc86-20260703 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_inbox_diagnostics.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --input-mode source_push_plan \
    --variants full,copy_release_only,compute_only_local \
    --repeat-runs 24 --warmup 2 --steps 5 --separate-compile --no-progress-events \
    --git-sha eb27dbc86 --jsonl scratch/source_push_w13_tail_diag_eb27dbc86_20260703.jsonl
  ```
- Summary rows:

  | variant | repeats | median time | p90 | p95 | max | rounded W13 TFLOP/s/rank | useful W13 TFLOP/s/rank | min useful | slow repeats `<160` |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `full` | 24 | `8.395 ms` | `11.390 ms` | `12.955 ms` | `13.826 ms` | `217.11` | `204.64` | `124.26` | `3/24 = 12.5%` |
  | `copy_release_only` | 24 | `5.611 ms` | `5.711 ms` | `7.032 ms` | `10.037 ms` | `324.84` | `306.19` | `171.17` | `0/24 = 0.0%` |
  | `compute_only_local` | 24 | `6.571 ms` | `6.621 ms` | `6.633 ms` | `7.596 ms` | `277.38` | `261.45` | `226.16` | `0/24 = 0.0%` |

- Shared queue metadata:
  - `live_entries_total=17382`, `payload_send_entries_total=17382`, `masked_rows_total=63872`.
  - `send_masked_row_fraction=0.057416`, `slot_empty_waits=17382`, `slot_full_waits=86910`.
  - `tail_entries_total=2032`, `max_slot_reuse_depth=23`, drops `0`.
- Slowest repeats:
  - `full`: `12.556 ms`, `13.025 ms`, `13.826 ms` (`136.8`, `131.9`, `124.3` useful TFLOP/s/rank).
  - `copy_release_only`: `7.264 ms`, `10.037 ms` (`236.5`, `171.2` useful TFLOP/s/rank).
  - `compute_only_local`: one `7.596 ms` repeat (`226.2` useful TFLOP/s/rank); otherwise the slowest cluster is
    `6.596-6.635 ms`.
- Interpretation:
  - The full path reproduces the slow-tail trigger (`3/24 = 12.5%` below `160` useful TFLOP/s/rank), matching the
    prior 9-repeat and 48-repeat evidence.
  - `copy_release_only` has a milder tail but does not cross the slow threshold; `compute_only_local` is tighter still.
    That argues against either standalone remote copy/semaphore traffic or standalone local WGMMA being sufficient to
    explain the severe full-path tail.
  - Median full time (`8.395 ms`) is much less than the serial sum of median `copy_release_only + compute_only_local`
    (`12.182 ms`), so the combined kernel is overlapping work on the median path. The remaining issue is a combined-path
    interaction tail, likely from queue-slot/arrival skew plus compute scheduling rather than from a pure copy-only or
    compute-only floor.
- Next action:
  - Keep this as the current slow-tail decomposition baseline for PR #6841. If more perf work is needed before landing,
    the next diagnostic should instrument per-slot/per-phase wait distribution or split producer/consumer scheduling,
    not blindly tune the existing full kernel.

## 2026-07-03 08:37 - Current-head full-forward target gate

Refreshed the full source-push forward target gate on current PR #6841 head after the public opt-in adapter and later
W13 reporting changes. This uses the staged W13 -> W2-return -> combine path, which remains the safer and faster
current full-forward structure than the single-JIT path.

- Commit Hash: `754fe3005`
- Job: `/dlwh/source-push-forward-gate-754fe3005-20260703`
- Cluster: `cw-us-east-02a`
- Iris summary: succeeded, one task, `exit_code=0`.
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-forward-gate-754fe3005-20260703 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s bash -lc 'set -euo pipefail
  JSONL=scratch/source_push_forward_gate_754fe3005_20260703.jsonl
  rm -f "$JSONL"
  COMMON="uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-check --no-progress-events --git-sha 754fe3005 --jsonl $JSONL"
  $COMMON --routing balanced --capacity-factor 1.0
  $COMMON --routing balanced --capacity-factor 1.25
  $COMMON --routing roughly_balanced --capacity-factor 1.25
  '
  ```
- Full-forward rows:

  | routing | capacity | repeats | median total | p90 total | rounded forward TFLOP/s/rank | useful forward TFLOP/s/rank | row efficiency | drops |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | 5 | `15.257 ms` | `15.342 ms` | `168.90` | `168.90` | `1.000000` | 0 |
  | `balanced` | `1.25` | 5 | `15.310 ms` | `15.415 ms` | `168.32` | `168.32` | `1.000000` | 0 |
  | `roughly_balanced` | `1.25` | 5 | `16.443 ms` | `22.694 ms` | `166.27` | `156.72` | `0.942584` | 0 |

- Stage medians:

  | routing | capacity | W13 | W2 return | source combine |
  | --- | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | `8.103 ms` | `5.716 ms` | `1.341 ms` |
  | `balanced` | `1.25` | `8.073 ms` | `5.795 ms` | `1.342 ms` |
  | `roughly_balanced` | `1.25` | `8.664 ms` | `6.190 ms` | `1.379 ms` |

- Repeat times:
  - balanced cf1.0 total: `[15.359, 15.315, 15.165, 15.257, 15.212] ms`.
  - balanced cf1.25 total: `[15.427, 15.397, 15.175, 15.310, 15.264] ms`.
  - rough-balanced cf1.25 total: `[15.954, 16.089, 17.615, 26.080, 16.443] ms`.
  - rough-balanced cf1.25 stage rows:
    - W13: `[8.456, 8.664, 8.652, 11.866, 8.781] ms`.
    - W2 return: `[6.054, 5.989, 7.441, 11.397, 6.190] ms`.
    - combine: `[1.345, 1.341, 1.426, 2.665, 1.379] ms`.
- Interpretation:
  - Current-head balanced full-forward target rows are materially faster than the older `bc0c377c1` full-forward rows
    (`~15.3 ms` now vs `~17.4-17.6 ms` then), while preserving zero drops and exact row efficiency.
  - Rough-balanced median remains close to the prior staged baseline (`16.443 ms` now vs `16.137 ms` at `8b988579d`),
    but one repeat (`26.080 ms`) shows that the rough-balanced slow-tail condition also affects the full W13/W2/combine
    chain.
  - The slow repeat includes simultaneous W13, W2-return, and combine inflation, so full-forward tail analysis should
    use the W13 combined-path tail decomposition above as the current baseline before adding more tuning knobs.
- Next action:
  - Treat `15.3 ms` balanced and `16.4 ms` rough-balanced staged full-forward medians as the current PR #6841 target
    full-forward baseline. The next production-relevant optimization remains reducing W2-return transport cost or
    replacing the current staged return boundary with a real producer/consumer return path.

## 2026-07-03 09:06 - Full-forward benchmark summary rows

Added aggregate summary rows to the package-private full-forward source-push benchmark output so target-gate reporting
does not require manual JSONL post-processing.

- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - `run_source_push_forward_source_plan` now appends `row_type="summary"` rows for `total`, `w13`, `w2_return`, and
    `combine` when repeat rows are present.
  - Summary rows include median metrics, min/max/p90/p95 `steady_state_time`, flattened queue stats, and W13 slow-repeat
    accounting against the existing `<160 useful TFLOP/s/rank` threshold.
  - The slow W13 threshold is now defined once in `source_push_inbox.py` and shared by diagnostics plus full-forward
    summaries.
  - Error rows remain unchanged; summary rows are added only for successful repeat rows.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'forward_adds_summary or source_push_forward_runner_returns_structured_validation_errors or source_push_forward_bench_cli_imports or source_push_forward_cli_passes_profile_defaults'`
    - Result: `4 passed, 11 warnings`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q -n 0`
    - Result: `47 passed, 1 warning`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- Interpretation:
  - This is reporting-only and does not change the W13/W2/combine kernels.
  - The next target H100 full-forward gate will emit the median/p90/p95 and slow-tail fields directly in JSONL.

## 2026-07-03 09:36 - Current-head full-forward summary gate

Ran the target full-forward source-push gate on the current PR #6841 head after adding direct benchmark summary rows.
This verifies that the summary rows are emitted at target shape and refreshes the staged full-forward numbers.

- Commit Hash: `831465357`
- Job: `/dlwh/source-push-forward-summary-gate-831465357-20260703`
- Cluster: `cw-us-east-02a`
- Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=167594`.
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-forward-summary-gate-831465357-20260703 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s bash -lc 'set -euo pipefail
  JSONL=scratch/source_push_forward_summary_gate_831465357_20260703.jsonl
  rm -f "$JSONL"
  COMMON="uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-check --no-progress-events --git-sha 831465357 --jsonl $JSONL"
  $COMMON --routing balanced --capacity-factor 1.0
  $COMMON --routing balanced --capacity-factor 1.25
  $COMMON --routing roughly_balanced --capacity-factor 1.25
  '
  ```
- Full-forward summary rows:

  | routing | capacity | repeats | median total | p90 total | p95 total | rounded forward TFLOP/s/rank | useful forward TFLOP/s/rank | row efficiency | drops |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | 5 | `15.219 ms` | `15.256 ms` | `15.264 ms` | `169.33` | `169.33` | `1.000000` | 0 |
  | `balanced` | `1.25` | 5 | `15.334 ms` | `15.632 ms` | `15.701 ms` | `168.06` | `168.06` | `1.000000` | 0 |
  | `roughly_balanced` | `1.25` | 5 | `15.805 ms` | `15.960 ms` | `15.972 ms` | `172.98` | `163.04` | `0.942584` | 0 |

- Stage summary rows:

  | routing | capacity | W13 median | W13 useful TFLOP/s/rank | W13 slow repeats `<160` | W2-return median | W2 TFLOP/s/rank | combine median | combine GB/s/rank |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | `8.090 ms` | `212.36` | `0/5` | `5.695 ms` | `150.83` | `1.351 ms` | `1614.21` |
  | `balanced` | `1.25` | `8.158 ms` | `210.58` | `0/5` | `5.733 ms` | `149.83` | `1.367 ms` | `1595.68` |
  | `roughly_balanced` | `1.25` | `8.434 ms` | `203.69` | `0/5` | `5.974 ms` | `152.55` | `1.336 ms` | `1632.79` |

- Total repeat times:
  - balanced cf1.0: `[15.272, 15.219, 15.128, 15.232, 15.144] ms`.
  - balanced cf1.25: `[15.424, 15.287, 15.770, 15.282, 15.334] ms`.
  - roughly-balanced cf1.25: `[15.985, 15.922, 15.793, 15.762, 15.805] ms`.
- Interpretation:
  - The new summary rows work at target shape and directly report the median/p90/p95 and W13 slow-tail fields required
    by the spec.
  - Current-head full-forward medians are stable against the prior `754fe3005` gate, with the rough-balanced row faster
    in this sample (`15.805 ms` vs `16.443 ms`) and no rough-balanced slow repeat in 5 repeats.
  - The broad bottleneck judgment is unchanged: W13 still meets the useful-throughput bar (`203.69 TFLOP/s/rank` useful
    on rough-balanced), while the staged full-forward path is dominated by W2-return (`~5.97 ms`) plus the memory-bound
    source combine (`~1.34 ms`) after the W13 stage.
- Next action:
  - Keep `831465357` as the current full-forward summary gate. The next production-relevant optimization remains W2-return
    transport or a producer/consumer return path; do not add more benchmark knobs unless they isolate that cost.

## 2026-07-03 10:27 - Direct source-visible W2 return experiment

Added an opt-in W2-return benchmark variant that computes W2 and writes each output block directly into the source-visible
return queue via Lane-lowered remote SMEM-to-GMEM stores. This tests whether the existing staged path's destination-local
W2 output plus separate return-copy kernel is paying avoidable local-GMEM staging cost.

- Commit Hash: `480e5ec3` plus local diff in `source_push_w2_return.py`, `bench_source_push_w2_return.py`, and
  `test_source_push_inbox.py`.
- Local verification:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py lib/levanter/scripts/bench/bench_source_push_w2_return.py lib/levanter/tests/grug/test_source_push_inbox.py`
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'w2_runner_tags or w2_bench_cli_imports or w2_destination_return_reorders_to_source_queue or w2_reference_uses_recv_metadata'`
    - Result: `5 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 smoke:
  - Job: `/dlwh/source-push-w2-direct-smoke-480e5ec3-20260703`
  - Cluster: `cw-us-east-02a`
  - Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=32972`.
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-w2-direct-smoke-480e5ec3-20260703 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 3600s uv run --package marin-levanter --group test python \
      lib/levanter/scripts/bench/bench_source_push_w2_return.py \
      --ep-size 8 --tokens-per-rank 16 --hidden-dim 128 --intermediate-dim 128 \
      --experts-per-rank 2 --topk 2 --capacity-factor 1.25 \
      --entries-per-rank 2 --inbox-slots 1 --block-m 64 --block-k 64 --block-n 128 \
      --n-group 1 --n-groups-per-job 1 --send-worker-programs-per-peer 1 \
      --worker-programs-per-peer 8 --send-pipeline-depth 1 --routing balanced \
      --hidden-input-mode w13_reference --direct-to-source --check \
      --warmup 0 --steps 1 --repeat-runs 1 --git-sha 480e5ec3 \
      --jsonl scratch/source_push_w2_direct_smoke_480e5ec3.jsonl
    ```
  - Result: `steady_state_time=1.798 ms`, `max_abs_diff=0.007897`, `source_queue_max_abs_diff=0.007897`,
    `dropped_routes=0`.
  - Note: after this run, the W2 benchmark CLI was consolidated from `--copy-to-source`/`--direct-to-source` booleans to
    `--return-mode {destination_local,separate_copy,direct_remote}`. The current equivalent for this smoke is
    `--return-mode direct_remote`.
- Target decomposition commands used the rough-balanced target profile:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_w2_return.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --hidden-input-mode synthetic <mode flag> --no-check \
    --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-progress-events \
    --git-sha 480e5ec3 --jsonl scratch/<mode>.jsonl
  ```
  The current equivalent mode flags are `--return-mode destination_local`, `--return-mode separate_copy`, and
  `--return-mode direct_remote`.
- Target rows:

  | mode | job | median time | min/max time | W2 TFLOP/s/rank | return GB/s/rank | drops |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | W2 local-only | `/dlwh/source-push-w2-local-target-480e5ec3-20260703` | `2.864 ms` | `2.864/2.876 ms` | `318.14` | `372.82` | 0 |
  | W2 + separate return copy | `/dlwh/source-push-w2-copy-target-480e5ec3-20260703` | `5.953 ms` | `5.842/6.120 ms` | `153.09` | `179.40` | 0 |
  | W2 direct-to-source | `/dlwh/source-push-w2-direct-target-480e5ec3-20260703` | `4.006 ms` | `3.978/4.138 ms` | `227.47` | `266.56` | 0 |

- Interpretation:
  - Lane-lowered remote stores from the W2 kernel are legal for this pattern; the checked smoke compiled and matched the
    source-visible reference queue.
  - Direct-to-source W2 return saves `~1.947 ms` versus the separate-copy return path on the target rough-balanced profile.
  - Direct-to-source still costs `~1.142 ms` over local W2-only, so remote return transport/store remains material, but the
    larger staged local-GMEM output plus second-kernel copy tax is avoidable.
  - Current staged rough-balanced full-forward W2-return was `~5.974 ms`; replacing it with the direct return kernel would
    put the expected W2-return component near `4.006 ms` before any full-forward integration effects.
- Next action:
  - Clean the opt-in benchmark path and integrate the direct-to-source W2-return variant into the staged full-forward
    source-push path for a new target full-forward gate.

## 2026-07-03 10:38 - Direct W2 return integrated into full-forward gate

Replaced the staged full-forward harness's W2 + separate return-copy stage with the direct source-visible W2-return kernel.
The package-private W2 benchmark still exposes local-only, separate-copy, and direct-return variants for decomposition, but
the full-forward path now uses the faster direct return by default and records `forward_mode=w13_w2_direct_return_combine`.

- Commit Hash: `480e5ec3` plus local diff.
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py`
  - `lib/levanter/scripts/bench/bench_source_push_w2_return.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/scripts/bench/bench_source_push_w2_return.py lib/levanter/tests/grug/test_source_push_inbox.py`
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'w2_runner_tags or w2_bench_cli_imports or w2_destination_return_reorders_to_source_queue or w2_reference_uses_recv_metadata or forward_inputs_share_one_plan or forward_adds_summary or forward_runner_returns_structured_validation_errors or forward_cli_passes_profile_defaults'`
    - Result: `9 passed, 11 warnings`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 integrated smoke:
  - Job: `/dlwh/source-push-forward-direct-smoke-480e5ec3-20260703`
  - Cluster: `cw-us-east-02a`
  - Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=33381`.
  - Result: `forward_mode=w13_w2_direct_return_combine`, total `4.443 ms`, W13 `2.293 ms`,
    W2-return `1.790 ms`, combine `0.228 ms`, `max_abs_diff=0.0078125`.
- Target gate:
  - Job: `/dlwh/source-push-forward-direct-gate-480e5ec3-20260703`
  - Cluster: `cw-us-east-02a`
  - Iris summary: succeeded, one task, `exit_code=0`, `duration_ms=288659`.
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-forward-direct-gate-480e5ec3-20260703 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 7200s bash -lc 'set -euo pipefail
    JSONL=scratch/source_push_forward_direct_gate_480e5ec3_20260703.jsonl
    rm -f "$JSONL"
    COMMON="uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-check --no-progress-events --git-sha 480e5ec3 --jsonl $JSONL"
    $COMMON --routing balanced --capacity-factor 1.0
    $COMMON --routing balanced --capacity-factor 1.25
    $COMMON --routing roughly_balanced --capacity-factor 1.25
    '
    ```
- Full-forward summary rows:

  | routing | capacity | repeats | median total | p90 total | p95 total | rounded forward TFLOP/s/rank | useful forward TFLOP/s/rank | row efficiency | drops |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | 5 | `13.640 ms` | `13.992 ms` | `14.002 ms` | `188.93` | `188.93` | `1.000000` | 0 |
  | `balanced` | `1.25` | 5 | `13.386 ms` | `13.485 ms` | `13.496 ms` | `192.51` | `192.51` | `1.000000` | 0 |
  | `roughly_balanced` | `1.25` | 5 | `13.992 ms` | `14.148 ms` | `14.185 ms` | `195.39` | `184.17` | `0.942584` | 0 |

- Stage summary rows:

  | routing | capacity | W13 median | W13 useful TFLOP/s/rank | W13 slow repeats `<160` | W2-return median | W2 TFLOP/s/rank | combine median | combine GB/s/rank |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | `8.090 ms` | `212.37` | `0/5` | `3.932 ms` | `218.45` | `1.341 ms` | `1626.92` |
  | `balanced` | `1.25` | `8.027 ms` | `214.02` | `0/5` | `3.932 ms` | `218.44` | `1.342 ms` | `1625.62` |
  | `roughly_balanced` | `1.25` | `8.536 ms` | `201.26` | `0/5` | `4.038 ms` | `225.66` | `1.337 ms` | `1631.54` |

- Comparison to prior current-head gate (`/dlwh/source-push-forward-summary-gate-831465357-20260703`):

  | routing | capacity | previous total | direct-return total | delta | previous W2-return | direct W2-return | W2 delta |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `balanced` | `1.0` | `15.219 ms` | `13.640 ms` | `-1.579 ms` | `5.695 ms` | `3.932 ms` | `-1.763 ms` |
  | `balanced` | `1.25` | `15.334 ms` | `13.386 ms` | `-1.948 ms` | `5.733 ms` | `3.932 ms` | `-1.801 ms` |
  | `roughly_balanced` | `1.25` | `15.805 ms` | `13.992 ms` | `-1.813 ms` | `5.974 ms` | `4.038 ms` | `-1.936 ms` |

- Interpretation:
  - Direct W2 return is a meaningful full-forward speedup, not just a standalone W2 benchmark win.
  - The rough-balanced target path improves from `163.04` useful TFLOP/s/rank to `184.17` useful TFLOP/s/rank.
  - The W2-return component now lines up with the standalone direct-return median (`~4.0 ms`), so the integration did not
    reintroduce the separate-copy tax.
  - Remaining staged bottlenecks are W13 `~8.5 ms`, combine `~1.34 ms`, and the residual direct-return overhead over local
    W2-only (`~1.17 ms` in the target decomposition).
- Next action:
  - Commit and push the direct W2-return path, then update PR #6841 status. A later pass can decide whether direct return
    should be fused with combine or whether source combine needs a lower-memory deterministic variant.

## 2026-07-03 11:28 - Current-head public source-push H100 pytest after direct W2 return

Verified the public opt-in `implementation="pallas_mgpu_source_push"` path at current PR head after the direct
source-visible W2-return commits.

- Commit Hash: `e3d4d4ff9`
- Job: `/dlwh/source-push-public-direct-h100-pytest-e3d4d4ff9-20260703-182515`
- Cluster: `cw-us-east-02a`
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-public-direct-h100-pytest-e3d4d4ff9-20260703-182515 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 3600s uv run --package marin-levanter --group test pytest \
    lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
  ```
- Iris summary:
  - State: `succeeded`
  - Task count: `1`
  - Exit code: `0`
  - Duration: `114903 ms`
  - Failure count: `0`
  - Preemption count: `0`
- Pytest result:
  - `3 passed, 19 deselected, 1 warning in 92.39s`
- Coverage:
  - Public source-push backend matches public `ragged_all_to_all` and `ring` baselines on the H100 smoke shape with zero
    drops and bf16 tolerances.
  - Repeated public source-push calls under fixed inputs are deterministic.
  - Tail blocks, empty local experts, and `topk=4` are covered by the H100 edge smoke.
- Interpretation:
  - The direct W2-return integration did not regress the checked public H100 source-push forward correctness path.
  - The remaining source-push work is performance/integration refinement, not a correctness regression from the direct
    return change.

## 2026-07-03 11:55 - Stage-specific H100 source-push pytest coverage

Added a checked H100-only pytest for the individual source-push forward stages. The existing public full-forward H100
smoke covered the integrated `pallas_mgpu_source_push` path, but the spec also calls out stage-level coverage for W13
expert-major placement, W2 return identity, and source combine.

- Commit Hash: `253e8b989` plus local diff in `lib/levanter/tests/grug/test_grugformer_moe.py`.
- Code:
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Added `test_source_push_stage_kernels_match_references_on_h100`.
  - Shape: `EP=8`, `T/rank=64`, `K=2`, `D=128`, `I=128`, `E_local=2`.
  - The test runs:
    - `run_source_push_inbox_source_plan(..., check=True)` and asserts source-plan metadata, zero metadata mismatches,
      expert-major W13 hidden diff within bf16 tolerance, and zero unwritten-row diff.
    - `run_source_push_w2_return_source_plan(..., hidden_input_mode="w13_reference", return_mode="direct_remote",
      check=True)` and asserts direct source-visible return plus source-queue diff within bf16 tolerance.
    - `run_source_push_combine_source_plan(..., check=True)` and asserts route-buffer gather/sum combine diff within
      bf16 tolerance.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k 'source_push_stage_kernels_match_references_on_h100'`
    - Result: `1 skipped, 22 deselected, 1 warning` on non-H100 local hardware.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
    - Result: `1 passed, 3 skipped, 19 deselected, 1 warning` on non-H100 local hardware.
- H100 verification:
  - Job: `/dlwh/source-push-stage-h100-pytest-253e8b989-local-20260703-185057`
  - Cluster: `cw-us-east-02a`
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-stage-h100-pytest-253e8b989-local-20260703-185057 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 3600s uv run --package marin-levanter --group test pytest \
      lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
    ```
  - Iris summary:
    - State: `succeeded`
    - Task count: `1`
    - Exit code: `0`
    - Duration: `124687 ms`
    - Failure count: `0`
    - Preemption count: `0`
  - Pytest result:
    - `4 passed, 19 deselected, 1 warning in 100.56s`
- Interpretation:
  - The source-push H100 pytest selection now directly exercises the W13, direct W2-return, and source-combine stages in
    addition to the public full-forward backend smokes.
  - This closes the objective's stage-specific H100 correctness coverage gap for the current staged forward path.

## 2026-07-03 12:20 - Add small-EP H100 source-push stage coverage

Extended the stage-specific H100 source-push pytest to cover the spec-literal small EP case in addition to the existing
full H100 mesh case.

- Commit Hash: `5b9ce15e` plus local diff in `lib/levanter/tests/grug/test_grugformer_moe.py`.
- Code:
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Parameterized `test_source_push_stage_kernels_match_references_on_h100` over `EP=2` and `EP=8`.
  - Both cases run the W13 source-plan kernel, direct W2-return kernel, and route-buffer source-combine kernel with
    reference checks enabled.
  - This directly covers the small-EP H100 stage requirement while preserving the previous full-mesh stage coverage.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k 'source_push_stage_kernels_match_references_on_h100'`
    - Result: `2 skipped, 22 deselected, 1 warning` on non-H100 local hardware.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
    - Result: `1 passed, 4 skipped, 19 deselected, 1 warning` on non-H100 local hardware.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 verification:
  - Job: `/dlwh/source-push-stage-ep2-h100-pytest-5b9ce15e-local-20260703-191712`
  - Cluster: `cw-us-east-02a`
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-stage-ep2-h100-pytest-5b9ce15e-local-20260703-191712 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 3600s uv run --package marin-levanter --group test pytest \
      lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
    ```
  - Iris summary:
    - State: `succeeded`
    - Task count: `1`
    - Exit code: `0`
    - Duration: `125920 ms`
    - Failure count: `0`
    - Preemption count: `0`
  - Pytest result:
    - `5 passed, 19 deselected, 1 warning in 105.24s`.
- Interpretation:
  - The H100 source-push pytest now includes both the small EP=2 stage case requested by the spec and the full EP=8
    stage case used by the target mesh.
  - This is a coverage/proof-strength patch only; it does not change kernel behavior or performance.

## 2026-07-03 12:44 - Add top-k=4 H100 source-push stage coverage

Extended the stage-specific H100 source-push pytest matrix to include the spec's `top_k=4` source-combine case.

- Commit Hash: `3af5b5322` plus local diff in `lib/levanter/tests/grug/test_grugformer_moe.py`.
- Code:
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - `test_source_push_stage_kernels_match_references_on_h100` now runs:
    - `EP=2`, `topk=2`
    - `EP=2`, `topk=4`
    - `EP=8`, `topk=2`
  - The new `EP=2`, `topk=4` case runs the same stage-level W13, direct W2-return, and route-buffer source-combine
    checks with references enabled.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k 'source_push_stage_kernels_match_references_on_h100'`
    - Result: `3 skipped, 22 deselected, 1 warning` on non-H100 local hardware.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
    - Result: `1 passed, 5 skipped, 19 deselected, 1 warning` on non-H100 local hardware.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 verification:
  - Job: `/dlwh/source-push-stage-topk4-h100-pytest-3af5b532-local-20260703-194134`
  - Cluster: `cw-us-east-02a`
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-stage-topk4-h100-pytest-3af5b532-local-20260703-194134 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 3600s uv run --package marin-levanter --group test pytest \
      lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
    ```
  - Iris summary:
    - State: `succeeded`
    - Task count: `1`
    - Exit code: `0`
    - Duration: `136904 ms`
    - Failure count: `0`
    - Preemption count: `0`
  - Pytest result:
    - `6 passed, 19 deselected, 1 warning in 111.01s`.
- Interpretation:
  - The H100 source-push stage coverage now includes `topk=2` and `topk=4` source-combine reference checks, plus the
    small EP=2 W13/W2 round-trip case and the full EP=8 stage case.
  - This is a coverage/proof-strength patch only; it does not change kernel behavior or performance.

## 2026-07-03 13:12 - Current-head source-push full-forward target gate

Refreshed the three target full-forward rows on the current PR head after the source-push wrapper, production-kernel
cleanup, and H100 smoke coverage commits.

- Commit Hash: `89f3267fc`
- Job: `/dlwh/source-push-forward-current-head-gate-89f3267fc-20260703-200306`
- Cluster: `cw-us-east-02a`
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-forward-current-head-gate-89f3267fc-20260703-200306 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s bash -lc 'set -euo pipefail
  JSONL=scratch/source_push_forward_current_head_gate_89f3267fc_20260703.jsonl
  rm -f "$JSONL"
  COMMON="uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-check --no-progress-events --git-sha 89f3267fc --jsonl $JSONL"
  $COMMON --routing balanced --capacity-factor 1.0
  $COMMON --routing balanced --capacity-factor 1.25
  $COMMON --routing roughly_balanced --capacity-factor 1.25
  '
  ```
- Iris summary:
  - State: `succeeded`
  - Task count: `1`
  - Exit code: `0`
  - Failure count: `0`
  - Preemption count: `0`

Summary medians over 5 repeats:

| routing | cf | stage | median time | useful TFLOP/s/rank | rounded TFLOP/s/rank | row efficiency | masked rows | drops |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | 1.0 | total | `13.462 ms` | `191.43` | `191.43` | `1.000000` | `0.000000` | `0` |
| balanced | 1.0 | W13 | `8.070 ms` | `212.88` | `212.88` | `1.000000` | `0.000000` | `0` |
| balanced | 1.0 | W2 return | `3.943 ms` | - | - | `1.000000` | `0.000000` | `0` |
| balanced | 1.0 | combine | `1.331 ms` | - | - | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | total | `13.448 ms` | `191.63` | `191.63` | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | W13 | `8.057 ms` | `213.23` | `213.23` | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | W2 return | `3.910 ms` | - | - | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | combine | `1.338 ms` | - | - | `1.000000` | `0.000000` | `0` |
| roughly_balanced | 1.25 | total | `13.876 ms` | `185.72` | `197.03` | `0.942584` | `0.057416` | `0` |
| roughly_balanced | 1.25 | W13 | `8.415 ms` | `204.17` | `216.60` | `0.942584` | `0.057416` | `0` |
| roughly_balanced | 1.25 | W2 return | `4.050 ms` | - | - | `0.942584` | `0.057416` | `0` |
| roughly_balanced | 1.25 | combine | `1.338 ms` | - | - | `0.942584` | `0.057416` | `0` |

- Slow W13 repeats: `0` for all three rows.
- Interpretation:
  - Current PR head still clears the initial W13 rough-balanced gate: useful `204.17 TFLOP/s/rank` and `8.415 ms`
    median, with no row drops.
  - The full staged forward remains around `185.72` useful TFLOP/s/rank on rough-balanced because the measured total
    includes W13, direct W2 return, and source combine.
  - The balanced rows remain stable against the earlier direct-return gate; rough-balanced improves slightly relative to
    the prior `8.536 ms` W13 median and `13.992 ms` full-forward median.

## 2026-07-03 13:43 - Add exact expert-major W13 layout proof for block-aligned plans

Added a private exact expert-major input builder for the W13 source-push inbox path and row-layout accounting fields
shared by the W13 and full-forward benchmark rows. The production/source-padded performance path remains unchanged.

- Commit Hash: `4fda5d101`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Added row layout labels:
    - `source_padded_expert_major`
    - `exact_expert_major`
  - Added counters separating useful/exact rows, WGMMA-rounded rows, layout rows, layout padding, and hidden-capacity
    unused rows.
  - Added `_make_exact_source_push_plan_inputs`, a private exact-layout builder that uses the plan's count-derived
    `expert_base + src_base + local_row_start` addressing when every live block is full.
  - The exact builder rejects tail-block plans because the current Lane/WGMMA kernel stores full `block_m` rows and would
    clobber a following exact source slice otherwise.
  - Added a Hopper-only W13 smoke that proves the exact count-derived row-start path lowers and matches reference for a
    block-aligned EP=2 case.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_plan_inputs or exact_source_push_plan_inputs'`
    - Result: `3 passed, 11 warnings in 22.25s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q`
    - Result: `41 passed, 11 warnings in 24.61s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push`
    - Result: `1 passed, 6 skipped, 19 deselected, 1 warning in 4.47s` on non-H100 local hardware.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 verification:
  - Job: `/dlwh/source-push-exact-layout-h100-pytest-1a7c5743c-local-20260703-203844`
  - Cluster: `cw-us-east-02a`
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-exact-layout-h100-pytest-1a7c5743c-local-20260703-203844 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 3600s uv run --package marin-levanter --group test pytest \
      lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
    ```
  - Iris summary:
    - State: `succeeded`
    - Task count: `1`
    - Exit code: `0`
    - Duration: `131388 ms`
    - Failure count: `0`
    - Preemption count: `0`
  - Pytest result:
    - `7 passed, 19 deselected, 1 warning in 109.95s`.
- Interpretation:
  - This does not switch the target rough-balanced path away from source-padded expert-major rows.
  - It proves the exact count-derived row-start path is valid for block-aligned cases, including balanced target-style
    counts, while keeping the source-padded layout as the only safe current path for tail blocks.

## 2026-07-03 14:16 - Add exact expert-major W2 return layout proof

Extended the W2 return path so it can consume exact expert-major hidden rows addressed by compact block metadata plus
destination-local `expert_base` and `src_base_by_expert`. The production/source-padded path remains the default; this
patch proves the count-derived exact W2 addressing contract for block-aligned plans.

- Commit Hash: `44b0ce16d`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - W2 reference and Pallas direct-return kernels now accept destination-local `expert_base` and
    `src_base_by_expert`.
  - Added `make_w2_return_exact_source_plan_inputs`, which builds W2 inputs from the private exact source-push plan and
    tags rows with `w2_input_mode=exact_source_push_plan`.
  - Exact W2 row starts are computed as
    `expert_base[local_expert] + src_base_by_expert[src_rank, local_expert] + local_row_start_within_src_expert`.
  - Existing full-forward/source-padded call sites now pass the bases through to the W2 return kernels while keeping
    `use_exact_expert_major=False`.
- Local verification:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_grugformer_moe.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'w2_exact or exact_source_push_plan_inputs or w2_source_plan_inputs or w2_reference'`
    - Result: `6 passed, 11 warnings in 21.95s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k 'source_push_exact_expert_major_w2_return_matches_reference_on_h100'`
    - Result: `1 skipped, 26 deselected, 1 warning in 0.08s` on non-H100 local hardware.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 verification:
  - Narrow exact-W2 job: `/dlwh/source-push-exact-w2-h100-pytest-44b0ce16d-20260703-210956`
    - Cluster: `cw-us-east-02a`
    - Command:
      ```bash
      uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
        --job-name source-push-exact-w2-h100-pytest-44b0ce16d-20260703-210956 \
        --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
        --enable-extra-resources --extra gpu -- \
        timeout 3600s uv run --package marin-levanter --group test pytest \
        lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 \
        -k source_push_exact_expert_major_w2_return_matches_reference_on_h100
      ```
    - Iris summary: `JOB_STATE_SUCCEEDED`, exit code `0`, failure count `0`, preemption count `0`.
    - Pytest result: `1 passed, 26 deselected, 1 warning in 8.25s`.
  - Broader source-push job: `/dlwh/source-push-exact-w2-h100-source-push-44b0ce16d-20260703-211305`
    - Cluster: `cw-us-east-02a`
    - Command:
      ```bash
      uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
        --job-name source-push-exact-w2-h100-source-push-44b0ce16d-20260703-211305 \
        --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
        --enable-extra-resources --extra gpu -- \
        timeout 3600s uv run --package marin-levanter --group test pytest \
        lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
      ```
    - Iris summary: `JOB_STATE_SUCCEEDED`, exit code `0`, duration `135263 ms`, failure count `0`, preemption count
      `0`.
    - Pytest result: `8 passed, 19 deselected, 1 warning in 112.82s`.
- Interpretation:
  - W2 direct return now has the same count-derived exact row-start proof as W13 for block-aligned plans.
  - The source-padded production path still passes the existing H100 source-push smoke after the W2 signature change.
  - Exact full forward remains a separate integration step because the full-forward host input builder still fixes
    `use_exact_expert_major=False` for the current production/source-padded path.

## 2026-07-03 14:45 - Add exact expert-major full-forward proof

Added a package-private exact full-forward path for block-aligned plans. The default public/source-padded path remains
unchanged; the new exact path lets the full staged forward use the same count-derived row-start contract already proven
for W13 and W2.

- Commit Hash: `183a6ee2d`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Change:
  - Added `make_source_push_forward_exact_source_plan_inputs` and
    `run_source_push_forward_exact_source_plan`.
  - Added a full-forward `use_exact_expert_major` bit through host inputs, device inputs, sharding, staged timing, and
    single-JIT kernel creation.
  - Exact full forward uses `plan.send_meta`, `plan.recv_meta`, `plan.expert_base`, and `plan.src_base_by_expert`
    directly, and rejects tail-block plans because current W13 stores full `block_m` tiles.
  - Source-padded full forward still uses `source_push_source_padded_row_bases` and remains the default path.
- Local verification:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_grugformer_moe.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'forward_exact or forward_inputs_share_one_plan or forward_real_inputs'`
    - Result: `4 passed, 11 warnings in 37.35s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k 'source_push_exact_expert_major_forward_matches_reference_on_h100'`
    - Result: `1 skipped, 27 deselected, 1 warning in 0.10s` on non-H100 local hardware.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 verification:
  - Job: `/dlwh/source-push-exact-forward-h100-source-push-183a6ee2d-20260703-214141`
  - Cluster: `cw-us-east-02a`
  - Command:
    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-exact-forward-h100-source-push-183a6ee2d-20260703-214141 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
      --enable-extra-resources --extra gpu -- \
      timeout 3600s uv run --package marin-levanter --group test pytest \
      lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 -k source_push
    ```
  - Iris summary:
    - State: `JOB_STATE_SUCCEEDED`
    - Task count: `1`
    - Exit code: `0`
    - Duration: about `134451 ms`
    - Failure count: `0`
    - Preemption count: `0`
  - Pytest result:
    - `9 passed, 19 deselected, 1 warning in 113.08s`.
- Interpretation:
  - W13, W2 return, and source combine now round-trip successfully in one staged exact-layout full-forward path for
    block-aligned plans.
  - This still does not make exact layout the production default because tail-block plans require the source-padded
    layout until the W13 store path masks or splits partial tiles.

## 2026-07-03 15:10 - Verify repro wrapper and planner capacity contracts

Confirmed the source-push queue repro wrapper imports the benchmark `main` from
`lib/levanter/scripts/bench/bench_source_push_inbox.py`; the stale package-private `source_push_inbox.main` import is
not present in the branch.

- Code:
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - Added planner coverage for count-derived expert/source offsets, matching accepted assignments back to token/route
    ids.
  - Added capacity clipping coverage against `_clip_receiver_group_sizes`.
  - Added a hard-error check for undersized source-push queue capacity.
- Local verification:
  - `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --help`
    - Result: passed; the wrapper prints the source-push benchmark CLI help.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'repro or source_push_plan_offsets or source_push_plan_capacity or source_push_plan_rejects_queue'`
    - Result: `4 passed, 11 warnings in 40.01s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_plan'`
    - Result: `7 passed, 11 warnings in 36.03s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.

## 2026-07-03 15:35 - Cover source-push stable queue order and padding mask

Added a concrete SourcePushPlan planner regression for the spec's transport order contract. The new case checks that a
source queue is ordered by destination-local expert then stable assignment id, that multi-block local row starts advance
within one `(src, dst, expert)` run, and that tail/unused queue rows stay invalid.

- Code:
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - Added `test_source_push_plan_uses_stable_expert_order_and_masks_padding`.
  - The test covers full, tail, and empty queue entries in one hand-built routing pattern.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_plan_uses_stable_expert_order or source_push_plan_offsets or source_push_plan_capacity or source_push_plan_rejects_queue'`
    - Result: `4 passed, 11 warnings in 12.08s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_plan'`
    - Result: `8 passed, 11 warnings in 12.11s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.

## 2026-07-03 15:55 - Cover combine masking of padded return rows

Added a source combine regression for the invariant that padded queue rows must not affect source-token output. The test
fills invalid return queue rows with large sentinel values and verifies the deterministic route-buffer combine result is
unchanged.

- Code:
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Change:
  - Added `test_source_push_combine_ignores_invalid_padded_queue_rows`.
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_combine_ignores_invalid_padded_queue_rows or source_push_combine_inputs_invert_queue_rows_to_route_slots'`
    - Result: `2 passed, 11 warnings in 11.65s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_plan or source_push_combine'`
    - Result: `13 passed, 11 warnings in 19.89s`.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.

## 2026-07-03 16:04 - Benchmark refactored SourcePushPlan W13 path

Benchmarked the current PR head after the `SourcePushPlan` refactor and planner/combine regression tests. This is the
production profile fed by production-like compact routing metadata, not the older exact-balanced synthetic path.

- Commit Hash: `e35e9357b355573ad4c657b0c19d7b9d512eaf14`
- Job: `/dlwh/source-push-plan-w13-target-e35e9357b-20260703-1555`
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-plan-w13-target-e35e9357b-20260703-1555 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 3600s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_inbox.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --input-mode source_push_plan \
    --warmup 2 --steps 5 --repeat-runs 9 \
    --separate-compile --no-check --no-progress-events \
    --git-sha e35e9357b \
    --jsonl scratch/source_push_plan_w13_target_e35e9357b.jsonl
  ```
- Iris summary:
  - State: `JOB_STATE_SUCCEEDED`
  - Exit code: `0`
  - Task duration: `54084 ms`
  - Failure count: `0`
  - Preemption count: `0`
- Median over 9 repeats:

  | metric | value |
  | --- | ---: |
  | steady_state_time | `8.307660 ms` |
  | useful_w13_tflops_per_rank | `206.795531` |
  | rounded_w13_tflops_per_rank | `219.392084` |
  | effective send bandwidth per rank | `85.700033 GB/s` |
  | plan_row_efficiency | `0.942584283` |
  | plan_masked_row_fraction | `0.057415717` |
  | dropped_routes | `0` |

- Planner shape:
  - `live_entries_total`: `17382`
  - `plan_useful_rows_total`: `1048576`
  - `plan_rounded_rows_total`: `1112448`
  - `plan_layout_padding_rows_total`: `63872`
- Repeat rows:

  | repeat | time ms | useful TFLOP/s/rank | rounded TFLOP/s/rank | send GB/s/rank |
  | ---: | ---: | ---: | ---: | ---: |
  | 0 | `8.307660` | `206.795531` | `219.392084` | `85.700033` |
  | 1 | `8.259195` | `208.008998` | `220.679468` | `86.202917` |
  | 2 | `8.260961` | `207.964531` | `220.632292` | `86.184489` |
  | 3 | `8.328544` | `206.276986` | `218.841954` | `85.485138` |
  | 4 | `8.253787` | `208.145289` | `220.824060` | `86.259399` |
  | 5 | `8.353879` | `205.651396` | `218.178257` | `85.225882` |
  | 6 | `8.327901` | `206.292899` | `218.858836` | `85.491733` |
  | 7 | `8.243126` | `208.414493` | `221.109662` | `86.370962` |
  | 8 | `8.315828` | `206.592406` | `219.176587` | `85.615854` |
- Interpretation:
  - The refactored `SourcePushPlan` path preserves the known rough-balanced W13 performance envelope.
  - This is slightly faster than the prior `8.415 ms` rough-balanced reference and comparable to the previous
    precomputed SourcePushPlan row (`8.441 ms`), with no drops and the expected `5.74%` padded-row tax.

## 2026-07-03 16:31 - Current-head source-push full-forward gate

Reran the three target full-forward rows on current PR head after the exact-layout, planner, and combine-mask commits.
This exercises the integrated `W13 -> W2 direct return -> deterministic source combine` path with the production
source-padded expert-major layout.

- Commit Hash: `0f5e3fa0fcebc1be81267cc57b59be00bef78878`
- Job: `/dlwh/source-push-forward-current-head-gate-0f5e3fa0fc-20260703-232731`
- Cluster: `cw-us-east-02a`
- Command:
  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-forward-current-head-gate-0f5e3fa0fc-20260703-232731 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 7200s bash -lc 'set -euo pipefail
  JSONL=scratch/source_push_forward_current_head_gate_0f5e3fa0fc_20260703.jsonl
  rm -f "$JSONL"
  COMMON="uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_forward.py --source-push-profile hopper_source_push_inbox_rough_balanced_216 --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 5 --separate-compile --no-check --no-progress-events --git-sha 0f5e3fa0fc --jsonl $JSONL"
  $COMMON --routing balanced --capacity-factor 1.0
  $COMMON --routing balanced --capacity-factor 1.25
  $COMMON --routing roughly_balanced --capacity-factor 1.25
  '
  ```
- Iris summary:
  - State: `succeeded`
  - Exit code: `0`
  - Task duration: `167099 ms`
  - Failure count: `0`
  - Preemption count: `0`

Summary medians over 5 repeats:

| routing | cf | stage | median time | useful TFLOP/s/rank | rounded TFLOP/s/rank | row efficiency | masked rows | drops |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | 1.0 | total | `13.461 ms` | `191.44` | `191.44` | `1.000000` | `0.000000` | `0` |
| balanced | 1.0 | W13 | `8.072 ms` | `212.82` | `212.82` | `1.000000` | `0.000000` | `0` |
| balanced | 1.0 | W2 return | `3.929 ms` | - | - | `1.000000` | `0.000000` | `0` |
| balanced | 1.0 | combine | `1.341 ms` | - | - | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | total | `13.487 ms` | `191.07` | `191.07` | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | W13 | `8.133 ms` | `211.24` | `211.24` | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | W2 return | `3.899 ms` | - | - | `1.000000` | `0.000000` | `0` |
| balanced | 1.25 | combine | `1.336 ms` | - | - | `1.000000` | `0.000000` | `0` |
| roughly_balanced | 1.25 | total | `14.100 ms` | `182.76` | `193.90` | `0.942584` | `0.057416` | `0` |
| roughly_balanced | 1.25 | W13 | `8.529 ms` | `201.44` | `213.71` | `0.942584` | `0.057416` | `0` |
| roughly_balanced | 1.25 | W2 return | `4.112 ms` | - | - | `0.942584` | `0.057416` | `0` |
| roughly_balanced | 1.25 | combine | `1.337 ms` | - | - | `0.942584` | `0.057416` | `0` |

- Interpretation:
  - Current head still clears the W13 rough-balanced target gate: useful `201.44 TFLOP/s/rank` and median `8.529 ms`,
    with no row drops.
  - The integrated full forward is a little slower than the prior `89f3267fc` gate on rough-balanced total
    (`13.876 ms` -> `14.100 ms`) and W13 (`8.415 ms` -> `8.529 ms`), but remains within the W13 acceptance bar.
  - W2 return and combine are stable at about `4.1 ms` and `1.34 ms`; the route-buffer combine remains a meaningful
    full-forward tax after proving invertibility.

## 2026-07-03 16:56 - Replace route-buffer combine with direct gather-sum

Replaced the package-private source-push combine Pallas kernel with a direct deterministic gather-sum kernel. The old
kernel scattered weighted route outputs into a dense `[T, K, D]` route buffer and then summed over `K`; the new kernel
launches over `(token_block, D_tile)`, gathers each token's `K` returned route rows, accumulates in fixed route-slot
order, and writes `[T, D]` directly. This removes the large route-buffer write/read and keeps the planner reference
combine as the correctness oracle.

- Commit Hash: `c32e5dd06`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_combine.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
  - `lib/levanter/tests/grug/test_grugformer_moe.py`
- Local verification:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_combine.py lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_grugformer_moe.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'source_push_combine or source_push_forward_inputs_share_one_plan'`
    - Result: `6 passed, 11 warnings in 14.36s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'source_push_stage_kernels_match_references_on_h100'`
    - Result: `3 skipped, 11 warnings in 10.14s` locally, as expected off H100.
  - `./infra/pre-commit.py --changed-files --fix`
    - Result: all checks passed.
- H100 correctness smoke:
  - Job: `/dlwh/source-push-direct-combine-smoke-a8bd45f6b8-20260703-234052`
  - Config: small EP8, `T/rank=64`, `D=128`, `K=4`, `--check`.
  - Result: succeeded, `combine_mode=direct_gather_sum`, `dropped_routes=0`, `max_abs_diff=0.001953125`,
    `mean_abs_diff=0.000266954`.
- Target combine-only timing:
  - Job: `/dlwh/source-push-direct-combine-target-a8bd45f6b8-20260703-234353`
  - Config: `hopper_source_push_inbox_rough_balanced_216`, `routing=roughly_balanced`, `capacity_factor=1.25`,
    `warmup=2`, `steps=5`, `repeat_runs=9`, `--separate-compile`, `--no-check`.
  - Result: succeeded, `dropped_routes=0`, `row_efficiency=0.942584283`, `route_buffer_elements_per_rank=0`.
  - Median over 9 repeats:

    | metric | value |
    | --- | ---: |
    | combine steady_state_time | `0.579442 ms` |
    | direct combine GB/s/rank | `1447.705470` |
    | compile_time | `0.096366 s` |
    | first_run_time | `2.984593 ms` |

- Full-forward gate:
  - Job: `/dlwh/source-push-forward-direct-combine-gate-a8bd45f6b8-20260703-234648`
  - Config: same three target rows as the previous gate, `execution_mode=staged_host_sync`, `warmup=1`, `steps=3`,
    `repeat_runs=5`, `--separate-compile`, `--no-check`.
  - Result: succeeded, no drops/errors.

  | routing | cf | stage | median time | useful TFLOP/s/rank | rounded TFLOP/s/rank | previous time | delta |
  | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
  | balanced | 1.0 | total | `12.709 ms` | `202.77` | `202.77` | `13.461 ms` | `-0.752 ms` |
  | balanced | 1.0 | W13 | `8.080 ms` | `212.62` | `212.62` | `8.072 ms` | `+0.008 ms` |
  | balanced | 1.0 | W2 return | `3.948 ms` | - | - | `3.929 ms` | `+0.019 ms` |
  | balanced | 1.0 | combine | `0.600 ms` | - | - | `1.341 ms` | `-0.741 ms` |
  | balanced | 1.25 | total | `12.592 ms` | `204.66` | `204.66` | `13.487 ms` | `-0.895 ms` |
  | balanced | 1.25 | W13 | `8.020 ms` | `214.21` | `214.21` | `8.133 ms` | `-0.113 ms` |
  | balanced | 1.25 | W2 return | `3.888 ms` | - | - | `3.899 ms` | `-0.011 ms` |
  | balanced | 1.25 | combine | `0.584 ms` | - | - | `1.336 ms` | `-0.752 ms` |
  | roughly_balanced | 1.25 | total | `13.282 ms` | `194.02` | `205.84` | `14.100 ms` | `-0.818 ms` |
  | roughly_balanced | 1.25 | W13 | `8.509 ms` | `201.90` | `214.19` | `8.529 ms` | `-0.020 ms` |
  | roughly_balanced | 1.25 | W2 return | `4.079 ms` | - | - | `4.112 ms` | `-0.033 ms` |
  | roughly_balanced | 1.25 | combine | `0.583 ms` | - | - | `1.337 ms` | `-0.754 ms` |

- Full-forward checked smoke:
  - Job: `/dlwh/source-push-forward-direct-combine-check-a8bd45f6b8-20260703-235229`
  - Config: small EP8, `T/rank=64`, `D=128`, `K=4`, `execution_mode=staged_host_sync`, `--check`.
  - Result: succeeded, `combine_mode=direct_gather_sum`, `dropped_routes=0`, `max_abs_diff=0.015625`,
    `mean_abs_diff=0.000549275`, total `steady_state_time=4.333 ms`.
- Interpretation:
  - The source-side route-buffer combine was a real integrated tax. Removing it cuts target combine from about
    `1.34 ms` to `0.58-0.60 ms` and improves rough-balanced full-forward total from `14.100 ms` to `13.282 ms`.
  - W13 and W2 timings are effectively unchanged, so this isolates the win to source combine rather than measurement
    noise in the compute stages.
  - Compare combine times, not combine GB/s, across the route-buffer and direct kernels: the direct path intentionally
    reports fewer bytes because it no longer writes and rereads `[T, K, D]`.

## 2026-07-03 17:13 - Define MLP-level H checkpoint contract

Added a package-private reference boundary for the revised source-push MoE MLP direction. The new boundary treats the
whole expert MLP as the differentiable unit:

```text
x, route_assignments, route_weights, w13, w2 -> source_push_moe_mlp(...) -> y
```

The stable forward residual is now modeled as W13 preactivation `H_expert_major`, not post-SwiGLU `A`. In the reference
contract this is shaped `[dst, local_expert, expert_capacity, 2 * intermediate_dim]`, matching the intended per-rank
production layout `[local_expert, capacity, 2I]` with an outer destination-rank axis for the host/JAX oracle.

- Commit Hash: `e8ace802f`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_mlp.py`
  - `lib/levanter/tests/grug/test_source_push_mlp.py`
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_mlp.py -q`
    - Result: `3 passed, 11 warnings in 10.71s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_mlp.py lib/levanter/tests/grug/test_source_push_mlp.py`
    - Result: all checks passed, including Pyrefly.
- Contract covered by tests:
  - `H_expert_major` stores raw W13 `[gate, up]` rows before `silu(gate) * up`.
  - Forward output matches an independent loop reference.
  - The MLP-level custom VJP matches JAX autodiff of the reference for `x`, `route_weights`, `w13`, and `w2`; the
    `d_route_weights` path is explicitly nonzero and checked.
- Current divergence from production kernels:
  - Existing source-push W13 still stores post-SwiGLU hidden with shape `[rows, I]`.
  - Existing W2 return consumes post-SwiGLU hidden and source combine applies route weights after W2.
  - Next implementation step is to move production W13/W2 to the H boundary: W13 stores `[E, capacity, 2I]`, W2 loads H,
    computes `silu(gate) * up`, applies route weight before W2, and returns/combines source-owned `y`.

## 2026-07-03 17:23 - Add shared H-forward reference target

Added SourcePushPlan-level reference helpers for the staged forward shape the production kernels need to implement:

```text
packed source queue x + W13 -> H_expert_major
H_expert_major + route_weight + W2 -> source return rows
preweighted source return rows -> y
```

This makes the H-boundary target available in the same plan/layout vocabulary as the existing staged W13/W2/combiner
harness. The existing Pallas path is still unchanged and still uses the legacy post-SwiGLU hidden contract; this patch
adds the reference target and tests needed before changing those kernels.

- Commit Hash: `eb896c674`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_plan.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_plan.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- New helpers:
  - `source_push_w13_h(...)`: writes raw W13 `[gate, up]` preactivation to `[dst, local_expert, expert_capacity, 2I]`.
  - `source_push_w2_from_h_return(...)`: recomputes `A = silu(gate) * up`, applies `route_weight * A` before W2, and
    emits source-queue return rows.
  - `source_push_combine_preweighted(...)`: sums return rows that already include route weights.
  - `reference_source_push_forward_h(...)`: source-push forward reference using the new H boundary.
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_mlp.py -q`
    - Result: `13 passed, 11 warnings in 11.38s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'h_forward or forward_h_reference or forward_inputs_share_one_plan'`
    - Result: `12 passed, 11 warnings in 10.23s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_plan.py lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_inbox.py`
    - Result: all checks passed, including Pyrefly.
- Interpretation:
  - The revised MLP boundary is now represented at both the MLP-vectorized reference layer and the shared
    SourcePushPlan/staged-forward reference layer.
  - The remaining forward migration is specifically in Mosaic kernels: W13 must store H instead of post-SwiGLU A, W2
    must load H and queue route weights, and source combine must run unweighted for that path.

## 2026-07-03 17:32 - Add and smoke-test Mosaic W13-H entrypoint

Added a dedicated W13-H Mosaic entrypoint that reuses the source-push inbox transport but stores raw W13 preactivation
instead of post-SwiGLU hidden. The legacy W13 activation-output path is unchanged. The new entrypoint emits flat
expert-major H rows with shape `[dst, hidden_rows_per_rank, 2 * intermediate_dim]`; the existing row metadata maps those
flat rows back to local expert/source offsets.

- Commit Hash: `e0f7fb52d`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- New package-private kernel entrypoints:
  - `_make_w13_h_kernel(...)`
  - `_sharded_w13_h_kernel(...)`
- Local verification:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/tests/grug/test_source_push_inbox.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'w13_h_reference or forward_h_reference'`
    - Result: `2 passed, 11 warnings in 9.67s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/tests/grug/test_source_push_inbox.py`
    - Result: all checks passed, including Pyrefly.
- H100 compile/correctness smoke:
  - Job: `/dlwh/source-push-w13-h-smoke-e0f7fb52d-20260703-1732`
  - Cluster: `cw-us-east-02a`
  - Config: `ep_size=2`, `tokens_per_rank=128`, `hidden_dim=128`, `intermediate_dim=128`, `experts_per_rank=2`,
    `topk=1`, `block_m=64`, `block_k=64`, `block_n=64`, `n_group=1`.
  - Result row:

    ```json
    {"expected_shape": [2, 512, 256], "kernel": "source_push_w13_h", "live_rows": 512, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "shape": [2, 512, 256]}
    ```

- Interpretation:
  - The source-push W13 Mosaic kernel can now produce the H checkpoint directly and exactly matches the bf16 reference
    on a small H100 smoke.
  - This is still not the full production MLP path: W2 must next consume flat H, compute SwiGLU, apply queue route
    weights before W2, and combine without applying weights a second time.

## 2026-07-03 18:00 - Unblock staged H forward W2-from-H lowering

Fixed the staged source-push H forward path so W2 consumes the W13 preactivation checkpoint on H100 for the
target-compatible tile shape. The core issue was Mosaic Lane lowering rejecting a standalone `block_m=64` route-weight
vector in the W2-from-H prologue. The working patch expands receive-order route weights to a `[block_m, block_k]` tile
for the device path and stages gate, up, route-weight tile, and W2 through WGMMA-compatible SMEM before forming
`silu(gate) * up * route_weight`.

- Commit Hash: `e8fbfa85e`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_mlp.py -q -k 'recv_route_weights or forward_inputs_share_one_plan or forward_h_reference or w13_h_reference or source_push_moe_mlp'`
    - Result: `6 passed, 11 warnings in 10.42s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/src/levanter/grug/_moe/source_push_w2_return.py lib/levanter/tests/grug/test_source_push_inbox.py`
    - Result: all checks passed, including Pyrefly.
- Failed H100 lowering attempts:

  | job | outcome |
  | --- | --- |
  | `/dlwh/source-push-forward-h-smoke-ae1c9aed1-20260703-1738` | W2-from-H failed on route-weight vector load: `memref<64xbf16>` must have a multiple of 128 elements. |
  | `/dlwh/source-push-forward-h-smoke-route-weight-tiles-20260703-1802` | Duplicating weights to `[M,2]` loaded, but slicing `[:, 0:1]` failed: `Only arrays with tiled layouts can be sliced`. |
  | `/dlwh/source-push-forward-h-smoke-route-weight-reduce-20260703-1810` | Reducing the duplicated lanes failed: `No support for axes yet`. |
  | `/dlwh/source-push-forward-h-smoke-route-weight-k-tile-20260703-1816` | Expanding weights to `[M,K]` in GMEM failed when directly assigning a strided fragment into WGMMA SMEM: `WGStridedFragLayout(shape=(64, 64), vec_size=4)`. |
  | `/dlwh/source-push-forward-h-smoke-w2h-smem-20260703-1824` | SMEM-staged W2-from-H passed far enough to reach combine; combine then failed for non-target `block_n=64` on the same 64-vector load constraint. |

- Successful H100 smoke:
  - Job: `/dlwh/source-push-forward-h-smoke-w2h-smem-bn128-20260703-1830`
  - Cluster: `cw-us-east-02a`
  - Config: `ep_size=2`, `tokens_per_rank=128`, `hidden_dim=128`, `intermediate_dim=128`, `experts_per_rank=2`,
    `topk=1`, `block_m=64`, `block_k=64`, `block_n=128`, `n_group=1`, staged host-sync execution, `--check`.
  - Result: succeeded, `dropped_routes=0`, `max_abs_diff=0.00048828125`, `mean_abs_diff=0.0000321401749`.

  | stage | steady_state_time | metric |
  | --- | ---: | ---: |
  | total | `1.889415 ms` | `0.138743 GB/s/rank`, `0.013319 rounded TFLOP/s/rank` |
  | W13/H | `0.845100 ms` | `0.019852 W13 TFLOP/s/rank` |
  | W2-from-H return | `0.406873 ms` | `0.020617 W2 TFLOP/s/rank` |
  | combine | `0.605616 ms` | `0.108214 GB/s/rank` |

- Interpretation:
  - The staged H forward now has a working Mosaic path at the MLP boundary: W13 stores H, W2 consumes H and applies
    route weights inside the W2 stage, return writes source-owned rows, and combine runs unweighted.
  - `block_n=64` remains unsupported for direct combine in this path because Mosaic rejects 64-element vector loads.
    The stable/target source-push profile already uses `block_n=128`, so this is not a target-shape blocker.
  - The current route-weight `[M,K]` expansion is a correctness/lowering unblock, not a performance-optimal metadata
    representation. Target-shape forward timing is still needed to quantify the extra route-weight tile traffic and
    SMEM staging cost.

## 2026-07-03 18:04 - Target timing for staged H forward

Ran the staged H-forward path at the stable source-push target profile after the W2-from-H lowering fix.

- Code Hash: `afed92f44`
- Job: `/dlwh/source-push-forward-h-target-afed92f44-20260703-1902`
- Cluster: `cw-us-east-02a`
- Command:

  ```bash
  uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
    --job-name source-push-forward-h-target-afed92f44-20260703-1902 \
    --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 \
    --enable-extra-resources --extra gpu -- \
    timeout 3600s uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_source_push_forward.py \
    --source-push-profile hopper_source_push_inbox_rough_balanced_216 \
    --execution-mode staged_host_sync --warmup 1 --steps 3 --repeat-runs 3 \
    --no-check --git-sha afed92f44 --jsonl scratch/source_push_forward_h_target_afed92f44.jsonl
  ```

- Config: `ep_size=8`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`,
  `experts_per_rank=32`, `topk=4`, `capacity_factor=1.25`, `routing=roughly_balanced`, `block_m=64`,
  `block_k=128`, `block_n=128`, `entries_per_rank=288`, `n_groups_per_job=2`.
- Queue stats: `dropped_routes=0`, `plan_row_efficiency=0.942584283`, `valid_rows_per_rank_mean=131072`,
  `rounded_rows_per_rank_mean=139056`.
- Median over 3 repeats:

  | stage | median steady_state_time | throughput |
  | --- | ---: | ---: |
  | total | `16.581843 ms` | `155.409765 useful TFLOP/s/rank`, `164.876253 rounded TFLOP/s/rank` |
  | W13/H | `8.561189 ms` | `212.895046 W13 TFLOP/s/rank` |
  | W2-from-H return | `7.047878 ms` | `129.303800 W2 TFLOP/s/rank` |
  | combine | `0.895950 ms` | `936.281131 GB/s/rank` |

- Comparison to previous post-SwiGLU source-push forward (`roughly_balanced`, `cf=1.25`, direct combine):

  | stage | H-forward | previous post-SwiGLU path | delta |
  | --- | ---: | ---: | ---: |
  | total | `16.582 ms` | `13.282 ms` | `+3.300 ms` |
  | W13/H | `8.561 ms` | `8.509 ms` | `+0.052 ms` |
  | W2 return | `7.048 ms` | `4.079 ms` | `+2.969 ms` |
  | combine | `0.896 ms` | `0.583 ms` | `+0.313 ms` |

- Interpretation:
  - The H checkpoint forward is production-relevant and still much faster than the older public `pallas_mgpu` target
    forward baseline (`~38.5 ms`), but it gives back about `3.3 ms` versus the post-SwiGLU source-push path.
  - The slowdown is almost entirely in W2-from-H: extra route-weight tile traffic plus gate/up/weight SMEM staging and
    SwiGLU in the W2 prologue add about `3.0 ms`.
  - W13/H remains essentially unchanged from the previous source-push W13 timing, so the W13 H checkpoint itself is not
    the bottleneck.
  - Next optimization target is the W2-from-H prologue: avoid full `[M,K]` route-weight expansion if Mosaic exposes a
    legal row-broadcast load, or move route scaling to an output-tile equivalent if we accept that forward-only
    algebraic rewrite for performance while preserving the MLP-level custom VJP math.

## 2026-07-03 18:18 - Expose staged forward H residual

Added a package-private `source_push_forward_with_h(...)` entrypoint that returns the staged source-push forward output
plus the W13 preactivation checkpoint from the production W13-H buffer. The returned H is the flat source-padded
expert-major layout `[destination_rank, source_padded_expert_major_row, 2 * intermediate_dim]`; it is before SwiGLU and
before route-weight scaling. This does not complete the MLP-level custom VJP integration, but it removes the immediate
forward API gap where the Pallas staged path computed H internally and discarded it.

- Commit Hash: `26d382e06`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_mlp.py -q -k 'forward_with_h or forward_h_reference or w13_h_reference or source_push_moe_mlp'`
    - Result: `6 passed, 11 warnings in 9.75s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/tests/grug/test_source_push_inbox.py`
    - Result: all checks passed, including Pyrefly.
- H100 smoke:
  - Job: `/dlwh/source-push-forward-with-h-smoke-20260703-1815`
  - Cluster: `cw-us-east-02a`
  - Config: `ep_size=2`, `tokens_per_rank=128`, `hidden_dim=128`, `intermediate_dim=128`,
    `experts_per_rank=2`, `topk=1`, `block_m=64`, `block_k=64`, `block_n=128`, staged host-sync execution.
  - Result row:

    ```json
    {"dropped_routes": 0, "h_max_abs_diff": 0.00390625, "h_mean_abs_diff": 1.049041748046875e-05, "kernel": "source_push_forward_with_h_smoke", "max_abs_diff": 0.00048828125, "mean_abs_diff": 3.214017488062382e-05, "observed_h_shape": [2, 256, 256], "observed_y_shape": [2, 128, 128]}
    ```

- Interpretation:
  - The staged Pallas path can now expose the same H checkpoint it uses for W2-from-H, with no extra W13 invocation.
  - The reference test also verifies that changing route weights changes `y` but not returned H, so this API is aligned
    with the intended residual boundary rather than saving post-SwiGLU `A`.
  - Next integration step is to route the production source-push MLP custom VJP forward rule through this staged
    with-H entrypoint and adapt the backward residual to consume the flat production H layout or convert it to the
    compact `[Dst, E, C, 2I]` route-table layout.

## 2026-07-03 18:24 - Add JAX plan gathers for VJP integration

Added pure-JAX helpers that pack source tokens and gather route weights from a fixed `SourcePushPlan` without
host-side `device_get` of differentiable arrays. This is a dependency for making the MLP-level custom VJP traceable:
the plan can remain nondifferentiable/static, while `x` and `route_weights` can flow through JAX gathers into the
staged source-push kernels.

- Commit Hash: `dbe8dbffd`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_plan.py`
  - `lib/levanter/tests/grug/test_source_push_plan.py`
- New helpers:
  - `pack_source_push_tokens_jax(x, plan)`
  - `source_push_queue_route_weights_jax(route_weights, plan)`
  - `source_push_recv_route_weights_jax(route_weights, plan)`
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py -q -k 'jax_pack_and_route_weight_gathers or packs_tokens'`
    - Result: `2 passed, 11 warnings in 7.84s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_plan.py lib/levanter/tests/grug/test_source_push_plan.py`
    - Result: all checks passed, including Pyrefly.
- Interpretation:
  - Under eager `jax.grad`, a custom VJP forward rule sees concrete arrays; under `jax.jit(jax.grad(...))`, it sees
    tracers. The old host helpers therefore cannot sit inside the production MLP VJP forward rule.
  - These helpers match the existing host plan exactly and are covered under `jax.jit`, so the next patch can build a
    preplanned source-push MLP VJP path whose differentiable inputs are packed/gathered on device.

## 2026-07-03 18:37 - Add preplanned staged forward inputs

Added `device_source_push_forward_inputs_from_plan(...)` and `source_push_forward_with_h_from_plan(...)`. These take
fixed `SourcePushForwardHostInputs` metadata plus dynamic `x`, `route_weights`, `w13`, and `w2`; the dynamic arrays are
packed/gathered with the JAX helpers from `dbe8dbffd` before entering the staged Pallas W13-H/W2/combine path. This is
the callable surface needed by the future MLP-level custom VJP forward rule, because it avoids rebuilding `packed_x` or
receive-order route weights on the host from differentiable arrays.

- Commit Hash: `f6a5d026e`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_forward.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_plan.py -q -k 'device_inputs_from_plan_use_dynamic_jax_arrays or jax_pack_and_route_weight_gathers or forward_with_h'`
    - Result: `3 passed, 11 warnings in 8.09s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_forward.py lib/levanter/tests/grug/test_source_push_inbox.py`
    - Result: all checks passed, including Pyrefly.
- H100 verification attempts:

  | job | outcome |
  | --- | --- |
  | `/dlwh/source-push-forward-with-h-from-plan-f6a5d026e-20260703-1828` | Failed before running the smoke: `ImportError: cannot import name 'barrier_test'` from `jax._src.pallas.mosaic_gpu.primitives`. |
  | `/dlwh/source-push-forward-with-h-from-plan-f6a5d026e-20260703-1831` | Same pre-smoke Mosaic import mismatch as the first attempt. |
  | `/dlwh/source-push-forward-with-h-from-plan-f6a5d026e-gpuextra-20260703-1834` | Adding inner `uv run --extra gpu` fixed the Mosaic import mismatch, but failed before the smoke with CuDNN runtime mismatch: runtime `9.10.2`, JAX source compiled with `9.12.0`, followed by `dnn_support != nullptr`. |

- Interpretation:
  - The preplanned path is locally covered for the key contract: dynamic arrays flow into kernel inputs from JAX values,
    not from host buffers captured when the plan was built.
  - H100 verification of this new entrypoint is not complete because all three H100 attempts failed in environment setup
    or first JAX device operation before executing the Pallas source-push smoke. The earlier `source_push_forward_with_h`
    smoke remains valid for the staged H-returning kernel path itself.
  - Next code step is still to wire the MLP-level custom VJP forward through `source_push_forward_with_h_from_plan(...)`
    and use a flat-H-aware backward residual.

## 2026-07-03 18:40 - Add flat-H MLP backward helper

Added a flat production-H backward helper for the source-push MLP reference/custom-VJP path. The compact
`[Dst, E, C, 2I]` backward and flat `[Dst, rows, 2I]` backward now share the same gradient body; the flat path gathers
route H rows with `expert_base[dst, expert] + expert_row`.

- Commit Hash: `97fa31e42`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_mlp.py`
  - `lib/levanter/tests/grug/test_source_push_mlp.py`
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_mlp.py -q -k 'flat_h_backward or custom_vjp or saves_w13'`
    - Result: `3 passed, 11 warnings in 9.46s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_mlp.py lib/levanter/tests/grug/test_source_push_mlp.py`
    - Result: all checks passed, including Pyrefly.
- Interpretation:
  - The MLP backward can now consume the same flat H layout returned by the staged Pallas forward, without converting
    the entire H buffer to compact expert-major form first.
  - The test uses deliberately nonzero `expert_base` offsets and checks the full gradient tuple against compact-H
    backward, so it would catch missing base-offset handling.

## 2026-07-04 00:19 - Add preplanned source-push MLP custom VJP

Added the MLP-level custom VJP surface that consumes a prebuilt `SourcePushForwardHostInputs`/`SourcePushPlan`, runs the
forward through the staged H-returning source-push path, and saves the flat production H buffer for backward. The public
`pallas_mgpu_source_push` adapter now builds one plan, derives the MLP route table from that plan, and enters through
this MLP boundary instead of calling the older forward-only adapter directly.

- Commit Hash: `e121f1d9e`
- Code:
  - `lib/levanter/src/levanter/grug/_moe/source_push_mlp.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_public.py`
  - `lib/levanter/tests/grug/test_source_push_mlp.py`
- New API:
  - `source_push_moe_mlp_from_plan(...)`
  - `source_push_moe_mlp_reference_with_h_flat(...)`
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_mlp.py -q -k 'from_plan or flat_h or custom_vjp or saves_w13'`
    - Result: `4 passed, 11 warnings in 22.52s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_mlp.py -q`
    - Result: `6 passed, 11 warnings in 12.17s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'source_push_backend_requires_concrete_expert_mesh'`
    - Result: `1 passed, 11 warnings in 14.16s`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/_moe/source_push_mlp.py lib/levanter/src/levanter/grug/_moe/source_push_public.py lib/levanter/tests/grug/test_source_push_mlp.py`
    - Result: all checks passed, including Pyrefly.
- H100:
  - No new H100 job launched for this patch. This is a structural VJP/API wiring checkpoint; the staged H-returning
    kernels were already smoked in `/dlwh/source-push-forward-with-h-smoke-20260703-1815`, and the last from-plan H100
    attempts failed before running kernels due environment mismatches documented in the 2026-07-03 18:37 entry.
- Interpretation:
  - The source-push production path now has the intended MLP-level differentiable boundary. The forward residual is flat
    H `[Dst, rows, 2I]`; backward gathers route rows with `expert_base[dst, expert] + expert_row`.
  - The checked-in gradient test runs the new from-plan custom VJP in reference mode and compares `dx`, `d_route_weights`,
    `dw13`, and `dw2` against the independent compact-H MLP reference.
  - Remaining production work is H100 verification of the Pallas from-plan forward under an environment with matching
    JAX/Mosaic/CuDNN bits, then target backward/perf measurement through this MLP boundary.

## 2026-07-04 00:39 - Public MLP VJP H100 smoke

Added a public-boundary compare mode and fixed the sharding annotations needed for dynamic JAX plan gathers under an
explicit expert mesh. The public `pallas_mgpu_source_push` adapter now runs through the preplanned MLP custom VJP on H100
for the reduced smoke shape and matches the package-private staged source-push forward exactly.

- Commit Hashes:
  - `3e27b8d52` - allow `pallas_mgpu_source_push` in the public compare benchmark.
  - `0449c1af9` - add explicit output sharding to source-major dynamic plan gathers.
  - `cd5d2ddba` - add explicit output sharding for receive-order per-source queue slices.
  - `0e7e1ba75` - use `jax.sharding.reshard(...)` for the final receive-order route-weight layout under explicit mesh axes.
- Code:
  - `lib/levanter/scripts/bench/bench_source_push_forward_public_compare.py`
  - `lib/levanter/src/levanter/grug/_moe/source_push_plan.py`
  - `lib/levanter/tests/grug/test_source_push_inbox.py`
- Local verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'public_compare_cli_passes_profile_defaults or public_compare_bench_cli_imports'`
    - Result: `2 passed, 11 warnings in 13.98s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_inbox.py lib/levanter/tests/grug/test_source_push_mlp.py -q -k 'jax_pack_and_route_weight_gathers or device_inputs_from_plan_use_dynamic_jax_arrays or from_plan'`
    - Result: `3 passed, 11 warnings in 17.56s`.
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_plan.py lib/levanter/tests/grug/test_source_push_inbox.py -q -k 'jax_pack_and_route_weight_gathers or device_inputs_from_plan_use_dynamic_jax_arrays'`
    - Result after final patch: `2 passed, 11 warnings in 14.41s`.
  - Touched-file `./infra/pre-commit.py --fix ...`
    - Result: all checks passed, including Pyrefly on Python files.
- H100 failed smoke attempts:

  | job | commit | outcome |
  | --- | --- | --- |
  | `/dlwh/source-push-public-mlp-vjp-smoke-3e27b8d525-20260704-0021` | `3e27b8d52` | Structured failure row: `ShardingTypeError` in `source_push_queue_route_weights_jax` for the first dynamic gather from `route_weights[source, token, slot]` under source-sharded public inputs. |
  | `/dlwh/source-push-public-mlp-vjp-smoke-0449c1af9c-20260704-0031` | `0449c1af9` | Structured failure row: first gather fixed, then `ShardingTypeError` on `queue_weights[src, send_dst_ord]` while reordering to receive order. |
  | `/dlwh/source-push-public-mlp-vjp-smoke-cd5d2ddbaa-20260704-0037` | `cd5d2ddba` | Structured failure row: receive-order tensor built, then explicit mesh `with_sharding_constraint` asserted because the value was replicated `P(None, None, None, None)` rather than destination-sharded `P('expert', None, None, None)`. |

- Successful H100 smoke:
  - Job: `/dlwh/source-push-public-mlp-vjp-smoke-0e7e1ba759-20260704-0041`
  - Cluster: `cw-us-east-02a`
  - Command:

    ```bash
    uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
      --job-name source-push-public-mlp-vjp-smoke-0e7e1ba759-20260704-0041 \
      --cpu 16 --memory 128GB --disk 16GB --gpu H100x2 --reserve H100x2 \
      --enable-extra-resources --extra gpu -- \
      timeout 1800s uv run --package marin-levanter --group test python \
      lib/levanter/scripts/bench/bench_source_push_forward_public_compare.py \
      --ep-size 2 --entries-per-rank 4 --inbox-slots 2 \
      --hidden-dim 128 --intermediate-dim 128 \
      --block-m 64 --block-k 64 --block-n 128 \
      --experts-per-rank 2 --send-worker-programs-per-peer 1 --worker-programs-per-peer 4 \
      --routing balanced --tokens-per-rank 128 --topk 1 --capacity-factor 1.25 \
      --source-push-implementation pallas_mgpu --source-push-execution-mode staged_host_sync \
      --public-implementations pallas_mgpu_source_push \
      --git-sha 0e7e1ba759 --jsonl scratch/source_push_public_mlp_vjp_smoke_0e7e1ba759.jsonl
    ```

  - Result row:

    ```json
    {"dropped_route_delta": 0, "error": null, "git_sha": "0e7e1ba759", "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "output_shape": [2, 128, 128], "public_dropped_routes": 0, "public_implementation": "pallas_mgpu_source_push", "source_push_dropped_routes": 0, "source_push_execution_mode": "staged_host_sync", "source_push_implementation": "pallas_mgpu"}
    ```

- Interpretation:
  - The public adapter now exercises the intended MLP custom VJP forward path on H100, not just the package-private
    staged forward harness.
  - The dynamic plan gathers need explicit sharding because public inputs are source-sharded under an explicit `expert`
    mesh. The receive-order route-weight gather necessarily repartitions metadata from source-major to destination-major
    layout before W2-from-H consumes it.
  - This is a reduced forward-only smoke. Target-shape forward/backward timing through the public MLP VJP remains to be
    measured by the fwd/bwd measurement workstream.

## 2026-07-04 00:45 - JIT custom VJP gradient check

Promoted the from-plan source-push MLP custom VJP gradient check from eager-only coverage to `jax.jit(jax.grad(...))`
coverage in reference mode. This is a small but important training-path check: the MLP custom VJP closes over the fixed
plan metadata and differentiates only `x`, `route_weights`, `w13`, and `w2`, so the VJP needs to remain valid once the
loss/gradient is compiled rather than only when run eagerly.

- Commit Hash: `dba20156d`
- Code:
  - `lib/levanter/tests/grug/test_source_push_mlp.py`
- Verification:
  - `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_source_push_mlp.py -q -k 'from_plan or flat_h or custom_vjp or saves_w13'`
    - Result: `5 passed, 11 warnings in 14.37s`.
  - `./infra/pre-commit.py --fix lib/levanter/tests/grug/test_source_push_mlp.py`
    - Result: all checks passed.
- Interpretation:
  - Local CPU/reference evidence now covers both eager and JIT-transformed custom VJP gradients for the preplanned MLP
    boundary.
  - This does not replace H100 fwd/bwd measurement through `pallas_mgpu_source_push`; that remains assigned to the
    measurement workstream.

## 2026-07-05 22:05 - Blackwell target-shape forward tuning

Measured the stacked Blackwell source-push forward path at the target local EP=8 shape with 65K tokens/rank,
D3072/I3072, topk=4, experts_per_rank=32, and the 576-entry copy profile.

- Code state:
  - Branch: `codex/blackwell-source-push-stack`
  - Stack base: source-push inbox production PR branch
  - Stack commit under test: `50d093ef3`
- B300 full-forward baseline:
  - Artifact: `blackwell_forward_baseline_b_64920.jsonl`
  - Median: 666.98 useful / 687.30 rounded TFLOP/s/rank
  - Median time: 22.25 ms
  - Dropped routes: 0
- B300 queue-parameter sweep:
  - Artifact: `blackwell_forward_tune_b_64921.jsonl`
  - Best median: 668.53 useful / 688.89 rounded TFLOP/s/rank
  - Best median time: 22.20 ms
  - Best config delta: `inbox_slots=48`; `entries_per_rank=576`, `send_worker_programs_per_peer=4`,
    `worker_programs_per_peer=32`, `n_groups_per_job=2`, `block_m=64`, `block_k=128`, `block_n=128`
  - Interpretation: the gain is under 0.3% versus the reproduced baseline, so it is not enough evidence to change the
    checked-in profile default. Larger `entries_per_rank` values regressed due extra padded hidden capacity.
- B200 full-forward baseline:
  - Artifact: `blackwell_forward_b200_known_env_64929.jsonl`
  - Median: 637.97 useful / 657.40 rounded TFLOP/s/rank
  - Median time: 23.27 ms
  - Dropped routes: 0
  - Interpretation: the known-good environment runs on B200. The earlier fresh-environment B200 attempt failed with a
    CUDA illegal-address error before producing timing rows, so that failure is an environment/toolchain datapoint rather
    than a source-push performance result.
- Stage timing:
  - Patched `bench_blackwell_source_push_forward_smoke.py` to use an explicit mesh, matching the working fwd/bwd harness.
  - After the mesh fix, standalone B300 stage timing hit a PTX/toolchain compile failure for `sm_103a` before writing
    timing rows. No stage timing result is recorded from that harness.

Current conclusion: full forward is still around 657-689 rounded TFLOP/s/rank depending on B200/B300, below the
800 TFLOP/s/rank goal. More blind queue tuning is unlikely to close the gap; the next useful step is a decomposed
Blackwell timing path that runs under the same explicit-mesh fwd/bwd benchmark harness.

## 2026-07-05 23:22 - B200 W2 tile tuning and stage split

Added a Blackwell-compatible decomposed forward timing path to the fwd/bwd harness and fixed the standalone Blackwell
forward smoke stage harness to use the same explicit mesh and replicated destination-transport base arrays as the
production path. These harness fixes produced B200 stage attribution and guided a focused W2 tile change.

- Code state:
  - Branch: `codex/blackwell-source-push-stack`
  - Base commit before this entry: `90cdbb0aa`
- B200 decomposed forward:
  - Artifact: `blackwell_forward_decomp_b200_64935.jsonl`
  - Staged compute/transport medians, excluding un-jitted pack diagnostic:
    - destination+W13 bucket: 10.79 ms
    - W2+return bucket: 9.05 ms
    - combine: 1.52 ms
  - Interpretation: the gap is split between the destination/W13 side and W2/return side; queue depth and worker count
    are not the main limit.
- B200 fine-stage forward:
  - Artifact: `blackwell_forward_fine_stage_b200_64937.jsonl`
  - Prepared-input median: 20.37 ms, 750.79 useful TFLOP/s/rank
  - Fine-stage medians:
    - destination transport: 6.39 ms
    - W13: 5.66 ms
    - W2: 4.76 ms
    - return transport: 4.50 ms
    - combine: 1.49 ms
- B200 queue/worker sweep:
  - Artifact: `blackwell_forward_tune_b200_64938.jsonl`
  - Best median: 637.85 useful / 657.28 rounded TFLOP/s/rank with `inbox_slots=32`
  - Baseline replicate: 637.14 useful / 656.54 rounded TFLOP/s/rank
  - `entries_per_rank=512` overflowed queue capacity; `block_m=128` regressed to 425.89 useful / 452.87 rounded.
  - Interpretation: queue/worker knobs are flat and do not move toward 800.
- B200 W2 tile tuning:
  - Artifact: `blackwell_w2n128_b200_64939.jsonl`
  - Change: W2 `tile_n=128` instead of 64, with `tile_m=128`, `tile_k=64`, `max_concurrent_steps=6`,
    `epilogue_tile_n=64`
  - Full-forward median: 665.22 useful / 685.48 rounded TFLOP/s/rank, 22.31 ms, 0 dropped routes
  - Prepared-input median: 19.19 ms, 797.15 useful TFLOP/s/rank
  - Fine-stage W2 improved from 4.76 ms to 3.98 ms; other stages were effectively unchanged.
  - `epilogue_tile_n=128` was neutral/slightly worse at 665.18 useful / 685.44 rounded.
  - Larger W2 variants failed shared-memory limits:
    - `tile_n=128,tile_k=128`: 409,780 bytes > 232,448
    - `tile_n=256,tile_k=64`: 311,476 bytes > 232,448
    - `tile_n=128,tile_k=64,max_concurrent_steps=8`: 278,756 bytes > 232,448
- B300 status:
  - The W2 `tile_n=128` config fails fresh B300 compilation with the PTX 8.7 / `sm_103a` toolchain error.
  - The checked-in change therefore selects the W2 `tile_n=128` config only on B200 and keeps `tile_n=64` as the default
    fallback for other Blackwell devices. A B300 selector probe confirmed the fallback returns the `tile_n=64` config,
    but fresh B300 full-forward compilation still hit the same PTX toolchain error in this environment.

Current conclusion: B200 full forward improved from about 657 rounded to about 685 rounded TFLOP/s/rank. The
prepared-input path is now at the 800 useful TFLOP/s/rank boundary, but the full forward path remains below 800 due
remaining input-prep/transport overhead. The next useful experiment is to reduce the destination and return transport
costs, not more queue-depth tuning.

## 2026-07-06 00:00 - B200 transport block tuning

Tuned the B200 destination and return transport copy tile dimensions after the W2 tile win. These knobs are in
`PushInboxConfig`: destination X transport uses `block_k`, return transport uses `block_n`.

- Code state:
  - Branch: `codex/blackwell-source-push-stack`
  - Base commit before this entry: `2d7aad079`
- B200 transport sweep:
  - Artifact: `blackwell_transport_sweep_b200_64951.jsonl`
  - Best median: 674.87 useful / 695.43 rounded TFLOP/s/rank with `block_k=256`, `block_n=256`
  - Baseline in same sweep: 664.75 useful / 684.99 rounded with `block_k=128`, `block_n=128`
  - `block_k=512` and `block_n=512` variants failed the async-copy limit of 256 elements per dimension.
- B200 combined sweep:
  - Artifact: `blackwell_transport_combo_b200_64952.jsonl`
  - Best median: 677.33 useful / 697.95 rounded TFLOP/s/rank with `block_k=256`, `block_n=256`,
    `inbox_slots=32`, `n_groups_per_job=1`
  - `inbox_slots=48` and `inbox_slots=32,n_groups_per_job=2` were worse.
- Checked-in profile update:
  - The Blackwell 65K/D3072/I3072 profile now defaults to `block_k=256`, `block_n=256`, `inbox_slots=32`,
    `n_groups_per_job=1`.
  - B200 named-profile verification artifact: `blackwell_profile_verify_b200_64953.jsonl`
  - Verified median: 675.08 useful / 695.64 rounded TFLOP/s/rank, 21.99 ms, 0 dropped routes.

Current conclusion: the best B200 full-forward profile is now about 696 rounded TFLOP/s/rank. The 800 full-forward goal
is still not met. The remaining gap is likely in the Lane transport stages plus input preparation; larger transport
tiles beyond 256 are not viable because of async-copy limits.

## 2026-07-06 03:20 - B200 full-forward verification and block_m sweep

Re-synced the clean `codex/blackwell-source-push-stack` source copy at `4d4a04889` and re-ran B200 forward timing on the
current Blackwell 65K/D3072/I3072 profile.

- Code state:
  - Branch: `codex/blackwell-source-push-stack`
  - Commit: `4d4a04889`
- Fine-stage diagnostic:
  - Artifact: `bw-fwd-stage-64957.out`
  - Prepared-input median: 17.90 ms, 854.71 useful TFLOP/s/rank
  - Stage medians including dynamic input prep:
    - input prep: 58.88 ms
    - destination transport: 5.41 ms
    - W13: 5.66 ms
    - W2: 3.72 ms
    - return transport: 3.98 ms
    - combine: 1.09 ms
  - Interpretation: the staged-device-sync diagnostic is useful for prepared-input and per-stage attribution, but its
    un-jitted input-prep timing should not be compared to the production-style outer-JIT forward number.
- MLP forward verification:
  - Artifact: `blackwell_mlp_forward_outerjit_64960.jsonl`
  - Median: 696.28 useful / 717.49 rounded TFLOP/s/rank, 21.32 ms, 0 dropped routes
  - Non-outer-JIT control artifact: `blackwell_mlp_forward_64959.jsonl`
  - Non-outer-JIT median: 168.78 useful / 173.92 rounded TFLOP/s/rank, 87.95 ms
  - Interpretation: use `--outer-jit true` for production-style full-forward timing; otherwise the benchmark mostly
    measures staged Python/dispatch overhead.
- `block_m` sweep with capacity held roughly constant:
  - Artifact: `blackwell_mlp_blockm_sweep_64961.jsonl`
  - `block_m=32`, `entries_per_rank=1152`, `inbox_slots=64`: 699.58 useful / 710.28 rounded TFLOP/s/rank, 21.22 ms
  - `block_m=48`, `entries_per_rank=768`, `inbox_slots=48`: 698.02 useful / 714.19 rounded TFLOP/s/rank, 21.27 ms
  - Both had 0 dropped routes.
  - Interpretation: smaller `block_m` improves row efficiency but adds queue-entry overhead, so it does not materially
    close the gap to 800. The best useful result is now about 700 TFLOP/s/rank; the best rounded result remains about
    717 TFLOP/s/rank.

Current conclusion: the checked-in B200 profile is still the most conservative default. A `block_m=32` profile is a
small exploratory win on useful TFLOP/s/rank but a rounded regression; do not promote it without more evidence. The
remaining full-forward gap is about 14% versus the 800 useful TFLOP/s/rank target, while the prepared-input path is
already above 800. The next worthwhile direction is structural reduction of compiled input preparation / dispatch
overhead or further transport fusion, not simple queue/block-size sweeps.

## 2026-07-06 03:45 - B200 raw-token destination transport

Replaced the dynamic Blackwell staged full-forward input path with a raw-token destination transport. The previous
outer-JIT full-forward path gathered `x` into `[src, dst, entry, row, d]` with `pack_source_push_tokens_jax` before the
Blackwell destination copy. The new Blackwell-only path carries raw source-major `x` plus static `token_ids` into a
Lane transport kernel and gathers each valid row directly while writing destination-local `x`. Prepared-input and
non-Blackwell paths still use the packed layout.

- Code state:
  - Branch: `codex/blackwell-source-push-stack`
  - Base before this entry: `8051b92e4`
- Smoke:
  - Artifact: `blackwell-rawx-forward-64967.out`
  - Small B200 staged-device-sync smoke passed with 0 drops and matching output/H tolerances.
  - The raw-token Lane transport lowered and ran on B200.
- Target full-forward benchmark:
  - Artifact: `blackwell_rawx_forward_64967.jsonl`
  - Command shape: Blackwell 65K/D3072/I3072 profile, `source_push_blackwell_staged`, `forward`, `--outer-jit true`,
    `--separate-compile`, 3 repeat rows.
  - Repeat useful TFLOP/s/rank: 893.30, 887.25, 895.16
  - Repeat rounded TFLOP/s/rank: 920.51, 914.27, 922.42
  - Median: 893.30 useful / 920.51 rounded TFLOP/s/rank, 16.62 ms, 0 dropped routes
  - Compile split: 4.89 s lower compile, 2.66 s first run, 7.54 s total first call
- Post-commit confirmation:
  - Artifact: `blackwell_rawx_target_64968.jsonl`
  - Commit: `4c575fd6c`
  - Median: 894.67 useful / 921.92 rounded TFLOP/s/rank, 16.59 ms, 0 dropped routes
  - Compile split: 2.98 s lower compile, 2.57 s first run, 5.55 s total first call

Current conclusion: the full B200 outer-JIT forward path now exceeds the 800 useful TFLOP/s/rank target. The structural
win came from removing the compiled packed-token gather from the Blackwell dynamic path, not from another tile sweep.
Next useful check is a small regression test that verifies the raw-token Blackwell path remains numerically aligned with
the packed prepared-input path.
