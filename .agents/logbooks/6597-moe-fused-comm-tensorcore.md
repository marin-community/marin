---
topic: Fused communication and tensor-core expert-parallel MoE
issue: https://github.com/marin-community/marin/issues/6597
pull_request: https://github.com/marin-community/marin/pull/6841
description: Persistent fused Hopper kernels built from the JAX-native semantic source-push plan.
author: David Hall
---

# 6597 Fused Communication and Tensor-Core MoE Logbook

Architecture spec: `.agents/projects/20260710_moe_fused_comm_tensorcore.md`

Prior logbook: `.agents/logbooks/6597-moe-mgpu-forward.md`

## Fixed Target

```text
EP ranks:               8
tokens/rank:            32768
top-k:                  4
experts/rank:           32
hidden dimension:       2560
intermediate dimension: 1280
routing:                roughly balanced
target:                 250 useful TFLOP/s/rank aggregate forward + backward
```

## Hypothesis Queue

| ID | Hypothesis | Status | Decisive evidence |
| --- | --- | --- | --- |
| FUSED-MOE-001 | B256 sends feeding B64 WGMMA consumers overlap semantic permutation with W13. | Running | Target H100 fused, copy-only, compute-only, and split medians. |
| FUSED-MOE-002 | Streaming W2 output directly to source-owned return storage removes the dense semantic return tax. | Proposed | Fused W2+return+combine target median and parity. |
| FUSED-MOE-003 | Source-owned dcombine producers can feed W2 backward without replicated `dy`. | Proposed | No all-gather in lowered graph plus target timing and parity. |
| FUSED-MOE-004 | W13 backward can stream source x and return dX while accumulating dW13. | Proposed | Fused dX/dW13 target timing and parity. |

## 2026-07-10 FUSED-MOE-001 - Semantic fused permute + W13 implementation snapshot

Implemented the first persistent fused stage from the architecture spec. The
new semantic kernel lowers four B64 expert-row blocks into one B256 send
generation, partitions each generation into parallel B64xK256 producer tiles,
and runs Lane-lowered WGMMA consumers directly from the rolling destination
inbox. Producer completion, full publication, consumer completion, and slot
release use cumulative generation semaphore targets.

The package-private semantic API builds all route/chunk metadata with JAX and
returns source-padded expert-major W13 preactivation and validity. The benchmark
alias `semantic_permute_w13` includes the optimized fused mode, the existing
split inbox path, copy-only pack, and compute-only prepacked W13 baseline. No
new production tuning flags were added.

Local verification:

```text
semantic raw-token adapter tests:       7 passed
semantic decomposition harness tests: 32 passed
parallel fused W13 kernel tests:        4 passed
optimized fused benchmark smoke:       1 passed
combined focused suite:                43 passed
scoped pre-commit:                     passed
```

No GPU claim is attached to this snapshot. Next run: one H100x8 target-shape
compile/correctness/decomposition comparison of the optimized parallel producer,
the older raw-token fused adapter, and the split copy/compute baselines.

## 2026-07-10 FUSED-MOE-001 - First B256 target H100 run

Job:

```text
cluster:  cw-rno2a
job:      /dlwh/bench-semantic-fused-w13-b256-target-20260710-1325
commit:   3b334d7510
Iris:     JOB_STATE_SUCCEEDED, exit 0, failures 0, preemptions 0
task 0:   succeeded, exit 0
submitted: 2026-07-10T20:24:02.801Z
started:   2026-07-10T20:24:05.360Z
finished:  2026-07-10T20:26:09.273Z
resources: H100x8, 16 CPU, 128 GiB memory, 16 GiB disk
```

Exact task command recorded by Iris `GetJobStatus`:

```bash
timeout 3600s bash -lc 'set -euo pipefail; uv pip install --reinstall nvidia-cudnn-cu13==9.19.0.56; exec uv run --no-sync --package marin-levanter --extra gpu --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_permute_w13_pallas,semantic_permute_w13_compare,w13_source_padded_inbox_pallas,source_padded_inbox_pack_pallas,w13_expert_major_prepacked_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha 3b334d7510 --jsonl scratch/semantic_fused_w13_b256_target_3b334d7510.jsonl'
```

The Iris job succeeded because the harness records mode failures as JSON rows
and continues. The new fused target and its correctness comparison both failed
before kernel compilation, so this run does not establish fused correctness or
performance:

```text
mode: semantic_permute_w13_pallas
rows: 0 repeats, 1 error; summary = all repeats failed

mode: semantic_permute_w13_compare
rows: 0 repeats, 1 error; summary = all repeats failed
```

First actionable failure: `_source_push_semantic_fused_w13_sharded` supplies
`shard_map` input partition specs one rank longer than the corresponding global
arrays. In particular, `token_ids_local` is rank 5 but receives a rank-6 spec;
`send_valid_local`, `recv_expert_local`, `recv_row_local`, `recv_valid_local`,
and `weights_local` are rank 4 but receive rank-5 specs. JAX raises:

```text
ValueError: shard_map ... in_specs entry ... is too long to be compatible with
the corresponding input value
```

Routing metadata was otherwise healthy in both error rows:

```text
dropped_routes=0
routing_dropped_routes=0
metadata_overflow_routes=0
semantic_live_pairs=64
semantic_useful_rows=1048576
semantic_rounded_rows=1310720
semantic_row_efficiency=0.8
semantic_masked_row_fraction=0.2
```

Every numeric timing row emitted by the successful controls:

| Mode | Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `w13_source_padded_inbox_pallas` | 0 | 22.814061 | 75.303864 | 94.129830 | 1305986465792.0 |
| `w13_source_padded_inbox_pallas` | 1 | 22.845354 | 75.200715 | 94.000893 | 1305986465792.0 |
| `w13_source_padded_inbox_pallas` | 2 | 22.813910 | 75.304361 | 94.130451 | 1305986465792.0 |
| `source_padded_inbox_pack_pallas` | 0 | 4.888674 | n/a | n/a | 3337592.75 |
| `source_padded_inbox_pack_pallas` | 1 | 4.865558 | n/a | n/a | 3337592.75 |
| `source_padded_inbox_pack_pallas` | 2 | 4.865371 | n/a | n/a | 3337592.75 |
| `w13_expert_major_prepacked_pallas` | 0 | 6.307397 | 272.376518 | 340.470648 | 478198136832.0 |
| `w13_expert_major_prepacked_pallas` | 1 | 6.319413 | 271.858637 | 339.823296 | 478198136832.0 |
| `w13_expert_major_prepacked_pallas` | 2 | 6.299846 | 272.703016 | 340.878770 | 478198136832.0 |

Repeat medians and ranges:

| Mode | Median (ms) | Min-max (ms) | Median useful TFLOP/s/rank | Median rounded TFLOP/s/rank | Compile / first run (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `w13_source_padded_inbox_pallas` | 22.814061 | 22.813910-22.845354 | 75.303864 | 94.129830 | 10.392467 / 6.656227 |
| `source_padded_inbox_pack_pallas` | 4.865558 | 4.865371-4.888674 | n/a | n/a | 0.515017 / 0.022169 |
| `w13_expert_major_prepacked_pallas` | 6.307397 | 6.299846-6.319413 | 272.376518 | 340.470648 | 0.374055 / 0.008236 |

All three successful control modes reported `dropped_routes=0`,
`routing_dropped_routes=0`, and `metadata_overflow_routes=0`. The inbox and
pack modes additionally reported zero queue-entry, queue-route, and layout-row
overflow errors in every repeat. No fused comparison error metric exists because
the compare mode failed at `shard_map` validation.

Decision: keep FUSED-MOE-001 running. Shorten each `in_specs` tuple by one axis
to describe the global argument rank, rerun focused tests, then repeat this exact
target benchmark. The non-fatal 12.50 GiB allocator warnings should be watched,
but they were not the first failure and the control modes completed.

## 2026-07-10 FUSED-MOE-002 - B256 shard-spec retry

Job:

```text
cluster:  cw-rno2a
job:      /dlwh/bench-semantic-fused-w13-b256-shardspec-20260710-1331
commit:   867d1f6928
Iris:     succeeded, exit 0, failures 0, preemptions 0
task 0:   succeeded, exit 0, duration 1m55.31s
submitted: 2026-07-10T20:30:52.722Z
resources: H100x8, 16 CPU, 128 GiB memory, 16 GiB disk
```

The retry used the same target-shape benchmark command as FUSED-MOE-001, with
the shard-spec fix at `867d1f6928` and a new JSONL destination:

```bash
timeout 3600s bash -lc 'set -euo pipefail; uv pip install --reinstall nvidia-cudnn-cu13==9.19.0.56; exec uv run --no-sync --package marin-levanter --extra gpu --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_permute_w13_pallas,semantic_permute_w13_compare,w13_source_padded_inbox_pallas,source_padded_inbox_pack_pallas,w13_expert_major_prepacked_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha 867d1f6928 --jsonl scratch/semantic_fused_w13_b256_shardspec_867d1f6928.jsonl'
```

The fused B256 path compiled and ran. Every fused timing row:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Checksum |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 29.998302 | 57.269471 | 71.586839 | 1305986465792.0 |
| 1 | 30.034369 | 57.200699 | 71.500874 | 1305986465792.0 |
| 2 | 29.959819 | 57.343034 | 71.678793 | 1305986465792.0 |

Summary: median `29.998302 ms`, range `29.959819-30.034369 ms`, median
`57.269471` useful and `71.586839` rounded TFLOP/s/rank. Compilation was
`10.327229 s`; first run was `6.704749 s`. Every timing repeat reported:

```text
dropped_routes=0
routing_dropped_routes=0
metadata_overflow_routes=0
queue_overflow_entry_error_count=0
queue_overflow_route_error_count=0
layout_overflow_row_error_count=0
semantic_useful_rows=1048576
semantic_rounded_rows=1310720
semantic_row_efficiency=0.8
```

The compare mode also emitted all three repeats. It measures the full comparison
path, not fused-kernel performance:

| Repeat | Time (ms) | z max / mean abs diff | h max / mean abs diff | Valid errors | Overflow errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 791.713158 | 0.03125 / 0.000018104 | 25.965332 / 1.877868 | 0 | 0 |
| 1 | 791.447083 | 0.03125 / 0.000018104 | 25.965332 / 1.877868 | 0 | 0 |
| 2 | 793.388347 | 0.03125 / 0.000018104 | 25.965332 / 1.877868 | 0 | 0 |

Compare summary: median `791.713158 ms`, range
`791.447083-793.388347 ms`; `valid_error_count=0`, all observed and expected
nonfinite counts were zero, and queue-entry, queue-route, layout-row, and
layout-overflow-mismatch counts were zero. The sampled live counts were 3,776
of 4,096 for both `z` and `h`.

Correctness conclusion: `z` is within the established bf16 tolerance, but the
post-SwiGLU `h` result is not correct (`max_abs_diff=25.965332`,
`mean_abs_diff=1.877868`). Therefore this run establishes compilation and a
stable approximately 30 ms fused timing, but not fused permute+W13 correctness.
The next action is to isolate the `h` activation/output-placement mismatch before
interpreting or tuning the 57.27 useful TFLOP/s/rank result.

## 2026-07-10 FUSED-MOE-003 - Full fused-stage scaffold checkpoint

Added package-private persistent-kernel scaffolds for the remaining three fused
stages in the architecture spec:

```text
source_push_semantic_fused_w2_return.py
source_push_semantic_fused_w2_backward.py
source_push_semantic_fused_w13_backward.py
```

Each stage has a JAX semantic reference/interpret path and a Lane-lowered Mosaic
GPU path using explicit `mgpu.wgmma`. The forward W2 stage writes bf16 route
values directly into source-owned storage and combines top-k on source. The W2
backward stage consumes source-sharded `dy`, computes route-weight gradients,
and sends weighted dy chunks without an all-gather. The W13 backward stage
rematerializes source x, returns dX into bounded source-owned slots, and combines
top-k on source.

Before sealing the checkpoint, removed rectangular selector scans from the
forward W13 and W2-return schedules. For W2 backward, changed dW2 ownership from
all 32 expert tiles per chunk to the at-most-four B64 expert blocks represented
by a B256 chunk; the first block for a repeated expert accumulates every matching
block. This reduces per-chunk dW2 jobs at the target shape from 6,400 to 800.
W13 backward already assigns one persistent owner per dW13 tile, so its
unnecessary fp32 GMEM atomic was replaced with owner-local read/add/store.
Sharded-wrapper partition specs were corrected to match global argument ranks.

Local verification after these changes:

```text
fused W13 forward:       5 passed
fused W2 return/combine: 4 passed
fused W2 backward:       5 passed
fused W13 backward:      4 passed
combined:               18 passed
scoped pre-commit:      passed
```

The FUSED-MOE-002 `h` comparison needs a precision-contract check before being
classified as output-placement corruption: observed `h` is recomputed from the
bf16 stored `z`, while the sampled reference may apply SwiGLU before the same
bf16 boundary. The `z` mean/max differences were `1.81e-5`/`0.03125`, validity
and placement counters were exact, and both sides were finite. A fair compare
must quantize the reference at the production boundary or compare the kernel's
intended direct-H output.

No H100 claim is attached to the three new stages. Next experiment: compile-only
target runs for W2-return, W2-backward, and W13-backward, followed by timing only
for stages that compile and pass correctness.

## 2026-07-10 FUSED-MOE-004 - Direct consumer-job indexing target rerun

Job `/dlwh/bench-semantic-fused-w13-direct-jobs-20260710-1348` ran on
`cw-rno2a` at commit `cc588798a8` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 1 minute 33.53 seconds. No retry was launched.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name bench-semantic-fused-w13-direct-jobs-20260710-1348 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu -- timeout 3600s bash -lc 'set -euo pipefail; uv pip install --reinstall nvidia-cudnn-cu13==9.19.0.56; exec uv run --no-sync --package marin-levanter --extra gpu --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_permute_w13_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha cc588798a8 --jsonl scratch/semantic_fused_w13_direct_jobs_cc588798a8.jsonl'
```

Every fused timing row:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Checksum |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 30.307594 | 56.685031 | 70.856288 | 1305986465792.0 |
| 1 | 29.704949 | 57.835041 | 72.293801 | 1305986465792.0 |
| 2 | 29.942560 | 57.376087 | 71.720108 | 1305986465792.0 |

Summary: median `29.942560 ms`, range `29.704949-30.307594 ms`, median
`57.376087` useful and `71.720108` rounded TFLOP/s/rank. Compilation was
`10.805683 s`; first run was `6.716892 s`. Every repeat reported:

```text
dropped_routes=0
routing_dropped_routes=0
metadata_overflow_routes=0
queue_overflow_entry_error_count=0
queue_overflow_route_error_count=0
layout_overflow_row_error_count=0
semantic_useful_rows=1048576
semantic_rounded_rows=1310720
semantic_row_efficiency=0.8
```

Compared with FUSED-MOE-002's `29.998302 ms` median, direct consumer-job
indexing is `0.055742 ms` (`0.19%`) faster. This is neutral at three repeats,
not a meaningful speedup. The checksum is unchanged. This timing-only run did
not rerun `semantic_permute_w13_compare`, so it adds no new correctness evidence
and does not supersede the known `h` comparison issue in FUSED-MOE-002.

The allocator logged two failed 12.50 GiB allocation attempts before execution
and then recovered; Iris and the benchmark both completed successfully. The
direct indexing change therefore removes scheduler arithmetic without moving
the approximately 30 ms fused bottleneck. Do not tune this factor further.

## 2026-07-10 FUSED-MOE-005 - Flat fused-W13 output-store rerun

Job `/dlwh/bench-semantic-fused-w13-flat-output-20260710-1353` ran on
`cw-rno2a` at commit `422c95f9bb` and reached terminal
`JOB_STATE_SUCCEEDED`. Its single task exited 0 after 1 minute 33.52 seconds,
with no failures or preemptions. No retry was launched.

Exact launch command reconstructed from the controller's `submit_argv`:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name bench-semantic-fused-w13-flat-output-20260710-1353 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu -- timeout 3600s bash -lc 'set -euo pipefail; uv pip install --reinstall nvidia-cudnn-cu13==9.19.0.56; exec uv run --no-sync --package marin-levanter --extra gpu --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_permute_w13_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha 422c95f9bb --jsonl scratch/semantic_fused_w13_flat_output_422c95f9bb.jsonl'
```

Every fused timing and drop/overflow row:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Dropped routes | Routing drops | Metadata overflow | Queue entry overflow | Queue route overflow | Layout row overflow | Checksum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 29.923961 | 57.411748 | 71.764685 | 0 | 0 | 0 | 0 | 0 | 0 | 1305986465792.0 |
| 1 | 29.749349 | 57.748722 | 72.185903 | 0 | 0 | 0 | 0 | 0 | 0 | 1305986465792.0 |
| 2 | 29.791293 | 57.667418 | 72.084272 | 0 | 0 | 0 | 0 | 0 | 0 | 1305986465792.0 |

Summary: median `29.791293 ms`, range `29.749349-29.923961 ms`, median
`57.667418` useful and `72.084272` rounded TFLOP/s/rank. The summary row
reported three repeat rows, zero error rows, zero dropped/routing-dropped
routes, zero metadata overflow routes, and median queue-entry, queue-route, and
layout-row overflow counts of zero. Compilation was `10.760242 s`; first run
was `6.599283 s`. Semantic row efficiency remained `0.8` with 1,048,576 useful
and 1,310,720 rounded rows.

Compared with FUSED-MOE-004's direct-job median of `29.942560 ms`, flattening
the fused output store is `0.151267 ms` (`0.51%`) faster. This is small and
needs more repeats before treating it as a real improvement. The checksum is
unchanged. This timing-only run did not execute the compare mode, so it adds no
new correctness evidence and does not supersede the known fair-precision
comparison requirement for `h`.

As in the direct-job run, the allocator logged two failed 12.50 GiB allocation
attempts before execution and recovered. Iris and all three measured repeats
completed successfully.

## 2026-07-10 FUSED-MOE-006 - Backward queue protocol correctness

The semantic fused backward scaffolds now use production-relevant queue
boundaries and bounded-slot ordering:

- fused W2 backward consumes the bf16 forward `return_y[S,DstOrd,Q,M,H]`
  residual directly instead of a dense token/route tensor;
- its destination consumers traverse the inverse source ordinal in the same
  rotating-peer phase order used by producers, preventing finite-slot reuse
  deadlock;
- fused W13 backward consumes returned dX tiles in physical
  `(dst_ordinal, chunk, block, hidden_tile)` order, scatters them into
  source-owned fp32 dX, and releases each slot generation in production order;
- route validity is clipped against the rows actually sent, and the sharded
  wrapper now uses rank-correct partition specifications.

The benchmark adds isolated `semantic_fused_w2_backward_{pallas,compare}`
modes. Its saved bf16 `return_y` and expert-major activation inputs are built
outside the timed callable, so the row measures only fused dcombine/dy-route
and W2 backward.

Local verification:

```text
pytest benchmark + fused W2 return/backward + fused W13 backward: 49 passed
scoped pre-commit: passed (Ruff, Black, license, pyrefly, AST, whitespace)
```

This is a local correctness milestone, not H100 correctness or performance
evidence. The next gate is one compile-first H100 target run for W2 return, W2
backward, and W13 backward before custom-VJP integration.

## 2026-07-10 FUSED-MOE-007 - Backward compile-first H100 gate

Job `/dlwh/bench-semantic-fused-backward-compile-20260710-1425` ran on
`cw-rno2a` at commit `8e3dd211f3` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 51.06 seconds, with no Iris failures or
preemptions. No retry was launched. The successful job state means the benchmark
driver completed; every requested benchmark mode itself returned an error row.

No mode reached compilation or timing. The per-mode result is:

| Mode | Result | First actionable source location |
| --- | --- | --- |
| `semantic_fused_w2_return_pallas` | Mosaic GPU could not infer the output layout of an iota | `source_push_semantic_fused_w2_return.py:521`, `jnp.arange(config.compute_m, dtype=jnp.int32)[:, None]`; apply an immediate `plgpu.layout_cast` |
| `semantic_fused_w2_backward_pallas` | XLA GEMM autotuning exhausted memory while allocating 3.14 GiB | `bench_source_push_semantic_plan.py:1106` calls the reference path, which reaches `source_push_semantic_fused_w2_return.py:238` `jnp.einsum` |
| `semantic_fused_w13_backward_pallas` | `Incompatible FragmentedArray layouts` | `source_push_semantic_fused_w13_backward.py:689`, adding `dw_ref[dw_index]` and `acc_ref[...]` |

The complete unmodified Iris output is captured in
`scratch/20260710-1425_bench-semantic-fused-backward-compile.raw.log`. The six
benchmark JSON lines total 32,719 bytes because each error row embeds its full
traceback. This compact projection preserves the timing and error fields in the
logbook:

```jsonl
{"row_type":"error","mode":"semantic_fused_w2_return_pallas","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"error_type":"RuntimeError","error_message":"Failed to infer the output layout of the iota. Please apply plgpu.layout_cast to its output right after its creation.","repeat_rows":null,"error_rows":null,"output_checksum":null,"rounded_tflops_per_rank":null}
{"row_type":"summary","mode":"semantic_fused_w2_return_pallas","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"error_type":null,"error_message":null,"error":"all repeats failed","repeat_rows":0,"error_rows":1,"output_checksum":null,"rounded_tflops_per_rank":null}
{"row_type":"error","mode":"semantic_fused_w2_backward_pallas","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"error_type":"JaxRuntimeError","error_message":"RESOURCE_EXHAUSTED: Autotuning failed for the f32[256,4608,2560] Triton GEMM fusion: out of memory while trying to allocate 3.14GiB.","repeat_rows":null,"error_rows":null,"output_checksum":null,"rounded_tflops_per_rank":null}
{"row_type":"summary","mode":"semantic_fused_w2_backward_pallas","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"error_type":null,"error_message":null,"error":"all repeats failed","repeat_rows":0,"error_rows":1,"output_checksum":null,"rounded_tflops_per_rank":null}
{"row_type":"error","mode":"semantic_fused_w13_backward_pallas","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"error_type":"ValueError","error_message":"Incompatible FragmentedArray layouts","repeat_rows":null,"error_rows":null,"output_checksum":null,"rounded_tflops_per_rank":null}
{"row_type":"summary","mode":"semantic_fused_w13_backward_pallas","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"error_type":null,"error_message":null,"error":"all repeats failed","repeat_rows":0,"error_rows":1,"output_checksum":null,"rounded_tflops_per_rank":null}
```

The first actionable traceback begins:

```text
File "/app/lib/levanter/src/levanter/grug/_moe/source_push_semantic_fused_w2_return.py", line 521, in _intermediate_loop
  row = jnp.arange(config.compute_m, dtype=jnp.int32)[:, None]
RuntimeError: Failed to infer the output layout of the iota. Please apply plgpu.layout_cast to its output right after its creation.
```

The W2-backward reference-input construction also produced repeated failed
12.50 GiB allocator attempts before GEMM autotuning failed. This run is a
meaningful negative compile gate: fix the two Mosaic layout errors and reduce
or restructure the W2-backward reference-input memory requirement before
resubmitting the same target shape.

## 2026-07-10 FUSED-MOE-008 - Backward compile-first H100 follow-up

Job `/dlwh/bench-semantic-fused-backward-compile2-20260710-1435` ran on
`cw-rno2a` at commit `586a0b1ef0`. It was stopped after W13 produced no result
row and task logs were silent for more than six minutes. Iris reached terminal
`killed` (`Terminated by user`); task 0 exited 0 after 9 minutes 49.29 seconds,
with zero failures and one preemption. No retry or duplicate was launched.

No mode reported compile or timing values:

| Mode | Result |
| --- | --- |
| `semantic_fused_w2_return_pallas` | `ValueError: Incompatible FragmentedArray layouts` during Mosaic GPU lowering; all timing fields null |
| `semantic_fused_w2_backward_pallas` | `ValueError`: cannot broadcast `float32[8@expert,8,72,4,64]` to `float32[8,8,72,4,64]` at `source_push_semantic_fused_w2_backward.py:361` in the `d_route.at[...].add(...)`; all timing fields null |
| `semantic_fused_w13_backward_pallas` | No result row; treated as a runtime hang after the final log at `21:30:13Z` and stopped at `21:37:15Z` |

Repeated failed GPU allocator requests (mostly 12.50 GiB) preceded both W2
error rows. Complete Iris output is in
`scratch/6597-compile2/final-logs.txt`; terminal status and task summary are in
`scratch/6597-compile2/final-status.txt` and
`scratch/6597-compile2/final-summary.txt`.

## 2026-07-10 FUSED-MOE-009 - W2 compile-first H100 follow-up

Job `/dlwh/bench-semantic-fused-w2-compile3-20260710-1450` ran once on
`cw-rno2a` at commit `8460737417` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 34.22 seconds, with zero failures and
preemptions. No stop, retry, or duplicate launch was needed.

Both requested modes returned error rows with `compile_time`,
`lower_compile_time`, `first_call_time`, `first_run_time`,
`steady_state_time`, `output_checksum`, and `rounded_tflops_per_rank` all null:

| Mode | First actionable error |
| --- | --- |
| `semantic_fused_w2_return_pallas` | `ValueError: Incompatible FragmentedArray layouts` at `source_push_semantic_fused_w2_return.py:530`, while evaluating `(jax.nn.silu(gate) * up * row_valid).astype(dtype)` |
| `semantic_fused_w2_backward_pallas` | `ValueError: Incompatible types for broadcasting: input type=float32[8@expert,8,72,4,64] and requested type=float32[8,8,72,4,64]` at `source_push_semantic_fused_w2_backward.py:364`, in the `d_route.at[...].add(...)` |

Each mode's summary row reported `all repeats failed`, with zero repeat rows
and one error row. Repeated failed allocator requests, mostly 12.50 GiB and one
6.25 GiB request, preceded the rows. Complete Iris output is in
`scratch/6597-compile3/final-logs.txt`; terminal status and task summary are in
`scratch/6597-compile3/final-status.txt` and
`scratch/6597-compile3/final-summary.txt`.

## 2026-07-10 FUSED-MOE-010 - Reduced W13 backward H100 timing

Job `/dlwh/bench-semantic-fused-w13b-reduced-20260710-1500` ran once on
`cw-rno2a` at commit `753445d04f` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 22.83 seconds, with zero failures or
preemptions. No stop, retry, or duplicate was needed.

`semantic_fused_w13_backward_pallas` completed without an error or hang:
compile/first-call 69.6007 s, lowering/compile 67.5735 s, first run 2.0272 s,
and steady state 69.1662 ms. It reported 7.7620 rounded and 6.2096 useful
TFLOP/s/rank, checksum 27,812,178, 0.8 row efficiency, and no queue, layout,
metadata, routing, or dropped-route errors. CUDA VMM emitted non-fatal
`CUDA_ERROR_NOT_PERMITTED` warnings before retrying with simpler handle types.
Complete Iris output is in `scratch/6597-w13b-reduced/raw.log`; terminal status
and task summary are in `scratch/6597-w13b-reduced/final-status.txt` and
`scratch/6597-w13b-reduced/final-summary.txt`.

## 2026-07-10 FUSED-MOE-011 - W2 compile-first H100 follow-up

Job `/dlwh/bench-semantic-fused-w2-compile4-20260710-1500` ran once on
`cw-rno2a` at commit `753445d04f` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 39.03 seconds, with zero failures and
preemptions. No stop, retry, restart, or duplicate was issued.

Both requested modes returned error rows with `compile_time`,
`lower_compile_time`, `first_call_time`, `first_run_time`,
`steady_state_time`, `output_checksum`, and `rounded_tflops_per_rank` all null:

| Mode | Exact error |
| --- | --- |
| `semantic_fused_w2_return_pallas` | `NotImplementedError: WGStridedFragLayout(shape=(64, 64), vec_size=4)` at `source_push_semantic_fused_w2_return.py:515`, assigning `(jax.nn.silu(gate) * up).astype(dtype)` to `h_smem[:, :]` |
| `semantic_fused_w2_backward_pallas` | `ValueError: Incompatible FragmentedArray layouts` at `source_push_semantic_fused_w2_backward.py:831`, adding `old + acc_ref[...]` |

Each summary row reported `all repeats failed`, with zero repeat rows and one
error row. Repeated failed allocator requests of 12.50 GiB, plus one 6.25 GiB
request, preceded the rows. Complete Iris output is in
`scratch/6597-compile4/final-logs.txt`; compact exact rows are in
`scratch/6597-compile4/result-rows.compact.jsonl`; terminal status and task
summary are in `scratch/6597-compile4/final-status.txt` and
`scratch/6597-compile4/final-summary.txt`.

## 2026-07-10 FUSED-MOE-012 - Tiny W13 backward compare gate

Job `/dlwh/bench-semantic-fused-w13b-tiny-compare-20260710-1510` ran once on
`cw-rno2a` at commit `d6749066bd` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 27.48 seconds, with zero failures or preemptions.
No stop, retry, restart, or duplicate was issued.

`semantic_fused_w13_backward_compare` failed before compilation in the JAX
reference path with `ShardingTypeError`: the gather at
`source_push_semantic_fused_w13_backward.py:239` requires an explicit
`.at[...].get(out_sharding=...)`. The summary reported zero repeat rows and one
error row. Consequently, no `dx`/`dw13` comparison metrics, checksums, or timing
values were emitted; compile, first-call, first-run, and steady-state timing
fields are all null. Reported `dropped_routes`, `routing_dropped_routes`, and
`metadata_overflow_routes` were zero. Queue/layout overflow counters were not
emitted because execution did not reach the fused kernel. Complete output and
exact result rows are in `scratch/6597-w13b-tiny-compare/raw.log` and
`scratch/6597-w13b-tiny-compare/result-rows.jsonl`; terminal status and task
summary are alongside them.

## 2026-07-10 FUSED-MOE-015 - W2 compile-first H100 follow-up

Job `/dlwh/bench-semantic-fused-w2-compile6-20260710-1520` ran once on
`cw-rno2a` at commit `a2637f641f` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 6.56 seconds, with zero failures or
preemptions. No code change, stop, restart, resubmit, or duplicate was issued.

Both modes succeeded with one repeat row and zero error rows. Return compile,
lowering, first-run, and steady-state times were `5.132605796097778`,
`3.0626156540820375`, `2.0699901420157403`, and `0.10240102896932513` seconds;
throughput was `10.485654634599875` rounded and `8.3885237076799` useful
TFLOP/s/rank; checksum was `4300405760.0`. Backward values were
`11.267498516943306`, `10.905288514913991`, `0.36221000202931464`, and
`0.22411802096758038` seconds; throughput was `9.581932049590257` rounded and
`7.665545639672207` useful TFLOP/s/rank; checksum was `5054136320.0`.

Both rows reported 0.8 semantic row efficiency and zero queue/layout/metadata
overflow, routing drops, or dropped routes. Logs contained three non-fatal
12.50 GiB allocation warnings and 80 non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM
fallback warnings. Complete raw output, exact rows, terminal summary, and monitor
state are in `scratch/20260710-1506_bench_semantic_fused_w2_compile6_raw.txt`,
`scratch/20260710-1506_bench_semantic_fused_w2_compile6_rows.jsonl`,
`scratch/20260710-1506_bench_semantic_fused_w2_compile6_terminal.txt`, and
`scratch/20260710-1506_bench_semantic_fused_w2_compile6_monitoring_state.json`.

## 2026-07-10 FUSED-MOE-013 - W2 compile-first H100 follow-up

Job `/dlwh/bench-semantic-fused-w2-compile5-20260710-1510` ran once on
`cw-rno2a` at commit `d6749066bd` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 56.47 seconds, with zero failures or
preemptions. No stop, retry, restart, or duplicate was issued.

`semantic_fused_w2_return_pallas` completed successfully: compile/first-call
5.1131718170 s, lowering/compile 3.0800009021 s, first run 2.0331709150 s,
steady state 102.5259479648 ms, 10.4728787718 rounded and 8.3783030175 useful
TFLOP/s/rank, and checksum 4,300,405,760. It reported one repeat row, zero
error rows, 0.8 row efficiency, and no queue, layout, metadata, routing, or
dropped-route errors.

`semantic_fused_w2_backward_pallas` returned one `JaxRuntimeError` row and no
repeat row. Its exact error was `RESOURCE_EXHAUSTED: Out of memory while trying
to allocate 12.50GiB. [executable_name='jit_semantic_fused_w2_backward_pallas']
[tf-allocator-allocation-error='']`; all timing, checksum, and throughput fields
were null, and its summary reported `all repeats failed`. Logs contained nine
failed 12.50 GiB allocation warnings, one failed 6.25 GiB allocation warning,
and 32 non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings. Complete Iris
output is in `scratch/6597-compile5/final-logs.txt`; compact exact rows are in
`scratch/6597-compile5/result-rows.compact.jsonl`; terminal status, task summary,
and monitor cadence are in `scratch/6597-compile5/final-status.txt`,
`scratch/6597-compile5/final-summary.txt`, and
`scratch/6597-compile5/monitor-status.txt`.

## 2026-07-10 FUSED-MOE-014 - Tiny W13 backward dx/dw13 comparison

Job `/dlwh/bench-semantic-fused-w13b-tiny-compare2-20260710-1520` ran once on
`cw-rno2a` at commit `a2637f641f` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 49.47 seconds, with zero failures or preemptions.
The benchmark progressed beyond startup within the six-minute guard, so no stop,
retry, restart, or duplicate was issued.

`semantic_fused_w13_backward_compare` emitted one successful repeat row. The
`dw13` comparison was close, with max absolute difference
`1.52587890625e-05` and mean absolute difference `4.991304649593076e-07`.
The `dx` comparison was materially discrepant, with max absolute difference
`420.63323974609375` and mean absolute difference `35.01224136352539`.
Expected and observed nonfinite error counts were zero for both `dx` and
`dw13`. Queue overflow, layout overflow, metadata overflow, routing drops, and
dropped routes were also all zero.

Compile/first-call time was `7.853159782011062` s, lowering/compile time was
`6.970990175032057` s, first-run time was `0.882169606979005` s, and steady
state time was `1.8387140007689595` ms. The row reported
`0.18248858705577575` rounded and `0.1459908696446206` useful TFLOP/s/rank,
checksum `455.6455078125`, and row efficiency `0.8`. CUDA VMM emitted non-fatal
`CUDA_ERROR_NOT_PERMITTED` warnings before retrying with simpler handle types.
Complete Iris output and exact result rows are in
`scratch/6597-w13b-tiny-compare2/raw.log` and
`scratch/6597-w13b-tiny-compare2/result-rows.jsonl`; terminal status and task
summary are alongside them.

## 2026-07-10 FUSED-MOE-016 - Tiny W13 backward dx-fix validation

Job `/dlwh/bench-semantic-fused-w13b-dxfix-20260710-1521` ran once on
`cw-rno2a` at commit `07b08cd709` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 43.34 seconds, with zero failures or
preemptions. No code change, stop, restart, resubmit, or duplicate was issued.

`semantic_fused_w13_backward_compare` emitted one successful repeat row and
zero error rows. The `dx` discrepancy remained large: max absolute difference
`420.63323974609375` and mean absolute difference `34.988548278808594`.
The `dw13` comparison remained close: max absolute difference
`1.52587890625e-05` and mean absolute difference
`4.986190447198169e-07`. Expected and observed nonfinite error counts were zero
for both tensors.

Compile/first-call time was `8.298504103091545` s, lowering/compile time was
`7.4632771600736305` s, first-run time was `0.8352269430179149` s, and steady
state time was `0.0020637259585782886` s. Queue overflow, layout overflow,
metadata overflow, routing drops, and dropped routes were all zero. The row
reported four live semantic pairs, 2,048 useful rows, 2,560 rounded rows, 0.8
row efficiency, `0.16259151008167685` rounded and `0.13007320806534148` useful
TFLOP/s/rank, and checksum `455.621826171875`. Twelve non-fatal CUDA VMM handle
fallback warnings occurred. Complete Iris output and exact result rows are in
`scratch/6597-w13b-dxfix/raw.log` and
`scratch/6597-w13b-dxfix/result-rows.jsonl`; terminal status, task summary, and
monitor state are alongside them.

## 2026-07-10 FUSED-MOE-017 - Parallel queue-consumer W2 backward H100

Job `/dlwh/bench-semantic-fused-w2b-parallel-20260710-1521` ran once on
`cw-rno2a` at commit `07b08cd709` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 2 minutes 11.71 seconds, with zero failures or
preemptions. No duplicate, stop, resubmit, kernel edit, or Iris restart was
issued.

`semantic_fused_w2_backward_pallas` emitted one successful repeat row and zero
error rows. Exact steady-state time was `0.2073946320451796` seconds, with
`8.283661449953765` useful and `10.354576812442206` rounded TFLOP/s/rank;
checksum was `5054136320.0`. Compile/first-call, lowering/compile, and first-run
times were `11.448272167006508`, `9.255549566005357`, and
`2.192722601001151` seconds.

Semantic row efficiency was 0.8: 1,048,576 useful rows, 1,310,720 rounded rows,
64 live pairs, and masked-row fraction 0.2. Queue and layout overflow error
counts, metadata overflow routes, routing dropped routes, and dropped routes
were all zero. Logs contained two non-fatal 12.50 GiB allocator warnings and 48
non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings; there was no
traceback or actionable failure. Complete raw Iris routine output and terminal
monitor state are in `scratch/6597-w2b-parallel/raw-routine-output.txt` and
`scratch/6597-w2b-parallel/monitoring-state.json`.

## 2026-07-10 FUSED-MOE-018 - Split W2 direct-return floor

Job `/dlwh/bench-semantic-w2-return-split-floor-20260710-1524` ran once on
`cw-rno2a` at commit `6ce3554953` and reached terminal `JOB_STATE_SUCCEEDED`.
Its single task exited 0 after 1 minute 34.42 seconds, with zero failures or
preemptions. No duplicate, stop, resubmit, kernel edit, or Iris restart was
issued.

`semantic_fused_w2_return_pallas` emitted three successful repeat rows and zero
error rows. Steady-state time had median `36.978593988654516` ms, min
`36.89697233494371` ms, and max `37.02089935541153` ms. Useful TFLOP/s/rank
had median `23.229478640089713`, min `23.20293332026891`, and max
`23.2808657415633`; rounded TFLOP/s/rank had median `29.036848300112144`, min
`29.003666650336136`, and max `29.10108217695413`.

All three repeats produced checksum `3915118848.0`. Semantic row efficiency
was 0.8, with 1,048,576 useful rows, 1,310,720 rounded rows, and 64 live pairs.
Queue and layout overflow error counts, metadata overflow routes, routing
dropped routes, and dropped routes were all zero. Compile/first-call,
lowering/compile, and first-run times were `5.083134362939745`,
`3.0745417720172554`, and `2.0085925909224898` seconds. Logs contained two
non-fatal 12.50 GiB allocator warnings and 48 non-fatal
`CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings; there was no traceback or
actionable failure. Complete Iris output and exact result rows are in
`scratch/6597-w2-return-split-floor/raw.log` and
`scratch/6597-w2-return-split-floor/result-rows.jsonl`; terminal status and task
summary are alongside them.

## 2026-07-10 FUSED-MOE-019 - Tiny W13 backward dX scale diagnostics

Job `/dlwh/bench-semantic-w13b-dxscale-20260710-1527` ran once on `cw-rno2a`
at commit `afc18931a9` and reached terminal `JOB_STATE_SUCCEEDED`. Its single
task exited 0 after 49.58 seconds, with zero failures and preemptions. No
duplicate, stop, resubmit, kernel edit, or Iris restart was issued.

The expected and observed `dX` absolute sums were 17,072,636 and 16,742,244,
respectively, while the least-squares scale was 0.5872155428 and cosine
similarity was 0.5925955176. Thus the discrepancy is not a simple global scale
mismatch despite the similar absolute sums. `dX` max and mean absolute
differences were 419.6332397461 and 34.6273040771. `dW13` remained close, with
max and mean absolute differences of 1.52587890625e-05 and
5.010289214624208e-07. Expected and observed nonfinite counts were zero for
both tensors.

Compile/first-call time was 10.5734108520 seconds, lowering/compile time was
9.6598809430 seconds, first-run time was 0.9135299090 seconds, and steady-state
time was 2.1707150154 ms. The row reported 4 live pairs, 2,048 useful rows,
2,560 rounded rows, 0.8 row efficiency, 0.1236622284 useful and 0.1545777855
rounded TFLOP/s/rank, and checksum 33,815,336. Queue and layout overflow error
counts, metadata overflow routes, routing drops, and dropped routes were all
zero. Twelve non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings
occurred. Complete raw output, exact rows, terminal status, and task summary are
in `scratch/6597-w13b-dxscale/`.

## 2026-07-10 FUSED-MOE-020 - Top-k=1 W13 backward route diagnostic

Job `/dlwh/bench-semantic-w13b-topk1-20260710-1531` ran once on `cw-rno2a`
at commit `afc18931a9` and reached terminal `JOB_STATE_SUCCEEDED`. Its single
task exited 0 after 40.76 seconds, with zero failures and preemptions. No
duplicate, stop, resubmit, kernel edit, or Iris restart was issued.

With EP2 and top-k=1, expected `dX` had absolute sum `10,787,750`, while
observed `dX` contained two nonfinite values versus zero expected. Consequently
the observed absolute sum, least-squares scale, cosine similarity, max absolute
difference, and mean absolute difference were all `NaN`. `dW13` remained close:
max absolute difference was `3.814697265625e-06` and mean absolute difference
was `3.316472430014983e-08`, with zero expected or observed nonfinite values.
The failure therefore does not require multi-route top-k accumulation; this
run leaves the route-mapped `dX` path implicated but does not alone prove the
specific mapping fault.

Compile/first-call time was `7.9865564740030095` seconds, lowering/compile time
was `7.144973514019512` seconds, first-run time was `0.8415829599834979`
seconds, and steady-state time was `1.9043060019612312` ms. The row reported
four live pairs, 1,024 useful rows, 1,280 rounded rows, 0.8 row efficiency,
`0.07048117679709569` useful and `0.08810147099636961` rounded TFLOP/s/rank.
Queue and layout overflow error counts, metadata overflow routes, routing
drops, and dropped routes were all zero. Twelve non-fatal
`CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings occurred. Complete raw output,
terminal status, task summary, and monitor state are in
`scratch/6597-w13b-topk1/`.

## 2026-07-10 FUSED-MOE-021 - Nonfinite-safe top-k=1 W13 backward diagnostic

Job `/dlwh/bench-semantic-w13b-topk1-finite-20260710-1535` ran once on
`cw-rno2a` at commit `0404bd5f12` and reached terminal
`JOB_STATE_SUCCEEDED`. Its single task exited 0 after 40.91 seconds, with zero
failures and preemptions. No duplicate, stop, resubmit, code edit, or Iris
restart was issued.

After excluding the two observed nonfinite `dX` entries, the finite-masked
expected absolute sum was `10,787,750`, but the finite-masked observed absolute
sum still overflowed the float32 reduction to `Infinity`. The least-squares
scale and cosine similarity remained `NaN`; the unmasked max and mean absolute
differences were also `NaN`. Expected `dX` had zero nonfinite entries; observed
`dX` had two.
`dW13` remained close, with max and mean absolute differences of
`3.814697265625e-06` and `3.3054845971491886e-08`, and zero expected or
observed nonfinite entries. The nonfinite mask alone is thus insufficient: the
remaining finite observed `dX` values still overflow float32 aggregate metrics.

Compile/first-call time was `8.013033227995038` seconds, lowering/compile time
was `7.178556734928861` seconds, first-run time was `0.8344764930661768`
seconds, and steady-state time was `1.7895640339702368` ms. The row reported
four live pairs, 1,024 useful rows, 1,280 rounded rows, 0.8 row efficiency,
`0.07500023774071458` useful and `0.09375029717589324` rounded TFLOP/s/rank.
Queue and layout overflow error counts, metadata overflow routes, routing
drops, and dropped routes were all zero. Twelve non-fatal
`CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings occurred; there was no
traceback or actionable runtime failure. Complete raw output, exact result
rows, terminal status, task summary, and monitor state are in
`scratch/6597-w13b-topk1-finite/`.

## 2026-07-10 FUSED-MOE-022 - Resident peer-local global dX init barrier

Job `/dlwh/bench-semantic-w13b-initbarrier-20260710-1538` ran once on
`cw-rno2a` at commit `38a77112f8` and reached terminal
`JOB_STATE_SUCCEEDED`. Its single task exited 0 after 41.52 seconds, with zero
failures and preemptions. No duplicate, stop, resubmit, code edit, or Iris
restart was issued.

Replacing the per-token init waits with a resident peer-local global `dX` init
barrier did not fix correctness. The finite-masked expected and observed `dX`
absolute sums were `10787750.0` and `Infinity`; least-squares scale and cosine
similarity were both `NaN`, as were max and mean absolute differences. Expected
`dX` had zero nonfinite entries and observed `dX` had two. `dW13` remained
close: max and mean absolute differences were `3.814697265625e-06` and
`3.263437520217849e-08`, with zero expected and observed nonfinite entries.

Compile/first-call time was `8.154060366912745` seconds, lowering/compile time
was `7.291952597908676` seconds, first-run time was `0.8621077690040693`
seconds, and steady-state time was `0.0018587580416351557` seconds. Useful and
rounded throughput were `0.07220828369997431` and `0.0902603546249679`
TFLOP/s/rank. The row reported four live pairs, 1,024 useful rows, 1,280 rounded
rows, 0.8 row efficiency, and `0.19999999999999996` masked-row fraction. Queue
overflow routes,
layout overflow rows, metadata overflow routes, routing drops, and dropped
routes were all zero. There was no deadlock or runtime exception; this is a
terminal correctness failure. Exact rows, raw logs, terminal status, task
summary, and monitor state are in `scratch/6597-w13b-initbarrier/`.

## 2026-07-10 FUSED-MOE-023 - Persistent W2 return and aggregated W2 backward tiny compare

Job `/dlwh/bench-semantic-fused-w2-tiny-compare-20260710-1542` ran once on
`cw-rno2a` at commit `f63a684742` and reached terminal Iris state `succeeded`.
Its single task exited 0 after 41.95 seconds, with zero failures and
preemptions. No duplicate, stop, resubmit, code edit, or Iris restart was
issued.

The persistent W2 return completed but failed correctness. `y` max and mean
absolute differences were `31.21875` and `2.2772815227508545`; expected and
observed `y` nonfinite counts were both zero. `return_y` max and mean absolute
differences were `NaN` because all `196608` observed elements were nonfinite,
versus zero expected. The output checksum was `NaN`. Valid, queue-overflow,
and layout-overflow error counters were zero; metadata-overflow routes,
routing-dropped routes, and dropped routes were also zero.

Return compile/first-call time was `2.65317596308887` seconds, lowering/compile
time was `1.814075340051204` seconds, first-run time was
`0.8391006230376661` seconds, and steady-state time was
`0.0012550349347293377` seconds. Useful and rounded throughput were
`0.05347170994445089` and `0.06683963743056362` TFLOP/s/rank.

The aggregated W2 backward comparison failed during lowering before producing
any `d_h`, `d_w2`, or `d_route_weight` comparison, nonfinite, counter, or
timing metrics. The first failure was `jax._src.core.ShardingTypeError` at
`source_push_semantic_fused_w2_backward.py:253`: the gather
`dy.at[source, metadata.token_ids].get()` lacked an unambiguous output
sharding for operand `bfloat16[2@expert,512,256]` and indices
`int32[2,2,4,4,64,2]`. Twelve non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM
fallback warnings preceded the results. Complete raw output, exact JSON rows,
terminal status, task summary, and monitor state are in
`scratch/6597-w2-persistent-tiny-compare/`.

## 2026-07-10 FUSED-MOE-025 - Direct-store W2 fragment and backward sharding tiny compare

Job `/dlwh/bench-semantic-fused-w2-tiny-compare2-20260710-1548` ran once on
`cw-rno2a` at commit `e05a3aa826` and reached terminal Iris state `succeeded`.
Its single task exited 0 after 35.52 seconds, with zero failures and
preemptions. No duplicate, stop, resubmit, code edit, or Iris restart was
issued.

The direct-store W2 WGMMA fragment materially improved source-visible `y` but
did not fix persistent `return_y`. `y` max and mean absolute differences were
`0.125` and `0.00547508429735899`, down from `31.21875` and
`2.2772815227508545` in FUSED-MOE-023; expected and observed `y` nonfinite
counts were both zero. `return_y` max and mean absolute differences remained
`NaN` because all `196608` observed elements were nonfinite, versus zero
expected. The output checksum was `NaN`. Valid, queue-overflow, and
layout-overflow error counters were zero; metadata-overflow routes,
routing-dropped routes, and dropped routes were also zero.

Return compile/first-call time was `2.6340669308556244` seconds,
lowering/compile time was `1.7945377879077569` seconds, first-run time was
`0.8395291429478675` seconds, and steady-state time was
`0.0012683930108323693` seconds. Useful and rounded throughput were
`0.05290857283734206` and `0.06613571604667756` TFLOP/s/rank.

The backward reference-sharding fix advanced past the previous `dy` gather,
but the comparison still failed during lowering before producing any `d_h`,
`d_w2`, or `d_route_weight` comparison, nonfinite, counter, or timing metrics.
The first failure was `ValueError: Incompatible types for broadcasting: input
type=float32[2@expert,2,4,4,64,128] and requested
type=float32[2,2,4,4,64,128]` at
`source_push_semantic_fused_w2_backward.py:278`, in the reference
`d_h.at[scatter_destination, scatter_expert, scatter_row].set(...)`. Twelve
non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings preceded the
results. Structured rows, relevant raw output, terminal status, task summary,
and monitor state are in `scratch/6597-w2-direct-stores-tiny-compare2/`.

## 2026-07-10 FUSED-MOE-024 - W13 backward direct dX fragment-to-peer-GMEM

Job `/dlwh/bench-semantic-w13b-direct-gmem-20260710-1548` ran once on
`cw-rno2a` at commit `e05a3aa826` and reached terminal Iris state `succeeded`.
Its single task exited 0 after 41.5 seconds, with zero failures and
preemptions. No duplicate, stop, resubmit, code edit, or Iris restart was
issued.

Storing each WGMMA dX fragment directly to peer GMEM removed the prior severe
dX corruption. Expected and observed dX absolute sums were `10787750.0` and
`10787916.0`; least-squares scale was `0.999991774559021` and cosine similarity
was `0.9999985098838806`. dX max and mean absolute differences were
`0.4999237060546875` and `0.06049002707004547`. dW13 remained close, with max
and mean absolute differences of `3.814697265625e-06` and
`3.301985884718306e-08`. Expected and observed nonfinite counts were zero for
both dX and dW13.

Compile/first-call time was `8.22770067199599` seconds, lowering/compile time
was `7.354656650917605` seconds, first-run time was `0.8730440210783854`
seconds, and steady-state time was `0.001655775005929172` seconds. Useful and
rounded throughput were `0.0810603660034601` and `0.10132545750432512`
TFLOP/s/rank. The row reported four live pairs, 1,024 useful rows, 1,280 rounded
rows, 0.8 row efficiency, and checksum `21575668.0`. Queue and layout overflow
error counts, metadata overflow routes, routing drops, and dropped routes were
all zero. Twelve non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM handle fallback
warnings occurred. Exact repeat and summary rows, terminal evidence, and
monitor state are in `scratch/6597-w13b-direct-gmem/`.

## 2026-07-10 FUSED-MOE-027 - Top-k=2 W13 backward direct dX to peer GMEM

Job `/dlwh/bench-semantic-w13b-topk2-direct-gmem-20260710-1551` ran once on
`cw-rno2a` at commit `b59fbd6cab` and succeeded. Its task exited 0 after 1
minute and 3.66 seconds, with zero failures and preemptions. No duplicate,
stop, resubmit, code edit, or Iris restart was issued.

Expected and observed dX absolute sums were `17072636.0` and `17072528.0`;
least-squares scale was `0.9999837279319763`, cosine similarity was
`0.9999985098838806`, and max/mean absolute differences were
`0.927032470703125`/`0.10450340807437897`. dW13 max/mean absolute differences
were `1.52587890625e-05`/`4.983673989045201e-07`. Expected and observed
nonfinite counts were zero for both dX and dW13.

Compile/first-call, lowering/compile, first-run, and steady-state times were
`8.548144964996027`, `7.701112067996291`, `0.8470328969997354`, and
`0.0018552929977886379` seconds. Useful/rounded throughput was
`0.14468628745969167`/`0.18085785932461462` TFLOP/s/rank. All overflow and
dropped-route counters were zero. Twelve non-fatal `CUDA_ERROR_NOT_PERMITTED`
VMM fallback warnings occurred; there was no traceback or runtime failure.

## 2026-07-10 FUSED-MOE-026 - Masked W2 return and replicated backward reference

Job `/dlwh/bench-semantic-fused-w2-tiny-compare3-20260710-1551` ran once on
`cw-rno2a` at commit `b59fbd6cab` and reached terminal Iris state `succeeded`.
Its single task exited 0 after 34.62 seconds, with zero failures and
preemptions. No duplicate, stop, resubmit, code edit, or Iris restart was
issued.

The masked return comparison was finite: `y` max/mean absolute differences
were `0.125`/`0.00547508429735899`, and `return_y` max/mean absolute
differences were `0.06487846374511719`/`0.008156927302479744` across `524288`
live elements. Expected and observed nonfinite counts were zero for both
outputs; checksum was `524288.1875`. Valid, queue-overflow, layout-overflow,
metadata-overflow, routing-drop, and dropped-route counters were all zero.
Compile/first-call, lowering, first-run, and steady-state times were
`2.7683367299905512`, `1.8655651479930384`, `0.9027715819975128`, and
`0.0015391109918709844` seconds. Useful/rounded throughput was
`0.04360235509618489`/`0.05450294387023111` TFLOP/s/rank; row efficiency was
0.8 (`2048` useful, `2560` rounded, four live pairs).

The replicated backward reference advanced to the `d_route` scatter but failed
during lowering before producing backward correctness or timing metrics:
`ValueError: Incompatible types for broadcasting: input
type=float32[2@expert,2,4,4,64] and requested type=float32[2,2,4,4,64]` at
`source_push_semantic_fused_w2_backward.py:322`, in
`d_route.at[source, metadata.token_ids, metadata.route_slots].add(queue_d_route)`.
The summary reported one error row, zero repeat rows, and `all repeats failed`.
Twelve non-fatal `CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings preceded the
results.

## 2026-07-10 FUSED-MOE-028 - W2 backward source-sharded scatter follow-up

Job `/dlwh/bench-semantic-fused-w2b-tiny-compare4-20260710-1555` ran once on
`cw-rno2a` at commit `41d400db5b` and reached terminal Iris state `succeeded`.
The kernel comparison did not lower because the final reference-only
`d_route_weight` scatter attempted to broadcast a source-sharded
`float32[2@expert,2,4,4,64]` update to an unsharded
`float32[2,2,4,4,64]` slice. No kernel timing or gradient comparison metrics
were emitted. This follows successful lowering of the preceding expert-space
reference gathers, scatter, and reductions. The follow-up replaces the
three-index scatter with independent per-source `[T,K]` scatters under
`jax.vmap`. The job had no duplicate launch or Iris restart.

## 2026-07-10 FUSED-MOE-029 - W2 backward per-source scatter vmap

Job `/dlwh/bench-semantic-fused-w2b-tiny-compare5-20260710-1600` ran once on
`cw-rno2a` at commit `c9057744c3` and reached terminal Iris state `succeeded`;
its single task exited 0 after 31 seconds with zero failures or preemptions.
The benchmark emitted zero repeat rows and one error row: lowering failed at
`source_push_semantic_fused_w2_backward.py:326` in
`jax.vmap(_scatter_source_route_gradient)` with `ValueError: Mapped away
dimension of inputs passed to vmap should be sharded the same. Got inconsistent
axis specs: None vs expert`. No `d_h`, `d_w2`, or `d_route` correctness or
nonfinite metrics were emitted, and compile, lower-compile, first-call,
first-run, steady-state, and throughput fields were null. Route-drop and
overflow counters were zero. No duplicate, stop, resubmit, code edit, or Iris
restart was issued.

## 2026-07-10 FUSED-MOE-030 - Target EP8 W13 backward direct-GMEM benchmark

Job `/dlwh/bench-semantic-w13b-target-direct-gmem-20260710-1602` ran on
`cw-rno2a` at commit `c9057744c3` and reached terminal Iris state `succeeded`.
Its H100x8 task exited 0 after 2 minutes and 39.49 seconds, with zero Iris
failures or preemptions. Setup began six seconds after submission, so the
eight-minute start-only stop condition did not apply. No duplicate, stop,
resubmit, code edit, or Iris restart was issued.

The target EP8 W13 backward Pallas mode failed before producing any of the
three requested repeats. JAX raised `RESOURCE_EXHAUSTED` from
`jit_semantic_fused_w13_backward_pallas` while trying to allocate 12.50 GiB.
The harness emitted one error row, zero repeat rows, and an `all repeats failed`
summary. Median/min/max steady-state time, useful and rounded TFLOP/s/rank, and
output checksum are therefore unavailable. This was an allocation failure, not
a deadlock.

The error row reported 64 live pairs, 1,048,576 useful rows, 1,310,720 rounded
rows, 0.8 row efficiency, and 0.2 masked-row fraction. Dropped routes,
routing-dropped routes, and metadata-overflow routes were all zero. Terminal
evidence, structured result rows, and closed monitoring state are in
`scratch/6597-w13b-target-direct-gmem/`.

## 2026-07-10 FUSED-MOE-031 - W2 backward aligned vmap input shardings

Job `/dlwh/bench-semantic-fused-w2b-tiny-compare6-20260710-1604` ran once on
`cw-rno2a` at commit `5d78f1b010` and reached terminal Iris state `succeeded`;
its single task exited 0 after 43.35 seconds with zero failures or preemptions.
The benchmark emitted one repeat row and one summary row with zero error rows.

The aligned vmap input shardings fixed the reference lowering failure and the
kernel matched exactly. Max and mean absolute differences were both `0.0` for
`d_h`, `d_w2`, and `d_route_weight`. Expected and observed nonfinite counts
were all `0.0`. Valid, queue-overflow-route, and layout-overflow-row error
counts were `0.0`; metadata-overflow routes, routing drops, and dropped routes
were all zero.

Compile/first-call, lowering/compile, first-run, and steady-state times were
`7.38956371700624`, `6.424557343008928`, `0.965006373997312`, and
`0.0015471859951503575` seconds. Useful/rounded throughput was
`0.08674957530685026`/`0.10843696913356282` TFLOP/s/rank. The random-routing
tiny shape had 2,048 useful and 2,560 rounded rows (0.8 efficiency), four live
pairs, and zero masked-data correctness failures. Sixteen non-fatal
`CUDA_ERROR_NOT_PERMITTED` VMM fallback warnings occurred; `uv` also fell back
from hardlinks to copies. There was no traceback or actionable runtime error.
Complete raw logs, exact JSON rows, terminal status, task summary, monitor
state, and a standalone report are in `scratch/6597-w2b-tiny-compare6/`. No
duplicate, stop, resubmit, code edit, or Iris restart was issued.

## 2026-07-10 FUSED-MOE-032 - Remove W13 backward capacity-sized fp32 mask

The target W13 backward failure in FUSED-MOE-030 was traced to the API-boundary
padding mask `jnp.where(valid[..., None], dz13, 0)`. Type promotion caused it
to materialize a per-rank fp32 tensor with shape `[32, 40960, 2560]`, exactly
12.50 GiB. This allocation was not part of the persistent queue or direct-GMEM
dX path.

The dense mask is removed. The dW13 consumer now zeros only invalid rows in
each 64-by-128 `dz` SMEM tile before issuing WGMMA. Invalid dX route rows are
not consumed, so the direct accumulator-to-GMEM dX implementation is unchanged.
The dedicated W13 backward tests pass (`6 passed`), the combined semantic MLP
and fused backward test selection passes (`24 passed`), scoped pre-commit and
Pyrefly pass, and `git diff --check` is clean. Target H100 validation remains
required.

## 2026-07-10 FUSED-MOE-033 - Target EP8 W13 backward SMEM-mask validation

Job `/dlwh/bench-semantic-w13b-target-masktile-20260710-1610` ran once on
`cw-rno2a` at commit `c5db9262e1` and reached terminal Iris state `succeeded`.
Its H100x8 task exited 0 after 1 minute and 29.63 seconds, with zero Iris
failures or preemptions. No duplicate, stop, resubmit, code edit, Iris restart,
or cluster bounce was issued.

The target EP8 random-routing `semantic_fused_w13_backward_pallas` benchmark
produced zero of the three requested repeats and one error row. The first
actionable failure was `TypeError: Unsupported index type:
<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>`. It localizes
to the new padding loop's `dz_smem[row, :]` assignment at
`source_push_semantic_fused_w13_backward.py:670`; the Pallas loop index is a
dynamic tracer that this SMEM ref indexing path rejects. Median/min/max
steady-state time and useful/rounded TFLOP/s/rank are unavailable because all
repeats failed.

The error row reported 64 live pairs, 1,048,576 useful rows, 1,310,720 rounded
rows, 0.8 row efficiency, 0.2 masked-row fraction, and zero dropped routes,
routing-dropped routes, and metadata-overflow routes. The allocator also twice
attempted and failed to allocate 12.50 GiB (13,421,772,800 bytes) before the
TypeError, so the target-size dense-allocation symptom remained observable in
this run. Raw logs, structured rows, terminal status, task summary, closed
monitor state, and a standalone report are in
`scratch/6597-w13b-target-masktile/`.

## 2026-07-10 FUSED-MOE-034 - Integrated fused MLP benchmark boundary

The semantic benchmark harness now exposes the actual fused custom-VJP
boundary instead of requiring stage-time arithmetic. New modes are
`semantic_fused_mlp_forward_pallas`, `semantic_fused_mlp_forward_compare`,
`semantic_fused_mlp_forward_backward_pallas`, and
`semantic_fused_mlp_forward_backward_compare`; alias `semantic_fused_mlp`
runs the two performance modes. Forward accounting includes W13, SwiGLU, and
W2; forward-plus-backward accounting uses three times that math. The compare
mode reports y, dX, route-weight, dW13, and dW2 errors and nonfinite counts.

The two focused benchmark tests pass and scoped pre-commit passes. No H100
claim is attached yet; the next integrated run follows isolated target-stage
validation.

## 2026-07-10 FUSED-MOE-035 - Port stable inbox producer topology to semantic W13

The semantic fused W13 producer schedule is aligned with the stable source-push
inbox profile from open PR #6840 and its cleaned successor in #6841. The old
`num_sms=32` knob represented the total peer-local worker grid; the current
profile names this `worker_programs_per_peer=32`, partitioned into two send
workers and 30 compute workers.

Previously the semantic kernel used 16 tile-fragment producers plus 32
consumers. A B256 chunk was split across 40 copy tiles, paying repeated empty
waits, 40 `send_done` signals, and one fan-in wait before publication. The new
schedule alternates complete chunks between two producers: the owner waits once,
copies all four B64 blocks and K tiles, then publishes `full` directly. The
12-slot cumulative generation protocol and concurrent consumers are unchanged.
At the target geometry this removes about 46,080 regular semaphore operations
per rank and reduces the grid from 384 to 256 CTAs/rank.

Five dedicated tests, target-shape kernel construction, and scoped pre-commit
pass. Target H100 timing is required before accepting the expected speedup.

## 2026-07-10 FUSED-MOE-036 - Target EP8 W13 backward bulk-mask validation

Job `/dlwh/bench-semantic-w13b-target-bulkmask-20260710-1615` ran once on
`cw-rno2a` at commit `ac5a885bc5` and reached terminal Iris state `succeeded`.
Its H100x8 task exited 0 after 2 minutes and 53.46 seconds, with zero Iris
failures or preemptions. No duplicate, stop, resubmit, code edit, Iris restart,
or cluster bounce was issued.

The target EP8 random-routing `semantic_fused_w13_backward_pallas` benchmark
produced zero of the three requested repeats and one error row. The first
actionable failure was `RESOURCE_EXHAUSTED: Out of memory while trying to
allocate 12.50GiB` in `jit_semantic_fused_w13_backward_pallas`. The allocator
logged two failed 13,421,772,800-byte requests before the structured error.
Median/min/max steady-state time, useful and rounded TFLOP/s/rank, and output
checksum are unavailable because all repeats failed.

The error row reported 64 live pairs, 1,048,576 useful rows, 1,310,720 rounded
rows, 0.8 row efficiency, 0.19999999999999996 masked-row fraction, and zero
dropped routes, routing-dropped routes, and metadata-overflow routes. Queue-
and layout-overflow counters were not emitted. Structured rows, terminal
status, task summary, closed monitor state, and a standalone report are in
`scratch/6597-w13b-target-bulkmask/`.

## 2026-07-10 FUSED-MOE-037 - Isolate W13 backward from pair scatter

The repeated 12.50 GiB request was outside the persistent W13 backward kernel.
The isolated benchmark first scattered pair-flat `dz_pair` into a 6.25 GiB
bf16 expert-major tensor; XLA required another full-capacity buffer for that
scatter. The production fused custom VJP already computes dSwiGLU in
expert-major form and does not use this conversion.

The W13 backward stage modes now receive a dedicated input bundle with
source-sharded `x` and destination-sharded expert-major `dz13` and `w13`, built
directly with `jax.make_array_from_callback` at target shape. This makes the
isolated row measure the kernel rather than a diagnostic pair-layout
conversion. Three focused benchmark tests and scoped pre-commit pass. Target
H100 timing remains required.

## 2026-07-10 FUSED-MOE-038 - Target W13 forward/backward inbox benchmark

Job `/dlwh/bench-semantic-w13-fwd-bwd-inbox-20260710-1626` ran once on
`cw-rno2a` at commit `c59a39e299` and reached terminal Iris state `succeeded`.
Its single H100x8 task exited 0 after 1 minute and 52.92 seconds, with zero
failures and preemptions. No duplicate, stop, resubmit, code edit, Iris restart,
or cluster bounce was issued.

The old-inbox 2-send-worker plus 30-compute-worker B256-to-B64
`semantic_permute_w13_pallas` topology completed all three repeats. Exact
median/min/max steady-state time was `55.011869000736624`/
`54.9034156720154`/`55.28170434990898` ms. Median useful/rounded throughput was
`31.229386487068737`/`39.036733108835925` TFLOP/s/rank. The output checksum was
`1305986465792.0`. Dropped routes, routing drops, metadata overflow, queue entry
overflow, queue route overflow, and layout row overflow were all zero.

The direct expert-major-dz `semantic_fused_w13_backward_pallas` mode produced
zero repeats and one error row. The first actionable failure was
`AttributeError: 'SemanticFusedW13BackwardBenchInputs' object has no attribute
'dy'`; median/min/max steady-state time and useful/rounded throughput are
therefore unavailable. Dropped-route, routing-drop, and metadata-overflow
counters were zero. Non-fatal logs also contained failed 12.50 GiB allocation
attempts and CUDA VMM handle fallback warnings. Exact structured evidence,
terminal status, closed monitor state, and a standalone report are in
`scratch/6597-w13-fwd-bwd-inbox/`.

## 2026-07-10 FUSED-MOE-038A - Target EP8 W2 return/backward benchmark

Job `/dlwh/bench-semantic-fused-w2-target-20260710-1608` ran once on `cw-rno2a`
at commit `5d78f1b010`. It reached terminal Iris state `killed` after `22m54s`:
the job was stopped after the W2 backward stage emitted no progress or result
for 20 minutes. There was no duplicate launch, resubmission, Iris restart, or
cluster bounce.

The target EP8 random-routing `semantic_fused_w2_return_pallas` mode completed
all three repeats. Exact median/min/max steady-state time was
`0.06503098636555175`/`0.06500880800498028`/`0.06515120734305431` seconds
(`65.03098636555175`/`65.00880800498028`/`65.15120734305431` ms). Median
useful/rounded throughput was `13.20898708765436`/`16.51123385956795`
TFLOP/s/rank, and the output checksum was `3915118848.0`. The three repeat
useful throughputs were `13.20898708765436`, `13.213493456674257`, and
`13.184613059846484` TFLOP/s/rank; rounded throughputs were
`16.51123385956795`, `16.51686682084282`, and `16.480766324808105`
TFLOP/s/rank. Dropped routes, routing drops, metadata overflow, queue-route
overflow, and layout-row overflow were all zero. The mode reported 64 live
pairs, 1,048,576 useful rows, 1,310,720 rounded rows, and 0.8 row efficiency.

`semantic_fused_w2_backward_pallas` was start-only. At
`23:09:12.540967 UTC`, it logged a failed 12.50 GiB allocator request and then
emitted no repeat, summary, structured error, or further progress before the
stop threshold. Median/min/max steady-state time, useful/rounded throughput,
checksum, and backward counters are unavailable. Raw logs, terminal status,
task summary, closed monitor state, and a standalone report are in
`scratch/6597-w2-target-20260710-1608/`.

The forward result rejects a literal two-owner port for semantic raw-token
gathers: it regressed from the previous `29.791293` ms / `57.667418` useful
TFLOP/s/rank row to `55.011869` ms / `31.229386`. The old inbox topology was
fast because its senders read prepacked contiguous rows. Semantic producers
must retain enough within-chunk gather parallelism while keeping the old
rolling-slot and fixed-wait protocol.

## 2026-07-10 FUSED-MOE-039 - W2 return as compute plus rolling return/combine

The W2 return stage now uses the shared compute-plus-return/combine template.
For each source peer it has two return-chunk owners, 30 fixed W2 workers, 12
rolling B64 outbox slots, grouped hidden-tile jobs, and cumulative start/done/
ready generations. The source combine derives the maximum required generation
for each `(token block, destination ordinal, slot)` from semantic inverse-route
metadata, waits only for those generations, then performs fp32 top-k accumulation
from compact bf16 `return_y`.

This removes the previous all-entry readiness barrier while preserving distinct
compact storage for every logical entry; outbox slot reuse cannot overwrite
`return_y`. Five dedicated tests, target-shape kernel construction, and scoped
pre-commit pass. Target H100 timing and correctness remain required.

## 2026-07-10 FUSED-MOE-040 - Rolling B64 W2 return target result

Job `/dlwh/bench-semantic-w2-return-rolling-20260710-1633` ran once on
`cw-rno2a` at commit `e8b76742a5` and reached terminal Iris state `succeeded`.
Its single H100x8 task exited 0 after 1 minute and 45.21 seconds, with zero
failures and preemptions. No duplicate, stop, resubmit, Iris restart, or cluster
bounce was issued.

The target EP8 random-routing `semantic_fused_w2_return_pallas` mode completed
all three repeats with the rolling B64 return outbox, two chunk owners plus 30
W2 workers per source, and generation-aware source combine. Exact median/min/max
steady-state time was `79.3795056718712`/`79.37796365392084`/
`79.43745500718553` ms. Median useful/rounded throughput was
`10.821350573166791`/`13.52668821645849` TFLOP/s/rank, and the output checksum
was `4097386496.0`.

Dropped routes, routing drops, metadata overflow, queue-route overflow, and
layout-row overflow were all zero. The mode reported 64 live pairs, 1,048,576
useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and 0.2 masked-row
fraction. Logs contained non-fatal failed 12.50 GiB allocator attempts and CUDA
VMM handle fallback warnings before the successful rows. The closed monitor
record is `scratch/20260710-1633_bench_semantic_w2_return_rolling_monitoring_state.json`.

## 2026-07-10 FUSED-MOE-041 - Remove W2 return local outbox payload staging

The 79.38 ms result localizes the rolling-return regression to payload staging,
not semantic readiness. W2 compute workers wrote accumulator fragments to a
local GMEM outbox, after which chunk owners copied the full B64-by-H payload to
peer `return_y`. This repeated the local-GMEM tax that the direct-GMEM
correctness work had already made unnecessary.

Compute workers now store bf16 accumulator fragments directly into local or
peer compact `return_y`. Chunk owners retain only start, fixed completion,
ready, and release generation management; source combine readiness is
unchanged. Five dedicated tests, target kernel construction, and scoped
pre-commit including Pyrefly pass. Target H100 timing remains required.

## 2026-07-10 FUSED-MOE-042 - Hierarchical W2 backward send plus compute

W2 backward now uses two chunk lifecycle owners without forcing those two CTAs
to perform all semantic gather/preparation. The existing 30 fixed compute CTAs
also prepare the 40 independent B64-by-256 `dy_route` tiles inside each B256
send. Owners wait for fixed cumulative helper completion before publishing the
slot; helpers then consume B64 WGMMA jobs. Preparation and consumption
interleave per chunk across the 12 rolling slots.

This retains independent B256 `send_m` and B64 `compute_m`, removes the old
16-fragment sender pool and 40-way ad hoc publication path, and keeps dW2 as an
expert-local atomic reduction side output. Six dedicated tests and scoped
pre-commit pass. Target H100 correctness and timing remain required.

## 2026-07-10 FUSED-MOE-043 - Direct-generation W2 return target result

Job `/dlwh/bench-semantic-w2-return-directgen-20260710-1638` ran once on
`cw-rno2a` at commit `1048f62b64` and reached terminal Iris state `succeeded`.
Its single H100x8 task exited 0 after 1 minute and 42.92 seconds, with zero
failures and preemptions. No duplicate, stop, resubmit, Iris restart, or cluster
bounce was issued.

The target EP8 random-routing `semantic_fused_w2_return_pallas` mode completed
all three repeats after removing local outbox payload staging and storing WGMMA
fragments directly to local or peer compact `return_y`. Exact median/min/max
steady-state time was `79.7906296599346`/`79.65751533629373`/
`80.16991099187483` ms. Median useful/rounded throughput was
`10.765593188836908`/`13.456991486046135` TFLOP/s/rank, and the output checksum
was `3915118848.0`.

Dropped routes, routing drops, metadata overflow, queue-route overflow, and
layout-row overflow were all zero. The mode reported 64 live pairs, 1,048,576
useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and
0.19999999999999996 masked-row fraction. No structured benchmark error was
emitted. Logs contained non-fatal failed 12.50 GiB allocator attempts and CUDA
VMM handle fallback warnings before the successful rows. Exact repeat metrics,
counters, terminal evidence, and the closed monitor state are in
`scratch/20260710-1640_bench_semantic_w2_return_directgen_report.md` and
`scratch/20260710-1640_bench_semantic_w2_return_directgen_monitoring_state.json`.

## 2026-07-10 FUSED-MOE-044 - Hierarchical semantic permute plus W13

The literal two-CTA raw-token gather is replaced by hierarchical preparation.
Two chunk owners retain B256 slot lifecycle and publication, while all 30 WGMMA
consumer CTAs also prepare independent B64-by-256 source tiles. Owners wait for
fixed cumulative helper completion before publishing `full`; consumers then
run their fixed B64 jobs and release through the existing cumulative fan-in.

This keeps B256 `send_m` independent of B64 `compute_m`, restores within-chunk
gather parallelism, and preserves direct compact expert-major output. Five
dedicated tests, target kernel construction, and scoped pre-commit including
Pyrefly pass. Target H100 timing remains required.

## 2026-07-10 FUSED-MOE-045 - Hierarchical W13 backward direct return

W13 backward now uses 30 helper/compute CTAs to prepare the 80 B64-by-128
source-X tiles inside each B256 chunk, while two owners retain empty/prepare/
full/release generation coordination. dX WGMMA accumulator fragments store
directly into local or peer source-owned compact `dx_return`; the local GMEM
outbox and owner payload recopy are removed. dW13 remains an expert-local
reduction side output, and source top-k combine uses semantic inverse-route
readiness.

The isolated stage harness now supplies source-sharded X and destination-sharded
expert-major dZ13/W13 directly, avoiding the diagnostic pair scatter. Nine
focused kernel/harness tests and scoped pre-commit including Pyrefly pass.
Target H100 correctness and timing remain required.

## 2026-07-10 FUSED-MOE-047 - Producer-first coarse W2 return experiment

The rolling W2-return protocol is replaced with the earlier coarse-readiness
shape to isolate residency ordering. Program IDs 0 through 159 are direct-peer
W2 producers indexed by `(source ordinal, hidden tile)`; IDs 160 through 191
are source combine consumers. Producers scan live entries, write accumulator
fragments directly to compact peer `return_y`, and emit one readiness signal.
Combine performs only 256 coarse waits before fp32 top-k accumulation.

This removes rolling outbox/generation metadata and ensures blocked combine
CTAs cannot occupy the first residency wave. Four dedicated tests and scoped
pre-commit including Pyrefly pass. Target H100 timing remains required.

## 2026-07-10 FUSED-MOE-046 - Hierarchical semantic permute plus W13 target run

Observation-only babysitting began for
`/dlwh/bench-semantic-w13-hierarchical-20260710-1648` on `cw-rno2a` at commit
`50667d88fe`. The target is `semantic_permute_w13_pallas`, three repeats, with
two B256 slot owners plus 30 B64 helper/compute CTAs. Iris reported `running`,
submitted at `2026-07-10 16:45:12 PDT`. No duplicate, resubmit, restart, stop,
or cluster bounce is permitted. Monitoring state is recorded in
`scratch/20260710-1648_bench_semantic_w13_hierarchical_monitoring_state.json`.

Iris reached terminal state `succeeded`; the single H100x8 task exited 0 after
2 minutes and 1.2 seconds with zero failures and preemptions. All three repeats
completed. Exact median/min/max steady-state time was
`27.372991006510954`/`27.038667001761496`/`27.54601902173211` ms. Median
useful/rounded throughput was `62.76211898039782`/`78.45264872549728`
TFLOP/s/rank, and the output checksum was `1305986465792.0`.

Dropped routes, routing drops, metadata overflow, queue-entry overflow,
queue-route overflow, and layout-row overflow were all zero. The mode reported
64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, 0.19999999999999996 masked-row fraction, and 288 queue entries per
rank. There were three repeat rows and zero error rows. Logs contained
non-fatal failed 12.50 GiB allocator attempts and CUDA VMM handle fallback
warnings before successful output. Exact repeat metrics, counters, terminal
evidence, and the closed monitor state are in
`scratch/20260710-1648_bench_semantic_w13_hierarchical_results.json`,
`scratch/20260710-1648_bench_semantic_w13_hierarchical_report.md`,
`scratch/20260710-1648_bench_semantic_w13_hierarchical_terminal.txt`, and the
monitoring state linked above.

## 2026-07-10 FUSED-MOE-048 - Hierarchical W2 backward target hang

Job `/dlwh/bench-semantic-w2b-hierarchical-20260710-1644` ran once on
`cw-rno2a` at commit `5fe60e67f0`. It targeted three repeats of
`semantic_fused_w2_backward_pallas` with two B256 chunk lifecycle owners, 30
B64 helper/compute CTAs, and the expert-local dW2 side reduction.

The first failed 12.50 GiB allocator request was logged at
`2026-07-10T23:44:05.020516Z`, followed by a second at
`2026-07-10T23:44:15.022004Z`. No repeat, summary, structured error, or other
benchmark progress appeared by the 10-minute deadline at
`2026-07-10T23:54:05.020516Z`, so the job was stopped under the requested hang
policy. Iris reached terminal state `killed`; its single H100x8 task exited 0
after 11 minutes and 44.5 seconds with zero failures and one preemption. No
duplicate, resubmit, task kick, Iris restart, or cluster bounce was issued.

The run produced zero structured repeat rows, zero structured summary rows,
and zero structured error rows. Timing, throughput, checksum, semantic row
counts, and all backward/drop/overflow counters are therefore unavailable.
Exact timestamps, terminal evidence, and the closed monitor state are in
`scratch/20260710-1644_bench_semantic_w2b_hierarchical_report.md`,
`scratch/20260710-1644_bench_semantic_w2b_hierarchical_results.json`,
`scratch/20260710-1644_bench_semantic_w2b_hierarchical_terminal.txt`, and
`scratch/20260710-1644_bench_semantic_w2b_hierarchical_monitoring_state.json`.

## 2026-07-10 FUSED-MOE-049 - Independent send and compute row readiness

The semantic lowering now treats `send_m` and `compute_m` as distinct physical
units. The initial Hopper profile retains B256 slot allocation and reuse to
amortize queue and semaphore overhead, while four B64 row blocks become ready
for WGMMA independently. A B64 consumer waits only for the K-copy tiles that
populate its own rows; it does not wait for the remaining rows in the B256
slot. Slot release still waits for all compute jobs across all four blocks.

This refines the previous hierarchical implementations, which exposed separate
B256/B64 configuration but still used one full-slot readiness event. Forward
W13, W2 backward, and W13 backward workers were redirected to implement
block-local readiness with distinct helper and WGMMA CTA roles. The intended
partial order is now copy tiles for B64 block `b` before WGMMA jobs for `b`,
with all jobs in the B256 slot before slot reuse.

## 2026-07-10 FUSED-MOE-054 - Independent-readiness W13 backward target hang

Job `/dlwh/bench-semantic-fused-bc8c-w13b-20260710-1709` ran once on
`cw-rno2a` at commit `bc8c101222`. It targeted
`semantic_fused_w13_backward_pallas` on one H100x8 task. Iris initially
reported `pending` with pending scheduler feedback, then `running` after the
task began setup at `2026-07-10 17:09:21 PDT`.

The benchmark logged failed 12.50 GiB BFC allocator requests at
`2026-07-10 17:10:21.662714 PDT` and `2026-07-10 17:10:31.664153 PDT`. No
structured repeat, summary, error, or other benchmark progress appeared in the
following ten minutes. Under the explicit hang policy, the existing job was
stopped at `2026-07-10 17:21:16 PDT`; no duplicate, resubmit, task kick, Iris
restart, or cluster bounce was issued.

Iris reached terminal state `killed` with reason `Terminated by user`. Its
single task was killed with exit 0 after 12 minutes and 21.55 seconds, with
zero failures and one preemption. The run produced zero structured repeat
rows, zero structured summary rows, and zero structured error rows. Median,
minimum, and maximum timing; useful and rounded TFLOP/s; correctness metrics;
and drop, routing, metadata, queue, and layout counters are therefore
unavailable.

## 2026-07-10 FUSED-MOE-047 - Producer-first coarse W2 return target result

Job `/dlwh/bench-semantic-fused-bc8c-w2f-20260710-1709` ran once on
`cw-rno2a` at commit `bc8c101222` and reached terminal Iris state `succeeded`.
Its single H100x8 task exited 0 after 1 minute and 52.29 seconds, with zero
failures and preemptions. No duplicate, stop, resubmit, Iris restart, or cluster
bounce was issued.

The requested EP8 random-routing `semantic_fused_w2_return_pallas` mode emitted
three structured repeat rows and one structured summary row, with zero error
rows. Exact median/min/max steady-state time was `0.06641123168325673`/
`0.06628185832717766`/`0.0665090666540588` seconds
(`66.41123168325673`/`66.28185832717766`/`66.5090666540588` ms). Median
useful/rounded throughput was `12.934460593908323`/`16.1680757423854`
TFLOP/s/rank, and the output checksum was `3915118848.0` in every repeat.
Repeat 0/1/2 steady-state times were `0.0665090666540588`/
`0.06628185832717766`/`0.06641123168325673` seconds; useful throughput was
`12.915433976362662`/`12.959706937603853`/`12.934460593908323` TFLOP/s/rank,
and rounded throughput was `16.14429247045333`/`16.199633672004815`/
`16.1680757423854` TFLOP/s/rank.

Dropped routes, routing drops, metadata overflow, queue-route overflow, and
layout-row overflow were all zero. The mode reported 64 live pairs, 1,048,576
useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and
0.19999999999999996 masked-row fraction. Block sizes were K64, N128, M64, with
16 producer programs per peer and 32 combine programs. Logs contained two
non-fatal failed 12.50 GiB allocator attempts and CUDA VMM FABRIC+POSIX_FD
handle fallback warnings before the successful structured rows.

## 2026-07-10 FUSED-MOE-053 - Independent-readiness W2 backward target run

Observation-only babysitting began for
`/dlwh/bench-semantic-fused-bc8c-w2b-20260710-1709` on `cw-rno2a`, targeting
`semantic_fused_w2_backward_pallas` at commit `bc8c101222`. Iris reported one
matching job in `pending` state, submitted at `2026-07-11T00:07:55.970Z`, with
reason `Pending scheduler feedback`; no task logs were present at the initial
check. No duplicate, resubmit, restart, stop, task kick, or cluster bounce is
authorized without a concrete main-thread handoff.

The task began setup at `2026-07-11T00:09:21Z`. Its first failed 12.50 GiB
allocator request was logged at `2026-07-11T00:10:22.054153Z`, followed by a
second at `2026-07-11T00:10:32.055560Z`. At `2026-07-11T00:21:14Z`, more than
10 minutes after the first warning, Iris still reported `running` and the logs
still contained no structured repeat, error, or summary row. The job was
therefore stopped under the requested hang policy; no duplicate or resubmit
was issued.

Iris reached terminal state `killed` with reason `Terminated by user`. Its
single H100x8 task exited 0 after 12 minutes and 28.91 seconds, with zero
failures and one preemption. The run produced zero structured repeat rows,
zero structured error rows, and zero structured summary rows. Exact
median/min/max timing, useful/rounded TFLOP/s, output checksum, correctness
counters, and drop/overflow counters are unavailable because the benchmark
never emitted a result row.

## 2026-07-10 FUSED-MOE-051 - Independent-readiness W13 forward target run

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-bc8c-w13f-20260710-1709` on `cw-rno2a`, targeting
`semantic_permute_w13_pallas` at commit `bc8c101222`. Iris reported terminal
state `succeeded`; the single H100x8 task succeeded with exit 0 after 1 minute
and 37.59 seconds, with zero failures and preemptions. No duplicate, resubmit,
restart, stop, task kick, or cluster bounce was issued.

The run emitted three structured repeat rows and one summary row, with zero
error rows. Repeat 0 reported `29.973478328126173` ms and
`57.31690194887708`/`71.64612743609635` useful/rounded TFLOP/s/rank; repeat 1
reported `30.171285344598193` ms and
`56.94112460832183`/`71.1764057604023`; repeat 2 reported
`29.910045986374218` ms and `57.438457940942115`/`71.79807242617764`.
The exact median/min/max steady-state time was
`29.973478328126173`/`29.910045986374218`/`30.171285344598193` ms. Median
useful/rounded throughput was `57.31690194887708`/`71.64612743609635`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`11.60969651886262`, `4.589810455916449`, and `7.019886062946171` seconds.

The output checksum was `1305986465792.0`; every repeat had `error: null`.
Dropped routes, routing drops, metadata overflow, queue-entry overflow,
queue-route overflow, and layout-row overflow were all zero. The mode reported
64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, 0.19999999999999996 masked-row fraction, and 288 queue entries per
rank. Logs contained two non-fatal failed 12.50 GiB allocator attempts and
CUDA VMM handle fallback warnings before all repeat and summary rows completed.

## 2026-07-10 FUSED-MOE-052 - Independent-readiness W2 return target run

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-8769-w2f-20260710-1715` on `cw-rno2a`, targeting
`semantic_fused_w2_return_pallas` with no synthetic scratch at commit
`87694ad4dc`. Iris reached exact terminal state `succeeded`; its single H100x8
task succeeded with exit 0 after 1 minute and 37.82 seconds, with zero failures
and preemptions. No duplicate, resubmit, restart, stop, task kick, or cluster
bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.06629139033611864` seconds (`66.29139033611864` ms) and
`12.957843467222927`/`16.19730433402866` useful/rounded TFLOP/s/rank. Repeat 1
reported `0.0661479876531909` seconds (`66.1479876531909` ms) and
`12.98593486628256`/`16.2324185828532` useful/rounded TFLOP/s/rank. Repeat 2
reported `0.06624067900702357` seconds (`66.24067900702357` ms) and
`12.96776349633916`/`16.20970437042395` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.06624067900702357`/`0.0661479876531909`/`0.06629139033611864` seconds
(`66.24067900702357`/`66.1479876531909`/`66.29139033611864` ms). Median
useful/rounded throughput was `12.96776349633916`/`16.20970437042395`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`5.079508807975799`, `2.9137780469609424`, and `2.1657307610148564` seconds.

The output checksum was `3915118848.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, metadata overflow,
queue-route overflow, and layout-row overflow were all zero. The mode reported
64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, and 0.19999999999999996 masked-row fraction. Block sizes were K64,
N128, and compute M64, with 16 producer programs per peer and 32 combine
programs. Logs contained two non-fatal failed 12.50 GiB allocator attempts and
CUDA VMM FABRIC+POSIX_FD handle fallback warnings before all repeat and summary
rows completed.

## 2026-07-10 FUSED-MOE-053 - 32-worker W13 forward target run

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-8769-w13f-20260710-1715` on `cw-rno2a`, targeting
`semantic_permute_w13_pallas` with the 32-worker split at commit `87694ad4dc`.
Iris reached exact terminal state `succeeded`; its single H100x8 task succeeded
with exit 0 after 1 minute and 38.87 seconds, with zero failures and
preemptions. No duplicate, resubmit, restart, stop, task kick, or cluster
bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.02595914698516329` seconds (`25.95914698516329` ms) and
`66.18040721376167`/`82.72550901720207` useful/rounded TFLOP/s/rank. Repeat 1
reported `0.0258783856794859` seconds (`25.8783856794859` ms) and
`66.38694312999085`/`82.98367891248856` useful/rounded TFLOP/s/rank. Repeat 2
reported `0.025977071995536487` seconds (`25.977071995536487` ms) and
`66.13474061646336`/`82.6684257705792` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.02595914698516329`/`0.0258783856794859`/`0.025977071995536487` seconds
(`25.95914698516329`/`25.8783856794859`/`25.977071995536487` ms). Median
useful/rounded throughput was `66.18040721376167`/`82.72550901720207`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`11.547150690923445`, `4.232943148934282`, and `7.314207541989163` seconds.

The output checksum was `1305986465792.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, metadata overflow, queue-entry
overflow, queue-route overflow, and layout-row overflow were all zero. The mode
reported 64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, 0.19999999999999996 masked-row fraction, and 288 queue entries per
rank. The reported source-push profile was
`hopper_source_push_inbox_rough_balanced_216` with block M64. Logs contained
two non-fatal failed 12.50 GiB allocator attempts and CUDA VMM
FABRIC+POSIX_FD handle fallback warnings before all repeat and summary rows
completed.

## 2026-07-10 FUSED-MOE-055 - Fused-SwiGLU W2 backward 32-worker target run

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-1072-w2b-20260710-1727` on `cw-rno2a`, targeting
`semantic_fused_w2_backward_pallas` with 32 workers per peer at commit
`1072877a0f`. Iris reached exact terminal state `succeeded`; its single H100x8
task succeeded with exit 0 after 2 minutes and 17.57 seconds, with zero
failures and preemptions. No duplicate, resubmit, restart, stop, task kick, or
cluster bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.12864630300706872` seconds (`128.64630300706872` ms) and
`13.354343484752933`/`16.692929355941168` useful/rounded TFLOP/s/rank. Repeat
1 reported `0.13548123801592737` seconds (`135.48123801592737` ms) and
`12.680626067190431`/`15.85078258398804` useful/rounded TFLOP/s/rank. Repeat 2
reported `0.12819995067548007` seconds (`128.19995067548007` ms) and
`13.400839152807784`/`16.75104894100973` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.12864630300706872`/`0.12819995067548007`/`0.13548123801592737` seconds
(`128.64630300706872`/`128.19995067548007`/`135.48123801592737` ms). Median
useful/rounded throughput was `13.354343484752933`/`16.692929355941168`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`19.733724814956076`, `17.54621010296978`, and `2.187514711986296` seconds.

The output checksum was `124060752.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, metadata overflow,
queue-route overflow, and layout-row overflow were all zero. The mode reported
64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, and 0.19999999999999996 masked-row fraction. Block sizes were
compute M64, hidden 128, intermediate 128, send M256, and send hidden 256,
with 12 inbox slots.

The first failed 12.50 GiB allocator request was logged at
`2026-07-11T00:27:36.529145Z`, followed by a second at
`2026-07-11T00:27:46.530708Z`. Structured repeat and summary rows arrived at
`2026-07-11T00:28:28Z`, about 51.47 seconds after the first warning and well
inside the 10-minute no-progress cutoff. Logs also contained non-fatal CUDA
VMM FABRIC+POSIX_FD handle fallback warnings before the structured rows.

## 2026-07-10 FUSED-MOE-056 - 32-worker W13 backward target run

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-1072-w13b-20260710-1727` on `cw-rno2a`, targeting
`semantic_fused_w13_backward_pallas` with 32 workers per peer at commit
`1072877a0f`. Iris reached exact terminal state `succeeded`
(`JOB_STATE_SUCCEEDED`); its single H100x8 task succeeded with exit 0 after
2 minutes and 28.2 seconds, with zero failures and preemptions. No duplicate,
resubmit, restart, stop, task kick, or cluster bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.15760997500425825` seconds (`157.60997500425825` ms) and
`21.800484624828908`/`27.250605781036132` useful/rounded TFLOP/s/rank. Repeat
1 reported `0.15527769367326982` seconds (`155.27769367326982` ms) and
`22.127929360091233`/`27.659911700114044` useful/rounded TFLOP/s/rank. Repeat
2 reported `0.15693655433521295` seconds (`156.93655433521295` ms) and
`21.89403132593849`/`27.36753915742311` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.15693655433521295`/`0.15527769367326982`/`0.15760997500425825` seconds
(`156.93655433521295`/`155.27769367326982`/`157.60997500425825` ms). Median
useful/rounded throughput was `21.89403132593849`/`27.36753915742311`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`16.07843820200651`, `13.806995096994797`, and `2.2714431050117128` seconds.

The output checksum was `20131827712.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, metadata overflow, queue-route
overflow, and layout-row overflow were all zero. The mode reported 64 live
pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and
0.19999999999999996 masked-row fraction. Block sizes were hidden 128, output
128, compute M64, send K256, send M256, with 12 inbox slots.

The first failed 12.50 GiB BFC allocator request appeared at
`2026-07-11T00:27:43.906995Z`, followed by a second at
`2026-07-11T00:27:53.908432Z`. Structured rows appeared at `00:28:30Z`, well
before the 10-minute no-progress cutoff. CUDA VMM FABRIC+POSIX_FD handle
fallback warnings appeared on all eight ranks before the successful rows.

## 2026-07-10 FUSED-MOE-057 - W2 return with two adjacent N tiles per job

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-4967-w2f-group2-20260710-1730` on `cw-rno2a`,
targeting `semantic_fused_w2_return_pallas` with two adjacent N tiles per job
at commit `4967039edc`. Iris reached exact terminal state `succeeded`; its
single H100x8 task succeeded with exit 0 after 1 minute and 44.08 seconds,
with zero failures and preemptions. No duplicate, resubmit, restart, stop,
task kick, or cluster bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.06273619000179072` seconds (`62.73619000179072` ms) and
`13.692152156123623`/`17.11519019515453` useful/rounded TFLOP/s/rank. Repeat
1 reported `0.06255813567743947` seconds (`62.55813567743947` ms) and
`13.731123056945277`/`17.163903821181595` useful/rounded TFLOP/s/rank.
Repeat 2 reported `0.06223163831358155` seconds (`62.23163831358155` ms) and
`13.803163189623623`/`17.25395398702953` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.06255813567743947`/`0.06223163831358155`/`0.06273619000179072` seconds
(`62.55813567743947`/`62.23163831358155`/`62.73619000179072` ms). Median
useful/rounded throughput was `13.731123056945277`/`17.163903821181595`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`5.880454795085825`, `3.843960358062759`, and `2.036494437023066` seconds.

The output checksum was `3915118848.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, metadata overflow,
queue-route overflow, and layout-row overflow were all zero. The mode reported
64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, and 0.19999999999999996 masked-row fraction. Block sizes were K64,
N128, and compute M64, with 16 producer programs per peer and 32 combine
programs. Logs contained two non-fatal failed 12.50 GiB allocator attempts and
CUDA VMM FABRIC+POSIX_FD handle fallback warnings before all repeat and summary
rows completed.

## 2026-07-10 FUSED-MOE-058 - W13 forward 8-row gather

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-86c7-w13f-gather8-20260710-1743` on `cw-rno2a`,
targeting the W13 forward 8-row gather at commit `86c7de325c`. Iris reached
exact terminal state `succeeded`; its single H100x8 task exited 0 after 1 minute
and 35.86 seconds, with zero Iris failures and preemptions. No duplicate,
resubmit, restart, stop, task kick, or cluster bounce was issued.

The benchmark failed before producing a repeat. Its first and only structured
error row was emitted at `2026-07-11T00:40:28Z` with exact error type
`RuntimeError` and exact error text `Failed to infer the output layout of the
iota. Please apply plgpu.layout_cast to its output right after its creation.`
The traceback identifies
`lib/levanter/src/levanter/grug/_moe/source_push_semantic_fused_w13.py:529`,
where `_copy_scope` evaluates
`hidden_offsets = k_start + jnp.arange(config.send_k, dtype=jnp.int32)`.
The structured summary reported exact `repeat_rows: 0`, `error_rows: 1`, and
`error: "all repeats failed"`.

For the error row, `compile_time`, `lower_compile_time`, `first_call_time`,
`first_run_time`, `steady_state_time`, `useful_tflops_per_rank`,
`rounded_tflops_per_rank`, and `output_checksum` were all `null`. Exact counters
were `dropped_routes: 0`, `routing_dropped_routes: 0`, and
`metadata_overflow_routes: 0`. The mode reported 64 live pairs, 1,048,576
useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and
0.19999999999999996 masked-row fraction. Block settings were `block_m: 64`,
`entries_per_rank: 288`, and
`source_push_profile: "hopper_source_push_inbox_rough_balanced_216"`.

Before the structured failure, failed 12.50 GiB BFC allocator attempts were
logged at `2026-07-11T00:40:17.025742Z` and
`2026-07-11T00:40:27.027126Z`. They did not change the Iris task's successful
exit status.

## 2026-07-10 FUSED-MOE-059 - W2 return with four adjacent N tiles per job

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-86c7-w2f-group4-20260710-1743` on `cw-rno2a`,
targeting `semantic_fused_w2_return_pallas` with four adjacent N tiles per job
at commit `86c7de325c`. Iris reached exact terminal state `succeeded`; its
single H100x8 task succeeded with exit 0 after 1 minute and 53.66 seconds, with
zero failures and preemptions. No duplicate, resubmit, restart, stop, task
kick, or cluster bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.06529241466584305` seconds (`65.29241466584305` ms) and
`13.156098814788852`/`16.445123518486064` useful/rounded TFLOP/s/rank. Repeat
1 reported `0.06531062566985686` seconds (`65.31062566985686` ms) and
`13.152430410668314`/`16.44053801333539` useful/rounded TFLOP/s/rank. Repeat 2
reported `0.06528477397902559` seconds (`65.28477397902559` ms) and
`13.157638555599101`/`16.447048194498876` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.06529241466584305`/`0.06528477397902559`/`0.06531062566985686` seconds
(`65.29241466584305`/`65.28477397902559`/`65.31062566985686` ms). Median
useful/rounded throughput was `13.156098814788852`/`16.445123518486064`
TFLOP/s/rank. Compile, lower-compile, and first-run times were
`9.12641014996916`, `6.970872813020833`, and `2.1555373369483277` seconds.

The output checksum was `3915118848.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, and metadata overflow were all
zero; queue-route overflow and layout-row overflow were both `0.0`. The mode
reported 64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, and 0.19999999999999996 masked-row fraction. Block sizes were K64,
N128, and compute M64, with 16 producer programs per peer and 32 combine
programs.

The first failed 12.50 GiB allocator request was logged at
`2026-07-11T00:40:18.305880Z`, followed by a second at
`2026-07-11T00:40:28.307572Z`. Structured repeat and summary rows arrived at
`2026-07-11T00:40:47Z`, about 28.69 seconds after the first warning and well
inside the 10-minute no-progress cutoff. CUDA VMM FABRIC+POSIX_FD handle
fallback warnings appeared on all eight ranks before the successful rows.

## 2026-07-10 FUSED-MOE-060 - W13 forward 8-row gather iota layout-cast fix

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-276e-w13f-gather8fix-20260710-1747` on
`cw-rno2a`, targeting the W13 forward 8-row gather with immediate iota layout
casts at commit `276ec40c2b`. Iris reached exact terminal state `succeeded`;
its single H100x8 task exited 0 after 1 minute and 25.93 seconds, with zero
Iris failures and preemptions. No duplicate, resubmit, restart, stop, task
kick, or cluster bounce was issued.

The benchmark failed before producing a repeat. Its first and only structured
error row was emitted at `2026-07-11T00:47:04Z` with exact error type
`TypeError` and exact error text `TiledLayout.__init__() missing 4 required
positional arguments: 'tiling', 'warp_dims', 'lane_dims', and 'vector_dim'`.
The traceback identifies
`lib/levanter/src/levanter/grug/_moe/source_push_semantic_fused_w13.py:530`,
where `_copy_scope` evaluates
`jnp.arange(config.send_k, dtype=jnp.int32)`. The structured summary reported
exact `repeat_rows: 0`, `error_rows: 1`, and `error: "all repeats failed"`.

For the error row, `compile_time`, `lower_compile_time`, `first_call_time`,
`first_run_time`, `steady_state_time`, `useful_tflops_per_rank`,
`rounded_tflops_per_rank`, and `output_checksum` were all `null`. Exact
counters were `dropped_routes: 0`, `routing_dropped_routes: 0`, and
`metadata_overflow_routes: 0`. The mode reported 64 live pairs, 1,048,576
useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and
0.19999999999999996 masked-row fraction. Block settings were `block_m: 64`,
`entries_per_rank: 288`, and
`source_push_profile: "hopper_source_push_inbox_rough_balanced_216"`.

Before the structured failure, failed 12.50 GiB BFC allocator attempts were
logged at `2026-07-11T00:46:53.418448Z` and
`2026-07-11T00:47:03.419983Z`. They did not change the Iris task's successful
exit status.

## 2026-07-10 FUSED-MOE-061 - Compact expert-major W2 backward with fused dZ13

Observation-only babysitting completed for
`/dlwh/bench-semantic-fused-3188-w2b-compact-20260710-1751` on `cw-rno2a`,
targeting compact expert-major dy staging with owned dW2 tiles and fused dZ13
at commit `3188ab69b0`. Iris reached exact terminal state `succeeded`; its
single H100x8 task succeeded with exit 0 after 1 minute and 51.12 seconds,
with zero failures and preemptions. No duplicate, resubmit, restart, stop,
task kick, or cluster bounce was issued.

The run emitted three structured repeat rows and one structured summary row,
with zero error rows. Repeat 0 reported exact steady-state time
`0.11894947732798755` seconds (`118.94947732798755` ms) and
`14.442996783103778`/`18.05374597887972` useful/rounded TFLOP/s/rank. Repeat
1 reported `0.11895087400140862` seconds (`118.95087400140862` ms) and
`14.44282719923231`/`18.05353399904039` useful/rounded TFLOP/s/rank. Repeat 2
reported `0.1188461153069511` seconds (`118.8461153069511` ms) and
`14.455558046326129`/`18.06944755790766` useful/rounded TFLOP/s/rank.

The exact summary median/min/max steady-state time was
`0.11894947732798755`/`0.1188461153069511`/`0.11895087400140862` seconds
(`118.94947732798755`/`118.8461153069511`/`118.95087400140862` ms). Median
useful/rounded throughput was `14.442996783103778`/`18.05374597887972`
TFLOP/s/rank. Compile, lower-compile, first-call, and first-run times were
`7.368648039060645`, `5.334333621081896`, `7.368648039060645`, and
`2.0343144179787487` seconds.

The output checksum was `127795200.0` in every repeat, and every repeat had
`error: null`. Dropped routes, routing drops, and metadata overflow were all
zero; queue-route overflow and layout-row overflow were both `0.0`. The mode
reported 64 live pairs, 1,048,576 useful rows, 1,310,720 rounded rows, 0.8 row
efficiency, and 0.19999999999999996 masked-row fraction. Block settings were
compute M64, hidden 128, intermediate 128, send-hidden 256, send M256, and 12
inbox slots.

Failed 12.50 GiB BFC allocator attempts were logged at
`2026-07-11T00:49:42.973704Z` and `2026-07-11T00:49:52.975131Z`. Structured
repeat and summary rows were received by the polling terminal at
`2026-07-11T00:50:34Z`, about 51 seconds after the first warning and well
inside the 10-minute no-progress cutoff. CUDA VMM FABRIC+POSIX_FD handle
fallback warnings appeared on all eight ranks before the successful rows.

## 2026-07-10 FUSED-MOE-062 - Separate transport rows from compute rows

Commits `e6b8aad7b9` and `f7c77d78ea` make the semantic queue's row
granularities explicit in the fused kernels. Transport reserves and reuses an
aggregate `send_m` slot, while readiness is published independently for each
`compute_m` block in that slot. The only semantic constraint is
`send_m % compute_m == 0`; the current Hopper profile remains B256 transport
feeding four independently ready B64 WGMMA blocks. W13 backward now also has a
separate `send_hidden_block`, removing the accidental use of the transport row
count as a hidden-dimension copy width.

The compact W2-backward target result in FUSED-MOE-061 improves the previous
32-worker result from `128.646303` ms to `118.949477` ms, a 7.54% latency
reduction, but remains structurally slow. Source inspection shows that each
consumer finishes its dZ13 stream before entering dW2 ownership, producing an
effective phase boundary under balanced routing. The owned dW2 tile then scans
all compact B64 blocks and reloads gate/up plus recomputes SwiGLU once per
hidden tile. At the target shape this is approximately 409,600 preparations
instead of about 20,480 unique `(expert, intermediate_tile, B64_block)`
preparations, roughly 20x amplification.

Next experiment: consume each independently ready B64 block once, form H once
per intermediate tile, reuse each dy tile for explicit-WGMMA dH and dW2, and
atomically accumulate the B64 dW2 partial. This intentionally trades back some
destination-local atomic cost to remove the much larger phase split and
recomputation tax. Target H100 runs for compact W13 backward and the concrete
8-row W13 forward gather layouts are active and separately babysat.

## 2026-07-10 FUSED-MOE-063 - Compact W13 backward target benchmark

Launched exactly one H100x8 benchmark on `cw-rno2a` for commit
`e6b8aad7b9`. Job:
`/dlwh/bench-semantic-fused-e6b8-w13b-compact-20260710-1758`.

Exact command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name bench-semantic-fused-e6b8-w13b-compact-20260710-1758 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu -- timeout 3600s bash -lc 'set -euo pipefail; uv pip install --reinstall nvidia-cudnn-cu13==9.19.0.56; exec uv run --no-sync --package marin-levanter --extra gpu --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha e6b8aad7b9 --jsonl scratch/bench-semantic-fused-e6b8-w13b-compact.jsonl'
```

Iris reached terminal `succeeded`: exit 0, zero failures, zero preemptions,
and one task completed in 1 minute and 49.97 seconds. All rows arrived at
`2026-07-11T00:59:18Z`, inside the 10-minute no-progress cutoff. No
duplicate, resubmit, restart, stop, task kick, or cluster bounce was issued.
The 12.50 GiB BFC allocation attempts and CUDA VMM fallback warnings were
non-fatal.

Exact numeric rows:

```jsonl
{"backend": "gpu", "backend_env": {"JAX_PLATFORMS": null, "JAX_PLATFORM_NAME": null, "XLA_FLAGS": null, "XLA_PYTHON_CLIENT_MEM_FRACTION": null, "XLA_PYTHON_CLIENT_PREALLOCATE": null}, "block_sizes": {"block_hidden": 128, "block_output": 128, "compute_m": 64, "inbox_slots": 12, "send_k": 256, "send_m": 256}, "compile_time": 5.879103085986571, "config": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "device_count": 8, "device_type": "NVIDIA H100 80GB HBM3", "dropped_routes": 0, "dtype": "bfloat16", "error": null, "error_message": null, "error_type": null, "first_call_time": 5.879103085986571, "first_run_time": 2.063109170005191, "git_sha": "e6b8aad7b9", "implementation": "pallas_mgpu", "kernel": "source_push_semantic_plan", "layout_overflow_row_error_count": 0.0, "lower_compile_time": 3.8159939159813803, "metadata_overflow_routes": 0, "mode": "semantic_fused_w13_backward_pallas", "output_checksum": 20131262464.0, "queue_overflow_route_error_count": 0.0, "repeat_run": 0, "repeat_runs": 3, "rounded_tflops_per_rank": 59.26884179539688, "routing_dropped_routes": 0, "row_type": "repeat", "semantic_live_pairs": 64, "semantic_masked_row_fraction": 0.19999999999999996, "semantic_rounded_rows": 1310720, "semantic_row_efficiency": 0.8, "semantic_useful_rows": 1048576, "shape": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "steady_state_time": 0.07246585500737031, "useful_tflops_per_rank": 47.415073436317506, "xla_flags": null}
{"backend": "gpu", "backend_env": {"JAX_PLATFORMS": null, "JAX_PLATFORM_NAME": null, "XLA_FLAGS": null, "XLA_PYTHON_CLIENT_MEM_FRACTION": null, "XLA_PYTHON_CLIENT_PREALLOCATE": null}, "block_sizes": {"block_hidden": 128, "block_output": 128, "compute_m": 64, "inbox_slots": 12, "send_k": 256, "send_m": 256}, "compile_time": 5.879103085986571, "config": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "device_count": 8, "device_type": "NVIDIA H100 80GB HBM3", "dropped_routes": 0, "dtype": "bfloat16", "error": null, "error_message": null, "error_type": null, "first_call_time": 5.879103085986571, "first_run_time": 2.063109170005191, "git_sha": "e6b8aad7b9", "implementation": "pallas_mgpu", "kernel": "source_push_semantic_plan", "layout_overflow_row_error_count": 0.0, "lower_compile_time": 3.8159939159813803, "metadata_overflow_routes": 0, "mode": "semantic_fused_w13_backward_pallas", "output_checksum": 20131262464.0, "queue_overflow_route_error_count": 0.0, "repeat_run": 1, "repeat_runs": 3, "rounded_tflops_per_rank": 58.763489323937776, "routing_dropped_routes": 0, "row_type": "repeat", "semantic_live_pairs": 64, "semantic_masked_row_fraction": 0.19999999999999996, "semantic_rounded_rows": 1310720, "semantic_row_efficiency": 0.8, "semantic_useful_rows": 1048576, "shape": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "steady_state_time": 0.07308904466723713, "useful_tflops_per_rank": 47.01079145915022, "xla_flags": null}
{"backend": "gpu", "backend_env": {"JAX_PLATFORMS": null, "JAX_PLATFORM_NAME": null, "XLA_FLAGS": null, "XLA_PYTHON_CLIENT_MEM_FRACTION": null, "XLA_PYTHON_CLIENT_PREALLOCATE": null}, "block_sizes": {"block_hidden": 128, "block_output": 128, "compute_m": 64, "inbox_slots": 12, "send_k": 256, "send_m": 256}, "compile_time": 5.879103085986571, "config": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "device_count": 8, "device_type": "NVIDIA H100 80GB HBM3", "dropped_routes": 0, "dtype": "bfloat16", "error": null, "error_message": null, "error_type": null, "first_call_time": 5.879103085986571, "first_run_time": 2.063109170005191, "git_sha": "e6b8aad7b9", "implementation": "pallas_mgpu", "kernel": "source_push_semantic_plan", "layout_overflow_row_error_count": 0.0, "lower_compile_time": 3.8159939159813803, "metadata_overflow_routes": 0, "mode": "semantic_fused_w13_backward_pallas", "output_checksum": 20131262464.0, "queue_overflow_route_error_count": 0.0, "repeat_run": 2, "repeat_runs": 3, "rounded_tflops_per_rank": 59.33272125962285, "routing_dropped_routes": 0, "row_type": "repeat", "semantic_live_pairs": 64, "semantic_masked_row_fraction": 0.19999999999999996, "semantic_rounded_rows": 1310720, "semantic_row_efficiency": 0.8, "semantic_useful_rows": 1048576, "shape": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "steady_state_time": 0.07238783600041643, "useful_tflops_per_rank": 47.466177007698285, "xla_flags": null}
{"backend": "gpu", "block_sizes": {"block_hidden": 128, "block_output": 128, "compute_m": 64, "inbox_slots": 12, "send_k": 256, "send_m": 256}, "config": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}, "device_count": 8, "device_type": "NVIDIA H100 80GB HBM3", "dropped_routes": 0, "dtype": "bfloat16", "error": null, "error_rows": 0, "implementation": "pallas_mgpu", "kernel": "source_push_semantic_plan", "max_steady_state_time": 0.07308904466723713, "median_compile_time": 5.879103085986571, "median_first_run_time": 2.063109170005191, "median_layout_overflow_row_error_count": 0.0, "median_lower_compile_time": 3.8159939159813803, "median_queue_overflow_route_error_count": 0.0, "median_rounded_tflops_per_rank": 59.26884179539688, "median_steady_state_time": 0.07246585500737031, "median_useful_tflops_per_rank": 47.415073436317506, "metadata_overflow_routes": 0, "min_steady_state_time": 0.07238783600041643, "mode": "semantic_fused_w13_backward_pallas", "repeat_rows": 3, "routing_dropped_routes": 0, "row_type": "summary", "semantic_rounded_rows": 1310720, "semantic_row_efficiency": 0.8, "semantic_useful_rows": 1048576, "shape": {"capacity_factor": 1.25, "dtype": "bfloat16", "ep_size": 8, "experts_per_rank": 32, "hidden_dim": 2560, "intermediate_dim": 1280, "plan_builder": "jax", "routing": "random", "routing_seed": 0, "rows_per_src_dst_capacity": 20480, "tokens_per_rank": 32768, "topk": 4}}
```

## 2026-07-10 FUSED-MOE-064 - Decoupled W13 transport target failure

Exactly one target H100x8 benchmark was launched on `cw-rno2a` for commit
`f7c77d78ea` as job
`/dlwh/bench-semantic-fused-f7c7-w13f-gather8-20260710-1759`. The requested
conceptual mode `semantic_fused_w13_pallas` is spelled
`semantic_permute_w13_pallas` in the benchmark harness at this commit. The
exact launch command was:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name bench-semantic-fused-f7c7-w13f-gather8-20260710-1759 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --topk 4 --experts-per-rank 32 --hidden-dim 2560 --intermediate-dim 1280 --capacity-factor 1.25 --routing random --modes semantic_permute_w13_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha f7c77d78ea --jsonl scratch/bench-semantic-fused-f7c7-w13f-gather8.jsonl
```

Iris reached terminal state `succeeded`; its single task exited 0 after
1 minute and 23.53 seconds, with zero Iris failures and preemptions. No
duplicate, resubmit, restart, cluster bounce, task kick, or kernel edit was
issued.

The benchmark failed before producing a numeric repeat row. Its first
structured error row arrived at `2026-07-11T01:00:43Z` with exact error type
`ValueError` and exact error text
`memref<8xi32, strided<[1], offset: ?>> must have a number of elements that is a multiple of 128 (got 8)`.
The traceback identifies
`lib/levanter/src/levanter/grug/_moe/source_push_semantic_fused_w13.py:550`,
where the 8-row gather loads `token_ids_ref`. The structured summary reported
exact `repeat_rows: 0`, `error_rows: 1`, and `error: "all repeats failed"`.

For the error row, `compile_time`, `lower_compile_time`, `first_call_time`,
`first_run_time`, `steady_state_time`, `useful_tflops_per_rank`,
`rounded_tflops_per_rank`, and `output_checksum` were all `null`. Exact
counters were `dropped_routes: 0`, `routing_dropped_routes: 0`, and
`metadata_overflow_routes: 0`. The mode reported 64 live pairs, 1,048,576
useful rows, 1,310,720 rounded rows, 0.8 row efficiency, and
0.19999999999999996 masked-row fraction. Failed 12.50 GiB BFC allocator
attempts were logged at `2026-07-11T01:00:32.175519Z` and
`2026-07-11T01:00:42.176949Z` before the structured failure, well inside the
10-minute no-progress cutoff.

## 2026-07-10 FUSED-MOE-065 - B64 atomic W2 backward target failure

Launched exactly one H100x8 benchmark on `cw-rno2a` for commit
`f08be40c5a`. Job:
`/dlwh/bench-semantic-fused-f08b-w2b-b64-atomic-20260710-1807`.

Exact command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name bench-semantic-fused-f08b-w2b-b64-atomic-20260710-1807 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu -- timeout 3600s bash -lc 'set -euo pipefail; uv pip install --reinstall nvidia-cudnn-cu13==9.19.0.56; exec uv run --no-sync --package marin-levanter --extra gpu --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w2_backward_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha f08be40c5a --jsonl scratch/bench-semantic-fused-f08b-w2b-b64-atomic.jsonl'
```

Iris reached terminal state `succeeded`; its single task exited 0 after
1 minute and 46.63 seconds, with zero Iris failures and preemptions. No
duplicate, resubmit, restart, cluster bounce, task kick, or kernel edit was
issued.

The benchmark failed during Mosaic GPU lowering before producing a numeric
repeat row. The first structured failure arrived at `2026-07-11T01:08:32Z`:

```jsonl
{"row_type":"error","mode":"semantic_fused_w2_backward_pallas","git_sha":"f08be40c5a","error_type":"TypeError","error_message":"Expected WGMMAAbstractAccumulatorRef got Ref<regs>{float32[128,128]}","compile_time":null,"lower_compile_time":null,"first_call_time":null,"first_run_time":null,"steady_state_time":null,"output_checksum":null,"useful_tflops_per_rank":null,"rounded_tflops_per_rank":null,"dropped_routes":0,"routing_dropped_routes":0,"metadata_overflow_routes":0,"semantic_live_pairs":64,"semantic_useful_rows":1048576,"semantic_rounded_rows":1310720,"semantic_row_efficiency":0.8,"semantic_masked_row_fraction":0.19999999999999996}
{"row_type":"summary","mode":"semantic_fused_w2_backward_pallas","error":"all repeats failed","repeat_rows":0,"error_rows":1}
```

The first actionable source location is
`lib/levanter/src/levanter/grug/_moe/source_push_semantic_fused_w2_backward.py:828`
inside `_dw2_acc_scope`, where `mgpu.wgmma` received a discharged
`Ref<regs>{float32[128,128]}` instead of a `WGMMAAbstractAccumulatorRef`.
Failed 12.50 GiB BFC allocator attempts were logged at
`2026-07-11T01:07:54.405280Z` and `2026-07-11T01:08:04.406734Z`; the structured
failure arrived about 38 seconds after the first warning, inside the 10-minute
no-progress cutoff.

## 2026-07-10 FUSED-MOE-066 - Aligned W13 index SMEM lowering failures

Job `/dlwh/bench-semantic-fused-596b-w13f-index-smem-20260710-2037` at
`596b024ae3` reached terminal Iris state `succeeded` but emitted no repeat row.
The aligned 128-int GMEM-to-SMEM metadata copy lowered; the subsequent 8-int
SMEM vector load failed in `plan_tiled_transfer` with `ZeroDivisionError:
integer modulo by zero`. Job
`/dlwh/bench-semantic-fused-2856-w13f-index-smem-20260710-2042` at
`2856b8882f` replaced that load with scalar SMEM loads plus `jnp.stack`; it also
reached terminal `succeeded` but failed lowering because `stack` is not
implemented for Lane lowering with warpgroup user semantics. Both jobs had
zero routing drops and no timing row. The next retry assembles the vector with
layout-preserving compare/select operations and no `stack` primitive.

## 2026-07-10 FUSED-MOE-067 - B64 atomic W2 backward comparison

Job `/dlwh/bench-semantic-fused-2856-w2b-b64-atomic-20260710-2042` at
`2856b8882f` completed three target repeats with zero drops and overflows. Times
were `123.716365`, `123.032341`, and `122.844066` ms; the median was
`123.032341` ms, with `13.963702` useful and `17.454627` rounded
TFLOP/s/rank. The output checksum was `127795200` in every repeat.

This is 3.43% slower than the atomic-free compact-owner result
`118.949477` ms / `14.442997` useful TFLOP/s/rank in FUSED-MOE-061. Removing
the effective dZ13-then-dW2 phase split and repeated SwiGLU preparation did not
repay B64-granularity fp32 dW2 atomic accumulation and accumulator resets. Keep
the atomic-free owned-tile schedule as the selected production candidate; the
B64 atomic schedule is a negative diagnostic result.

## 2026-07-10 FUSED-MOE-068 - W13 contiguous-row semantic send target

Job `/dlwh/bench-semantic-fused-c4f6-w13f-rowloads-20260710-2050` at
`c4f6696002` completed three target repeats on `cw-rno2a` with zero drops and
overflows. Times were `26.063831`, `25.922804`, and `25.743525` ms. Median
throughput was `66.273189` useful and `82.841487` rounded TFLOP/s/rank; output
checksum was `1305986465792` in every repeat. The aligned 128-int token metadata
DMA followed by grouped contiguous 256-element source-row loads is therefore
the selected supported semantic W13 send path. It is only 0.14% faster than the
previous `25.9591` ms row-loop baseline and remains far behind the prepacked
inbox W13/H path.

## 2026-07-10 FUSED-MOE-069 - Integrated selected-kernel target boundary

Job `/dlwh/bench-semantic-fused-c4f6-integrated-20260710-2053` at
`c4f6696002` ran selected-kernel integrated modes. Forward completed three
repeats at `48.745531`, `49.051833`, and `49.351792` ms. The median was
`49.051833` ms with `52.563225` useful and `65.704032` rounded TFLOP/s/rank,
zero dropped routes, and no structured kernel error. The synthetic target
inputs overflowed the bf16 output checksum to infinity, so a finite-scale
correctness comparison remains required.

Forward+backward failed before launching backward kernels. The custom VJP
returned source-sharded `dx` (`8@expert,32768,2560`) for a replicated primal
`x` (`8,32768,2560`). This is an API-boundary sharding type mismatch, not a
kernel failure. The next commit normalizes source tensors and destination
weights to their production shardings before entering `custom_vjp`, so every
returned gradient matches the custom rule's primal type.

## 2026-07-10 FUSED-MOE-070 - Keep global semantic routing metadata replicated

Commit `766d775637` fixed the first custom-VJP mismatch by source-sharding all
leading-axis inputs. The target retry
`/dlwh/bench-semantic-fused-766d-fwd-bwd-20260710-2101` reached terminal Iris
state `succeeded` but failed before a kernel timing row with:

```text
ShardingTypeError: slicing on sharded dims where out dim (1) is not divisible
by mesh axes (8) with spec (expert) is not implemented
```

The failure was in `build_source_push_semantic_plan_jax` while evaluating
`receiver_counts[sender_index]`. The plan builder intentionally sees the full
global routing tensors and cannot slice a source-sharded leading dimension in
its current JAX formulation.

Commit `4a3922ad1f` therefore keeps `selected_experts` and `route_weights`
replicated while building the global semantic plan, source-shards only `x`,
destination-shards the expert weights, and reshards the returned route-weight
gradient to match its replicated primal. Focused semantic MLP and benchmark
boundary tests passed (`10 passed`), and the changed source passed the required
Marin pre-commit checks. Next action: rerun the target integrated forward and
backward mode at `4a3922ad1f`; if it reaches the kernels, use its first numeric
row or first kernel-specific failure as the next integration decision.

## 2026-07-10 FUSED-MOE-071 - First complete fused semantic forward and backward boundary

Job `/dlwh/bench-semantic-fused-76698-fwd-bwd-20260710-2130` at
`76698c304a` completed the selected fused semantic custom-VJP boundary on the
target H100x8 shape. Iris state was `succeeded`, exit 0, with one successful
task, zero failures, and zero preemptions. The three steady-state rows were:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---:|---:|---:|---:|
| 0 | 242.277932 | 31.926010 | 39.907512 |
| 1 | 242.826982 | 31.853823 | 39.817278 |
| 2 | 242.802626 | 31.857018 | 39.821273 |

The median is `242.802626 ms`, `31.857018` useful TFLOP/s/rank, and
`39.821273` capacity-rounded TFLOP/s/rank. Routing counters were exactly zero:
`dropped_routes=0`, `routing_dropped_routes=0`, and
`metadata_overflow_routes=0`. The output checksum was stable at
`-1.1766269542473693e+26` across all repeats.

This proves that the selected four fused-stage implementations compose through
one custom VJP with consistent shardings. It is not a performance milestone:
the result is 7.85x slower than the `250` useful TFLOP/s/rank target. The
dominant next work remains physical schedule replacement, beginning with the
forward W13 path: preserve semantic metadata, but restore the proven inbox
shape with two aggregated send producers, approximately 30 WGMMA consumers,
B256 allocation, and independent B64 readiness.

## 2026-07-10 FUSED-MOE-072 - W13 backward role split regresses

Commit `0f130df981` split the 64 CTAs that previously performed all raw-x
staging and then dX combine into separate resident groups of 32 staging and 32
combine CTAs. It also changed dX publication from whole-B256 readiness to
per-hidden-tile readiness. Job
`/dlwh/bench-semantic-fused-0f130-w13b-overlap-20260710-2120` completed three
target repeats with zero drops and overflows:

```text
times_ms:             88.264978, 89.238979, 88.997078
median_ms:            88.997078
useful_TFLOP/s/rank:  38.607715
rounded_TFLOP/s/rank: 48.259644
checksum:             20130811904
```

The previous selected compact-owner path was `72.465855 ms` / `47.415073`
useful TFLOP/s/rank. The role split is 22.8% slower: halving raw-x staging
residency costs more than overlapping source combine and publishing hidden
tiles earlier. Commit `30334c4928` reverted this candidate. Keep the 64-CTA
stage-then-combine allocation until staging itself is made cheaper; do not tune
the 32/32 split further.

## 2026-07-10 FUSED-MOE-073 - First inbox-shaped forward and adjacent-N ablations

Job `/dlwh/bench-semantic-fused-30334-three-candidates-20260710-2125` tested
three target candidates at `30334c4928`, with three repeats and zero route
drops or metadata/layout overflows in every mode:

| Candidate | Median ms | Useful TFLOP/s/rank | Prior ms | Decision |
|---|---:|---:|---:|---|
| W13 direct raw gather, 2 producers + 30 consumers | 53.762686 | 31.955005 | 25.922804 | reject; 107.4% slower |
| W2 return, per-adjacent-N completion cohorts | 62.575497 | 13.727313 | 62.558136 | reject; no measurable change |
| W2 backward, reuse gate SMEM for SwiGLU | 120.194554 | 14.293384 | 118.949477 | reject; 1.05% slower |

The W13 checksum was `1305986465792`, W2 return checksum `3915106560`, and W2
backward checksum `127795200`, matching their prior target rows. The W13 result
isolates why the old `216.949` inbox profile does not transfer directly: two
send CTAs are enough for prepacked contiguous B64 rows, but not for semantic
raw-token gathers. The old semantic 2-owner + 10-helper arrangement supplies
far more gather bandwidth despite leaving only 20 WGMMA consumers.

Commits `1b284f02d7` and `7a1a651d94` reverted the neutral W2-return and slower
W2-backward changes. The next W13 experiment keeps the direct B256/B64 protocol
but uses eight gather producers and 24 WGMMA consumers, testing whether added
raw-gather residency beats the old helper schedule without returning to twelve
producer-side programs.

## 2026-07-10 FUSED-MOE-074 - Direct W13 producer-residency curve

Two additional target jobs measured the fixed direct raw-gather protocol while
changing only the 32-program peer-local producer/consumer split:

| Producers / consumers | Times (ms) | Median ms | Useful TFLOP/s/rank |
|---|---|---:|---:|
| 2 / 30 | 53.762686, 54.194796, 53.705101 | 53.762686 | 31.955005 |
| 8 / 24 | 27.905820, 27.718614, 27.787361 | 27.787361 | 61.826199 |
| 12 / 20 | 25.995065, 25.855658, 25.786619 | 25.855658 | 66.445300 |

The 8/24 job was
`/dlwh/bench-semantic-fused-7a1a-w13-8p24c-20260710-2130` at
`7a1a651d94`; the 12/20 job was
`/dlwh/bench-semantic-fused-81c9-w13-12p20c-20260710-2135` at
`81c9b36490`. All repeats had zero route drops and queue/layout overflows and
the same `1305986465792` checksum.

The direct 12/20 schedule is 0.26% faster than the selected 2-owner + 10-helper
+ 20-consumer path (`25.922804 ms`, `66.273189` useful TFLOP/s/rank). This is a
small speedup, but it removes owner/helper handoff and proves that approximately
12 producer-side CTAs are required for raw semantic gathers. Keep 12/20 as the
current candidate. Next isolate its two-buffer send scope: two B64xK256 SMEM
tiles consume 64 KiB per producer CTA and may reduce co-resident WGMMA work;
compare against one-buffer sequential sends before integrating the candidate.

## 2026-07-10 FUSED-MOE-075 - Direct W13 helps forward but loses aggregate objective

The one-buffer 12/20 variant at `6bdeffd5cf` measured `26.056364`,
`25.863272`, and `25.753976 ms`, with median `25.863272 ms` and `66.425737`
useful TFLOP/s/rank. It is indistinguishable from and 0.03% slower than the
two-buffer `25.855658 ms` result. Commit `a6ab528f84` reverted the one-buffer
variant; the async send overlap repays its second 32 KiB producer tile.

Job `/dlwh/bench-semantic-fused-a6ab-integrated-20260710-2145` then measured
the two-buffer 12/20 schedule through the integrated boundary:

| Mode | Times (ms) | Median ms | Useful TFLOP/s/rank | Prior median ms |
|---|---|---:|---:|---:|
| forward | 48.740140, 48.517199, 48.682463 | 48.682463 | 52.962040 | 49.051833 |
| fwd+bwd | 243.649039, 243.081268, 244.860232 | 243.649039 | 31.746350 | 242.802626 |

Forward alone improves 0.75%, but fwd+bwd regresses 0.35%. A higher-confidence
fwd+bwd-only job,
`/dlwh/bench-semantic-fused-a6ab-fwd-bwd-repeat5-20260710-2150`, used two
warmups, five timed steps, and five repeat runs. Times were `244.726842`,
`245.273243`, `245.682916`, `244.498768`, and `244.382273 ms`; median was
`244.726842 ms`, `31.606536` useful TFLOP/s/rank. This is 0.79% slower than the
first complete selected-boundary baseline. All integrated runs had zero drops
and metadata overflows.

The project target is aggregate fwd+bwd throughput, so the direct producer
series is rejected despite its small forward-only win. Commits `b37f188355`,
`e067d9f2cf`, and `1f14e70fff` revert the 12/20, 8/24, and direct-producer
schedule commits, respectively. The selected W13 path returns to two chunk
owners + ten gather helpers + twenty WGMMA consumers. Future W13 work must make
the raw gather itself cheaper or more coalesced; redistributing the same gather
work among persistent CTAs has now been bracketed and does not improve the
aggregate objective.

## 2026-07-10 FUSED-MOE-076 - W2 fused-kernel decomposition

Job `/dlwh/bench-semantic-w2-decomposition-4d8f-20260710-2200` measured the
selected target shape with three repeats per mode:

| Mode | Median ms | Useful TFLOP/s/rank |
|---|---:|---:|
| W2 expert-major prepacked | 32.988112 | 26.039485 |
| Direct return to source | 6.691665 | 128.367665 |
| Source-owned combine | 2.932673 | n/a |
| Fused W2 + return + combine | 61.828298 | 13.893209 |

The split-stage median sum is `42.612450 ms`, leaving `19.215848 ms` of fused
execution tax. This is not launch overhead: the fused persistent consumer loop
serializes input staging, SwiGLU, each WGMMA K step, remote return, and slot
release. The isolated W2 mode is also much slower than the historical balanced
microbenchmark, so both tensor-core scheduling and overlap need work. The
direct-return synthetic checksum was NaN, so use that row for timing
decomposition only; integrated correctness remains the acceptance gate.

The next experiment is a manual two-stage SMEM K pipeline in the Lane-lowered
fused kernel. Peer-id refs prevent using the existing warp-specialized pipeline
helper, but the current K loop can still prefetch tile `k+1` while WGMMA consumes
tile `k` before the final wait and remote return.

## 2026-07-10 FUSED-MOE-077 - Manual WGMMA pipeline ablation

Job `/dlwh/bench-semantic-fused-pipelines-20260710-2215` at `68c6eb6f89`
completed successfully on `cw-rno2a` (`exit=0`, one task, 2m37s). It measured
the three manual two-stage SMEM pipeline candidates on the target random-routing
shape, with three repeats per mode:

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Selected baseline | Decision |
|---|---|---:|---:|---:|---|
| Fused W2 return | 62.135641, 62.654210, 62.459036 | 62.459036 | 13.752909 | 62.558136 ms / 13.731123 | neutral: 0.16% faster |
| Fused W2 backward | 129.846319, 131.135306, 129.984515 | 129.984515 | 13.216858 | 118.949477 ms / 14.442997 | reject: 9.28% slower |
| Fused W13 backward | 66.068675, 66.282220, 67.336677 | 66.282220 | 51.838545 | 72.465855 ms / 47.415073 | keep candidate: 8.53% less time, 9.33% more throughput |

Rounded-throughput medians were `17.191137`, `16.521073`, and `64.798181`
TFLOP/s/rank, respectively. Checksums were stable across all three repeats:
`3915118848` for W2 return, `127795200` for W2 backward, and `20131303424`
for W13 backward. Every repeat reported zero routing drops, metadata-overflow
routes, queue-overflow route errors, and layout-overflow row errors.

The W13 backward result validates overlapping SMEM refill with explicit WGMMA
for its dX and dW13 scans. The same technique does not automatically transfer
to W2 backward: its 128 KiB dW2 operand staging likely reduces occupancy enough
to outweigh the removed per-block drains. The W2-return change is below the
noise threshold and should not be selected on this run alone. Before changing
the integrated boundary, retain the W13 candidate for a full fwd+bwd comparison
and revert or redesign the W2-backward pipeline.

## 2026-07-10 FUSED-MOE-078 - K512 helper transport lowering failure

Job `/dlwh/bench-semantic-w13-k512-20260710-2225` tested
`semantic_permute_w13_pallas` at `efdacd1ef6` on `cw-rno2a`. Iris reached
`succeeded` (`exit=0`, one task, 1m22s), but the benchmark emitted one error
row and no repeat rows. Mosaic rejected the helper's SMEM-to-GMEM copy during
lowering:

```text
ValueError: Async copies only support copying <=256 elements along each
dimension, got (1, 1, 64, 512)
```

The failing call is `mgpu.copy_smem_to_gmem` in
`source_push_semantic_fused_w13.py`; therefore K512 produced no timing,
checksum, or drop/overflow counters to compare with the selected K256 baseline
(`25.922804 ms`, `66.273189` useful TFLOP/s/rank, checksum
`1305986465792`). Per the experiment rule, do not resubmit this form. A future
K512 experiment must keep each physical async-copy dimension at most 256, for
example by issuing two K256 copies, while preserving the wider logical helper
work unit.

## 2026-07-10 FUSED-MOE-079 - W13-backward pipeline wins integrated fwd+bwd

Job `/dlwh/bench-semantic-integrated-w13b-pipeline-20260710-2230` at
`139afe4e28` completed successfully on `cw-rno2a` (`exit=0`, one task,
3m39s). This commit retains only the measured W13-backward two-stage WGMMA
pipeline; the neutral W2-forward and slower W2-backward candidates were
already reverted.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Baseline | Result |
|---|---|---:|---:|---:|---|---|
| Integrated forward | 49.732339, 49.339032, 49.538596 | 49.538596 | 52.046743 | 65.058428 | 49.051833 ms / 52.563225 useful | 0.99% slower; reject as a forward-only change |
| Integrated fwd+bwd | 236.813932, 236.564280, 236.956558 | 236.813932 | 32.662638 | 40.828297 | 242.802626 ms / 31.857018 useful | keep: 5.988694 ms or 2.47% less time, 2.53% more useful throughput |

Every repeat reported zero dropped routes, zero routing drops, and zero
metadata-overflow routes. Forward emitted the existing integrated harness
checksum `Infinity` on all repeats. Forward+backward emitted the stable checksum
`-1.1766203134195027e+26` on all repeats. The process logged recoverable 12.5
GiB BFC allocation failures and FABRIC-handle VMM fallback warnings during
initialization, but both modes produced all three rows and Iris terminated
successfully.

The isolated W13-backward improvement (`72.465855 -> 66.282220 ms`) survives
the integrated boundary: fwd+bwd saves 5.99 ms even though forward-only timing
is about 0.49 ms noisier in this run. Keep the W13-backward pipeline selected;
the aggregate fwd+bwd objective is the acceptance criterion.

## 2026-07-10 FUSED-MOE-080 - Overlap round 2 isolates a W2-forward win

Job `/dlwh/bench-semantic-overlap-round2-20260710-2240` at `62de1c1541`
completed on `cw-rno2a` (`Iris succeeded`, exit 0, one task, 2m19s). It tested
the split-publication K512 W13 helper, a 128-worker W2 forward schedule with
cohort-local combine, and the low-SMEM W2-backward gate/up prefetch.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Baseline | Result |
|---|---|---:|---:|---:|---|---|
| Fused W13 forward, K512 gather | no repeat rows | n/a | n/a | n/a | 25.922804 ms / 66.273189 useful | reject: Mosaic lowering failure |
| Fused W2 return, 128-worker cohort | 44.099458, 44.087474, 44.085326 | 44.087474 | 19.483844 | 24.354805 | 62.558136 ms / 13.731123 useful | candidate: 18.470662 ms or 29.52% less time |
| Fused W2 backward, low-SMEM prefetch | 119.637949, 119.617371, 119.492700 | 119.617371 | 14.362353 | 17.952941 | 118.949477 ms / 14.442997 useful | reject: 0.667894 ms or 0.56% slower |

The K512 W13 helper passed the prior per-dimension size check after splitting
publication into two K256 copies, but each half is a strided view of the K512
SMEM tile. Mosaic rejected the first publication with:

```text
ValueError: async_copy needs the SMEM reference to be contiguous, but got
strides [512, 1] for shape [64, 256]
```

Therefore this W13 variant produced no checksum or drop/overflow counters. A
valid physical implementation needs two separately contiguous K256 publication
buffers, or a synchronous copy that supports the strided source view; do not
retry the current form.

The W2-forward candidate was stable across repeats and reported zero routing
drops, metadata-overflow routes, queue-overflow route errors, and layout-overflow
row errors. Its checksum was `3915118336` on every repeat, versus the supplied
selected-baseline checksum `3915106560` (difference `11776`, about 3.0e-6
relative). Treat the timing as a strong candidate but require the integrated
correctness check before selection. The W2-backward checksum remained exactly
`127795200`, with all drop and overflow counters zero; its small slowdown means
the low-SMEM prefetch does not repay its scheduling cost.

The W2-forward result is the first large improvement in this round and is
consistent with the earlier decomposition: the prior fused path had roughly
19.2 ms of execution tax beyond isolated W2, return, and combine. The cohort
schedule removes 18.47 ms, nearly all of that measured tax. Next run it through
integrated forward and fwd+bwd correctness/timing while retaining the selected
W13-backward pipeline.

## 2026-07-10 FUSED-MOE-081 - Reduced integrated W2 cohort compare blocked by reference sharding

Job `/dlwh/compare-semantic-integrated-w2-cohort-20260710-2250` at
`a8b05dc63e` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
task, and a 36.55-second task duration. The reduced EP8/T128/E4/K2 integrated
forward comparison did not reach either the cohort kernel comparison or its
numeric checks: the benchmark emitted one error row and zero repeat rows.

The first actionable failure is a `ShardingTypeError` in the interpret/reference
W13 path. `source_push_semantic_fused_w13_reference_jax` performs
`w_gate_up.at[destination, safe_expert].get()` on expert-sharded weights, and
JAX cannot infer the gather result sharding:

```text
Use `.at[...].get(out_sharding=)` to provide output PartitionSpec for the
gather indexing as out sharding could not be resolved unambiguously (or would
require collectives on inputs). Got
operand=ShapedArray(bfloat16[8@expert,4,2560,2560]),
indices=ShapedArray(int32[8,8,1,4,2])
```

The failing call is reached while constructing `expected_y` with
`semantic_fused_mlp_forward(inputs, interpret=True)`, before the comparison
metrics are calculated. Consequently this run produced no max or mean error,
error counts, dropped-route difference, output checksum, or timing. Its input
routing metadata reported 10 capacity-dropped/metadata-overflow routes and zero
routing-policy drops, but those are not candidate-versus-reference differences.

This is a comparison-harness/reference-sharding failure, not evidence for or
against the cohort W2 kernel. Keep the 44.087474 ms isolated timing as a
candidate only; the integrated correctness gate remains open. Per the run
instructions, no automatic retry or kernel change was made.

## 2026-07-10 FUSED-MOE-083 - W2 cohort win survives the integrated boundary

Job `/dlwh/bench-semantic-integrated-w2-cohort-20260710-2250` at
`a8b05dc63e` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
task, and a 3m30s task duration. This boundary retains the W2 cohort-local
combine and the selected W13-backward pipeline.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Selected baseline | Result |
|---|---|---:|---:|---:|---|---|
| Integrated forward | 30.636462, 30.669469, 30.852934 | 30.669469 | 84.068053 | 105.085067 | 49.538596 ms / 52.046743 useful | 18.869127 ms or 38.09% less time; 61.53% more useful throughput |
| Integrated fwd+bwd | 217.732366, 218.166632, 218.824850 | 218.166632 | 35.454403 | 44.318004 | 236.813932 ms / 32.662638 useful | 18.647300 ms or 7.87% less time; 8.55% more useful throughput |

Every repeat reported zero dropped routes, zero routing-policy drops, and zero
metadata-overflow routes. Forward emitted `Infinity` on all repeats, matching
the existing selected integrated-forward checksum. Forward+backward emitted
`-1.1766203134195027e+26` on all repeats, exactly matching the selected
integrated fwd+bwd checksum. The process logged the same recoverable 12.5 GiB
BFC allocation failures and FABRIC-handle VMM fallbacks seen in prior integrated
runs; both modes nevertheless produced all requested rows and Iris terminated
successfully.

The performance effect is real and nearly one-for-one with the isolated W2
improvement: the integrated forward saves 18.87 ms versus the isolated stage's
18.47 ms. This confirms that cohort-local combine removes fused W2 scheduling
tax rather than merely shifting it elsewhere in the graph. Retain the cohort
schedule as the current performance candidate.

Correctness is not fully closed. Matching `Infinity` shows no new checksum
regression relative to the selected path, but a non-finite checksum cannot
validate either path. The reduced finite-reference comparison in FUSED-MOE-081
failed before reaching the candidate due to ambiguous reference gather
sharding. The next gate is to repair that comparison boundary or add a finite
integrated error metric, then compare cohort output against the prior W2 path
before declaring this schedule production-selected.

## 2026-07-10 FUSED-MOE-082 - Reduced direct W2 cohort comparison passes

Job `/dlwh/compare-semantic-w2-cohort-direct-20260710-2255` at
`a8b05dc63e` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
task, and a 43.45-second task duration. This reduced EP8/T128/E4/K2 run bypassed
the unrelated integrated W13 reference-sharding failure and compared
`semantic_fused_w2_return_compare` directly.

| Quantity | Result |
|---|---:|
| `return_y` max absolute difference | 0.1654167175 |
| `return_y` mean absolute difference | 0.0064286510 |
| final `y` max absolute difference | 0.375 |
| final `y` mean absolute difference | 0.0049526617 |
| expected/observed `return_y` nonfinite errors | 0 / 0 |
| expected/observed `y` nonfinite errors | 0 / 0 |
| validity errors | 0 |
| queue-overflow route errors | 0 |
| layout-overflow row errors | 0 |
| live `return_y` elements | 5,217,280 |
| output checksum | 5,217,280.5 |

The random reduced shape had 10 metadata-capacity overflows and therefore 10
capacity-dropped routes; routing-policy drops were zero. These drops are shared
input-plan behavior, not a candidate/reference disagreement. The one-repeat
steady-state time was 4.949592 ms, but this reduced correctness shape is not a
target-shape performance measurement.

The cohort-local combine candidate therefore passes the direct numerical and
structural gate: all expected and observed outputs are finite, and there are no
validity, queue, or layout errors. The bf16 route-buffer comparison has small
mean error and bounded max error after the final weighted combine. Retain the
target-shape 44.087474 ms timing candidate and proceed to integrated
target-shape forward and fwd+bwd timing/correctness; the earlier integrated
reference harness still needs an explicit gather output sharding before it can
serve as that gate.

## 2026-07-10 FUSED-MOE-084 - Two-buffer K512 W13 transport wins

Job `/dlwh/bench-semantic-w13-k512-two-buffers-20260710-2300` at
`d2ce47ca35` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
task, and a 1m48s task duration. The semantic W13 helper shares one token-ID
load across two independent contiguous B64xK256 SMEM payloads and publishes
them as one logical K512 helper item.

| Quantity | Result |
|---|---:|
| Repeat times | 23.025404, 23.000391, 23.202345 ms |
| Median time | 23.025404 ms |
| Useful TFLOP/s/rank | 74.612671 |
| Rounded TFLOP/s/rank | 93.265839 |
| Min / max time | 23.000391 / 23.202345 ms |
| Output checksum | 1,305,986,465,792 |
| Dropped routes | 0 |
| Metadata-overflow routes | 0 |
| Queue entry / route overflows | 0 / 0 |
| Layout row overflows | 0 |

Against the selected K256 baseline of 25.922804 ms and 66.273189 useful
TFLOP/s/rank, K512 saves 2.897400 ms or 11.18% of stage time and raises useful
throughput by 12.58%. Its checksum exactly matches the K256 baseline. Keep the
two-buffer K512 helper transport as the W13-forward candidate and evaluate it
at the integrated forward and fwd+bwd boundaries before making it the selected
production schedule.

The process emitted recoverable 12.5 GiB BFC allocation warnings and FABRIC
handle VMM fallbacks before producing all requested rows. There was no Mosaic
lowering error, benchmark error row, or runtime correctness counter failure.

## 2026-07-10 FUSED-MOE-085 - Three-winner forward sets a new boundary; backward config is stale

Job `/dlwh/bench-semantic-integrated-three-winners-20260710-2310` at
`8cc14a507b` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
task, and a 2m44s task duration. The integrated candidate combines the
cohort-local W2 return/combine schedule, the two-buffer K512 W13 helper
transport, and the selected pipelined W13 backward kernel.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Result |
|---|---|---:|---:|---:|---|
| Integrated forward | 29.300053, 29.336042, 30.173681 | 29.336042 | 87.889243 | 109.861554 | New selected forward candidate |
| Integrated fwd+bwd | none | none | none | none | Failed before compilation: backward rebuilt W13 metadata with `send_k=256`, but the selected Hopper lowering requires 512 |

Against the W2-cohort/K256 integrated forward baseline of 30.669469 ms and
84.068053 useful TFLOP/s/rank, the K512 helper saves 1.333427 ms or 4.35% and
raises useful throughput by 4.55%. Against the original complete forward
baseline of 49.051833 ms, the three-winner forward saves 19.715791 ms or
40.19%.

All three forward repeats reported zero dropped routes, zero routing-policy
drops, and zero metadata-overflow routes. The output checksum was `Infinity`
on every repeat, matching the prior integrated forward boundary but remaining
non-diagnostic; the separate finite direct W2 comparison in FUSED-MOE-082 is
still the available numerical evidence. The integrated repeat schema did not
emit separate queue-entry, route, or layout-overflow counters.

The fwd+bwd mode produced one error row and no repeat rows, checksum, drop
counters, or throughput. The failure is a configuration-plumbing mismatch in
`source_push_semantic_fused_w13_backward_metadata_jax`: it constructs the
shared forward metadata path with `send_k=256`, which now fails the K512 Hopper
validation. This run therefore does not measure the aggregate effect of all
three winners. Do not compare it to the selected 218.166632 ms fwd+bwd boundary
until the backward metadata uses the selected W13 transport configuration.

The process emitted the same recoverable 12.5 GiB BFC allocation warnings and
FABRIC-handle VMM fallbacks as earlier integrated runs. No retry or kernel
change was made while babysitting this job.

## 2026-07-10 FUSED-MOE-086 - K512 W2 backward cuts the integrated boundary to 174.8 ms

Job `/dlwh/bench-semantic-w2b-k512-integrated-20260710-2320` at
`7528a3fa2c` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
task, and a 3m08s task duration. The candidate uses logical B64xK512 W2-backward
producer items published as two contiguous B64xK256 payloads, together with the
cohort-local W2 forward schedule, two-buffer K512 W13 forward transport,
pipelined W13 backward, and the shared-forward-config metadata fix.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---|---|---:|---:|---:|
| W2 backward | 76.733521, 77.179025, 76.837743 | 76.837743 | 22.358633 | 27.948292 |
| Integrated fwd+bwd | 174.834816, 175.590344, 174.689098 | 174.834816 | 44.241575 | 55.301969 |

The isolated W2-backward candidate saves 42.111734 ms or 35.40% against the
118.949477 ms baseline and raises useful throughput from 14.442997 to
22.358633 TFLOP/s/rank, a 54.81% increase. Its checksum is unchanged at
`127795200`. All repeats reported zero dropped routes, zero routing-policy
drops, zero metadata-overflow routes, zero queue-overflow route errors, and
zero layout-overflow row errors.

The integrated candidate saves 43.331816 ms or 19.86% against the selected
K256-forward 218.166632 ms boundary and raises useful throughput from
35.454 to 44.241575 TFLOP/s/rank. Against the original 242.802626 ms boundary,
it saves 67.967810 ms or 27.99%; useful throughput rises from 31.857018 to
44.241575 TFLOP/s/rank, a 38.88% increase. Integrated repeats reported zero
dropped routes, zero routing-policy drops, and zero metadata-overflow routes.
The integrated schema did not emit separate queue or layout counters.

The integrated checksum was stable within this run at
`-1.0907506275226644e26`, but differs from the prior selected boundary's
`-1.1766203134195027e26`. The isolated W2-backward checksum and the previously
validated forward components match their baselines, but this aggregate checksum
change means the full candidate still needs a finite numerical integrated
comparison before it is considered correctness-proven. Keep the K512 W2
backward schedule as the performance candidate; do not claim final integration
correctness from this timing run alone.

The process emitted recoverable 12.5 GiB BFC allocation warnings and
FABRIC-handle VMM fallbacks, then produced all requested repeat and summary
rows. There was no lowering error, benchmark error row, or retry.

## 2026-07-10 FUSED-MOE-088 - Reduced K512 W2-backward comparison hits an illegal address

Job `/dlwh/compare-semantic-w2b-k512-20260710-2330` at `f84e84957a`
completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one task, and a
1m04s task duration. The requested reduced-shape
`semantic_fused_w2_backward_compare` benchmark did not complete: the candidate
kernel triggered `CUDA_ERROR_ILLEGAL_ADDRESS`, which surfaced while JAX tried
to record the next CUDA event for the output checksum.

| Comparison | Max abs error | Mean abs error | Nonfinite / error count |
|---|---:|---:|---:|
| `d_z13` | unavailable | unavailable | unavailable |
| `d_w2` | unavailable | unavailable | unavailable |
| `d_route_weight` | unavailable | unavailable | unavailable |

The benchmark emitted one error row, zero comparison/repeat rows, and a summary
with `error="all repeats failed"`. It therefore produced no output checksum,
timing, or numerical comparison metrics. The reduced random-routing shape had
10 metadata-capacity drops, 10 total dropped routes, zero routing-policy drops,
2,038 useful rows, 2,560 rounded rows, and 79.609375% row efficiency. No queue
or layout-overflow counters were emitted before the device fault.

This contradicts correctness for the `f84e84957a` four-helper K512 W2-backward
candidate on the reduced comparison shape. The target-shape timing and stable
checksum do not supersede an illegal-address failure. Keep the target-shape
timing as performance evidence only; do not promote this schedule until the
memory fault is isolated and a finite `d_z13`/`d_w2`/route-weight comparison
succeeds. Per the bounded babysitting request, no kernel edit or retry was made.

## 2026-07-10 FUSED-MOE-089 - Four-helper K512 W2 backward nearly halves stage time

Job `/dlwh/bench-semantic-backward-aggregation-20260710-2330` at
`f84e84957a` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, and
one task. It measured the four-helper K512 W2-backward producer schedule and
the four-publication K512 W13-backward staging schedule at the target random
routing shape.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---|---|---:|---:|---:|
| W2 backward, four helpers per peer | 39.824625, 40.296294, 39.913566 | 39.913566 | 43.042682 | 53.803352 |
| W13 backward, four K128 publications | 65.808347, 65.906800, 66.310181 | 65.906800 | 52.133829 | 65.167286 |

The four-helper W2-backward schedule saves 36.924177 ms or 48.05% against the
five-helper K512 result of 76.837743 ms and raises useful throughput from
22.358633 to 43.042682 TFLOP/s/rank. Against the original 118.949477 ms
baseline, it saves 79.035911 ms or 66.44% and raises useful throughput from
14.442997 by 198.02%. Its checksum remains `127795200`. The reduced-shape
comparison in FUSED-MOE-088 nevertheless hit an illegal address, so quarantine
this performance candidate until that fault is isolated and finite numerical
correctness passes.

The aggregated W13-backward staging schedule saves only 0.375420 ms or 0.57%
against the selected 66.282220 ms pipeline and raises useful throughput from
51.838545 by 0.57%. Its checksum is `20131401728`, which is 98,304 above the
selected pipeline checksum `20131303424`. This is too small a performance gain
to justify promotion without a finite numerical comparison; retain the prior
W13-backward schedule for now.

Both modes reported zero dropped routes, zero routing-policy drops, zero
metadata-overflow routes, zero queue-overflow route errors, and zero
layout-overflow row errors. The process emitted recoverable 12.5 GiB BFC
allocation warnings and FABRIC-handle VMM fallbacks, then produced all six
repeat rows and both summaries without a benchmark error.

## 2026-07-10 FUSED-MOE-090 - Helper4 succeeds alone; composed comparison faults

Job `/dlwh/debug-w2b-helper4-compare-20260710-2340` at `a00cf3bb02`
completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one task, and a
54.82s task duration. The reduced no-drop shape used T512, E4/rank, top-k 2,
capacity factor 4.0, random routing, B64 compute, and logical K512 producer
transport. It ran `semantic_fused_w2_backward_pallas` first and then
`semantic_fused_w2_backward_compare` in the same benchmark process.

The Pallas-only mode completed its compile, first execution, and timed repeat:

| Metric | Result |
|---|---:|
| Lower/compile time | 5.089672s |
| First execution time | 2.070099s |
| Timed repeat | 4.743654 ms |
| Useful TFLOP/s/rank | 2.829416 |
| Capacity-rounded TFLOP/s/rank | 11.317666 |
| Output checksum | 1,832,960 |
| Dropped / metadata-overflow routes | 0 / 0 |
| Queue-overflow routes / layout-overflow rows | 0 / 0 |

This shape has 8,192 useful rows, 32,768 capacity-rounded rows, 25% row
efficiency, and a 75% masked-row fraction. The timing is diagnostic rather than
a target-shape performance result.

The subsequent composed comparison failed in
`jit_semantic_fused_w2_backward_compare` with `CUDA_ERROR_ILLEGAL_ADDRESS`
while `_block_until_ready` queried the CUDA stream. It emitted one error row,
zero repeat/comparison rows, and no `d_z13`, `d_w2`, or `d_route_weight`
max/mean-error metrics. Drops and metadata overflow were still zero at the
benchmark boundary, but queue/layout counters were unavailable after the
fault.

This run does not reproduce a fault when the helper4 Pallas kernel executes and
synchronizes alone. The failure is specific to the executable that composes the
interpreted reference and Pallas candidate, or to state/memory interaction
created by that composition. The standalone checksum is not a numerical
correctness proof. Keep the fast helper4 schedule quarantined until a comparison
that isolates reference and candidate execution produces finite
`d_z13`/`d_w2`/route-weight metrics. No retry or kernel edit was made during
this bounded diagnostic.

## 2026-07-10 FUSED-MOE-091 - Split W2-backward comparison passes exactly

Job `/dlwh/compare-split-w2b-helper4-20260710-2350` at `e4abeea048`
completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one of one task
succeeded, and a 50.17s task duration. The comparison ran the interpreted
reference and four-helper K512 Pallas candidate in separate executions, copied
each result to host, released device arrays between executions, and compared
the resulting gradients on host. This avoids the composed executable and
device-memory interaction that produced the illegal addresses in
FUSED-MOE-088 and FUSED-MOE-090.

| Gradient | Max abs error | Mean abs error | Expected nonfinite | Observed nonfinite |
|---|---:|---:|---:|---:|
| `d_z13` | 0.0 | 0.0 | 0 | 0 |
| `d_w2` | 0.0 | 0.0 | 0 | 0 |
| `d_route_weight` | 0.0 | 0.0 | 0 | 0 |

The reduced no-drop comparison shape used T512, E4/rank, top-k 2, capacity
factor 4.0, random routing, B64 compute, and logical K512 producer transport.
It reported zero dropped routes, zero routing-policy drops, zero
metadata-overflow routes, zero validity errors, zero queue-overflow route
errors, and zero layout-overflow row errors. The benchmark emitted one repeat
row and one summary row with no error.

This exact finite comparison clears the correctness quarantine on the
four-helper K512 W2-backward schedule. The earlier illegal addresses were a
comparison-composition problem rather than evidence of a faulty standalone
kernel. Keep the target-shape FUSED-MOE-089 result as its performance evidence:
39.913566 ms and 43.042682 useful TFLOP/s/rank, 66.44% faster than the original
118.949477 ms W2-backward baseline. No retry or kernel edit was made during
this bounded babysitting run.

## 2026-07-10 FUSED-MOE-092 - Adjacent-N W13 dW grouping regresses

Job `/dlwh/bench-w13b-adjacent-n-20260710-2350` at `e4abeea048`
completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one of one task
succeeded, and a 1m43.43s task duration. It measured the adjacent-N W13
gradient schedule from `930876778e`, where each dW13 owner shares an X tile
across two neighboring O128 output tiles.

| Metric | Adjacent-N result | Selected pipeline | Change |
|---|---:|---:|---:|
| Repeat times | 84.488504, 85.402947, 84.589032 ms | - | - |
| Median time | 84.589032 ms | 66.282220 ms | 27.62% slower |
| Useful TFLOP/s/rank | 40.619614 | 51.838545 | 21.64% lower |
| Capacity-rounded TFLOP/s/rank | 50.774518 | 64.798181 | 21.64% lower |
| Output checksum | 20,129,775,616 | 20,131,303,424 | -1,527,808 |

All repeats reported zero dropped routes, zero routing-policy drops, zero
metadata-overflow routes, zero queue-overflow route errors, and zero
layout-overflow row errors. The process emitted recoverable 12.5 GiB BFC
allocation warnings and FABRIC-handle VMM fallbacks, then produced all three
repeat rows and the summary without a benchmark error.

The shared X load does not repay the larger operand footprint and accumulator
pressure at this target shape. Reject adjacent-N grouping and retain the
selected one-output-tile W13-backward pipeline at 66.282220 ms and 51.838545
useful TFLOP/s/rank. The checksum difference independently means this variant
would require a finite numerical comparison even if its timing were favorable.
Per the bounded request, no retry or kernel edit was made.

## 2026-07-11 FUSED-MOE-098 - Integrated retry clears route sharding but faults in backward

Job `/dlwh/integrated-compare-d237` tested the reduced integrated forward and
fwd+bwd comparisons at `d237d620ea` on `cw-rno2a`. Exactly one H100x8 job was
launched. The shape was EP8, T128/rank, H2560, I1280, E4/rank, top-k 2,
capacity factor 4.0, random routing seed 0, bf16, JAX plan metadata, zero
warmups, one step, one repeat, and separately compiled reference and Pallas
executions.

Command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name integrated-compare-d237 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_mlp_forward_compare,semantic_fused_mlp_forward_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha d237d620ea --jsonl scratch/integrated-compare-d237.jsonl
```

Iris reported terminal state `succeeded`, exit 0, one of one task succeeded,
and a 2m02.81s duration. This is a harness-level false success for the backward
mode: the forward comparison emitted a repeat and summary row, then the
backward execution caused `CUDA_ERROR_ILLEGAL_ADDRESS` on all eight devices and
emitted no result row. The debug-exception wrapper still exited zero.

| Forward host metric | Result |
|---|---:|
| `y_max_abs_diff` | 27712.0 |
| `y_mean_abs_diff` | 122.16789245605469 |
| `y_expected_abs_sum` | 35432582398.5625 |
| `y_observed_abs_sum` | 35438733167.515625 |
| `y_least_squares_scale` | 0.9997958680757424 |
| `y_cosine_similarity` | 0.9996540069856641 |
| `expected_y_nonfinite_error_count` | 0.0 |
| `observed_y_nonfinite_error_count` | 0.0 |
| `dropped_routes_error_count` | 0.0 |
| candidate `dropped_routes` | 0 |
| candidate `routing_dropped_routes` | 0 |
| candidate `metadata_overflow_routes` | 0 |

The forward diagnostic took 57.963894 ms and reported 0.086878 useful and
0.347512 rounded TFLOP/s/rank. This reduced-shape comparison timing is not a
target-shape performance result. Its output checksum was `70871343104.0`.

No backward host metrics were produced:

| Gradient | Absolute difference | Scale / cosine | Nonfinite counts |
|---|---|---|---|
| `dx` | unavailable | unavailable | unavailable |
| `d_route_weights` | unavailable | unavailable | unavailable |
| `dw13` | unavailable | unavailable | unavailable |
| `dw2` | unavailable | unavailable | unavailable |

The first actionable backward failure was the device-wide
`CUDA_ERROR_ILLEGAL_ADDRESS`; subsequent module-unload, stream/event-destroy,
device-memory-free, and host-memory-unregister errors were cleanup fallout.
There was no Python traceback or structured backward error row to localize the
fault further. The absence of the prior custom-VJP sharding error shows that
`d237d620ea` reached runtime, but the integrated fwd+bwd correctness gate remains
uncleared. Do not interpret Iris's exit-0 state as kernel success. Per the
bounded request, no code was edited and no retry, duplicate run, or Iris
cluster mutation was performed.

## 2026-07-11 FUSED-MOE-096 - W13-backward role48 comparison blocked by reference gather sharding

Job `/dlwh/w13b-role48-compare-d237` at `d237d620ea` completed on
`cw-rno2a` with Iris state `succeeded`, exit 0, one of one task succeeded, and
a 43.49s task duration. Exactly one H100x8 job was launched. The reduced
random-routing comparison shape was T128 per rank, E4/rank, top-k 2, capacity
factor 4.0, H2560, I1280, bf16, and eight expert-parallel ranks. The command
was:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-role48-compare-d237 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha d237d620ea --jsonl scratch/w13b-role48-compare-d237.jsonl
```

The comparison produced no `dx` or `dw13` numerical metrics. The separately
compiled reference failed during lowering at
`source_push_semantic_fused_w13_backward_reference_jax` before the Pallas
candidate ran:

```text
ShardingTypeError: Use `.at[...].get(out_sharding=)` to provide output
PartitionSpec for the gather indexing as out sharding could not be resolved
unambiguously (or would require collectives on inputs). Got
operand=ShapedArray(bfloat16[8@expert,128,2560]),
indices=ShapedArray(int32[8,8,2,4,64,2])
```

Consequently `dx`/`dw13` absolute differences, scale, cosine similarity, and
nonfinite metrics are unavailable, as are candidate queue/layout overflow
metrics. The pre-run semantic metadata row reported zero dropped routes, zero
routing-policy drops, and zero metadata-overflow routes. This run neither
validates nor falsifies the 48-staging/40-dX/40-dW candidate. The next bounded
comparison must make the reference gather's output sharding explicit while
keeping reference and candidate compilation separate. Per the request, no code
edit, retry, duplicate, or Iris cluster mutation was performed.

## 2026-07-11 FUSED-MOE-096 - Integrated comparison still fails backward VJP sharding

Job `/dlwh/integrated-compare-45d6-20260711-0002` at `45d6402623`
completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one of one task
succeeded, and a 1m27.23s task duration. This was the single requested reduced
H100x8 correctness run: EP8, T128/rank, H2560, I1280, E4/rank, top-k 2,
capacity factor 4.0, random routing seed 0, bf16, zero warmups, one timed step,
one repeat, and separate reference/Pallas compilation.

Command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name integrated-compare-45d6-20260711-0002 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_mlp_forward_compare,semantic_fused_mlp_forward_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha 45d6402623 --jsonl scratch/integrated-compare-45d6-20260711-0002.jsonl
```

The forward host comparison completed with finite outputs and matching route
cardinality. The absolute error is large in isolation because the synthetic
outputs have an expected absolute sum of `3.54325823985625e10`; the normalized
metrics are scale `0.9994649892` and cosine similarity `0.9996123629`.

| Forward host metric | Result |
|---|---:|
| `y_max_abs_diff` | 34816.0 |
| `y_mean_abs_diff` | 121.29826354980469 |
| `y_expected_abs_sum` | 35432582398.5625 |
| `y_observed_abs_sum` | 35428997866.515625 |
| `y_least_squares_scale` | 0.9994649892214961 |
| `y_cosine_similarity` | 0.9996123629170338 |
| `expected_y_nonfinite_error_count` | 0.0 |
| `observed_y_nonfinite_error_count` | 0.0 |
| `dropped_routes_error_count` | 0.0 |
| candidate `dropped_routes` | 0 |
| candidate `routing_dropped_routes` | 0 |
| candidate `metadata_overflow_routes` | 0 |

The forward diagnostic time was 57.320585 ms, 0.087853 useful TFLOP/s/rank,
and 0.351412 rounded TFLOP/s/rank. It is a reduced-shape correctness diagnostic,
not a target-shape performance result.

The backward comparison failed while lowering the separately compiled
reference and emitted no host metrics for `dx`, `d_route_weights`, `dw13`, or
`dw2`. The first actionable failure remains the custom-VJP output sharding
mismatch:

```text
ValueError: Custom VJP bwd rule must produce an output with the same type as the args tuple of the primal function, but at output[2] the bwd rule produced an output of type float32[8@expert,128,2] corresponding to an input of type float32[8,128,2]
```

Raw forward repeat row:

```json
{"backend":"gpu","compile_time":15.293175374856219,"config":{"capacity_factor":4.0,"dtype":"bfloat16","ep_size":8,"experts_per_rank":4,"hidden_dim":2560,"intermediate_dim":1280,"plan_builder":"jax","routing":"random","routing_seed":0,"rows_per_src_dst_capacity":128,"tokens_per_rank":128,"topk":2},"device_count":8,"device_type":"NVIDIA H100 80GB HBM3","dropped_routes":0,"dropped_routes_error_count":0.0,"error":null,"expected_y_nonfinite_error_count":0.0,"first_call_time":15.293175374856219,"first_run_time":2.129469515872188,"git_sha":"45d6402623","lower_compile_time":13.16370585898403,"metadata_overflow_routes":0,"mode":"semantic_fused_mlp_forward_compare","observed_y_nonfinite_error_count":0.0,"output_checksum":70861619200.0,"repeat_run":0,"repeat_runs":1,"rounded_tflops_per_rank":0.3514120617739505,"routing_dropped_routes":0,"row_type":"repeat","semantic_live_pairs":64,"semantic_masked_row_fraction":0.75,"semantic_rounded_rows":8192,"semantic_row_efficiency":0.25,"semantic_useful_rows":2048,"steady_state_time":0.05732058500871062,"useful_tflops_per_rank":0.08785301544348763,"y_cosine_similarity":0.9996123629170338,"y_expected_abs_sum":35432582398.5625,"y_least_squares_scale":0.9994649892214961,"y_max_abs_diff":34816.0,"y_mean_abs_diff":121.29826354980469,"y_observed_abs_sum":35428997866.515625}
```

Raw backward error row, with the traceback elided after its first actionable
exception because it repeats the same type mismatch through the JIT boundary:

```json
{"compile_time":null,"dropped_routes":0,"error":"Custom VJP bwd rule must produce an output with the same type as the args tuple of the primal function, but at output[2] the bwd rule produced an output of type float32[8@expert,128,2] corresponding to an input of type float32[8,128,2]","error_message":"Custom VJP bwd rule must produce an output with the same type as the args tuple of the primal function, but at output[2] the bwd rule produced an output of type float32[8@expert,128,2] corresponding to an input of type float32[8,128,2]","error_type":"ValueError","git_sha":"45d6402623","metadata_overflow_routes":0,"mode":"semantic_fused_mlp_forward_backward_compare","routing_dropped_routes":0,"row_type":"error","steady_state_time":null}
```

Interpretation: the forward comparison now supplies useful normalized evidence
and has no nonfinite or route-count errors, but the integrated fwd+bwd
correctness gate remains unexecuted. The next fix must make the custom VJP
return `d_route_weights` with the same replicated sharding/type as its primal
argument at the reference boundary before rerunning this gate. No kernel or
harness code was edited and no retry or duplicate H100 job was launched.

## 2026-07-10 FUSED-MOE-093 - Fully selected helper4 path reaches 57.1 useful TFLOP/s/rank

Job `/dlwh/bench-semantic-integrated-helper4-selected-20260710-2400` at
`8c51ec3c93` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one
of one task succeeded, and a 3m22.57s task duration. This is the fully selected
target-shape path: two-buffer K512 semantic W13 forward, cohort-local W2 return
and source combine, four-helper K512 W2 backward, and the pipelined single-N
W13 backward.

| Mode | Repeat times (ms) | Median ms | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---|---|---:|---:|---:|
| Integrated forward | 28.105864, 27.744702, 27.964463 | 27.964463 | 92.199965 | 115.249957 |
| Integrated fwd+bwd | 135.395309, 135.414665, 135.673697 | 135.414665 | 57.120606 | 71.400757 |

Forward saves 1.371579 ms or 4.68% against the prior K512 selected candidate
at 29.336042 ms and raises useful throughput from 87.889243 to 92.199965
TFLOP/s/rank. Against the original complete-forward baseline of 49.051833 ms,
it saves 21.087370 ms or 42.99% and raises useful throughput by 75.41%.

Fwd+bwd saves 39.420151 ms or 22.55% against the five-helper K512 result of
174.834816 ms and raises useful throughput from 44.241575 to 57.120606
TFLOP/s/rank, a 29.11% increase. Against the original 242.802626 ms boundary,
it saves 107.387961 ms or 44.23%; useful throughput rises from 31.857018 by
79.30%.

All six repeats reported zero dropped routes, zero routing-policy drops, and
zero metadata-overflow routes. The integrated schema does not emit separate
queue-entry, route, or layout-overflow counters. The forward checksum was
`Infinity` on every repeat and remains non-diagnostic. The fwd+bwd checksum was
stable at `-9.452085837927071e25`, but differs from both the five-helper K512
checksum `-1.0907506275226644e26` and the original selected checksum
`-1.1766203134195027e26`. The four-helper W2-backward kernel passed the split
finite comparison exactly in FUSED-MOE-091, but this timing run is not a finite
end-to-end numerical comparison; retain that as the remaining aggregate
correctness caveat.

The process emitted recoverable 12.5 GiB BFC allocation warnings and
FABRIC-handle VMM fallbacks, then produced all six repeat rows and both summary
rows. There was no benchmark error, retry, or kernel edit. Promote this as the
current performance-selected integrated schedule while keeping finite
end-to-end comparison as the next correctness gate.

## 2026-07-10 FUSED-MOE-094 - Split integrated finite comparison fails its gate

Job `/dlwh/compare-split-integrated-selected-20260710-2410` at
`4625779973` completed on `cw-rno2a` with Iris state `succeeded`, exit 0, and
one of one task succeeded. The reduced random-routing comparison shape was T128
per rank, E4/rank, top-k 2, capacity factor 4.0, H2560, I1280, bf16, and eight
H100 ranks. Reference and Pallas executions were compiled and run separately;
the reference operands were replicated and the production candidate remained
sharded.

The forward comparison completed, but it did not pass numerically:

| Metric | Result |
|---|---:|
| `y` max absolute difference | 1024.0 |
| `y` mean absolute difference | 110.7229232788086 |
| Expected `y` nonfinite errors | 0 |
| Observed `y` nonfinite errors | 0 |
| Dropped-route difference count | 0 |
| Candidate dropped routes | 0 |
| Candidate routing-policy drops | 0 |
| Candidate metadata-overflow routes | 0 |

The forward candidate remained finite and preserved routing cardinality, but
the output error is far too large to treat the fully selected integrated path
as correct. The comparison's diagnostic timing was 35.387899 ms and is not a
target-shape performance result.

The fwd+bwd comparison failed before compilation and emitted no `dx`,
`d_route_weights`, `dw13`, or `dw2` error metrics. JAX rejected the custom VJP
because it returned `d_route_weights` with type
`float32[8@expert,128,2]` for a replicated primal input of type
`float32[8,128,2]`. Therefore:

| Gradient | Max/mean/nonfinite result |
|---|---|
| `dx` | unavailable; comparison did not compile |
| `d_route_weights` | unavailable; output sharding/type mismatch |
| `dw13` | unavailable; comparison did not compile |
| `dw2` | unavailable; comparison did not compile |

This run contradicts aggregate correctness rather than clearing the remaining
gate. Keep the FUSED-MOE-093 target-shape timing as performance evidence only;
do not promote the selected integrated path as numerically validated. Per the
bounded babysitting request, no retry or code edit was made.

## 2026-07-10 FUSED-MOE-095 - W13-backward 48/40/40 role split improves latency by 4.23%

Job `/dlwh/bench-w13b-role48-40-40-20260710-2410` at `4625779973`
completed on `cw-rno2a` with Iris state `succeeded`, exit 0, one of one task
succeeded, and a 1m37.13s task duration. The target-shape kernel used 48
staging, 40 dX, and 40 dW persistent programs while preserving K128 staging,
single-N dW ownership, and the two-stage WGMMA pipelines.

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---:|---:|---:|---:|
| 0 | 63.576811 | 54.044451 | 67.555564 |
| 1 | 63.099683 | 54.453108 | 68.066385 |
| 2 | 63.478141 | 54.128457 | 67.660571 |
| Median | **63.478141** | **54.128457** | **67.660571** |

Against the selected 64-staging/32-dX/32-dW schedule at 66.282220 ms and
51.838545 useful TFLOP/s/rank, the 48/40/40 split saves 2.804079 ms or 4.23%
and raises useful throughput by 2.289912 TFLOP/s/rank or 4.42%.

All repeats reported zero dropped routes, zero routing-policy drops, zero
metadata-overflow routes, zero queue-overflow route errors, and zero
layout-overflow row errors. The checksum was stable at `20131207168`, which is
`96256` below the selected schedule's `20131303424`; therefore this is a
performance candidate pending a finite numerical comparison, not yet a
correctness promotion. The process emitted recoverable 12.5 GiB BFC allocation
warnings and FABRIC-handle VMM fallbacks, then produced all requested rows.
Per the bounded request, no retry or kernel edit was made.

## 2026-07-11 FUSED-MOE-097 - Row-split full-K W13-forward candidate faults

Job `/dlwh/w13f-b32-fullk-d237` tested `semantic_permute_w13_pallas` at
`d237d620ea` on `cw-rno2a`. The target was EP8, T32768/rank, H2560, I1280,
E32/rank, top-k 4, capacity factor 1.25, random routing seed 0, bf16, JAX plan
metadata, one warmup, three timed steps, three repeat runs, and separate
compilation. Iris reported terminal state `succeeded`, exit 0, one of one task
succeeded, and a 1m35.71s duration, but this is a harness-level false success:
the debug-exception path returned zero after the candidate faulted.

The first actionable runtime failure was a device-wide
`CUDA_ERROR_ILLEGAL_ADDRESS` during the first candidate execution. The process
then emitted CUDA module-unload, stream-destroy, event-destroy, and allocation
cleanup errors on all eight ranks. Before the fault it also emitted two
recoverable 12.5 GiB BFC allocation warnings. No repeat row, summary row,
checksum, timing, TFLOP/s value, or drop/overflow counter was produced.

Therefore the row-split B32 full-K producer candidate is invalid at the target
shape and cannot be compared numerically with the selected 23.025404 ms /
74.612671 useful TFLOP/s/rank schedule. Retain the selected B64 K512/two-K256-
payload implementation and investigate the B32 remote-write indexing or
payload publication lifetime before reconsidering this schedule. Exactly one
H100x8 job was launched; no retry or kernel edit was made.

## 2026-07-11 FUSED-MOE-098 - Publishing W2-backward payload before route dot saves 3.98%

Job `/dlwh/w2b-publish-first-8b0f` tested commit `8b0f46e8c8` on `cw-rno2a`
with the target EP8, T32768/rank, H2560, I1280, E32/rank, top-k 4, capacity
factor 1.25, random production routing seed 0, bf16, JAX plan metadata, one
warmup, three timed steps, three repeat runs, and separate compilation. Iris
reported terminal state `succeeded`, exit 0, one of one task succeeded, and a
2m04.89s task duration.

The candidate publishes each K512 `dy_route` payload and signals
`compact_ready_sem` before computing the `hidden_tile == 0` route-gradient dot.
The route dot still completes before `helper_done_sem`; helper4 distribution,
buffers, arithmetic, and semaphore counts are unchanged.

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Checksum |
|---:|---:|---:|---:|---:|
| 0 | 38.325619 | 44.826071 | 56.032589 | 127795200 |
| 1 | 38.475174 | 44.651830 | 55.814788 | 127795200 |
| 2 | 38.268151 | 44.893387 | 56.116733 | 127795200 |
| Median | **38.325619** | **44.826071** | **56.032589** | **127795200** |

Against the selected 39.913566 ms / 43.042682 useful TFLOP/s/rank result, this
saves 1.587947 ms or 3.98% and raises useful throughput by 1.783389
TFLOP/s/rank or 4.14%. Every repeat reported zero dropped routes, zero routing-
policy drops, zero metadata-overflow routes, zero queue-overflow route errors,
and zero layout-overflow row errors. The checksum exactly matches the selected
result. The process emitted recoverable 12.5 GiB BFC allocation warnings and
FABRIC-handle VMM fallbacks, then produced all requested rows. This candidate
should replace the selected W2-backward schedule, subject to preserving the
already-established split numerical comparison coverage.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w2b-publish-first-8b0f --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w2_backward_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha 8b0f46e8c8 --jsonl scratch/w2b-publish-first-8b0f.jsonl
```

## 2026-07-11 FUSED-MOE-099 - Replicated W13-backward comparison reaches candidate lowering

Job `/dlwh/w13b-role48-compare-replicated-7c1` tested commit `7c1f9bca72`
on `cw-rno2a` with EP8, T128/rank, H2560, I1280, E4/rank, top-k 2,
capacity factor 4.0, random routing seed 0, bf16, JAX plan metadata, no warmup,
one timed step, one repeat, and separate compilation. Iris reported terminal
state `succeeded`, exit 0, one of one task succeeded, and a 43.26s duration.
This is a harness-level false success because the debug-exception path returned
zero after candidate lowering failed.

The replicated reference-input fix cleared the prior `ShardingTypeError`, but
the observed W13-backward kernel failed while lowering the `x_smem` zero-fill
at `source_push_semantic_fused_w13_backward.py:903`:

```text
NotImplementedError: WGSplatFragLayout(shape=(128, 128))
```

The failing operation was `x_smem[:, :] = jnp.zeros((128, 128), ...)`, lowered
through Mosaic GPU's swap/load-tiled path. No repeat completed, so `dx` and
`dw13` max/mean absolute differences, least-squares scales, cosine
similarities, and nonfinite counts are unavailable. Pre-run metadata reported
zero dropped routes, zero routing-policy drops, and zero metadata-overflow
routes; queue- and layout-overflow fields were not reached. Exactly one H100x8
job was launched, with no retry, source edit, duplicate, or Iris mutation.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-role48-compare-replicated-7c1 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha 7c1f9bca72 --jsonl scratch/w13b-role48-compare-replicated-7c1.jsonl
```

## 2026-07-11 FUSED-MOE-101 - W2-backward publish-first candidate is bit-exact

Job `/dlwh/w2b-publish-first-compare-d7db` tested commit `d7dbb9aa7c` on
`cw-rno2a` with EP8, T128/rank, H2560, I1280, E4/rank, top-k 2, capacity
factor 4.0, random routing seed 0, bf16, JAX plan metadata, no warmup, one
timed step, one repeat, and separate compilation. Iris reported terminal state
`succeeded`, exit 0, one of one task succeeded, and a 44.39s duration.

The separately compiled finite reference and publish-before-route-dot candidate
matched exactly:

| Output | Max abs diff | Mean abs diff | Expected nonfinite | Observed nonfinite |
|---|---:|---:|---:|---:|
| `d_z13` | 0.0 | 0.0 | 0 | 0 |
| `d_w2` | 0.0 | 0.0 | 0 | 0 |
| `d_route_weight` | 0.0 | 0.0 | 0 | 0 |

The run also reported zero validity errors, zero queue-overflow route errors,
zero layout-overflow row errors, zero dropped routes, zero routing-policy
drops, and zero metadata-overflow routes. This validates the selected
publish-before-route-dot schedule that measured 38.325619 ms and 44.826071
useful TFLOP/s/rank at the target shape in FUSED-MOE-098.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w2b-publish-first-compare-d7db --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w2_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha d7dbb9aa7c --jsonl scratch/w2b-publish-first-compare-d7db.jsonl
```

## 2026-07-11 FUSED-MOE-100 - Paired-B64 W13-backward dW candidate does not lower

Job `/dlwh/w13b-paired-k128-d7db` tested commit `d7dbb9aa7c` on `cw-rno2a`
with the target EP8, T32768/rank, H2560, I1280, E32/rank, top-k 4, capacity
factor 1.25, random production routing seed 0, bf16, JAX plan metadata, one
warmup, three timed steps, three repeat runs, and separate compilation. Iris
reported terminal state `succeeded`, exit 0, one of one task succeeded, and a
1m32.2s task duration. This is a harness-level false success because the
debug-exception path returned zero after lowering failed.

The paired-B64 candidate failed before its first execution while lowering the
B128 `x_smem` zero-fill at
`source_push_semantic_fused_w13_backward.py:903`:

```text
NotImplementedError: WGSplatFragLayout(shape=(128, 128))
```

The operation was `x_smem[:, :] = jnp.zeros((128, 128), ...)`, lowered through
Mosaic GPU's swap/load-tiled path. No repeat completed, so there is no timing,
useful or rounded TFLOP/s, or output checksum to compare with the current
48/40/40 candidate at 63.478141 ms / 54.128457 useful TFLOP/s/rank. Pre-run
metadata reported zero dropped routes, zero routing-policy drops, and zero
metadata-overflow routes. Queue- and layout-overflow counters were not reached.
The process also emitted two recoverable 12.5 GiB BFC allocation warnings.
Exactly one H100x8 job was launched; no retry, source edit, duplicate, or Iris
mutation was made.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-paired-k128-d7db --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha d7dbb9aa7c --jsonl scratch/w13b-paired-k128-d7db.jsonl
```

## 2026-07-11 FUSED-MOE-102 - Paired-B64 W13-backward comparison reaches runtime and faults

Job `/dlwh/w13b-paired-compare-055` tested commit `055159bed85c` on
`cw-rno2a` with EP8, T128/rank, H2560, I1280, E4/rank, top-k 2, capacity
factor 4.0, random routing seed 0, bf16, JAX plan metadata, no warmup, one
timed step, one repeat, and separate compilation. Iris reported terminal state
`succeeded`, exit 0, one of one task succeeded, and a 44.02s task duration.
This is a harness-level false success because the debug-exception path returned
zero after the candidate faulted.

The replicated reference and WGMMA-layout-safe B128 zero-fill both passed their
previous failure points. The observed paired-B64/K128 candidate then failed
when `jax.device_get` synchronized its outputs:

```text
jax.errors.JaxRuntimeError: INTERNAL: CUDA error: :
CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
```

No repeat completed, so `dx` and `dw13` max/mean absolute differences,
least-squares scales, cosine similarities, and nonfinite counts are
unavailable. Pre-run metadata reported zero dropped routes, zero
routing-policy drops, and zero metadata-overflow routes. Queue- and
layout-overflow fields were not reached. This single reduced retry therefore
does not validate the paired-B64 candidate; the runtime memory access remains
the first actionable failure. Exactly one H100x8 job was launched, with no
retry, source edit, duplicate, or Iris mutation.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-paired-compare-055 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha 055159bed85c --jsonl scratch/w13b-paired-compare-055.jsonl
```

## 2026-07-11 FUSED-MOE-103 - Paired-B64 W13-backward target timing retry

Job `/dlwh/w13b-paired-k128-zero-layout-055` tested commit `055159bed85c`
on `cw-rno2a` with the target EP8, T32768/rank, H2560, I1280, E32/rank,
top-k 4, capacity factor 1.25, random production routing seed 0, bf16, JAX
plan metadata, one warmup, three timed steps, three repeat runs, and separate
compilation. Iris reached terminal state `succeeded`: exit 0, zero failures,
zero preemptions, one of one task succeeded, and a 1m40.33s task duration.
Exactly one H100x8 job was launched; there was no retry, duplicate, source
edit, stop, resubmit, or Iris mutation.

The WGMMA-layout-safe paired-B64/K128 candidate completed all repeats:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank | Checksum |
|---:|---:|---:|---:|---:|
| 0 | 64.869818 | 52.967219 | 66.209023 | 20131459072 |
| 1 | 64.620335 | 53.171712 | 66.464640 | 20131459072 |
| 2 | 65.033155 | 52.834186 | 66.042733 | 20131459072 |

The median was 64.869818 ms, 52.967219 useful TFLOP/s/rank, and 66.209023
rounded TFLOP/s/rank; the min/max times were 64.620335/65.033155 ms. Relative
to the current 48/40/40 result at 63.478141 ms and 54.128457 useful
TFLOP/s/rank, pairing adjacent B64 reduction rows is 2.19% slower in time and
2.15% lower in useful throughput. Dropped routes, routing-policy drops,
metadata-overflow routes, queue-overflow route errors, and layout-overflow row
errors were all zero. The run emitted two recoverable 12.50 GiB BFC allocation
warnings before successful compilation and execution.

This is a timing result only. FUSED-MOE-102's reduced direct comparison reached
the candidate but faulted with `CUDA_ERROR_ILLEGAL_ADDRESS`, so the paired-B64
candidate remains unvalidated and should not replace the current 48/40/40
path.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-paired-k128-zero-layout-055 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha 055159bed8 --jsonl scratch/w13b-paired-k128-zero-layout-055.jsonl
```

## 2026-07-11 FUSED-MOE-104 - Restored 48/40/40 W13-backward comparison faults

Job `/dlwh/w13b-role48-selected-compare-2f04` tested the restored selected
48-stager/40-dX/40-dW schedule at commit `2f04c25dd3` on `cw-rno2a` with EP8,
T128/rank, H2560, I1280, E4/rank, top-k 2, capacity factor 4.0, random routing
seed 0, bf16, JAX plan metadata, no warmup, one timed step, one repeat, and
separate compilation. Iris reported terminal state `succeeded`, exit 0, one of
one task succeeded, and a 49.27s task duration. This is a harness-level false
success because the debug-exception path returned zero after device execution
faulted.

The restored selected candidate reached device execution and failed with:

```text
CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
```

No comparison row completed, so `dx` and `dw13` max/mean absolute differences,
least-squares scales, cosine similarities, nonfinite counts, and queue/layout
overflow counters are unavailable. The fault also prevented buffered pre-run
drop and metadata counters from being emitted. This single requested run does
not validate the selected 48/40/40 schedule. Exactly one H100x8 job was
launched; there was no retry, source edit, duplicate, stop, resubmit, or Iris
mutation. Concurrent W2-backward edits were left untouched.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-role48-selected-compare-2f04 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha 2f04c25dd3 --jsonl scratch/w13b-role48-selected-compare-2f04.jsonl
```

## 2026-07-11 FUSED-MOE-105 - W2-backward route-gradient-on-owner comparison faults

Job `/dlwh/w2b-route-owner-compare-2f75` tested commit `2f75259cb2` on
`cw-rno2a` with EP8, T128/rank, H2560, I1280, E4/rank, top-k 2, capacity
factor 4.0, random routing seed 0, bf16, JAX plan metadata, no warmup, one
timed step, one repeat, and separate compilation. Iris reached terminal state
`succeeded`: exit 0, zero failures, zero preemptions, one of one task
succeeded, and a 44.18s task duration. This is a harness-level false success
because the debug-exception path returned zero after device execution faulted.

The separately compiled reference completed, but the route-gradient-on-owner
candidate failed while `jax.device_get` materialized the observed outputs:

```text
CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
```

No comparison row completed. Therefore `d_z13`, `d_w2`, and
`d_route_weight` max/mean absolute differences and nonfinite counts are
unavailable, as are validity, queue-overflow, and layout-overflow metrics. The
error row reported zero dropped routes, zero routing-policy drops, and zero
metadata-overflow routes. This run does not validate the owner-route-gradient
schedule. Exactly one H100x8 job was launched; there was no retry, source edit,
duplicate, stop, resubmit, or Iris mutation.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w2b-route-owner-compare-2f75 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w2_backward_compare --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha 2f75259cb2 --jsonl scratch/w2b-route-owner-compare-2f75.jsonl
```

## 2026-07-11 FUSED-MOE-106 - W2-backward route-gradient-on-owner target timing faults

Job `/dlwh/w2b-route-owner-2f75` tested commit `2f75259cb2` on
`cw-rno2a` with the target EP8, T32768/rank, H2560, I1280, E32/rank,
top-k 4, capacity factor 1.25, random production routing seed 0, bf16, JAX
plan metadata, one warmup, three timed steps, three repeat runs, and separate
compilation. Iris reached terminal state `succeeded`: exit 0, zero failures,
zero preemptions, one of one task succeeded, and a 1m42.92s task duration.
This is a harness-level false success because device execution faulted.

The route-gradient-on-owner candidate failed with:

```text
CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
```

No repeat completed, so timing, useful or rounded TFLOP/s/rank, checksum, and
drop/metadata/queue/layout counters are unavailable. The candidate therefore
cannot be compared numerically with or replace the selected publish-first
schedule at 38.325619 ms and 44.826071 useful TFLOP/s/rank. Together with
FUSED-MOE-105's reduced comparison fault, this target failure rejects the
route-gradient-on-owner schedule in its current form. The run emitted two
12.50 GiB BFC allocation warnings before the device fault. Exactly one H100x8
job was launched; there was no retry, duplicate, source edit, stop, resubmit,
or Iris mutation.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w2b-route-owner-2f75 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --capacity-factor 1.25 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w2_backward_pallas --warmup 1 --steps 3 --repeat-runs 3 --separate-compile --debug-exceptions --git-sha 2f75259cb2 --jsonl scratch/w2b-route-owner-2f75.jsonl
```

## 2026-07-11 FUSED-MOE-107 - Restored W13 backward faults standalone at reduced shape

Job `/dlwh/w13b-role48-standalone-reduced-377` tested the restored selected
48-stager/40-dX/40-dW W13-backward kernel at commit `37729dc653` on
`cw-rno2a`. The run used EP8, T128/rank, H2560, I1280, E4/rank, top-k 2,
capacity factor 4.0, random routing seed 0, bf16, JAX plan metadata, no warmup,
one timed step, one repeat, and separate compilation. Iris reported terminal
state `succeeded`, exit 0, one of one task succeeded, and a 44.99s task
duration. This is a harness-level false success because the benchmark's
debug-exception path returned zero after device execution faulted.

The standalone `semantic_fused_w13_backward_pallas` mode failed with:

```text
CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
```

No repeat row completed. Timing, useful and rounded TFLOP/s/rank, output
checksum, and drop/metadata/queue/layout counters are therefore unavailable.
This proves the reduced-shape fault is in the selected candidate itself rather
than an interaction with the split comparison harness. Exactly one H100x8 job
was launched; there was no retry, duplicate, source edit, stop, resubmit, or
Iris mutation.

Exact launch command:

```bash
uv run --project /Users/dlwh/src/marin iris --cluster=cw-rno2a job run --no-wait --job-name w13b-role48-standalone-reduced-377 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu --sync-package marin-levanter --timeout 3600 -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/bench_source_push_semantic_plan.py --ep-size 8 --tokens-per-rank 128 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 4 --topk 2 --capacity-factor 4.0 --rows-per-src-dst-capacity auto --routing random --routing-seed 0 --dtype bfloat16 --plan-builder jax --modes semantic_fused_w13_backward_pallas --warmup 0 --steps 1 --repeat-runs 1 --separate-compile --debug-exceptions --git-sha 37729dc653 --jsonl scratch/w13b-role48-standalone-reduced-377.jsonl
```

## 2026-07-11 FUSED-MOE-108 - Isolated W13 backward cohorts succeed

Commit `c7585f7e2f` added package-private compile-time role diagnostics without
changing the production full path. Focused tests passed (`13 passed`) and the
four-file scoped pre-commit check passed. The branch-wide changed-file check
was otherwise clean but retained the known oversized historical
`.agents/logbooks/6597-moe-mgpu-forward.md` finding.

Job `/dlwh/w13b-role-diagnostics-c7585f7e2f` ran all three diagnostics on
`cw-rno2a` with EP8, T128/rank, H2560, I1280, E4/rank, top-k 2, capacity
factor 4.0, random routing seed 0, bf16, JAX metadata, no warmup, one timed
step, and separate compilation. Iris and its only task succeeded in 48.86s.

| Resident roles | Time (ms) | Output checksum |
|---|---:|---:|
| staging only, 48 CTAs | 3.187271 | 176914.390625 |
| staging + dX + source combine, 88 CTAs | 4.452068 | 19767040.0 |
| staging + dW, 88 CTAs | 3.982255 | 26391238.0 |

Every mode reported zero dropped routes, routing-policy drops, metadata
overflows, queue overflows, and layout-overflow rows. Both isolated compute
legs therefore survive the reduced geometry that faults in the 128-CTA full
kernel. Sparse dW blocks alone are not sufficient to reproduce the fault, and
dX plus source combine also succeeds. The next isolation is a 128-CTA
staging+dX+dW mode with source combine compiled out; this distinguishes a
dX/dW coexistence failure from a dW/source-combine interaction.

## 2026-07-11 FUSED-MOE-109 - Full compute cohorts pass without source combine

Job `/dlwh/w13b-dx-dw-no-combine-219c` tested commit `219c72568f` on
`cw-rno2a` at the same reduced EP8, T128/rank, H2560, I1280, E4/rank,
top-k 2, capacity-factor 4.0 random-routing geometry. The diagnostic retained
all 48 staging, 40 dX, and 40 dW CTAs, including dX return writes and normal
publication semaphore signals, but compiled source-owned dX combine out. Iris
and its only task succeeded with exit 0 in 76.92s.

The kernel took `4.836912 ms`, reported `1.387432` useful and `5.549728`
rounded TFLOP/s/rank under the diagnostic accounting, and produced checksum
`26391238.0`. Dropped routes, routing-policy drops, metadata overflows, queue
overflows, and layout-overflow rows were all zero.

Thus 128-CTA dX/dW coexistence is valid, as are the complete semaphore set and
dX publication protocol when no consumer reads the return queue. The reduced
illegal address requires source combine to execute concurrently with dW. The
next experiment preserves the full path but gates combine on completion of all
40 dW owners. A passing serialized-combine variant would identify an overlap
race; a failure would instead implicate the combine/dW memory footprints even
without temporal overlap.

## 2026-07-11 FUSED-MOE-110 - Serializing source combine removes the fault

Job `/dlwh/w13b-full-serial-combine-c992` tested commit `c9920db6f6` at the
same reduced geometry. It retained all full-path staging, dX return,
publication, dW, and source-combine operations, but made each staging/combine
CTA wait for completion signals from all 40 dW owners before reading the dX
return queue. Iris and its only task succeeded with exit 0 in 41.78s.

The kernel took `5.056301 ms`, reported `1.327232` useful and `5.308929`
rounded TFLOP/s/rank, and produced checksum `39498520.0`. All drop,
metadata-overflow, queue-overflow, and layout-overflow counters were zero. The
increment over staging+dX+dW without combine was only `0.219389 ms` at this
reduced geometry.

Together with FUSED-MOE-108 and FUSED-MOE-109, this identifies a temporal
overlap defect: each constituent cohort succeeds, all 128 compute CTAs coexist
successfully, and the full memory footprints succeed when combine follows dW.
The production `FULL` role profile now uses this ordering as a correctness
baseline. The next gates are the existing reduced finite split comparison and
then target-shape timing. Restoring overlap requires a separate handoff design;
the previously concurrent path is not safe merely because it happens to finish
at the target geometry.

## 2026-07-11 FUSED-MOE-111 - Production output contract still faults

Job `/dlwh/w13b-serial-full-compare-e50e` tested the production `FULL` role
profile at commit `e50e88d828`, with the dW-completion gate enabled, against
the separately compiled finite JAX reference at the reduced geometry. Iris and
its only task succeeded at the orchestration level in 43.21s, but the observed
kernel raised `CUDA_ERROR_ILLEGAL_ADDRESS` during `jax.device_get`. No dX or
dW comparison metrics were produced. Pre-run dropped-route, routing-drop, and
metadata-overflow counters were zero.

The successful `FULL_SERIAL_COMBINE` diagnostic and failing production `FULL`
kernel have the same 128 roles, operations, semaphore set, and ordering. Their
material difference is the `shard_map` output contract: the diagnostic exports
`x_expert`, dX, and dW, while production exported only dX and dW. The next
candidate adds a value-preserving scalar dependency from the remote-written
`x_expert` scratch output to dX, preventing XLA from treating that custom-call
output as dead or reusable without materializing a dense checksum.

## 2026-07-11 FUSED-MOE-112 - Scratch-liveness fix is bit-exact

Job `/dlwh/w13b-scratch-live-compare-b362` tested commit `b36296e1b7` on
`cw-rno2a` at the reduced finite-comparison geometry. The production `FULL`
path retained serialized source combine and added a scalar, value-preserving
dependency from `x_expert` to dX after the custom call. This keeps the
remote-written scratch output live through the `shard_map` boundary without a
dense checksum or API return. Iris and its only task succeeded with exit 0 in
55.29s.

The separately compiled reference and Pallas result matched exactly:

| Output | Max abs diff | Mean abs diff | Scale | Cosine | Nonfinite |
|---|---:|---:|---:|---:|---:|
| dX | 0.0 | 0.0 | 1.0 | 1.0 | 0 / 0 |
| dW13 | 0.0 | 0.0 | 1.0 | 1.0 | 0 / 0 |

Dropped routes, routing-policy drops, metadata overflows, queue overflows, and
layout-overflow rows were all zero. This proves the reduced illegal address was
caused by the production output-liveness contract rather than WGMMA arithmetic
or route metadata. The target-shape timing run is the next gate; restoring
combine/dW overlap remains separate work after the correct baseline is measured.

## 2026-07-11 FUSED-MOE-113 - Serialized target baseline is correct but slow

Job `/dlwh/w13b-scratch-live-target-b362` tested commit `b36296e1b7` on the
target EP8, T32768/rank, H2560, I1280, E32/rank, top-k 4,
capacity-factor 1.25 random-routing geometry. Iris and its only task succeeded
with exit 0 in 104.3s. All three repeats completed with a stable checksum of
`20131794944` and zero drop, metadata-overflow, queue-overflow, and
layout-overflow counters.

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---:|---:|---:|---:|
| 0 | 83.343990 | 41.226414 | 51.533017 |
| 1 | 84.130013 | 40.841237 | 51.051547 |
| 2 | 83.379320 | 41.208945 | 51.511182 |

The median was `83.379320 ms`, `41.208945` useful and `51.511182` rounded
TFLOP/s/rank, with a min/max of `83.343990/84.130013 ms`. This is 31.4% slower
than the earlier unvalidated concurrent target timing of `63.478141 ms`.

The prior isolation confounded ordering with output liveness: every successful
diagnostic exported `x_expert`, while the failing production path discarded it.
The next candidate therefore restores concurrent combine/dW execution while
retaining only the scalar `x_expert` liveness dependency that made the reduced
finite comparison bit-exact. If it passes, serialization was unnecessary and
the original overlap schedule can be recovered safely.

## 2026-07-11 FUSED-MOE-114 - Liveness alone does not fix overlap

Job `/dlwh/w13b-overlap-live-compare-8cb9` tested commit `8cb9d06989` at the
reduced finite-comparison geometry after restoring concurrent source combine
and dW while retaining the `x_expert` liveness dependency. Iris and its only
task succeeded at the orchestration level in 44.42s, but the observed kernel
again raised `CUDA_ERROR_ILLEGAL_ADDRESS` during `jax.device_get`. No dX or
dW comparison metrics were produced. Pre-run dropped-route, routing-drop, and
metadata-overflow counters were zero.

Therefore scratch liveness and combine/dW ordering are independent
requirements in the current role assignment. The next candidate preserves
overlap but moves source combine from the 48 staging CTAs to the 40 dX CTAs
after they publish all return chunks. Staging CTAs then perform only remote
`x_expert` writes and exit, removing the unsafe staging-write-to-combine role
transition while dW remains concurrent.

## 2026-07-11 FUSED-MOE-115 - dX-worker combine is bit-exact

Job `/dlwh/w13b-dx-worker-combine-compare-a0bb` tested commit `a0bba573e8`
on `cw-rno2a` at the reduced finite-comparison geometry. Staging CTAs only
remote-write `x_expert`; after all dX work and publication, the 40 dX CTAs
divide the source-combine jobs while the 40 dW owners remain concurrent. Iris
and its only task succeeded with exit 0 in 54.39s.

The separately compiled reference and candidate matched exactly:

| Output | Max abs diff | Mean abs diff | Scale | Cosine | Nonfinite |
|---|---:|---:|---:|---:|---:|
| dX | 0.0 | 0.0 | 1.0 | 1.0 | 0 / 0 |
| dW13 | 0.0 | 0.0 | 1.0 | 1.0 | 0 / 0 |

Dropped routes, routing-policy drops, metadata overflows, queue overflows, and
layout-overflow rows were all zero. This validates a safe overlap structure
without the 20 ms target serialization tax. Target timing is the next gate.

The target follow-up `/dlwh/w13b-dx-worker-combine-target-a0bb` reached Iris
state `succeeded` with exit 0 in 98.79s, but the benchmark raised
`CUDA_ERROR_ILLEGAL_ADDRESS` before any repeat. Pre-run dropped-route,
routing-drop, and metadata-overflow counters were zero; timing and numerical
rows were unavailable. The dX-worker role reuse is therefore reduced-correct
but target-unsafe and is rejected. Production `FULL` returns to the serialized,
bit-exact baseline while a separate post-kernel source combine is evaluated.

## 2026-07-11 FUSED-MOE-116 - No-combine target establishes two-kernel floor

Job `/dlwh/w13b-no-combine-target-a0bb` tested the 128-CTA staging+dX+dW
kernel at commit `a0bba573e8` with source combine compiled out. The target
geometry and routing matched the other target W13-backward runs. Iris and its
only task succeeded with exit 0; every repeat produced checksum `13505684480`
and zero drop, metadata-overflow, queue-overflow, and layout-overflow counters.

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---:|---:|---:|---:|
| 0 | 62.464365 | 55.006944 | 68.758680 |
| 1 | 62.032635 | 55.389778 | 69.237222 |
| 2 | 62.170948 | 55.266551 | 69.083188 |

The median was `62.170948 ms`, with min/max `62.032635/62.464365 ms`. This is
1.307 ms faster than the earlier unvalidated concurrent full timing and 21.208
ms faster than the correct in-kernel serialized baseline. The production
replacement now makes this queue explicit and launches a source-local Pallas
combine kernel after the fused compute/transport kernel completes.

## 2026-07-11 FUSED-MOE-117 - First standalone Pallas combine faults

Job `/dlwh/w13b-split-combine-compare-e716` tested commit `e716de715c` at the
reduced finite-comparison geometry. The first kernel returned the source-owned
dX queue and dW, then a two-dimensional Pallas grid gathered top-k queue rows
into dX. Iris and its only task succeeded at the orchestration level in 46.93s,
but the observed sequence raised `CUDA_ERROR_ILLEGAL_ADDRESS` during
`jax.device_get`. No dX or dW numerical metrics were produced; pre-run drop and
metadata-overflow counters were zero.

The next isolation keeps the same explicit queue-producing kernel but replaces
the standalone Pallas gather with a source-sharded JAX top-k gather/reduce. A
passing JAX combine proves the queue contract and confines the defect to the
new Pallas combine lowering rather than the first kernel.

## 2026-07-11 FUSED-MOE-118 - JAX combine validates queue but costs 25 ms

Job `/dlwh/w13b-jax-combine-compare-ade6` tested commit `ade668f16a` at the
reduced finite-comparison geometry. The explicit returned queue followed by a
source-sharded JAX top-k gather/reduce matched exactly: dX and dW13 both had
zero max/mean error, scale and cosine 1.0, no nonfinite values, and zero drop or
overflow counters.

Target job `/dlwh/w13b-jax-combine-target-ade6` also succeeded with stable
checksum `20131827712` and all counters zero:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---:|---:|---:|---:|
| 0 | 87.499361 | 39.268559 | 49.085699 |
| 1 | 87.112243 | 39.443065 | 49.303831 |
| 2 | 86.729823 | 39.616982 | 49.521228 |

The median was `87.112243 ms`, so JAX combine adds about `24.94 ms` over the
`62.170948 ms` queue producer. It is a correctness/isolation path, not a
performance candidate. The queue and first kernel are sound; the standalone
Pallas combine is now aligned with the proven source-combine kernel's 64-row
token block instead of the failed 256-row Lane-lowered loop.

## 2026-07-11 FUSED-MOE-119 - Pallas64 combine is exact and fast

Reduced job `/dlwh/w13b-pallas64-combine-compare-3884` tested commit
`388408ee81` after changing the standalone Lane-lowered source combine from a
256-row token loop to the proven 64-row shape. The separately compiled result
was bit-exact for dX and dW13: zero max/mean error, scale and cosine 1.0, no
nonfinite values, and zero drop or overflow counters.

Target job `/dlwh/w13b-pallas64-combine-target-3884` also succeeded with
stable checksum `19830183936` and all counters zero:

| Repeat | Time (ms) | Useful TFLOP/s/rank | Rounded TFLOP/s/rank |
|---:|---:|---:|---:|
| 0 | 63.123062 | 54.432940 | 68.041175 |
| 1 | 62.697307 | 54.802574 | 68.503218 |
| 2 | 62.420632 | 55.045483 | 68.806853 |

The median was `62.697307 ms`, with min/max `62.420632/63.123062 ms`. The
standalone Pallas combine adds only `0.526359 ms` over the `62.170948 ms`
no-combine kernel. This correct path is `20.682013 ms` faster than the
serialized in-kernel baseline and `0.780834 ms` faster than the previous
unvalidated 48/40/40 target timing. It is the selected W13-backward path.

## 2026-07-11 FUSED-MOE-120 - Integrated graph still faults after W13 fix

Job `/dlwh/mlp-fwd-bwd-split-w13-compare-d2a4` tested commit `d2a421fc55` at
the reduced integrated forward+backward comparison geometry. Iris and its only
task succeeded at the orchestration level, but the observed monolithic graph
raised `CUDA_ERROR_ILLEGAL_ADDRESS` during `jax.device_get`. No y, dX,
weight-gradient, or route-gradient metrics were produced, and the first failing
kernel could not be identified from asynchronous cleanup logs.

Forward, W2 backward, and the selected W13 backward are independently
validated. A benchmark-only `semantic_fused_mlp_stop_after_w2_backward_pallas`
mode now composes fused W13 forward, fused W2 return/combine, and fused W2
backward while returning every intermediate. This isolates whether the
remaining liveness fault occurs before W13 backward or specifically at the
W2-to-W13 boundary.

## 2026-07-11 FUSED-MOE-121 - Add an all-stages-live composition boundary

The next benchmark-only isolation mode,
`semantic_fused_mlp_all_stages_live_pallas`, composes fused W13 forward, fused
W2 return/combine, fused W2 backward, and the selected split W13 backward while
returning every forward and W2-backward intermediate. This distinguishes a
W2-to-W13 launch-boundary failure from custom-VJP output liveness: if the
stop-after-W2 mode passes but this mode faults, the failure is introduced by
the W13-backward launch; if both pass while the custom VJP faults, the custom
VJP graph is dropping or reusing a required buffer.

Local validation:

- `python -m py_compile lib/levanter/scripts/bench/bench_source_push_semantic_plan.py`
- `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_source_push_semantic_plan_bench.py -k 'fused_mlp_modes_use_target_shape_flops or bench_emits_full_fused_mlp_rows'`
  passed (`2 passed`, 11 warnings).
- `git diff --check` passed.
- Scoped `./infra/pre-commit.py` passed.

The reduced stop-after-W2 H100 run is delegated and active on `cw-rno2a`; no
duplicate job was launched. In parallel, a separate worker is adapting the
semantic metadata to the old B64 inbox physical protocol so forward W13 can
recover independent B64 readiness and the proven lightweight lifecycle.
