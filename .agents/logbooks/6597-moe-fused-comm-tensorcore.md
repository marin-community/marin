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
