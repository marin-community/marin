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
