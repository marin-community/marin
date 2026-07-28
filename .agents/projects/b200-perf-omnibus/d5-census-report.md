# D-5 collective-overlap schedule census at d6144

## TL;DR

The d6144 local-expert shape produced 12 MoE all-to-alls at every overlap
limit, matching the structural prediction. The SYNC census was `4, 0, 0, 0`
at limits `1, 2, 4, 8`. Limit 4 clears every MoE all-to-all at this shape.
The pre-registered exact census `3, 0, 0, 1` was falsified at limits 1 and 8,
and the earlier EP4 regressions at limits 2 and 8 did not reproduce.

## Pre-registration

Recorded before producing or reading the d6144 schedule dumps.
This text was carried forward verbatim after the first submission was cancelled,
before any d6144 result was seen.

The d6144 4-of-128 shape has two local experts per device at EP64. The same
dispatch code should therefore emit 2 forward-dispatch, 2 forward-combine,
4 backward-dispatch, and 4 backward-combine all-to-alls: 12 distinct MoE
all-to-all operations at every overlap limit. This is the structural count
reported independently for the d6144 profile in
[#7279 comment 5095217108](https://github.com/marin-community/marin/issues/7279#issuecomment-5095217108).

I predict the following SYNC all-to-all census:

| overlap limit | predicted MoE SYNC all-to-alls |
|---:|---:|
| 1 | 3 |
| 2 | 0 |
| 4 | 0 |
| 8 | 1 |

The limit-1 prediction uses the three of twelve d6144 operations observed on
the compute stream. With half as many local experts as the earlier EP4 harness,
I expect a budget of two to cover all twelve operations. Limit 4 should also
clear them. I expect the limit-8 heuristic regression from the EP4 census to
persist.

Any total other than 12 distinct MoE all-to-alls falsifies the structural
prediction. Any SYNC count different from the table falsifies the exact census
prediction. A nonzero count at limit 4 separately falsifies the operational
claim that limit 4 clears this class at d6144.

## Harness

Job: `/mwittmann/d5-census-d6144-v2-20260728`

The harness uses one 4×GB200 node on `cw-us-east-08a`. It preserves the
schedule-relevant d6144 configuration with a reduced EP4 mesh: d6144/i3072,
top-4, 8 routed experts over an expert axis of 4, and therefore two local
experts per device. Four scanned layers preserve the single compiled layer
schedule while avoiding a 48-layer compile. Batch 64 at sequence length 4096
gives each of the four devices the same 65,536 tokens as the 64-device,
batch-1024 reference leg.

All four subprocesses use `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, the
latency-hiding scheduler, and
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`.
Only `--xla_gpu_experimental_parallel_collective_overlap_limit` and the dump
directory change between subprocesses.

Exact submission:

```bash
IRIS_USER=mwittmann .venv/bin/iris --cluster=cw-us-east-08a job run --no-wait \
  --priority production \
  --enable-extra-resources --gpu GB200x4 --extra gpu \
  --cpu 32 --memory 256GB --disk 256GB \
  --job-name d5-census-d6144-v2-20260728 \
  -e RUN_ID d5-census-d6144-v2-20260728 \
  -e GRUG_RUN_INLINE 1 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 \
  -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 \
  -e SCALE_GPU_TYPE GB200 \
  -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 \
  -e SCALE_NUM_EXPERTS 8 \
  -e SCALE_TOP_K 4 \
  -e SCALE_HIDDEN_DIM 6144 \
  -e SCALE_NUM_LAYERS 4 \
  -e SCALE_SEQ_LEN 4096 \
  -e SCALE_BATCH 64 \
  -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 2 \
  -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_A2A_FIXED 1 \
  -e SCALE_A2A_CHUNKS 1 \
  -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_A2A_CUSTOM_ADJOINT 1 \
  -e SCALE_MOE_QB 1 \
  -e SCALE_OPTIMIZER muonh \
  -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 \
  -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER d5-census-d6144-v2-20260728.metrics \
  -e SCALE_REPORT_DROPS 1 \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- bash -c '
set -euo pipefail
for limit in 1 2 4 8; do
  dump="/tmp/d5-census-limit-${limit}"
  echo "D5_CENSUS_LIMIT_BEGIN limit=${limit} dump=${dump}"
  env \
    RUN_ID="d5-census-d6144-limit${limit}-20260728" \
    XLA_FLAGS="--xla_dump_to=${dump} \
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_experimental_parallel_collective_overlap_limit=${limit} \
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false" \
    /app/.venv/bin/python -m experiments.grug.moe.launch_cw_scale \
      --version d5-census-dev --run
  echo "D5_CENSUS_REPORT_BEGIN limit=${limit}"
  /app/.venv/bin/python -m experiments.grug.moe.schedule_report "${dump}"
  echo "D5_CENSUS_LIMIT_END limit=${limit}"
done'
```

The recorded submit arguments for both d6144 reference legs,
`/mwittmann/ep25d5-d6144-e128-bf16-120-0726-1140-v3` and
`/mwittmann/ep25d6-d6144-e128-dense-120-0726-1440`, contain no `XLA_FLAGS`
environment entry. Those legs therefore almost certainly compiled with the
default overlap limit of 1.

The report uses only the MoE dispatch and combine rows from
`schedule_report.py`. The earlier `reshard SYNC` column came from the reduced
four-shard mesh and does not exist on the EP64 rack mesh. Cover counts are
interpreted using the parser after commit `54809714c`; earlier cover output is
invalid.

## Result

The job succeeded with no failures or preemptions in 28 minutes 30 seconds. Each
subprocess completed two training steps with finite loss. Step-1 loss ranged
from 14.9898 to 14.9911 and step-1 drop fraction ranged from 0.0690 to 0.0712.

| overlap limit | predicted MoE SYNC | observed MoE operations | observed MoE SYNC |
|---:|---:|---:|---:|
| 1 | 3 | 12 | 4 |
| 2 | 0 | 12 | 0 |
| 4 | 0 | 12 | 0 |
| 8 | 1 | 12 | 0 |

The 12-operation structural prediction held at every limit: 2 forward dispatch,
2 forward combine, 4 backward dispatch, and 4 backward combine. At limit 1,
the four synchronous operations were one forward dispatch, two backward
dispatch, and one backward combine. Forward combine was fully asynchronous, as
expected.

The exact census prediction was falsified at limits 1 and 8. Limit 1 left four
operations synchronous instead of three, while limit 8 left zero instead of
one. Limits 2 and 4 matched the prediction and cleared all 12 operations. The
limit-2 count improved from 4 to 0 relative to limit 1, so the earlier ECHO
limit-2 regression did not reproduce in this census. The EP4 limit-8 regression
also did not reproduce. Limit 4 still satisfies the operational requirement
without relying on the higher setting.

## Scope

This census closes the unexplained three-of-twelve inline-collective
observation. It does not measure a throughput gain and does not make the
compute-stream collective duration recoverable one-for-one. At d5120, moving
2,961 ms per three steps off the compute stream reduced net exposed collective
time by only 463 ms because the same GEMMs then hid 40% more asynchronous
collective time. MFU improved by 0.12 percentage points with matched step-119
drop fractions of 0.0876 and 0.0882. Applying that measured conversion ratio to
d6144 suggests roughly 0.1 percentage points, not 2.8%. The D-5 dump measures
schedule structure only; it does not establish that throughput gain at d6144.
