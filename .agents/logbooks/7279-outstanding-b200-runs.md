---
topic: moe-hero-ep
issue: https://github.com/marin-community/marin/issues/7279
description: Outstanding GB200 (A08) runs still queued or in flight, with their exact job ids.
author: rav
---

# Outstanding B200 runs — check these

Snapshot at 2026-08-05 05:50 UTC. All four are queued on `cw-us-east-08a` and had not produced a
result yet. They retry on their own, so they can land at any time. W&B project `rav_moe`, entity
`marin-community`.

| run | job id | what it answers | state at snapshot |
| --- | --- | --- | --- |
| MHEP-021 | `/rav/mhep-021-wide-591b-e128-i5120-k4-p32579-20260805-coord` | 128 x i5120 top-4, 590.7 B total, 29.0 B active, 20,480 active routed neurons. Does width buy active compute cheaply at a fixed parameter budget? | building, failures=1, preemptions=4 |
| MHEP-022 | `/rav/mhep-022-fine-591b-e512-i1280-k4-p32580-20260805-coord` | 512 x i1280 top-4, 590.7 B total, 15.4 B active, 8 experts per device. The fine-grained end of the granularity axis. | building, failures=1, preemptions=3 |
| MHEP-023 | `/rav/mhep-023-xprof-10-p32581-20260805-coord` | XProf trace of steps 5 and 6 on rank 0, with the HLO proto, at the hero shape. | building, preemptions=2 |
| MHEP-024 | `/rav/mhep-024-nsys-10-p32582-20260805-coord` | Nsight Systems capture on task 0 (`IRIS_NSYS_TASKS=first`) at the hero shape. | building, preemptions=2 |

The GPU child of each job is `<coord>/grug-train-<run-id>`.

## How to collect a result

```bash
uv run iris --config lib/iris/config/marin.yaml job summary <job-id>
uv run iris --config lib/iris/config/marin.yaml job summary <job-id>/grug-train-<run-id>
```

Metrics land in `s3://marin-us-east-02a/marin/grug/<run-id>/2026.08.05/tracker_metrics.jsonl`. That
file is one line with `config` and `summary` keys, and `summary` holds `throughput/p50_mfu`,
`throughput/tokens_per_second`, `train/loss`, and `moe/drop_fraction`, with MFU already in percent.
Read it from outside the cluster with the `iris-task-env` credentials, the `https://cwobject.com`
endpoint, and virtual-host addressing.

## Known risks for these four

- MHEP-021 and MHEP-022 each carry one real failure. Their per-device memory estimates are 23.00 GiB
  and 19.25 GiB of top-k-scaled buffers, against 20.50 GiB for a configuration that passed and
  30.75 GiB for one that failed, so an out-of-memory failure is a plausible result rather than an
  infrastructure fault. Check the logs for `ncclAlltoAll` before assuming a retry will fix it.
- MHEP-023 carries a known risk: earlier work found that the JAX profiler deadlocked EP64 execution
  at step 0 across about 15 runs. That work used the custom-transport EP lane, not this pure-XLA
  `fixed_all_to_all` hero, so it may not repeat. If the run hangs before step 5, that is the known
  failure, and the recorded workaround was `--xla_gpu_enable_command_buffer=CUSTOM_CALL` rather than
  the fully disabled setting this hero uses.
- The A08 cluster was heavily contended through this session, with 12-rack runs cycling and evicting
  16-node gangs repeatedly. Preemption counts on these jobs are expected to keep climbing.
