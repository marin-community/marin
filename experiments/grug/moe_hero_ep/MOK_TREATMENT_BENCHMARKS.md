# MoK Expert Placement and PGLE Benchmarks

This record compares two scheduling treatments for the dropless Mixture-of-Kittens (MoK) backend
on one GB200 NVL72 rack:

1. Relabel routed experts so each two-expert device shard pairs a hot expert with a cold expert.
2. Replay one shared profile-guided latency estimator (PGLE) profile on every JAX process.

The treatments were measured separately and together. They preserve the training graph used by the
contiguous MoK baseline: model shape, router, two fused shared experts, optimizer and host-offloaded
Muon state, batch schedule, seed, process topology, and MoK tuning knobs are unchanged.

## Configuration

All runs use 16 trays with four GB200 GPUs per tray and one JAX process per GPU. The common settings
are:

| Setting | Value |
| --- | ---: |
| Routed experts | 128 |
| Routed top-k | 4 |
| Experts per device | 2 |
| Shared experts | 2, both fused into the MoK call |
| MoK minibatch | 8,192 tokens |
| MoK macrobatch | 131,072 tokens |
| Schedule-capacity multiplier | 0.5 |
| Forward communication SMs | 40 |
| Backward communication SMs | 28 |
| All-gather chunk | 2,048 bytes |
| Optimizer | MuonH with pinned-host state offload |
| Software | Torch 2.11+cu130, cuBLAS 13.2, JAX 0.11.0 |

MoK is dropless in every treatment. Drop-adjusted throughput therefore equals raw throughput.

## Treatments

### Hot/cold placement

The placement is a static, per-layer permutation derived from the final routing histograms of the
r9 contiguous run. Initialization applies the same relabeling to the router output columns and
expert weights before those weights are sharded. This preserves the seeded model function and lets
the normal optimizer initialization create Muon state for the final parameter layout without a
global gather.

### Shared PGLE replay

The replay profile was aggregated at the 90th percentile from 64 XPlanes captured over steps 40–42
of the exact contiguous executable. The training executable matched 2,127 profiled instructions and
missed 257, or 89.2% coverage. Every rank loads the same content-addressed profile:

```text
s3://marin-us-east-02a/tmp/ttl=30d/xprof/pgle-profiles/mok-pgle-all64-p90-cbd3d7f0d0d6ca3bdaf2ff12ce88416f8753ecfc282af4b6ebcaf7f8fd757e4b.pb
sha256:cbd3d7f0d0d6ca3bdaf2ff12ce88416f8753ecfc282af4b6ebcaf7f8fd757e4b
```

Manual replay is used instead of AutoPGLE. Multi-host AutoPGLE can recompile hosts independently
and desynchronize distributed ranks, while one shared file gives every process the same schedule at
startup.

## Throughput results

The r9 baseline captured XProf during steps 80–84. The primary comparisons therefore end at step 79
so profiler overhead cannot bias the baseline. The 60–79 window is the most warmed-up common window.

### Profile-free common windows

| Treatment | Mean TPS, steps 40–79 | vs r9 | Mean TPS, steps 60–79 | MFU, steps 60–79 | vs r9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| r9 contiguous | 257,324 | — | 264,316 | 22.05% | — |
| Hot/cold | 259,715 | +0.93% | 265,919 | 22.18% | +0.61% |
| Shared PGLE | 260,313 | +1.16% | 266,803 | 22.26% | +0.94% |
| Hot/cold + PGLE | 260,044 | +1.06% | 269,017 | 22.44% | +1.78% |

### Full-run tail

The three treatment runs completed 100 steps. The r9 baseline stopped after its step-84 profile, so
it has no matching tail.

| Treatment | Mean TPS, steps 90–99 | Mean MFU | Last-step TPS | Last-step MFU | Last loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hot/cold | 287,720 | 24.00% | 289,495 | 24.15% | 4.1985 |
| Shared PGLE | 286,931 | 23.94% | 287,954 | 24.02% | 4.1915 |
| Hot/cold + PGLE | 285,235 | 23.79% | 286,838 | 23.93% | 4.1896 |

At common step 84, losses were 4.4267 for r9, 4.4196 for hot/cold, 4.4183 for PGLE, and 4.4170
for the combined treatment. All four runs reported zero MoK drops.

## Load balance

Hot/cold placement improved physical-rank balance, but the throughput benefit was much smaller than
the trace-derived upper bound.

| Measurement | r9 contiguous, step 84 | Hot/cold, step 84 | Hot/cold, step 99 |
| --- | ---: | ---: | ---: |
| Mean per-layer rank max/mean | 1.2929 | 1.2205 | 1.1048 |
| Worst layer max/mean | 1.4798 | 1.3056 | 1.1598 |
| Mean coefficient of variation | 0.1077 | 0.0836 | 0.0398 |

The original placement model predicted a 1.528-second barrier reduction and about 6.4% throughput
gain. The measured profile-free gain was approximately 0.6–0.9%. Load balance is therefore real,
but it was not the sole cause of the observed barrier tail.

## PGLE and communication interpretation

The existing MoK binary already satisfies the two mechanical prerequisites suggested by the first
trace review:

- The MoK compute stream has priority 0 while the NCCL stream has signed priority -5.
- The SM100 cubin contains cluster-launch-control instructions.

The trace instead shows no NCCL work submitted while the opaque MoK megakernel is active. Manual
PGLE can improve scheduling of the surrounding XLA and FSDP work, but it cannot move collectives
inside the custom call. This is consistent with the measured gain of roughly 1% rather than the
larger overlap-derived upper bound.

The combined result is not reliably additive. It was best in the warmed common 60–79 window, but
its 90–99 tail was 0.86% below hot/cold alone and 0.59% below PGLE alone. These differences are near
the scale of rack-run variability. Neither treatment should become the default based on one run per
arm; repeated matched runs or the exact-treatment traces below should precede that decision.

## Exact-treatment XProf captures

Three regular Iris jobs were submitted on 2026-08-10 at interactive priority. Each job requests 16
non-preemptible GB200 trays, uses hard one-rack `nvlink.domain` coscheduling, trains through step 84,
and captures process 0 during steps 80–84.

| Treatment | Iris job | W&B run | XProf root after completion |
| --- | --- | --- | --- |
| Hot/cold | `/muchanem/mok-hotcold-xprof-r14-20260810` | `mark-mok-hotcold-xprof-r14-100-s80n5-20260810` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-hotcold-xprof-r14-100-s80n5-20260810` |
| Shared PGLE | `/muchanem/mok-pgle-xprof-r15-20260810` | `mark-mok-pgle-xprof-r15-100-s80n5-20260810` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-pgle-xprof-r15-100-s80n5-20260810` |
| Hot/cold + PGLE | `/muchanem/mok-hotcold-pgle-xprof-r16-20260810` | `mark-mok-hotcold-pgle-xprof-r16-100-s80n5-20260810` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-hotcold-pgle-xprof-r16-100-s80n5-20260810` |

These paths are lifecycle-managed for 30 days. Copy profiles needed for long-term comparison to a
non-TTL artifact prefix before the lifecycle deadline.

## Recommendation

Keep hot/cold placement and shared PGLE replay available as independent experimental controls.
Hot/cold placement has the stronger mechanistic result because it materially reduces rank imbalance
with no runtime dependency. Shared PGLE has a small positive result but requires an exact executable
profile and cannot optimize inside MoK. The next optimization target should be adapter-only memory
traffic or work submission from within the MoK call, followed by repeated rack-scale measurements.
