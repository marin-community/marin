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

## Profile catalog

The object-store paths below are the collaborator-facing source of truth. All captures use the
64-GPU, one-process-per-GPU topology unless an entry explicitly says otherwise. XProf directories
contain both the compressed Perfetto trace (`*.trace.json.gz`) and the raw XPlane protobuf
(`*.xplane.pb`). Use the XPlane when exact event metadata or PGLE conversion is required; use the
compressed trace for interactive timeline inspection.

All listed paths are under a `ttl=30d` lifecycle prefix. They are not permanent archives and should
be copied to a non-TTL prefix before early September 2026 if they are needed after this experiment.

### Which profile to use

| Question | Recommended artifact |
| --- | --- |
| Compare the original contiguous MoK timeline with DeepEP | Clean r9 XProf pair |
| Inspect CUDA API calls, allocation counts, memory traffic, or cross-stream overlap | Valid r8 Nsight pair and exported analysis |
| Rebuild or audit the shared PGLE estimator | All-rank r11 XProf source plus the content-addressed p90 protobuf |
| Compare expert placement with the contiguous r9 timeline | Hot/cold r14 XProf |
| Inspect the effect of PGLE replay on the contiguous executable | PGLE-only r15 XProf |
| Inspect hot/cold placement and PGLE together | Combined r16 XProf after that job completes |

### Clean rack-scale XProf captures

These are the primary timeline artifacts. Each process-0 treatment capture covers exactly five
training steps, 80–84 inclusive, under `plugins/profile/steps-80-to-85`.

| Capture | Status | Iris job | W&B run | S3 root | Objects |
| --- | --- | --- | --- | --- | --- |
| Contiguous MoK r9 | Complete, validated GPU plane | `/muchanem/mok-deepep-profile-100-s80n5-r9-xprof-20260808` | `mark-mok-profile-parity-mok-mb8192-100-s80n5-r9-xprof-20260808` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-profile-parity-mok-mb8192-100-s80n5-r9-xprof-20260808/plugins/profile/steps-80-to-85` | 22,690,554-byte trace; 145,483,412-byte XPlane |
| DeepEP r9 | Complete, validated GPU plane | `/muchanem/mok-deepep-profile-100-s80n5-r9-xprof-20260808` | `mark-mok-profile-parity-deepep-100-s80n5-r9-xprof-20260808` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-profile-parity-deepep-100-s80n5-r9-xprof-20260808/plugins/profile/steps-80-to-85` | 24,467,961-byte trace; 151,049,255-byte XPlane |
| Hot/cold r14 | Complete, 16/16 tasks, no failures or preemptions | `/muchanem/mok-hotcold-xprof-r14-20260810` | `mark-mok-hotcold-xprof-r14-100-s80n5-20260810` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-hotcold-xprof-r14-100-s80n5-20260810/plugins/profile/steps-80-to-85` | 22,152,910-byte trace; 145,527,090-byte XPlane |
| Shared PGLE r15 | Complete, 16/16 tasks, no failures or preemptions | `/muchanem/mok-pgle-xprof-r15-20260810` | `mark-mok-pgle-xprof-r15-100-s80n5-20260810` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-pgle-xprof-r15-100-s80n5-20260810/plugins/profile/steps-80-to-85` | 22,296,535-byte trace; 146,575,810-byte XPlane |
| Hot/cold + PGLE r16 | Pending after two priority preemptions; no profile object yet | `/muchanem/mok-hotcold-pgle-xprof-r16-20260810` | `mark-mok-hotcold-pgle-xprof-r16-100-s80n5-20260810` | `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-hotcold-pgle-xprof-r16-100-s80n5-20260810/plugins/profile/steps-80-to-85` | None as of 2026-08-10 |

The r9 pair has already been decoded and cross-checked. The MoK XPlane contains 281,805 CUDA kernel
events and the DeepEP XPlane contains 299,810. Each contains exactly five consecutive
`jit_train_step` executions. Mean execute time was 15.272 seconds for contiguous MoK and 12.923
seconds for DeepEP. Checksums are:

| Object | SHA-256 |
| --- | --- |
| MoK r9 trace | `3ad0f914a86d27ad23fd00093ecf2251deebd6a8594484c3f03f92812d53fa42` |
| MoK r9 XPlane | `d09ae542dbbf244be788ea8e59fe2890f252d9a4b34b62294748319324b5794d` |
| DeepEP r9 trace | `aba13ce3306e5676ea4fbf0c9b474d95d3739d1449235ba10f6ffbe847366c04` |
| DeepEP r9 XPlane | `3f9bbd1d2f5bad193a3122931107686b2b9b6ddf976e86b156863f3c5de010e0` |

The r14 and r15 object sizes have been checked against S3. Detailed event-count and critical-path
analysis has not yet been recorded for those two captures.

### All-rank PGLE source and replay profile

The r11 source capture recorded steps 40–42 from all 64 JAX processes. Its root is:

```text
s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-pgle-profile-all64-r11-100-s40n3-20260809
```

It contains `process-00000` through `process-00063`. Every process directory has one trace and one
XPlane under `plugins/profile/steps-40-to-43`, for 128 objects and 8,408,407,217 bytes total. The
per-process XPlanes are approximately 118 MB each. This capture is intentionally much larger than
the process-0 comparison traces because PGLE aggregation needs latency observations across the full
rack.

The 64 XPlanes were converted independently, then aggregated instruction-by-instruction at the
90th percentile. The replayable output is:

```text
s3://marin-us-east-02a/tmp/ttl=30d/xprof/pgle-profiles/mok-pgle-all64-p90-cbd3d7f0d0d6ca3bdaf2ff12ce88416f8753ecfc282af4b6ebcaf7f8fd757e4b.pb
size: 67,612 bytes
sha256: cbd3d7f0d0d6ca3bdaf2ff12ce88416f8753ecfc282af4b6ebcaf7f8fd757e4b
```

The r15 and r16 launchers verify that hash before setting
`--xla_gpu_pgle_profile_file_or_directory_path`. This protobuf is an XLA profiled-instructions
database, not an XProf timeline, so it is consumed by XLA rather than opened in the trace viewer.

### Nsight Systems pair and exports

The valid rank-0 Nsight pair is from r8 and covers the same five steps, 80–84. Unlike the r8 XProf
files described below, the Nsight reports have complete CUDA activity and remain the best artifacts
for allocation-call, copy-volume, CUDA API, and overlap analysis.

| Backend | Raw report |
| --- | --- |
| MoK | `s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808/mok/r00000-s38vxs64.nsys-rep` |
| DeepEP | `s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808/deepep/r00000-s38vxs64.nsys-rep` |

The MoK report is 58,136,827 bytes with SHA-256
`36e78a2918ee47ade2037a7e790ae6b6af641961d9849a8324aa2b388d840c3f`. The DeepEP report is
62,058,661 bytes with SHA-256
`7b1a67873889e6d13f11338571c50158163408c5423bbc93c8bfa3d099aa890e`.

Compressed SQLite exports and raw `nsys stats` CSV reports live under:

```text
s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/k3sc0re/mok-vs-deepep-20260808/analysis/r8
```

That directory includes CUDA API, GPU kernel, GPU memory time and size, NVTX, and OS runtime tables
for each backend. The Nsight analysis found identical allocation/free call counts, only about 52 ms
per step of extra MoK copy-engine time, and a much larger difference in serialized fused-kernel and
barrier time.

### Retained but excluded profiles

These objects are available for historical debugging but should not be used as graph-parity
evidence:

- The r8 MoK and DeepEP XProf captures under
  `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-profile-parity-mok-mb8192-100-s80n5-r8-20260808`
  and
  `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mark-mok-profile-parity-deepep-100-s80n5-r8-20260808`
  are structurally valid but host-only. XProf and Nsight competed for CUPTI in the same process, and
  Nsight won the GPU subscription. The clean r9 pair supersedes them.
- The older DeepEP XProf at
  `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mhprof-ep-jax-20260808/plugins/profile/steps-30-to-33`
  used one JAX process per tray controlling four GPUs and a 50-step optimizer horizon. Its trace is
  59,644,745 bytes and its XPlane is 263,158,646 bytes.
- The older DeepEP Nsight report at
  `s3://marin-us-east-02a/tmp/ttl=30d/iris-profiles/rav/mhep-024-nsys-10-p32582-20260805-coord/grug-train-mhep-024-nsys-10-p32582-20260805/r00000-s1b62nb4.nsys-rep`
  is 344,265,848 bytes and also predates the final process topology and optimizer horizon.

## Recommendation

Keep hot/cold placement and shared PGLE replay available as independent experimental controls.
Hot/cold placement has the stronger mechanistic result because it materially reduces rank imbalance
with no runtime dependency. Shared PGLE has a small positive result but requires an exact executable
profile and cannot optimize inside MoK. The next optimization target should be adapter-only memory
traffic or work submission from within the MoK call, followed by repeated rack-scale measurements.
