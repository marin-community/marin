---
title: JAX-Toolbox GB200 full-MoE comparison
author: dlwh
issue: https://github.com/marin-community/marin/issues/7519
created: 2026-07-22
---

# JAX-Toolbox GB200 full-MoE comparison

## Scope

Compare the current Iris GPU task stack with an NVIDIA JAX-Toolbox-derived task image on the exact full-MoE workload from [#7507](https://github.com/marin-community/marin/issues/7507). Report stable steady-state MFU and determine whether the Toolbox stack compiles and runs the same model when layers are not represented by `jax.lax.scan`.

## Baseline

The historical full-MoE run used source commit `31b221e1db02bf553488c903ed36d8ae1f424b63`, 16 four-GPU GB200 nodes, and scan layers. Excluding warmup and profiler steps, steps 5–11 and 15–19 had median 14.7946 seconds, 283,502 tokens/second, and 25.614% MFU. This is context only; the comparison will rerun both images from the same source revision.

## Hypotheses

- `JTB-001`: the Toolbox-derived image completes the scan workload with finite loss and steady-state throughput no more than 2% below a contemporaneous control.
- `JTB-002`: the Toolbox runtime compiles and completes the full 48-layer workload with `SCALE_SCAN_LAYERS=0`, avoiding the Cubin load failure reported in #7407/#7507.
- `JTB-003`: control → Toolbox → control MFU medians vary by less than 1% within an image after excluding steps 0–4 and profiler steps 12–14.

## Method

Use immutable image digests and unique W&B run IDs in `marin-community/marin_moe`. The primary sequence is control-scan → Toolbox-scan → control-scan. Add Toolbox-no-scan and control-no-scan as compile/correctness probes. Aggregate raw `throughput/mfu`, `throughput/duration`, and `throughput/tokens_per_second` samples over steps 5–11 and 15–19. Preserve raw samples and resolved package versions with the result.

## 2026-07-22: setup and single-node capability

The bare image `ghcr.io/nvidia/jax@sha256:15c3f15baf88af18e295b37c6417521f74ebf7bf2449ea09ab64d388be914343` completed a four-GPU BF16 matrix multiply on `/dlwh/jax-toolbox-gb200-canary-20260722`. JAX was `0.11.0.dev20260627+060c1f383`. The entrypoint warned that NVIDIA peer memory was not detected; the 16-node #7507 workload fits one NVL72 domain and does not validate cross-rack GPUDirect RDMA.

Prototype work started from the exact #7507 commit. The branch adds a nested GPU image override, restores a real scan/no-scan code path, and teaches Iris setup to honor image-owned distributions listed in `/etc/iris/preserved-python-packages` through a system-site-packages overlay venv.

Status: implementation and image build in progress; no full-MoE comparison launched yet.
