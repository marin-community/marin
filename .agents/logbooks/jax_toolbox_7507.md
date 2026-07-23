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

## 2026-07-22: first control and overlay failure

Control scan A completed on 16 `GB200x4` nodes with median 14.6695 seconds, 285,921 tokens/second, and 25.8327% MFU over steps 5–11 and 15–19. The 12 selected step durations had standard deviation 0.0745 seconds. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-control-scan-a-20260722-195616)

The first Toolbox normal-sync smoke failed before Sonic-CuTe. The overlay preserved every declared distribution, but `uv sync` installed `nvidia-cudnn-cu13==9.19.0.56` beside Toolbox's system cuDNN 9.22. XLA reported the runtime/source mismatch and failed `dnn_support != nullptr` on the first compiled array operation. The preserved list must also exclude CUDA wheel packages that represent image-owned system libraries even when the image has no Python distribution metadata for them.

The second smoke excluded the CUDA wheel families and reached four GPUs without the cuDNN mismatch. It failed while importing CUTLASS because the overlay installed `torch==2.11.0+cu128` without the excluded CUDA 12 wheels. The image now supplies and preserves `torch==2.11.0+cpu`; CUTLASS imports Torch for adapters, but Torch is not the accelerator runtime under test.

The third smoke preserved CPU Torch and every `nvidia-*` CUDA distribution. Normal workspace sync completed and JAX enumerated all four GB200 GPUs. CUTLASS then rejected its environment as older than CUDA 13.1 because the overlay had installed the separately named `cuda-toolkit==12.8.1` metapackage. Add that name to the image-owned exclusion list before the next smoke.

Further inspection showed the apparent CUDA-version check was an unconditional stub from an overlapping-wheel bug in CUTLASS DSL 4.5.2, not runtime detection. The current `quack-kernels==0.6.1` also imports IKET, which does not exist in 4.5.2. Repinning to the July 18 Toolbox image supplied JAX `0.11.1.dev20260718`, CUDA 13.3, and CUTLASS DSL 4.6.1. A normal-sync four-GPU Sonic-CuTe forward/backward smoke then succeeded with loss 6.9999.

The first full treatment launch failed during imports, before W&B initialization or compilation. The overlay installed `torchvision==0.26.0+cu128` beside preserved `torch==2.11.0+cpu`; Transformers' `AutoProcessor` import failed because `torchvision::nms` was unavailable. An ephemeral validation replacing it with `torchvision==0.26.0+cpu` imported `AutoProcessor` successfully. The derived image must preserve the matching CPU Torch/Torchvision pair.

## 2026-07-22: stable scan comparison

The exact control → Toolbox → control sequence completed on 16 four-GPU GB200 nodes without retries, preemptions, or failed workers. All three runs used the #7507 model shape (`hidden_dim=6144`, 48 layers, 128 experts, top-4 routing) and the same selected steady-state steps 5–11 and 15–19.

| Run | Median seconds/step | Median tokens/second | Median MFU |
| --- | ---: | ---: | ---: |
| [Control A](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-control-scan-a-20260722-195616) | 14.6695 | 285,920.95 | 25.8327% |
| [Toolbox](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-scan-c-20260722-1523) | 14.8055 | 283,293.64 | 25.5954% |
| [Control B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-control-scan-b-20260722-1539) | 14.7376 | 284,600.13 | 25.7134% |

Control A to B drift was -0.462% in throughput and MFU. Relative to the two-control midpoint, Toolbox was 0.690% slower; a linear interpolation at the treatment timestamp estimates a 0.505% throughput and MFU regression and a 0.507% step-time increase. The direction is consistently negative, but the effect is small and passes the proposed 2% adoption gate. The comparison changes the full image stack—JAX/XLA, CUDA, NCCL, CUTLASS, and base OS—rather than isolating one component.

One earlier Toolbox run used the wrong environment variable names and therefore ran `hidden_dim=3072`, 32 experts, and 128 GPUs. Its 10.67% MFU is excluded from the comparison.

## 2026-07-22: exact no-scan probes

The first exact 48-layer Toolbox treatment with `SCALE_SCAN_LAYERS=0` failed before step 0 after 16m34s in the first `jit_train_step`. CUDA attempted a 927,668,341,744-byte allocation and JAX reported `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 863.96GiB`; the allocator limit was 138.22 GiB. However, that run still initialized one `ArrayStacked[Block]` pytree and iterated over unstacked views, so it did not test the intended independent-module representation. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-noscan-a-20260722-1553)

After changing initialization to store a tuple of independently initialized `Block` modules, a corrected exact-shape run spent about 26 minutes in the first `jit_train_step` and failed before step 0. CUDA requested 972,028,334,224 bytes (905.27 GiB) against a 138.22 GiB allocator limit. The run used 16 hosts and 64 devices, had retries disabled, and produced neither MFU metrics nor an in-memory-CUBIN or null-module signature. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-noscan-modules-a-20260722-1640)

The corrected run reproduces the unrolled-layer temporary-arena pathology, not the distinct CUBIN-load failure. JAX-Toolbox gets past that known CUBIN surface in this case but does not make the full workload viable without scan, and one OOM-terminated probe cannot establish that it fixes CUBIN loading in general. A current-image no-scan repeat is unlikely to add useful performance evidence because the treatment terminates at the earlier arena-memory boundary.

## 2026-07-22: no-scan block-boundary barrier probe

Commit [`239b663c4d`](https://github.com/marin-community/marin/commit/239b663c4d) adds `jax.lax.optimization_barrier` between the 48 independently initialized blocks while retaining per-block rematerialization. A small CPU test confirmed that the barriers preserve model outputs.

The exact 64-GPU Toolbox treatment failed before step 0 after 26m37s in the first `jit_train_step`. Each local CUDA async allocator requested 972,028,203,152 bytes (905.27 GiB), versus 972,028,334,224 bytes without barriers: a reduction of only 131,072 bytes. The allocator limit remained 138.22 GiB, with 51.10 GiB in use. The run produced no CUBIN signature, history rows, or MFU metrics. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-noscan-barrier-a-20260722-1837)

Follow-up lowering inspection showed that this was not a clean barrier/no-barrier comparison. The baseline already contains StableHLO optimization barriers introduced by per-layer `eqx.filter_checkpoint`/`jax.checkpoint`. In a differentiated two-block CPU probe, the baseline contained two `stablehlo.optimization_barrier` ops and the explicit treatment contained four. Both optimized executables had the same fusion count and no remaining barrier ops. The 128 KiB arena delta therefore shows that an extra barrier outside each checkpoint is redundant; it does not independently test a previously unconstrained cross-block graph.
