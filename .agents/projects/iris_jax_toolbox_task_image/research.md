# Background Research Brief

- Effort: medium
- Stop rule: repository, issue, live-cluster, image-manifest, uv, Kubernetes, and W&B evidence converged on one design and benchmark ladder.
- Date: 2026-07-22

## Question

Can Iris use an NVIDIA JAX-Toolbox image as its GPU task base while continuing to synchronize Marin, Levanter, and Haliax from the submitted workspace? How should we compare that environment with the current PyPI CUDA/JAX stack on the communication-bound workload from [#7507](https://github.com/marin-community/marin/issues/7507)?

## Current Marin context

Iris has one task image. The `task` target starts from `python:3.12-slim`, installs build tools, uv, Node, Rust, IB userspace, and Nsight Systems, but no application source ([Dockerfile](https://github.com/marin-community/marin/blob/c26285a61654a9e6a9029cfdb3d018badc35d71c/lib/iris/Dockerfile#L176-L239)). At runtime, the submitter renders a `uv sync` script that creates an isolated workspace venv and installs the requested extras ([setup_scripts.py](https://github.com/marin-community/marin/blob/c26285a61654a9e6a9029cfdb3d018badc35d71c/lib/iris/src/iris/cluster/setup_scripts.py#L49-L118)). Marin and Levanter's `gpu` extras pin JAX 0.10.1 and CUDA 13 Python wheels, so changing only the Docker `FROM` line would leave the JAX-Toolbox stack shadowed by the venv.

Kubernetes selects `run_req.task_image` or one cluster default before it parses GPU resources ([tasks.py](https://github.com/marin-community/marin/blob/c26285a61654a9e6a9029cfdb3d018badc35d71c/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L712-L790)). It also replaces the image's OCI entrypoint with `bash -lc`, while the Docker backend preserves the image entrypoint. The successful one-node canary in [#7519](https://github.com/marin-community/marin/issues/7519) had to call `/opt/nvidia/nvidia_entrypoint.sh` explicitly.

## Internal prior work

### JAX-Toolbox GB200 canary

`ghcr.io/nvidia/jax:jax-2026-06-27` is a multi-architecture image. Its ARM64 manifest contains CUDA 13.2.1, NCCL 2.30.4, HPC-X 2.26, and JAX `0.11.0.dev20260627+060c1f383`. The image ran successfully on one four-GPU GB200 node in `cw-us-east-08a`: four devices enumerated, a BF16 4096×4096 matrix multiplication returned 4096, and the Iris job exited zero in 2m01.5s. NVIDIA's entrypoint warned that Mellanox hardware was present but the NVIDIA peer-memory driver was not detected. Single-node computation did not exercise inter-node GPUDirect RDMA.

### Issue #7507 baseline

The full-MoE reproducer at commit `31b221e1db02bf553488c903ed36d8ae1f424b63` ran on 16 four-GPU GB200 nodes. The worker gang succeeded in 10m11.43s with JAX 0.10.1 and NCCL 2.28.9. Excluding profiler steps 12–14, steps 5–11 and 15–19 had a median 14.7946 s/step, 283,502 tokens/s, 25.614% MFU, and 0.0462 s within-run step-time standard deviation. The [W&B run](https://wandb.ai/marin-community/marin_moe/runs/gb200-d6144-64gpu-b200minimal-sw512-chunk4) and [issue reproducer](https://github.com/marin-community/marin/issues/7507#issuecomment-5048563412) are durable references.

The attention-only variant had a median 3.03847 s/step and 1.3804M tokens/s. Its reported 124.7% MFU is invalid because the analytic FLOP counter still counts the disabled MoE. It remains useful for examining Q/K/V/O all-gather placement and idle gaps.

The CPU coordinator's `--task-image` does not set the nested Fray GPU image. The benchmark branch must pass an image through `ResourceConfig.image` where `launch_cw_scale.py` constructs the worker resources. Both arms must use the same source commit plus that one image-selection knob.

### No-scan and CUBIN history

The documented L48 no-scan failure in [#7282](https://github.com/marin-community/marin/issues/7282#issuecomment-5007936111) is an XLA planned temporary arena of 340–851 GiB because all 48 MoE layer buffers remain live. It is distinct from the scanned L48 CUBIN load failure in [#7407](https://github.com/marin-community/marin/issues/7407) and [#7421](https://github.com/marin-community/marin/issues/7421#issuecomment-5028250174). Later instrumentation found that all 64 failed `cuModuleLoadData` calls received a null module image, while successful calls received non-null images; replaying captured non-null CUBINs more than one million times succeeded ([#7421 result](https://github.com/marin-community/marin/issues/7421#issuecomment-5040056368)). The evidence does not show malformed CUBIN bytes or a wrong GPU architecture. A no-scan probe must report arena OOM and null-module load failure separately.

## External prior art

JAX-Toolbox builds JAX/JAXLIB components, TransformerEngine, CUTLASS DSL, and cuda-tile into the system Python environment ([Dockerfile.jax](https://github.com/NVIDIA/JAX-Toolbox/blob/45b708b1263f7d67cfb1dc3e12baebb5fdf8dc22/.github/container/Dockerfile.jax)). Its GPU performance guide recommends O1 scheduling and pipelined collectives for poor compute/communication overlap, and custom/address-computation fusion for scan-related dynamic-update-slice copies ([GPU performance guide](https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/GPU_performance.md)). Those flags are separate treatments; the first image A/B should not enable them.

uv supports `uv venv --system-site-packages` and repeated `uv sync --no-install-package NAME`. uv deliberately ignores system-site distributions during installation, so both mechanisms are required: visibility alone does not stop uv from installing a shadowing copy ([uv venv](https://docs.astral.sh/uv/reference/cli/#uv-venv--system-site-packages), [uv sync](https://docs.astral.sh/uv/reference/cli/#uv-sync--no-install-package)). Kubernetes preserves an OCI entrypoint when a pod supplies `args` without replacing `command` ([Kubernetes command and arguments](https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/)).

## Negative and failed leads

- `--no-sync` proved the raw image works, but it omits Marin, Levanter, Haliax, and Iris runtime integration.
- `uv sync --inexact` does not preserve the base stack; it still installs locked JAX-family packages into the venv.
- Preserving only `jax` and `jaxlib` can mix NVIDIA's nightly core with Marin's `jax-cuda13-plugin`, PJRT, TransformerEngine, CUTLASS, cuDNN, or NCCL distributions.
- Prepending system site-packages through `PYTHONPATH` can make import precedence disagree with distribution metadata and does not prevent compiled-package mixing.
- Installing workspace packages system-wide with `--no-deps` loses the locked non-GPU dependency environment and contaminates the image-owned Python environment.
- The historical #7507 run is useful prior evidence but not a sufficient control: source, task image, caches, and rack placement must be controlled in a new A/B.

## Prototype findings

The first normal-sync prototype preserved only distributions visible through Python metadata. `uv sync` then installed CUDA 12 and CUDA 13 cuDNN wheels from the lock, and the first GPU operation failed because JAX was compiled against cuDNN 9.22 while the overlay loaded cuDNN 9.19. This falsified the assumption that the protected boundary can be derived only from installed distributions.

The second prototype excluded all CUDA wheel names and reached four GB200 devices without a cuDNN mismatch. It still installed the lock's GPU-enabled Torch wheel without its excluded CUDA 12 dependencies. CUTLASS imported Torch and failed to load `libcudart.so.12` and `libcublasLt.so`. The derived image therefore needs to carry and preserve a CPU-only Torch build for CUTLASS's Python import path; Torch is not part of the accelerator runtime under test.

The third prototype preserved CPU Torch and every `nvidia-*` accelerator distribution, but the overlay installed the separately named `cuda-toolkit==12.8.1` metapackage. CUTLASS then classified the environment as older than CUDA 13.1 despite the container's CUDA 13.2.1 runtime. The protected boundary must be audited from the entire lock graph, not inferred from an `nvidia-*` naming convention.

Inspection then showed that the CUTLASS 4.5.2 error was an unconditional stub from overlapping base and cu13 wheel paths, not CUDA detection. That release also lacks IKET, which current QuACK imports. The July 18 Toolbox image provides JAX `0.11.1.dev20260718`, CUDA 13.3, and CUTLASS DSL 4.6.1; its normal-sync Sonic-CuTe forward/backward smoke passed. The first full launch exposed one more overlay boundary: preserved CPU Torch paired with overlay CUDA-12 Torchvision and broke the Transformers `AutoProcessor` import. A matching CPU Torch/Torchvision pair passed the import probe and belongs in the derived image.

The exact control → Toolbox → control sequence completed all 16 `GB200x4` workers per run without retries, preemptions, or failed workers. Over steps 5–11 and 15–19, control A measured 14.6695 seconds/step, 285,921 tokens/second, and 25.8327% MFU; Toolbox measured 14.8055 seconds/step, 283,294 tokens/second, and 25.5954% MFU; control B measured 14.7376 seconds/step, 284,600 tokens/second, and 25.7134% MFU. Control drift was 0.462%. Toolbox was 0.690% slower than the two-control midpoint, or 0.505% slower using linear time interpolation at the treatment timestamp. The result passes the proposed 2% adoption gate, although the consistently negative direction gives no performance reason by itself to switch stacks. [Control A](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-control-scan-a-20260722-195616), [Toolbox](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-scan-c-20260722-1523), [control B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-control-scan-b-20260722-1539).

The first 48-layer Toolbox no-scan probe did not use the intended parameter representation. It changed execution to a Python loop but retained one `ArrayStacked[Block]`, then iterated over `ArrayStacked.unstacked()` views. That graph failed before step 0 on an 863.96 GiB allocation, but the result does not characterize independently initialized layer modules. Commit [`09b2054d8e`](https://github.com/marin-community/marin/commit/09b2054d8e) selects a tuple of independent `Block` pytrees for no-scan while retaining per-layer rematerialization.

The corrected 64-GPU Toolbox run used those independent modules and the exact #7507 shape. It spent about 26 minutes in the first `jit_train_step`, then failed before step 0 when CUDA requested 972,028,334,224 bytes (905.27 GiB) against a 138.22 GiB allocator limit. It produced no MFU metrics and no null-module or in-memory-CUBIN signature. The Toolbox stack therefore gets past the previously observed CUBIN failure surface for this probe but does not make the full unrolled graph viable; this OOM does not establish that other CUBIN failures are fixed. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-noscan-modules-a-20260722-1640)

A follow-up inserted `jax.lax.optimization_barrier` between all 48 independent blocks. It failed before step 0 after 26m37s with a 972,028,203,152-byte request, only 131,072 bytes below the unbarriered request. The barriers produced no CUBIN signature or MFU metrics. This rules out cross-block fusion and optimization as material causes of the 905 GiB arena. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-noscan-barrier-a-20260722-1837)

## Evidence map

### Claim: an image-owned preserved-package list can retain the NVIDIA stack

- Support: uv exposes system packages at runtime and can omit named distributions during sync.
- Contradiction: uv warns that `--no-install-package` may produce a broken environment; a partial preserved set can silently mix binary stacks.
- Directness to Marin: exact mechanism used by `default_setup_script` and the proposed image.
- Confidence: the exclusion mechanism is established, but the complete boundary remains experimental until the corrected full workspace sync and kernel ladder pass.
- Action: audit installed entries outside the overlay and intentionally absent entries inside it after every sync.

### Claim: #7507 can distinguish scheduler/runtime changes

- Support: 64 GPUs, FSDP all-gathers, recorded XProf traces, and 0.3% within-run step-time variation.
- Contradiction: sequential runs may land on different racks; the full workload fits one NVL72 domain and does not test cross-rack IB.
- Directness to Marin: exact Grug MoE workload motivating improved communication overlap.
- Confidence: stable as an in-rack performance benchmark; not evidence for cross-rack GPUDirect RDMA.
- Action: compare immutable control and treatment images at one source commit, excluding profiler steps.

## Recommended experiments

### 1. Workspace overlay preflight

- Minimum experiment: normal Iris sync in the Toolbox-derived image using the `gpu` extra, followed by import/version/location checks for Marin, Levanter, Haliax, JAX, JAX plugins, TransformerEngine, CUTLASS DSL, and cuda-tile.
- Baseline/control: current `iris-task` sync.
- Expected signal: workspace packages resolve from `$IRIS_VENV`; preserved distributions resolve outside it at the versions recorded in the image.
- Falsifier: missing distribution, preserved package under `$IRIS_VENV`, Python ABI mismatch, or CUDA library conflict.
- Cost/risk: one CPU setup plus one GB200 node.

### 2. Kernel and distributed ladder

- Minimum experiment: one-node Sonic-CuTe forward/backward, one-node FA4 forward/backward, then two-node `iris.runtime.jax_init` and NCCL transport inspection.
- Baseline/control: identical source and commands in current `iris-task`.
- Expected signal: successful compilation, finite outputs/gradients, eight visible GPUs across two tasks, and intended NCCL transport.
- Falsifier: import/compile failure, hang, peer-memory/transport fallback, or numerical failure.
- Cost/risk: below one GPU-hour for the ladder.

### 3. Full-MoE A/B from #7507

- Minimum experiment: control then Toolbox treatment, 20 steps each, 16 `GB200x4` tasks.
- Baseline/control: immutable current `iris-task` digest.
- Primary metric: median step duration over steps 5–11 and 15–19; tokens/s and full-MoE MFU are equivalent views.
- Expected signal: no more than 2% throughput regression; adoption is stronger with reduced trace bubbles or positive throughput.
- Falsifier: correctness failure, compile failure, or more than 2% regression.
- Cost/risk: approximately 21.7 GPU-hours; control→treatment→control costs approximately 32.6 GPU-hours.

### 4. Attention-only trace diagnostic

- Minimum experiment: repeat both images with `SCALE_ATTN_ONLY=1` and the same profiler window.
- Baseline/control: current image.
- Primary metric: step duration and trace placement, not MFU.
- Expected signal: less idle time around Q/K/V/O projections or earlier all-gather overlap.
- Cost/risk: approximately 11.6 GPU-hours for A/B.

## Hypothesis queue update

- Add: a coherent system JAX stack plus newer XLA/NCCL reduces #7507's communication bubbles without changing model code.
- Add: the overlay venv can preserve the image stack while installing current workspace source and locked non-GPU dependencies.
- Revise: attention-only is a diagnostic, not the primary performance gate.
- Stop: do not tune O1, PGLE, fusion flags, or collective thresholds in the image A/B.

## Source ledger

| Source | Type | Location | Claim used for | Confidence |
|---|---|---|---|---|
| Iris task Dockerfile | Marin code | `lib/iris/Dockerfile:176-239` | Current image responsibilities | High |
| Iris setup script | Marin code | `lib/iris/src/iris/cluster/setup_scripts.py:49-118` | Current isolated uv sync | High |
| Iris K8s pod builder | Marin code | `lib/iris/src/iris/cluster/backends/k8s/tasks.py:712-806` | Image precedence and entrypoint replacement | High |
| #7519 | GitHub issue | https://github.com/marin-community/marin/issues/7519 | Single-node Toolbox compatibility | High |
| #7507 | GitHub issue/W&B | https://github.com/marin-community/marin/issues/7507 | Workload and historical metrics | High |
| #7282/#7407/#7421 | GitHub issues | links above | No-scan arena OOM and scanned null-module CUBIN signature | High |
| JAX-Toolbox Dockerfile | External code | pinned NVIDIA commit above | System stack contents | High |
| uv CLI reference | Official docs | links above | Overlay mechanics and caveat | High |
| NVIDIA GPU performance guide | Official docs | link above | Expected scheduler differences | Medium; flags are experimental |

## Handoff

- Suggested design: separate `iris-task-gpu`, image-owned preserved distribution list, system CUDA capability marker, GPU-aware K8s image selection, and OCI entrypoint preservation.
- Open questions: whether the preserved list itself implies system CUDA ownership; whether phase one is Kubernetes-only; whether the performance gate is A/B or control→treatment→control.
- Stop reason: remaining uncertainty requires a prototype image and measured runs, not more source search.
