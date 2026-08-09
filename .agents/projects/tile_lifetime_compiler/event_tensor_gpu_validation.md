# Event Tensor GPU validation plan

## Scope

This work keeps `TaskDependence` and `EventTensorPlan` target-independent. CUDA is a legalization of a selected plan, not an event semantic.

The first lowering accepts a verified, exact, CTA-scope plan with one consumer and at most 1,024 producers per event. It creates one fresh `cuda::barrier<cuda::thread_scope_block>` per CTA invocation. Producer tasks store FP32 partials and call `arrive`; the consumer waits before reading those partials. A generated CTA `__syncthreads` kernel and a two-kernel producer/finalizer sequence are controls.

The first checkpoint does not support phased persistent reuse, device-wide semaphores, coarsened events, or multiple consumers per event. Unsupported plans are rejected before source generation.

The benchmark currently uses a Torch C++ extension only as a rapid CUDA
compile/load harness. Torch is not part of the intended compiler contract. A
promoted lowering should expose the same generated body through Shuttle's JAX
typed-FFI registration path, with JAX owning model-level autodiff and Shuttle
recovering task dependencies from the differentiated program.

## Pre-hardware validation

```bash
uv run --frozen --package marin-tile-lifetime --group test pytest \
  lib/tile_lifetime/tests/test_event_dataflow.py \
  lib/tile_lifetime/tests/test_cuda_event_dataflow_codegen.py -q

uvx pyrefly check \
  lib/tile_lifetime/src/tile_lifetime/event_dataflow.py \
  lib/tile_lifetime/src/tile_lifetime/event_dataflow_examples.py \
  lib/tile_lifetime/src/tile_lifetime/cuda_event_dataflow_codegen.py \
  lib/tile_lifetime/tests/test_event_dataflow.py \
  lib/tile_lifetime/tests/test_cuda_event_dataflow_codegen.py
```

The scoped tests pass locally. The repository-wide pre-commit command still reports the base-branch Pyrefly error at `xla_hlo_recovery.py:662`.

## H100 request and commands

Phase 1 needs one H100 for less than one hour. No multi-GPU communication is involved.

```bash
export GPU_NAME="${USER}-shuttle-event-tensor"
uv run --package marin-iris --extra controller scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$GPU_NAME" allocate \
  --gpu-variant h100 \
  --gpus-per-node 1 \
  --priority batch \
  --timeout 1800
```

After connecting, synchronize the exact branch revision and enable GPU dependencies:

```bash
uv run --package marin-iris --extra controller scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$GPU_NAME" connect

cd /app
git fetch origin research/shuttle-event-tensor-prototype
git checkout --detach <phase-1-commit>
uv sync --all-packages --extra=gpu
export TORCH_CUDA_ARCH_LIST=9.0a
timeout 20m uv run python lib/tile_lifetime/benchmarks/h100_event_tensor_split_fold.py \
  --rows 4096 \
  --partitions 64 \
  --warmups 10 \
  --repeats 30 \
  --iterations 100 \
  --shuttle-revision <phase-1-commit> \
  --json-output /tmp/event-tensor-split-fold-h100.json
```

The holder submission currently requires an Iris client newer than the Shuttle branch's base revision. The holder may therefore use a separate current-Iris sparse checkout. The generated benchmark itself must check out and record the exact Shuttle revision independently; the result must record both revisions.

The first H100 request was submitted at batch priority on 2026-08-08. It remained
`SchedulingGated` without a pod for roughly nine minutes and was terminated before
any GPU allocation. This is a zero-use capacity blocker, not a benchmark result.
The bounded fallback uses the same generated plan on one GB200 at batch priority:

```bash
export GPU_NAME="${USER}-shuttle-event-tensor-gb200"
uv run --package marin-iris --extra controller scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$GPU_NAME" allocate \
  --gpu-variant gb200 \
  --gpus-per-node 1 \
  --priority batch \
  --timeout 1800
```

For this fallback, record the SM100/CUDA/toolchain details independently from the
original H100 intent. Use `TORCH_CUDA_ARCH_LIST=10.0a` only if the installed Torch
toolchain accepts that architecture spelling; otherwise record the exact supported
SM100 spelling used by the build.

The GB200 fallback also remained `SchedulingGated` without a pod for roughly ten
minutes. Kueue reported that no GPU node fit the request. It was terminated before
any GPU allocation, so this is a second zero-use capacity blocker rather than an
SM100 benchmark result.

The command records raw repeated-run samples, execution order, generated-source hash, plan fingerprint, kernel resource attributes, toolchain, driver, clocks, and power telemetry. Correctness runs use two producer permutations and nonzero producer delays. Repeated fresh kernel invocations verify the declared `per_invocation` generation policy.

Release immediately after copying the JSON and generated CUDA source:

```bash
uv run --package marin-iris --extra controller scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$GPU_NAME" release
```

## Later phases

Phase 2 will replace one existing MoE readiness record with a plan derived from `RelationPlan` and task decomposition. DeepEP may remain only as transport. Runtime event counts and source/event offsets must be physical inputs derived from the current `RelationPlan`; compile-time constants do not establish this result. Zero-count segments become initially ready or are omitted by a generic empty-task policy. A GB200 run will compare the replaced edge against the current generic schedule with identical routing and compute.

Phase 3 will replace one attention producer/consumer readiness edge. Candidate edges are routed-KV staging readiness or an exposed QK-to-state/PV stage edge. The selected proof must use generated task relations and generic attention bodies; calling a named attention kernel does not count. A one-shot edge is only an intermediate smoke. The final proof must exercise `phased` generation with circular-buffer reuse, or conclude explicitly that producer-event-consumer factorization lacks the structure needed to express the pipeline safely.

## Bounded dynamic-device validation

Before integrating either edge into a workload path, the generic device prototype
compiles two generated CUDA modules:

1. A runtime segmented-readiness module. Event counts, event/source offsets, and
   source indices are device tensor inputs derived from the current
   `RelationPlan`. One CTA realizes each segment. An empty segment follows the
   generic initially-ready identity path and does not initialize a zero-count
   barrier.
2. A phased Contract/Fold/Contract module. Separate worker warps publish first-
   Contract and Fold-state generations through CTA-visible semaphores. The
   finalizer performs a source-ordered normalized weighted Fold and releases each
   bounded slot for the next generation. Slot storage is reused while logical
   identity remains `(slot, generation)`.

These are validation templates, not production kernels. The runtime segmented
module deliberately maps one event to one CTA rather than claiming cross-CTA
dynamic scheduling. The phased module uses a conservative
finalize-before-next-generation reuse edge, so it validates safe generations but
does not claim an optimally overlapped attention pipeline.

Run both with one bounded allocation:

```bash
export TORCH_CUDA_ARCH_LIST=<architecture accepted by installed Torch>
timeout 20m uv run python \
  lib/tile_lifetime/benchmarks/gpu_event_tensor_dynamic_validation.py \
  --shuttle-revision <dynamic-device-commit> \
  --holder-revision <iris-holder-commit> \
  --json-output /tmp/event-tensor-dynamic-device.json
```

The runtime mutation changes relation counts and offsets while reusing the same
compiled module. The phased mutation changes generation count and pipeline depth
through the same compiled module. Both paths check correctness and repeat hashes
before timing.
