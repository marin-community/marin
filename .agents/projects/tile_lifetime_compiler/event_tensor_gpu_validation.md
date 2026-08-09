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
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$GPU_NAME" allocate --gpu-count 1 --timeout 1800
```

After connecting, synchronize the exact branch revision and enable GPU dependencies:

```bash
uv run scripts/iris/dev_gpu.py \
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

The command records raw repeated-run samples, execution order, generated-source hash, plan fingerprint, kernel resource attributes, toolchain, driver, clocks, and power telemetry. Correctness runs use two producer permutations and nonzero producer delays. Repeated fresh kernel invocations verify the declared `per_invocation` generation policy.

Release immediately after copying the JSON and generated CUDA source:

```bash
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$GPU_NAME" release
```

## Later phases

Phase 2 will replace one existing MoE readiness record with a plan derived from `RelationPlan` and task decomposition. DeepEP may remain only as transport. A GB200 run will compare the replaced edge against the current generic schedule with identical routing and compute.

Phase 3 will replace one attention producer/consumer readiness edge. Candidate edges are routed-KV staging readiness or an exposed QK-to-state/PV stage edge. The selected proof must use generated task relations and generic attention bodies; calling a named attention kernel does not count.
