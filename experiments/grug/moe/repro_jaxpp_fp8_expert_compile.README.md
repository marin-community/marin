# JaxPP FP8 expert-backward compile reproducer

This bounded reproducer isolates the compile stall seen in
`grug_1f1b_mb0_stage3_loss_backward`. The original one-device-per-stage
reproducer retained the FP8 expert GEMMs, delayed-scaling overwrite state,
`value_and_grad`, and microbatch accumulation, but it did not reproduce. Every
ramp through L2/m4/e8/t65536/h2560/i1280 passed; at the largest shape, direct
and distributed-direct compiled in about 5.3 seconds and JaxPP compiled and
executed in about 18.9 seconds.

The later ring gates also pass with two and four devices per stage. Eight
devices per stage needs two H100x8 Iris tasks, so `--worker-mode external`
runs one process per task and joins them through either Iris job-info discovery
or a complete JAX distributed environment. Local worker mode remains the
default for the single-task dps2/dps4 gates.

The `*_ring` modes add the smallest omitted production structure:

- a stage mesh named `(replica_dcn, data, expert, model)`, with all stage
  devices on `expert`;
- activations sharded as `P((replica_dcn, data, expert), None)` and expert
  weights sharded as `P(expert, None, None)`;
- the production `moe_mlp(..., implementation="ring")` `shard_map`, including
  `all_gather` dispatch and `psum_scatter` collection; and
- replicated FP8 overwrite state at the `shard_map` boundary, whose custom VJP
  performs the production stage-mesh `pmax` cotangent reduction.

Attention, learned routing, optimizer state, the language-model head, and the
pipeline scheduler remain excluded. Nothing has been filed upstream.

## Dps8 external-worker gate

Run the BF16 and FP8 gates as separate jobs. Each command generates a fresh
name, requests two gang-scheduled H100x8 tasks, and runs one JAX process with
all eight local devices per task. Iris job info supplies rank and coordinator
discovery; if all three of `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, and
`JAX_PROCESS_ID` are present, the script uses those values instead.

BF16 JaxPP ring control:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-fp8-ring-dps8-bf16-${STAMP}" \
  -- bash -c '
    set -euxo pipefail
    command -v ptxas
    command -v nvlink
    uv pip install --link-mode=symlink cupy-cuda13x
    uv pip install --link-mode=symlink --no-deps \
      "jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
    .venv/bin/python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --worker-mode external --runtime jaxpp --kernel bf16_ring \
      --devices-per-stage 8 --experts 8 --top-k 4 --tokens 256 \
      --hidden 128 --intermediate 128 --layers 1 --microbatches 1 \
      --amax-history 1024 --timeout 900 --stack-after 120 \
      --coordinator-port 5793 --dump-dir /tmp/jaxpp-fp8-ring/dps8-bf16
  '
```

FP8 JaxPP ring candidate:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-fp8-ring-dps8-fp8-${STAMP}" \
  -- bash -c '
    set -euxo pipefail
    command -v ptxas
    command -v nvlink
    uv pip install --link-mode=symlink cupy-cuda13x
    uv pip install --link-mode=symlink --no-deps \
      "jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
    .venv/bin/python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --worker-mode external --runtime jaxpp --kernel fp8_ring \
      --devices-per-stage 8 --experts 8 --top-k 4 --tokens 256 \
      --hidden 128 --intermediate 128 --layers 1 --microbatches 1 \
      --amax-history 1024 --timeout 900 --stack-after 120 \
      --coordinator-port 5793 --dump-dir /tmp/jaxpp-fp8-ring/dps8-fp8
  '
```

Production BF16 ring and direct FP8 evidence already isolate those controls.
If a matched multi-host direct control is needed, submit the FP8 command with a
fresh `jaxpp-fp8-ring-dps8-fp8-direct-${STAMP}` name and replace
`--runtime jaxpp` with `--runtime distributed_direct`. Rank 1 performs the
ordinary sharded JAX backward while rank 0 waits at the completion barrier.

Both the Iris timeout and the per-task watchdog are bounded. A worker emits
periodic Python stacks after 120 seconds and exits 124 after 900 seconds; a
peer left in distributed initialization or a barrier is bounded independently
by its own watchdog.

## Completed dps2 allocation

This is the first useful gate: two devices per stage and four H100s total. It
runs a distributed BF16 ring control, direct and distributed-direct FP8 ring
controls, a JaxPP BF16 ring control, and the matched JaxPP FP8 ring case. The
production FP8 history length is retained because its state is part of the
suspect custom VJP.

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --gpu=H100x4 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=3600 \
  --job-name=jaxpp-fp8-ring-compile-dps2 \
  -- bash -c '
    set -euxo pipefail
    command -v ptxas
    command -v nvlink
    uv pip install --link-mode=symlink cupy-cuda13x
    uv pip install --link-mode=symlink --no-deps \
      "jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
    SCRIPT=experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py
    COMMON="--devices-per-stage 2 --experts 4 --top-k 4 --tokens 128 \
      --hidden 128 --intermediate 128 --layers 1 --microbatches 1 \
      --amax-history 1024 --timeout 900 --stack-after 120"

    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u "$SCRIPT" \
      --runtime distributed_direct --kernel bf16_ring $COMMON \
      --coordinator-port 5791 --dump-dir /tmp/jaxpp-fp8-ring/bf16-distributed
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u "$SCRIPT" \
      --runtime direct --kernel fp8_ring $COMMON \
      --dump-dir /tmp/jaxpp-fp8-ring/fp8-direct
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u "$SCRIPT" \
      --runtime distributed_direct --kernel fp8_ring $COMMON \
      --coordinator-port 5792 --dump-dir /tmp/jaxpp-fp8-ring/fp8-distributed
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u "$SCRIPT" \
      --runtime jaxpp --kernel bf16_ring $COMMON \
      --coordinator-port 5793 --dump-dir /tmp/jaxpp-fp8-ring/bf16-jaxpp
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u "$SCRIPT" \
      --runtime jaxpp --kernel fp8_ring $COMMON \
      --coordinator-port 5794 --dump-dir /tmp/jaxpp-fp8-ring/fp8-jaxpp
  '
```

The command intentionally does not submit unless run by the parent
investigation owner. Use a non-login worker shell: `bash -l` can replace Iris's
staged CUDA toolchain path.

## Topology-first ramp

The dps2 and dps4 ring gates pass. The dps4 shape retained one local expert and
the minimum 128 FP8 assignments per device:

```text
--devices-per-stage 4 --experts 4 --top-k 4 --tokens 128
```

The next 16-GPU gate uses the external worker mode and eight devices per stage,
matching production's expert-axis width:

```text
--devices-per-stage 8 --experts 8 --top-k 4 --tokens 256
```

In local mode, direct sees one stage's devices while distributed-direct and
JaxPP see twice that number. In external mode, each Iris task sees one stage's
devices and the distributed mesh combines both tasks.

Only if the dps8 minimum passes, restore the reduced production stage-3 shape:

```text
--devices-per-stage 8 --layers 2 --microbatches 4 \
--experts 64 --top-k 4 --tokens 32768 \
--hidden 2560 --intermediate 1280 --amax-history 1024
```

Here `tokens=32768` is one b32/m4, sequence-4096 pipeline microbatch. Each
expert shard receives capacity for 16384 routed assignments, matching the
production ring capacity calculation.

## Interpretation

- Distributed-direct BF16 ring and JaxPP BF16 ring pass, direct and
  distributed-direct FP8 ring pass, but JaxPP FP8 ring stalls: the essential
  delta is JaxPP localization/`apply_task` around the FP8 ring custom VJP.
- JaxPP BF16 ring stalls: the generic JaxPP plus `shard_map` collective graph
  is sufficient; FP8 state is not required.
- Distributed-direct BF16 ring passes but distributed-direct FP8 ring stalls:
  the issue is below JaxPP in distributed JAX/XLA compilation of the FP8 ring
  custom VJP.
- Direct FP8 ring stalls: the issue is stage-local and below distributed setup.
- All dps8 cases pass at the reduced production shape: this isolated expert
  backward is still insufficient; the next candidates are full block remat,
  the stage-3 language-model head/loss, or the complete task output tree.

`--stop-after lower` proves Python tracing and JaxPP MPMD localization without
entering XLA compilation. The full direct paths log separate lower, compile,
and execute events. JaxPP's `eval_local` combines compile and execute for each
localized `apply_task`, so the corresponding event is
`jaxpp_eval_local_compile_execute_entered`. The watchdog emits periodic Python
stacks, records `watchdog_timeout`, and exits with status 124.

## Runtime pins

The production observation used JAX/JAXLIB 0.10.1 and NVIDIA/JaxPP revision
`7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`. The FP8 kernel uses E4M3 in both
directions because mixed E4M3/E5M2 Mosaic WGMMA requires JAX 0.11 or newer. A
CUDA toolchain containing `ptxas` and `nvlink` must be on `PATH`.
