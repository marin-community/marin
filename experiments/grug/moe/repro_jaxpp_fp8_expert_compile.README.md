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

The external dps8 BF16 minimum, FP8 minimum, and FP8 production expert shape
also passed. At L2/m4/e64/top-k4/t32768/h2560/i1280, rank 0 lowered in 1.819
seconds and returned from `eval_local` in 8.747 seconds; rank 1 lowered in 1.874
seconds and returned in 24.397 seconds. No watchdog fired. This rules out the
isolated expert backward, including production expert-axis width and FP8
overwrite state.

The `*_ring` modes add the smallest omitted production structure:

- a stage mesh named `(replica_dcn, data, expert, model)`, with all stage
  devices on `expert`;
- activations sharded as `P((replica_dcn, data, expert), None)` and expert
  weights sharded as `P(expert, None, None)`;
- the production `moe_mlp(..., implementation="ring")` `shard_map`, including
  `all_gather` dispatch and `psum_scatter` collection; and
- replicated FP8 overwrite state at the `shard_map` boundary, whose custom VJP
  performs the production stage-mesh `pmax` cotangent reduction.

`--loss-boundary next_token` adds the next production boundary without adding
attention or learned routing:

- sequence-shaped `[batch, sequence, hidden]` stage input and matching
  activation cotangent;
- the final RMSNorm and rank-128 gated norm parameters;
- shifted token labels, per-token loss weights, the replicated language-model
  head, and the production XLA fused linear cross-entropy path; and
- the complete last-stage `value_and_grad` result: scalar loss, dynamic
  `[layers, experts]` auxiliary output, the mixed expert/FP8/final/head
  parameter-gradient tree, and every microbatch input cotangent.

In fixed-routing mode, the auxiliary values are a minimal
activation-dependent stand-in for router betas. Nothing has been filed
upstream.

`--remat-mode save_moe` adds the exact non-effectful ring-MoE checkpoint
boundary used by `TransformerPipelineStage.run_block`: one
`eqx.filter_checkpoint` per block, default `prevent_cse=True`, with
`jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)`. The
ring backend already emits those production checkpoint names around dispatch,
expert hidden state, and dispatch output. `--remat-mode recompute_all` uses the
same boundary with no save policy. `--remat-mode none` remains the default so
all completed gates keep their previous lowering.

This matches the non-effectful branch selected by the production `ring`
backend. The separate DeepEP branch checkpoints attention and leaves the
effectful MoE outside remat, so it is intentionally not modeled here. Within
the checkpoint, this gate retains only the residual routed expert computation
and its auxiliary output. Attention, per-block RMS/gated norms, learned
routing, the shared expert, attention mask/static flags, optimizer state, and
the full pipeline schedule remain excluded.

`--routing-mode learned_qb` replaces each fixed expert layer with the actual
production `MoEMLP` and `MoEExpertMlp` modules. It adds the learned router and
the stage QB-beta input, applies the production centered negative QB bias,
computes FP32 router logits and probabilities, biased top-(K+1) selection,
sigmoid/renormalized top-k combine weights, QB beta and router statistics, and
then enters the existing 1.25-capacity ring dispatch. The ring implementation
owns assignment sorting, capacity clipping, group-size construction,
collectives, and FP8 ragged-dot state. Router z-loss is computed as part of the
actual statistics path but has coefficient zero, matching the production
launcher.

The learned-QB dps8 A/B also passed. BF16 job
`/dlwh/jaxpp-routing-dps8-bf16-20260722-221754` completed both tasks in 96.92
seconds per task including setup. FP8 job
`/dlwh/jaxpp-routing-dps8-fp8-20260722-222555` completed both tasks in 122.42
seconds per task including setup. Neither fired a watchdog or failed a task;
Finelog did not return the per-phase events. This rules out the complete
learned QB router and production ring dispatch in combination with the already
completed remat and loss boundaries.

`--block-boundary full` adds the next localized production boundary. Parameters
are an actual final `TransformerPipelineStage`, and its `block_range`,
`finalize_hidden`, and `hidden_next_token_loss` methods execute inside the same
`value_and_grad` task. Each production `Block` includes:

- attention RMSNorm and GatedNorm, Q/K/V projections, half-RoPE, XSA, head
  gating, output projection, and the attention residual;
- pre-MoE RMSNorm and GatedNorm, learned-QB `MoEMLP`, and the MoE residual;
- the model's production 1.0 routed-expert capacity factor (the earlier
  standalone routing gate retains its original 1.25-capacity control);
- the existing production `save_moe` block remat and complete router metrics;
  and
- final RMS/GatedNorm, the LM head, fused loss, parameter gradients, and stage
  input cotangent.

For the matched two-layer last stage, `--total-layers 8` makes the local
blocks global layers 6 and 7. Layer 6 uses the production 2048-token sliding
window; final layer 7 is full causal and disables long-layer RoPE. The target
geometry is 20 query heads, 5 KV heads, and head dimension 128. H100 uses
`gpu_fa4_cute`, the CuTe FA4 backend that supports this sliding-window
configuration. CPU gates select `reference` through the same attention API.
The shared expert remains excluded by setting its independently configurable
intermediate dimension to zero.

## Full-block CuTe FA4 dps8 gate

These commands add the production attention/norm/residual block to the
completed learned-routing boundary. They also install the patched multi-device
JAX TVM FFI dependency used by the production JaxPP launcher and verify the
CuTe/FA4 imports before starting the reproducer.

BF16 expert control:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-full-block-dps8-bf16-${STAMP}" \
  -- bash -c '
    set -euxo pipefail
    cd "$IRIS_WORKDIR"
    command -v ptxas
    command -v nvlink
    uv pip install --link-mode=symlink cupy-cuda13x
    rm -rf /tmp/jax-tvm-ffi
    git clone --quiet --filter=blob:none https://github.com/NVIDIA/jax-tvm-ffi.git /tmp/jax-tvm-ffi
    git -C /tmp/jax-tvm-ffi checkout --quiet e238a28483123efc8f56b9de358c2fb8b8de77e5
    git -C /tmp/jax-tvm-ffi apply "$IRIS_WORKDIR/experiments/grug/moe/jax_tvm_ffi_multidevice.patch"
    uv pip install --link-mode=symlink --force-reinstall --no-deps /tmp/jax-tvm-ffi
    uv pip install --link-mode=symlink --no-deps \
      "jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
    .venv/bin/python -c "import cutlass.cute, cutlass.jax, flash_attn.cute.flash_bwd_sm90"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
    .venv/bin/python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --worker-mode external --runtime jaxpp --kernel bf16_ring \
      --loss-boundary next_token --remat-mode save_moe --routing-mode learned_qb \
      --block-boundary full --attention-implementation gpu_fa4_cute \
      --num-heads 20 --num-kv-heads 5 --total-layers 8 --sliding-window 2048 \
      --devices-per-stage 8 --layers 2 --microbatches 4 \
      --experts 64 --top-k 4 --tokens 32768 --sequence-length 4096 \
      --vocab-size 8192 --hidden 2560 --intermediate 1280 --amax-history 1024 \
      --timeout 1200 --stack-after 120 --coordinator-port 5793 \
      --dump-dir /tmp/jaxpp-full-block/dps8-bf16
  '
```

FP8 expert candidate:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-full-block-dps8-fp8-${STAMP}" \
  -- bash -c '
    set -euxo pipefail
    cd "$IRIS_WORKDIR"
    command -v ptxas
    command -v nvlink
    uv pip install --link-mode=symlink cupy-cuda13x
    rm -rf /tmp/jax-tvm-ffi
    git clone --quiet --filter=blob:none https://github.com/NVIDIA/jax-tvm-ffi.git /tmp/jax-tvm-ffi
    git -C /tmp/jax-tvm-ffi checkout --quiet e238a28483123efc8f56b9de358c2fb8b8de77e5
    git -C /tmp/jax-tvm-ffi apply "$IRIS_WORKDIR/experiments/grug/moe/jax_tvm_ffi_multidevice.patch"
    uv pip install --link-mode=symlink --force-reinstall --no-deps /tmp/jax-tvm-ffi
    uv pip install --link-mode=symlink --no-deps \
      "jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
    .venv/bin/python -c "import cutlass.cute, cutlass.jax, flash_attn.cute.flash_bwd_sm90"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
    .venv/bin/python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --worker-mode external --runtime jaxpp --kernel fp8_ring \
      --loss-boundary next_token --remat-mode save_moe --routing-mode learned_qb \
      --block-boundary full --attention-implementation gpu_fa4_cute \
      --num-heads 20 --num-kv-heads 5 --total-layers 8 --sliding-window 2048 \
      --devices-per-stage 8 --layers 2 --microbatches 4 \
      --experts 64 --top-k 4 --tokens 32768 --sequence-length 4096 \
      --vocab-size 8192 --hidden 2560 --intermediate 1280 --amax-history 1024 \
      --timeout 1200 --stack-after 120 --coordinator-port 5793 \
      --dump-dir /tmp/jaxpp-full-block/dps8-fp8
  '
```

Optimizer state, Sonic materialization, incoming/outgoing pipeline transfers,
gradient accumulation across stages, and the outer 1F1B scheduler remain
excluded. The shared expert is not inseparable from `Block`; it is deliberately
disabled for this gate.

## Dynamic QB-routing dps8 gate

These matched jobs add only `learned_qb` routing to the completed next-token
and `save_moe` boundary. Each invocation creates a fresh Iris job name.

BF16 control:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-routing-dps8-bf16-${STAMP}" \
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
      --loss-boundary next_token --remat-mode save_moe \
      --routing-mode learned_qb --devices-per-stage 8 \
      --layers 2 --microbatches 4 --experts 64 --top-k 4 \
      --tokens 32768 --sequence-length 4096 --vocab-size 8192 \
      --hidden 2560 --intermediate 1280 --amax-history 1024 \
      --timeout 1200 --stack-after 120 --coordinator-port 5793 \
      --dump-dir /tmp/jaxpp-routing/dps8-bf16
  '
```

FP8 candidate:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-routing-dps8-fp8-${STAMP}" \
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
      --loss-boundary next_token --remat-mode save_moe \
      --routing-mode learned_qb --devices-per-stage 8 \
      --layers 2 --microbatches 4 --experts 64 --top-k 4 \
      --tokens 32768 --sequence-length 4096 --vocab-size 8192 \
      --hidden 2560 --intermediate 1280 --amax-history 1024 \
      --timeout 1200 --stack-after 120 --coordinator-port 5793 \
      --dump-dir /tmp/jaxpp-routing/dps8-fp8
  '
```

Attention, attention-side norms/gates, the pre-MoE RMS/gated norm, the shared
expert, optimizer state, and the full pipeline schedule remain excluded.

## Completed block-remat dps8 gate

Both `save_moe` jobs passed without watchdog, OOM, or error. BF16 job
`/dlwh/jaxpp-remat-dps8-bf16-20260722-215133` returned lower/`eval_local`
times of 0.972/5.985 seconds on rank 0 and 0.985/16.984 seconds on rank 1. FP8
job `/dlwh/jaxpp-remat-dps8-fp8-20260722-215622` returned 2.842/10.440
seconds on rank 0; rank 1 lowered in 2.837 seconds and returned from
`eval_local` in less than 30.474 seconds, but Finelog lost the exact return
event. The successful parent and tasks bound the missing time. This rules out
the isolated production `save_moe` policy around the fixed routed expert path.

The commands below retain the completed configuration for reference.

BF16 control:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-remat-dps8-bf16-${STAMP}" \
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
      --loss-boundary next_token --remat-mode save_moe \
      --devices-per-stage 8 --layers 2 --microbatches 4 \
      --experts 64 --top-k 4 --tokens 32768 --sequence-length 4096 \
      --vocab-size 8192 --hidden 2560 --intermediate 1280 \
      --amax-history 1024 --timeout 1200 --stack-after 120 \
      --coordinator-port 5793 --dump-dir /tmp/jaxpp-remat/dps8-bf16
  '
```

FP8 candidate:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-remat-dps8-fp8-${STAMP}" \
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
      --loss-boundary next_token --remat-mode save_moe \
      --devices-per-stage 8 --layers 2 --microbatches 4 \
      --experts 64 --top-k 4 --tokens 32768 --sequence-length 4096 \
      --vocab-size 8192 --hidden 2560 --intermediate 1280 \
      --amax-history 1024 --timeout 1200 --stack-after 120 \
      --coordinator-port 5793 --dump-dir /tmp/jaxpp-remat/dps8-fp8
  '
```

A separate `recompute_all` control remains available by using a fresh
`jaxpp-remat-all-dps8-fp8-${STAMP}` name and replacing `--remat-mode save_moe`
with `--remat-mode recompute_all`.

## Completed next-token dps8 gate

Both matched next-token jobs passed without watchdog, OOM, or error. BF16 job
`/dlwh/jaxpp-last-stage-dps8-bf16-20260722-213810` returned lower/`eval_local`
times of 0.532/6.005 seconds on rank 0 and 0.538/17.552 seconds on rank 1. FP8
job `/dlwh/jaxpp-last-stage-dps8-fp8-20260722-214009` returned 1.920/10.940
seconds on rank 0 and 1.922/30.621 seconds on rank 1. This rules out final
norms, the LM head, fused next-token loss, and the complete task result tree in
combination with the isolated FP8 expert backward.

The commands below retain the completed configuration for reference.

BF16 control:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-last-stage-dps8-bf16-${STAMP}" \
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
      --loss-boundary next_token --devices-per-stage 8 \
      --layers 2 --microbatches 4 --experts 64 --top-k 4 \
      --tokens 32768 --sequence-length 4096 --vocab-size 8192 \
      --hidden 2560 --intermediate 1280 --amax-history 1024 \
      --timeout 1200 --stack-after 120 --coordinator-port 5793 \
      --dump-dir /tmp/jaxpp-last-stage/dps8-bf16
  '
```

FP8 candidate:

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --replicas 2 --gpu=H100x8 \
  --cpu=32 --memory=256G --disk=256G --extra=gpu --timeout=2400 \
  --job-name="jaxpp-last-stage-dps8-fp8-${STAMP}" \
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
      --loss-boundary next_token --devices-per-stage 8 \
      --layers 2 --microbatches 4 --experts 64 --top-k 4 \
      --tokens 32768 --sequence-length 4096 --vocab-size 8192 \
      --hidden 2560 --intermediate 1280 --amax-history 1024 \
      --timeout 1200 --stack-after 120 --coordinator-port 5793 \
      --dump-dir /tmp/jaxpp-last-stage/dps8-fp8
  '
```

For a matched distributed-JAX control, use the FP8 command with a fresh
`jaxpp-last-stage-dps8-fp8-direct-${STAMP}` job name and replace `--runtime
jaxpp` with `--runtime distributed_direct`.

## Completed dps8 expert-only gate

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
- The completed expert-only, next-token, and block-remat dps8 cases pass at the
  reduced production shape. The next isolated candidate is learned QB routing
  inside the `save_moe` block. If it also passes, the minimum missing structure
  remains in the omitted per-block norms/shared expert, attention, their
  interaction with routing, or the full pipeline schedule.

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
