# MoE MGPU H100 Tryout Snapshot

This snapshot exposes `implementation="pallas_mgpu"` through the Grug MoE scale
launcher and adds `SCALE_MOE_CAPACITY_FACTOR` so full trainer runs can use the
same padded-capacity path validated by the H100 kernel tests.

## Known Good Evidence

- Public Grug MoE Pallas forward/grad validation:
  `/dlwh/iris-run-test_grugformer_moe-20260629-topology-hardening-public-refresh`
  passed `3 passed, 107 deselected, 1 warning in 218.99s`.
- Broader Hopper Pallas test slice:
  `/dlwh/iris-run-test_grugformer_moe-20260629-lint-cleanup-broad-plus-chunked`
  passed `11 passed, 98 deselected, 1 warning`.
- Module-boundary training-step smoke:
  `/dlwh/iris-run-test_grugformer_moe-20260629-module-training-step`
  passed `1 passed, 110 deselected, 1 warning in 63.41s`.
- Target forward+backward benchmark:
  `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-fwd-bwd-lint-cleanup`
  reported `steady_state_time=0.069388s`, `139.27 TFLOP/s/rank`,
  `14.08%` of nominal H100 bf16 roofline, and zero dropped routes.
- Current-commit one-node full-trainer target-shape smoke
  `/dlwh/grug-moe-pallas-mgpu-20step-current-20260629-150755` completed
  `20/20` steps from code-bearing commit `14fd25a73` on one 8xH100 node with
  `SCALE_WATCH_TARGETS=`, saved checkpoint `step-20`, and reported final MFU
  `20.19%`, mean MFU `20.11%`, and about `553k` tokens/s. Later branch commits
  through `50074c33b` only refreshed logbook/runbook/readiness artifacts, so
  this remains the current best full-trainer tryout recipe for the branch.
- Earlier one-node full-trainer target-shape smoke
  `/dlwh/grug-moe-pallas-mgpu-20step-smoke-2d87348e7` completed `20/20` steps
  on one 8xH100 node with `SCALE_WATCH_TARGETS=`, saved checkpoint `step-20`,
  and reported mean MFU `20.13%`, p50 MFU `20.15%`, and about `554k` tokens/s.
- Earlier one-node smoke `/dlwh/grug-moe-pallas-mgpu-20step-smoke-42ba9b7d4`
  reached `global_step=9/20`, then OOMed during the default step-10
  watch/per-parameter-norm path. Keep `SCALE_WATCH_TARGETS=` empty for
  throughput/full-run smokes unless diagnostics are explicitly needed.
- A 32-node 20-step attempt
  `/dlwh/grug-moe-pallas-mgpu-20step-scale-fe788a6e5-20260629-134609` was
  externally killed while the 32 child tasks were still building, before any
  training metrics appeared. It is not evidence about Pallas MGPU scale
  correctness or performance.
- A corrected 4-node target-shape attempt
  `/dlwh/grug-moe-pallas-mgpu-target-20step-r4-fixedaxis-20260629-143835` set
  `SCALE_REPLICA_AXIS=4` and got past the earlier Pallas MGPU data-axis guard,
  but failed before any train step completed because the multi-node distributed
  runtime could not load NVSHMEM. This does not contradict the single-node
  NVLink EP target in the spec, but it means the current multi-node tryout
  recipe is not known-good.
- A later B-tiled CE R=N scale sweep at the larger default 90B-ish shape
  (`hidden_dim=3072`, `num_layers=48`, `num_experts=128`) was killed after n4,
  n8, and n16 jobs hit pre-step GPU OOM while allocating 576 MiB during
  `int(state.step)`, before any train-step metrics. Do not relaunch that larger
  recipe unchanged; reduce model or optimizer-state memory first.

## Recommended 20-Step Integration Smoke

This is the first run to try from the snapshot. It uses one 8xH100 node, keeps
the full trainer path, and sets the local MoE token shape to the benchmark
target (`batch=128`, `seq=2048`, `8` expert-parallel ranks gives
`32768` tokens/rank). It uses two layers to keep the run cheap.

```bash
RUN_ID="grug-moe-pallas-mgpu-20step-smoke-$(date +%Y%m%d-%H%M%S)"
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
  --job-name "$RUN_ID" \
  --cpu=2 --memory=2G --disk=8G --extra=cpu \
  -- env \
    RUN_ID="$RUN_ID" \
    SCALE_GPU_REPLICAS=1 \
    SCALE_EXPERT_AXIS=8 \
    SCALE_REPLICA_AXIS=1 \
    SCALE_BATCH=128 \
    SCALE_SEQ_LEN=2048 \
    SCALE_STEPS=20 \
    SCALE_HIDDEN_DIM=2560 \
    SCALE_NUM_LAYERS=2 \
    SCALE_NUM_EXPERTS=256 \
    SCALE_TOP_K=4 \
    SCALE_MOE_IMPLEMENTATION=pallas_mgpu \
    SCALE_MOE_CAPACITY_FACTOR=1.25 \
    SCALE_REMAT=save_moe \
    SCALE_CHECKPOINTS=local \
    SCALE_TRACKER=json_logger \
    SCALE_WATCH_TARGETS= \
    uv run python -m experiments.grug.moe.launch_cw_scale
```

Use this as the pass/fail gate before trying the full 32-node shape. It passed
with run id `grug-moe-pallas-mgpu-20step-current-20260629-150755` on
code-bearing commit `14fd25a73`. Later branch commits through `50074c33b` only
updated project notes/logbooks; rerun this recipe if code changes behavior or
if you need a fresh trainer smoke.

## Experimental Multi-Node 20-Step Smoke

The target Pallas MGPU backend is single-node / single-NVLink-domain EP, so the
one-node smoke above is the current best integration gate. The command below is
only for checking that the full trainer can replicate that single-node EP group
across multiple nodes. It is not known-good as of
`/dlwh/grug-moe-pallas-mgpu-target-20step-r4-fixedaxis-20260629-143835`: the
corrected r4 run passed the Pallas MGPU data-axis guard but failed pre-step on a
missing NVSHMEM load in the multi-node distributed runtime.

The launcher has an experimental `SCALE_GPUS_PER_TASK` knob for decomposing the
logical GPU allocation into smaller Python processes. The current validated
Pallas MGPU path leaves it omitted, which defaults to `8` and keeps the whole
expert-parallel group local to one process. `SCALE_MOE_IMPLEMENTATION=pallas_mgpu`
now fails fast if `SCALE_GPUS_PER_TASK < SCALE_EXPERT_AXIS`, because the backend
requires all expert-parallel ranks to be visible local GPUs.

Do not relaunch this exact recipe unchanged unless the runtime image or JAX
distributed setup is known to provide the required NVSHMEM bits. If you do
retest it, set `SCALE_GPU_REPLICAS` to `4`, `8`, `16`, then `32` as capacity
allows; the command sets global batch to `128 * SCALE_GPU_REPLICAS` so each
device sees the same local token count as the passing one-node run. Keep
`SCALE_REPLICA_AXIS=$SCALE_GPU_REPLICAS` so the cross-node data mesh axis stays
size `1`, which the current Pallas MGPU backend requires.

Use the executor entrypoint shown below. Do not directly import
`scale_moe_step.config` and call `run_grug_moe_trial` from a Python heredoc: that
leaves `VersionedValue` fields unresolved and fails before trainer startup.

```bash
SCALE_GPU_REPLICAS=4
SCALE_BATCH=$((128 * SCALE_GPU_REPLICAS))
RUN_ID="grug-moe-pallas-mgpu-target-20step-r${SCALE_GPU_REPLICAS}-$(date +%Y%m%d-%H%M%S)"
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
  --job-name "$RUN_ID" \
  --cpu=2 --memory=2G --disk=8G --extra=cpu \
  -- env \
    RUN_ID="$RUN_ID" \
    SCALE_GPU_REPLICAS="$SCALE_GPU_REPLICAS" \
    SCALE_EXPERT_AXIS=8 \
    SCALE_REPLICA_AXIS="$SCALE_GPU_REPLICAS" \
    SCALE_BATCH="$SCALE_BATCH" \
    SCALE_SEQ_LEN=2048 \
    SCALE_STEPS=20 \
    SCALE_HIDDEN_DIM=2560 \
    SCALE_NUM_LAYERS=2 \
    SCALE_NUM_EXPERTS=256 \
    SCALE_TOP_K=4 \
    SCALE_MOE_IMPLEMENTATION=pallas_mgpu \
    SCALE_MOE_CAPACITY_FACTOR=1.25 \
    SCALE_REMAT=save_moe \
    SCALE_CHECKPOINTS=local \
    SCALE_TRACKER=json_logger \
    SCALE_WATCH_TARGETS= \
    uv run python -m experiments.grug.moe.launch_cw_scale
```

Babysit each submitted run to a terminal state. If this target-shape scaling
recipe OOMs before training, first try `SCALE_REMAT=recompute_all`; if parameter
or optimizer-state memory is still the blocker, try
`SCALE_MP=params=bfloat16,compute=bfloat16,output=bfloat16` and record that the
precision policy changed from the validated one-node evidence.

## Full-Scale 20-Step Run

This keeps the scale launcher's default 90B-total shape but switches the MoE
backend to Pallas MGPU and limits training to 20 steps. The current larger-shape
scale evidence is not clean: one 32-node attempt was killed while child tasks
were building, and a later R=N B-tiled sweep OOMed before the train iterator
started. Treat the command below as a template, not a known-good recipe; reduce
model or optimizer-state memory before relaunching and babysit it to a terminal
result.

```bash
RUN_ID="grug-moe-pallas-mgpu-20step-scale-$(date +%Y%m%d-%H%M%S)"
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
  --job-name "$RUN_ID" \
  --cpu=2 --memory=2G --disk=8G --extra=cpu \
  -- env \
    RUN_ID="$RUN_ID" \
    SCALE_GPU_REPLICAS=32 \
    SCALE_EXPERT_AXIS=8 \
    SCALE_REPLICA_AXIS=1 \
    SCALE_STEPS=20 \
    SCALE_MOE_IMPLEMENTATION=pallas_mgpu \
    SCALE_MOE_CAPACITY_FACTOR=1.25 \
    SCALE_REMAT=save_moe \
    SCALE_CHECKPOINTS=local \
    SCALE_TRACKER=json_logger \
    SCALE_WATCH_TARGETS= \
    uv run python -m experiments.grug.moe.launch_cw_scale
```

Switch `SCALE_TRACKER=wandb` and pass `WANDB_API_KEY` through the Iris job when
you want W&B metrics/artifacts instead of JSON logs.

## Tuning Knobs

- `SCALE_MOE_CAPACITY_FACTOR=1.25` is the robust setting from the target H100
  parity/perf runs. `1.125` was faster in balanced target benchmarks, but only
  use it after checking route drops in the run metrics.
- `SCALE_REMAT=save_moe` keeps MoE dispatch tensors for backward and avoids
  rerunning EP dispatch during recompute. Use `recompute_all` if memory is the
  blocker.
- `SCALE_BATCH` controls local tokens/rank. For one 8-GPU node at seq 2048,
  `SCALE_BATCH=128` gives `32768` tokens/rank; `64` gives `16384`.
- `SCALE_NUM_LAYERS` is the cheapest smoke knob. Use `2` for integration,
  `4` or `8` if compile/runtime looks stable, and the default `48` for the
  production-size run.
- `SCALE_MP=params=bfloat16,compute=bfloat16,output=bfloat16` can reduce
  FSDP parameter traffic, but the validated benchmark evidence used the default
  float32 parameter policy.
- `SCALE_WATCH_TARGETS` defaults to empty in the CoreWeave scale launcher.
  Leave it empty for throughput/full-run smoke tests. To opt in, set e.g.
  `SCALE_WATCH_TARGETS=grads,params`; keep
  `SCALE_WATCH_PER_PARAMETER_NORMS=false` unless the run has ample memory.
- `SCALE_PROFILER_STEPS=N SCALE_PROFILER_START=K` enables a JAX profile window.
  Pair this with `SCALE_TRACKER=wandb` so the profile uploads.

## Benchmark Rechecks

Target forward+backward benchmark:

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
  --job-name "bench-grug-moe-pallas-mgpu-target-fwd-bwd-$(date +%Y%m%d-%H%M%S)" \
  --cpu=16 --memory=128GB --disk=16GB --gpu=H100x8 --reserve=H100x8 \
  --enable-extra-resources --extra=gpu \
  -- uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py \
    --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 \
    --experts-per-rank 32 --topk 4 --capacity-factor 1.25 \
    --implementations pallas_mgpu --pass-mode forward_backward \
    --routing uniform --warmup 1 --steps 3 --fail-on-error \
    --git-sha "$(git rev-parse --short HEAD)" \
    --jsonl /tmp/moe_mgpu_target_fwd_bwd.jsonl
```

Backward stage breakdown:

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
  --job-name "bench-grug-moe-pallas-mgpu-bwd-stages-$(date +%Y%m%d-%H%M%S)" \
  --cpu=16 --memory=128GB --disk=16GB --gpu=H100x8 --reserve=H100x8 \
  --enable-extra-resources --extra=gpu \
  -- uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py \
    --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 \
    --experts-per-rank 32 --topk 4 --capacity-factor 1.25 \
    --implementations none --include-pallas-stages \
    --pallas-stages saved_backward_pipeline combine_bwd w2_bwd w13_bwd dx_unpermute_vector \
    --routing uniform --warmup 1 --steps 3 --fail-on-error \
    --git-sha "$(git rev-parse --short HEAD)" \
    --jsonl /tmp/moe_mgpu_bwd_stages.jsonl
```

Forward chunked `permute_up` remains an opt-in benchmark lane, not the default
production launcher path:

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait \
  --job-name "bench-grug-moe-pallas-mgpu-permute-up-chunked-$(date +%Y%m%d-%H%M%S)" \
  --cpu=16 --memory=128GB --disk=16GB --gpu=H100x8 --reserve=H100x8 \
  --enable-extra-resources --extra=gpu \
  -- uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py \
    --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 \
    --experts-per-rank 32 --topk 4 --capacity-factor 1.25 \
    --implementations none --include-pallas-stages --pallas-stages permute_up_compare \
    --routing balanced --dispatch-chunked-permute-up \
    --dispatch-chunk-copy-tile 256 --dispatch-chunk-copy-rows 1 \
    --warmup 1 --steps 3 --fail-on-error \
    --git-sha "$(git rev-parse --short HEAD)" \
    --jsonl /tmp/moe_mgpu_permute_up_chunked.jsonl
```
