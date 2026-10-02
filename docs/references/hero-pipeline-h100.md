# Hero pipeline on H100

The Hero pipeline runner supports a hardware adapter for the current Hero model
and optimizer configuration and a fresh Harrier data stream. The prepared
H100 gates use two layers at width 6144 with 384 experts, sequence length 4096,
PP2/EP8, FP32 parameters, BF16 compute, and scaled MuonH. The combined runner requires new bounded hardware gates. Results from the
separate H100 and GB200 launch snapshots do not validate this integration checkout.

These worker commands require the integration branch `codex/hero-pipeline-consolidation`,
based on main `89ac0d7705`, with the Hero pipeline and checkpoint changes.
Record `git rev-parse HEAD` before installing; main alone does not contain this
runner and runtime overlay.

`--main-hero-recipe` uses `HERO_MODEL_CONFIG` and `MoeHeuristic` from
`experiments/grug/moe_hero_ep`. H100 selects `gpu_fa4_cute` attention and
`fixed_pooled_wave_all_to_all` expert transport with six waves. Current-main
ragged transport requires the patched ARM PJRT build and SM100 expert kernels;
it cannot run on this x86 H100 stack. The adapter retains router/gate weight
decay 0.02 and z-loss 1e-4, and disables SYRK on H100.

## Runtime and startup

The prepared adapter runtime uses JAX/JAXlib and stock x86 CUDA 13 PJRT/plugin
0.11.1, FA4 4.0.0b28, Cutlass DSL 4.6.2, Quack 0.6.4, and CUDA Torch
2.11.0+cu128. JAXPP is pinned to
`328f75a80cecf22c7cc030a82d8941d3c1e220b6` with the checked-in
`experiments/grug/moe_hero_pipeline/runtime/jaxpp_host_startup.patch.gz` overlay
for forward-offload scheduling, pinned-host inference, and disposable startup.
Its decompressed SHA256 is
`d987c16ece399fe5eeb69d413703d549135635f19a57dc4b6c60cc0d1ea4b1a7`.
This JAXPP package declares JAX <=0.11.0; the 0.11.1 combination is an explicit
validation gate. A stock JAXPP installation does not supply those overlays.

Install from the integration checkout on each x86 Linux GPU worker. Ordinary
`uv sync` resets manual runtime overrides, so use `--no-sync` afterwards or
reapply this sequence:

```bash
uv sync --all-packages --extra pipeline --no-dev
uv pip install 'jax[cuda13]==0.11.1' jaxlib==0.11.1 \
  jax-cuda13-pjrt==0.11.1 jax-cuda13-plugin==0.11.1 \
  'flash-attn-4[cu13]==4.0.0b28' 'nvidia-cutlass-dsl[cu13]==4.6.2' \
  'quack-kernels[cu13]==0.6.4'
uv pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
uv pip install --no-deps --reinstall \
  'jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@328f75a80cecf22c7cc030a82d8941d3c1e220b6'
uv run --no-sync python experiments/grug/moe_hero_pipeline/runtime/apply_overlay.py
uv run --no-sync python -c \
  'from iris.cluster.setup_scripts import cuda_toolchain_setup_script; print(cuda_toolchain_setup_script())' \
  > /tmp/hero-cuda-toolchain.sh
IRIS_VENV="$PWD/.venv" IRIS_WORKDIR="$PWD" bash /tmp/hero-cuda-toolchain.sh
export PATH="$PWD/.venv/bin:$PATH"
```

The installer checks the JAXPP source revision and overlay bytes, and prints
installed versions. CUDA library lower bounds remain in the environment; this
is not a fully pinned standalone lock. Run one pipeline per process: startup
rebuilds a process-global compiled-task cache.

Enable both CUDA and CPU backends: checkpoint serialization needs pageable CPU arrays.
Require CUDA as the default backend and CUDA-enabled Torch before dispatch:

```bash
export JAX_PLATFORMS=cuda,cpu
uv run --no-sync python -c 'import jax; assert jax.default_backend() == "gpu", (jax.default_backend(), jax.devices()); assert jax.devices("cuda"); assert jax.devices("cpu"); print(jax.default_backend(), jax.devices("cuda"), jax.devices("cpu"))'
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_capability())'
JAX_PLATFORMS=cpu uv run --no-sync python -m experiments.grug.moe_hero_pipeline.pipeline_smoke --help
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest experiments/grug/moe_hero_pipeline/test_pipeline.py -q -n 0
```

JAX reports CUDA's default backend platform as `gpu`. The explicit `cuda`
device query and `JAX_PLATFORMS=cuda,cpu` restrict that accelerator backend to CUDA.
The explicit `--processes-per-task 8` matches the worker wrapper and disables
per-process PGLE by default; concurrent CUPTI sessions on one node collide.

Iris must allocate two eight-GPU H100 worker tasks in one job and expose its
endpoint registry, task context, and a `jax` port. Each worker launches eight
processes with one GPU per process. `iris.jax.multigpu_main` derives global rank
as `8 * task_index + local_rank`, sets `IRIS_MULTIGPU_PROCESS_COUNT=16`,
`IRIS_MULTIGPU_PROCESS_INDEX`, and `IRIS_MULTIGPU_LOCAL_DEVICE_IDS`. The runner's
`iris.jax.init.initialize_jax` registers rank zero's coordinator endpoint and
discovers it for the remaining ranks. The worker command requires this Iris job
context; it is not a standalone two-host shell launcher. Submit
the coordinator and GPU workers at `PRIORITY_BAND_BATCH` with a one-hour timeout
and no automatic retries. Retain per-step CUDA synchronization: the older
full-shape pipeline showed NaNs without it.

## Bounded synthetic gate

Run this command on each allocated worker with a unique shared run ID:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_PGLE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_MODULE_LOADING=EAGER \
JAXPP_DISABLE_SCHEDULE_TASK_FUSION=1 \
uv run --no-sync python -m iris.jax.multigpu_main --nproc 8 --devices-per-proc 1 -- \
  python -u -m experiments.grug.moe_hero_pipeline.pipeline_smoke \
    --main-hero-recipe --processes-per-task 8 --diagnostic-layers 2 \
    --attention-implementation gpu_fa4_cute \
    --moe-implementation fixed_pooled_wave_all_to_all --expert-waves 6 \
    --schedule standard_1f1b --stages 2 --expert-axis-size 8 \
    --batch-size 32 --microbatches 4 --sequence-length 4096 --steps 10 \
    --optimizer muonh --offload-opt-state --offload-activations \
    --park-state-during-warmup --synchronize-devices-after-step \
    --seed 0 --run-id <unique-run-id>
```

This gate processes 1,310,720 global training tokens and writes no checkpoint.
Require ten finite `pipeline_step` losses, an `optimizer_state_offloaded` event
with `memory_kind=pinned_host`, `pipeline_complete` with `steps=10`, and exit code zero for
every native process and Iris task. Record the resolved model, optimizer,
runtime versions, source commit, and launch manifest with the result.

## Fresh-data gate

After the synthetic gate passes, use the same worker command and source with
`--steps 20`, a new run ID, and these additional arguments:

```bash
--real-data --data-schedule-steps 390251 \
  --data-output-root s3://marin-us-east-02a/marin/users/<user>/<unique-run-id>
```

The data source is the current Harrier mixture adopted from
`s3://marin-us-east-02a/marin/datakit/store_4d2e363d`. It reads existing caches in that region
without simulated epoching. The mixture schedule uses the explicit 390251-step
horizon; the execution limit remains twenty steps. This preserves rare
components that would become empty if their views shrank to the trial budget.
The `real_data_view` event records every component's sequence count. The trial
processes 2,621,440 global training tokens from fresh seed zero.
Each process uses the existing 1 GB jagged-array read cache. The output
root owns mixture metadata; this command does not request a checkpoint.

The runner closes the disposable compilation-sample iterator before compiling,
then opens the training iterator at the checkpoint step. It closes that
iterator on success and failure. A pipeline resume keeps `--seed` and
`--data-schedule-steps` unchanged for random-access sample order, and keeps
`--steps` unchanged for the optimizer schedule. Real-data checkpoints record
the mixture horizon and reject a resume with a different horizon.

## Checkpoint boundary and storage

Before these adapter gates, complete a reduced BF16 pooled-transport
save/resume/uninterrupted-control comparison with one source and runtime.
The reduced model has width 256, four layers, 96 experts, sequence length 4096,
PP2/EP8, batch 16, two microbatches, and the historical constant MuonH schedule.
Use the runner's reduced default model with `--optimizer muonh --steps 2`.
Save with `--checkpoint-root <shared-scratch-root> --checkpoint-every-steps 1
--stop-after-step 1`. Resume with the same root, steps, model, and seed, omitting
both save-interval and stop flags. Run the two-step control with no checkpoint
root. Use the original pinned JAXPP46b runtime with its historical overlays for
all three jobs; its source and setup hashes must match.
Require restoration of step 1, a finite step 2, a matched step-2 loss, and zero
exit codes from every native process and Iris task. Keep one scratch checkpoint with a 30-day TTL; resume and
control do not write another checkpoint.
The scratch root must be under `s3://marin-us-east-02a/tmp/ttl=30d/`.
Compare step-2 losses with the existing pipeline-test tolerances
`rtol=1e-5, atol=1e-5`; record the actual absolute difference. A complete save
emits `pipeline_checkpoint_saved` with `step=1`; resume must emit
`pipeline_checkpoint_restored` with `step=1` before stepping.

The pipeline state tree uses global layer paths. Standard Hero FSDP checkpoints
use a different tree and have no validated conversion into this runner.
For the main FP32 recipe, CPU shape evaluation of the actual model and MuonH
state estimates 196,925,211,676 bytes for two full-width layers and
4,290,644,189,212 bytes for 48 layers. These estimates exclude metadata and
storage overhead. Measure completed checkpoint bytes before widening a storage
gate; the prepared synthetic and fresh-data gates write zero checkpoint bytes.
