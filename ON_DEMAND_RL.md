# ON_DEMAND_RL

This document is a handoff for running `exp2039` RL without a Ray cluster, using two manually created TPU VMs:
- one trainer TPU
- one sampler TPU

It also records what was changed, why those decisions were made, and where scripts/tests live for future agents.

## Goal

Replicate `experiments/exp2039_rl_math500.py` in a no-Ray multi-host setup while preserving high throughput and enabling fast sampler startup.

## Inputs Used

The implementation decisions were based on these local docs:
- `/Users/ahmed/code/tunix-pathways/marin_rl_design_doc.md`
- `/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md`
- `/Users/ahmed/code/marin/FAST_VLLM_LOAD_TPU.md`

## High-Level Decisions

1. Keep Levanter `launch.py` as the deployment path.
- We did **not** replace Levanter launch.
- On-demand scripts call `lib/levanter/infra/launch.py` directly for TPU create + docker run.

2. Add manual no-Ray roles to `exp2039`.
- Added `--mode trainer|sampler|launch-plan|executor`.
- `executor` remains the existing behavior.
- `trainer` and `sampler` run each role independently, sharing state via GCS + Arrow metadata.

3. Define four explicit deployment presets in same-zone trainer/sampler pairs.
- `v5p_east5a` -> `us-east5-a`, `v5p-8`
- `v5p_central1a` -> `us-central1-a`, `v5p-8`
- `v6e_euw4a` -> `europe-west4-a`, `v6e-8`
- `v6e_east1d` -> `us-east1-d`, `v6e-8`

4. Use filesystem-based Arrow coordinator for no-Ray coordination.
- Added `coordinator_backend` support (`actor` or `filesystem`).
- Manual mode uses `filesystem` with metadata JSON under shared root.

5. Add fast vLLM bootstrap from object storage.
- Start vLLM with `load_format=dummy` using staged minimal metadata.
- Read safetensors from GCS/S3 via `read_safetensors_fsspec`.
- Convert/inject weights with existing mapping and `sync_weights`.
- Continue online updates via ArrowFlight afterward.

6. Throughput default from no-Ray notes: target 256 concurrent completions per sampler.
- `--rollout-shape auto` derives prompts from `TARGET_CONCURRENCY_PER_SAMPLER=256`.
- `--rollout-shape exp2039` preserves old 64x16 shape.

7. Capacity mode default is Spot for on-demand scripts.
- `CAPACITY_MODE=spot` default.
- Can override to `on-demand`, `reserved`, or `best-effort`.

8. Fix Levanter TPU create flags for current gcloud behavior.
- Removed unsupported `--provisioning-model=SPOT` from queued-resources create.
- Keep `--spot`, `--best-effort`, `--reserved` only.

## What Changed

### 1) Experiment manual mode and presets

File:
- `experiments/exp2039_rl_math500.py`

Key changes:
- Added `DeploymentPreset` and four presets above.
- Added manual CLI surface:
  - `--mode`
  - `--deployment-preset`
  - `--run-id`
  - `--shared-root`
  - `--trainer-zone`, `--sampler-zone`
  - `--trainer-tpu-type`, `--sampler-tpu-type`
  - `--rollout-shape`
  - `--bootstrap-checkpoint-path`
- Added `launch-plan` mode that prints exact trainer/sampler launch commands.
- Manual mode now:
  - enforces `gs://` shared root
  - enforces shared-root region alignment with selected zone
  - constructs explicit shared paths under `shared-root/run-id/...`

### 2) Weight transfer no-Ray filesystem coordination

Files:
- `lib/marin/src/marin/rl/weight_transfer/base.py`
- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`

Key changes:
- `WeightTransferConfig` now supports:
  - `coordinator_backend: Literal["actor", "filesystem"]`
  - `coordinator_metadata_path: str`
- Added `FileSystemArrowFlightCoordinator`:
  - writes/reads coordinator metadata JSON
  - uses lock file for updates
  - ignores stale weight IDs
- Added coordinator factory logic to choose actor vs filesystem backend.

### 3) vLLM fast bootstrap path

File:
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`

Key changes:
- `vLLMInferenceContextConfig` adds:
  - `enable_fast_bootstrap: bool`
  - `bootstrap_checkpoint_path: str | None`
- Added bootstrap path:
  - stage metadata locally (`config.json`, tokenizer files, etc.)
  - initialize vLLM with `load_format="dummy"`
  - load safetensors shards from object store
  - inject into engine via `sync_weights` (or async fallback path)

### 4) .marin.yaml support for launch defaults

File:
- `lib/levanter/src/levanter/infra/cli_helpers.py`

Key changes:
- Config resolution now prefers `.marin.yaml`, then `.levanter.yaml`, then `.config`.
- Searches from cwd up to repo root.
- Warns when both `.marin.yaml` and `.levanter.yaml` are present in same directory.

### 5) TPU capacity flag compatibility fix

File:
- `lib/levanter/src/levanter/infra/tpus.py`

Key changes:
- For spot/preemptible, pass `--spot` only.
- For best-effort, pass `--best-effort` only.
- Removed unsupported `--provisioning-model=SPOT` from queued-resources create.

## On-Demand Script Location

All scripts are in:
- `on-demand-rl-scripts/`

Shared runner:
- `on-demand-rl-scripts/_run_exp2039.sh`

Preset scripts:
- `on-demand-rl-scripts/exp2039_useast5a.sh`
- `on-demand-rl-scripts/exp2039_uscentral1a.sh`
- `on-demand-rl-scripts/exp2039_euwest4.sh`
- `on-demand-rl-scripts/exp2039_useast1d.sh`

Legacy alias kept:
- `on-demand-rl-scripts/run_exp2039_v5p_east5a.sh` -> delegates to `exp2039_useast5a.sh`

## Script Defaults and Overrides

Default behavior in `_run_exp2039.sh`:
- `RUN_ID=exp2039-<timestamp>`
- `ROLLOUT_SHAPE=auto`
- `CAPACITY_MODE=spot`
- `RETRIES=10`
- TPU names:
  - `${RUN_ID}-trainer`
  - `${RUN_ID}-sampler`

Override environment variables:
- `RUN_ID`
- `SHARED_ROOT` (or `SHARED_BUCKET`)
- `CAPACITY_MODE` (`spot|on-demand|reserved|best-effort`)
- `RETRIES`
- `ROLLOUT_SHAPE` (`auto|exp2039`)
- `ZONE`, `TPU_TYPE`, `DEPLOYMENT_PRESET` (advanced/manual override)

## How To Run

Examples:

```bash
bash on-demand-rl-scripts/exp2039_euwest4.sh
bash on-demand-rl-scripts/exp2039_useast1d.sh
bash on-demand-rl-scripts/exp2039_useast5a.sh
bash on-demand-rl-scripts/exp2039_uscentral1a.sh
```

Force on-demand capacity:

```bash
CAPACITY_MODE=on-demand bash on-demand-rl-scripts/exp2039_euwest4.sh
```

Force classic exp2039 rollout shape:

```bash
ROLLOUT_SHAPE=exp2039 bash on-demand-rl-scripts/exp2039_euwest4.sh
```

## Launch/Infra Flow (Important)

The script calls:
- `lib/levanter/infra/launch.py`

Which calls:
- `lib/levanter/src/levanter/infra/tpus.py`

Which uses:
- `gcloud alpha compute tpus queued-resources create ...`

So seeing `gcloud ... queued-resources` in logs is expected; that is the normal Levanter launch path.

## Visibility / Tracking

The launcher is `--foreground`, so local terminal output includes build/push/TPU create/job run logs.

Useful checks during run:

```bash
gcloud alpha compute tpus queued-resources describe <RUN_ID>-trainer --zone=<ZONE>
gcloud alpha compute tpus queued-resources describe <RUN_ID>-sampler --zone=<ZONE>

gcloud alpha compute tpus tpu-vm ssh <RUN_ID>-trainer --zone=<ZONE> --worker=all --command='sudo docker logs -f --tail 200 levanter'
gcloud alpha compute tpus tpu-vm ssh <RUN_ID>-sampler --zone=<ZONE> --worker=all --command='sudo docker logs -f --tail 200 levanter'
```

## Tests Added/Updated

New tests:
- `lib/levanter/tests/test_cli_helpers.py`
  - `.marin.yaml` precedence
  - parent-dir discovery
  - `.levanter.yaml` fallback
  - `.config` warning fallback
  - do not search above repo root
- `lib/levanter/tests/test_tpus.py`
  - capacity flags for spot/preemptible/best-effort/reserved
  - ensures no `--provisioning-model=*`
  - on-demand uses no special capacity flag
- `tests/rl/test_vllm_fast_bootstrap.py`
  - shard discovery
  - metadata staging
  - state serialization helper

Updated tests:
- `tests/rl/test_weight_transfer.py`
  - filesystem Arrow coordinator roundtrip
  - stale update rejection

## Operational Notes for Future Agents

1. Keep trainer and sampler in same region for shared-root locality and cost.
2. Capacity failures are common; retry or switch preset/zone.
3. Default capacity is Spot in scripts.
4. If `.marin.yaml` exists at repo root, Levanter launch picks it up automatically.
5. For manual one-off command generation, use:

```bash
uv run python experiments/exp2039_rl_math500.py \
  --mode launch-plan \
  --deployment-preset v6e_euw4a \
  --run-id <RUN_ID> \
  --shared-root gs://<BUCKET>/tmp/exp2039/<RUN_ID>
```

## Quick Index (Files to Inspect First)

- Experiment manual-mode entrypoint:
  - `experiments/exp2039_rl_math500.py`
- On-demand scripts:
  - `on-demand-rl-scripts/_run_exp2039.sh`
  - `on-demand-rl-scripts/exp2039_*.sh`
- Launch config resolution:
  - `lib/levanter/src/levanter/infra/cli_helpers.py`
- TPU queued resource create logic:
  - `lib/levanter/src/levanter/infra/tpus.py`
- Filesystem Arrow coordinator:
  - `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
- Fast vLLM bootstrap:
  - `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
