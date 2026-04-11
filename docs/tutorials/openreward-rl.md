# OpenReward RL Smoke Run

> TL;DR: the current checked-in OpenReward path in Marin is a manifest-backed, text-only, single-turn RL flow on the Qwen + vLLM stack. Prepare manifests first, install the `rl` extra, then launch `experiments/openreward_qwen3_8b_smoke.py`.

This tutorial shows the narrow path that Marin supports today:

- manifest-backed OpenReward tasks
- text prompt blocks only
- one terminal tool call per rollout
- Qwen-family models through the vLLM inference path

It does not cover multi-turn ORS sessions yet.

## Prerequisites

Start from the normal Marin installation in [Installation](installation.md).

For the OpenReward smoke path, install Marin with both the accelerator extra you need and the `rl` extra:

```bash
uv sync --all-packages --extra=tpu --extra=rl
```

The checked-in smoke launcher is TPU-first. If you are using another supported accelerator setup, keep the matching hardware extra and still add `--extra=rl`.

You will also need:

- `OPENREWARD_API_KEY` for the OpenReward control plane
- `HF_TOKEN` if you need access to gated checkpoints
- any tool-secret environment variables required by the target OpenReward environment, such as `OPENAI_API_KEY`

## Step 1: Prepare Manifests

Marin does not discover OpenReward tasks live inside rollout workers. Instead, it snapshots prompts and tool schemas into deterministic manifest files ahead of training.

Prepare a small train slice:

```bash
uv run python lib/marin/src/marin/rl/scripts/prepare_openreward_task_manifest.py \
  --environment your-org/your-env \
  --split train \
  --start 0 \
  --stop 64 \
  --output /tmp/openreward-train.json
```

Prepare a held-out eval slice:

```bash
uv run python lib/marin/src/marin/rl/scripts/prepare_openreward_task_manifest.py \
  --environment your-org/your-env \
  --split train \
  --start 64 \
  --stop 80 \
  --output /tmp/openreward-eval.json
```

If your deployment uses variants or a non-default API endpoint, add `--variant ...` and `--base-url ...` to the manifest preparation command too.

## Step 2: Launch The Smoke Experiment

Set the credentials you want to expose to the rollout worker:

```bash
export OPENREWARD_API_KEY=...
export OPENAI_API_KEY=...
```

The smoke launcher validates those variables locally, forwards the selected names into the executor and RL worker environments, and keeps the secret values out of the serialized RL curriculum config.

Then launch the checked-in smoke example:

```bash
uv run python experiments/openreward_qwen3_8b_smoke.py \
  --train-manifest /tmp/openreward-train.json \
  --eval-manifest /tmp/openreward-eval.json \
  --tool-secret-env OPENAI_API_KEY \
  --run-name openreward-smoke
```

Useful flags:

- `--openreward-variant`: selects a deployment variant when the environment exposes one
- `--openreward-base-url`: points Marin at a non-default OpenReward API endpoint
- `--num-train-steps`: keeps the first run short while you validate the path
- `--train-tpu-type` and `--inference-tpu-type`: override the default TPU shape

For a minimal end-to-end validation, you can omit `--eval-manifest`; the smoke launcher will reuse the train manifest for eval.

## What The Launcher Wires Up

`experiments/openreward_qwen3_8b_smoke.py` does three important things:

1. Uses the checked-in manifest-backed `OpenRewardEnv`.
2. Pins the model preset to Qwen3 8B, which is the currently validated tool-calling path in Marin.
3. Sets `pip_dependency_groups=["rl"]`, so the rollout and trainer workers install the `openreward` SDK together with the existing RL extras.

## Passing Tool Secrets

OpenReward tools often need credentials that are separate from the OpenReward control-plane API key. The smoke launcher accepts repeated `--tool-secret-env NAME` flags and forwards those environment variables into the OpenReward session `secrets` mapping at runtime.

For example:

```bash
uv run python experiments/openreward_qwen3_8b_smoke.py \
  --train-manifest /tmp/openreward-train.json \
  --tool-secret-env OPENAI_API_KEY \
  --tool-secret-env ANTHROPIC_API_KEY
```

If one of the requested variables is missing, the launcher fails fast before submitting the RL job.

## Supported Scope

The current Marin OpenReward path supports:

- text prompt blocks
- single-turn, terminal-tool lessons
- manifest-backed task identity
- Qwen-family models on the vLLM inference path

The current path does not support:

- image prompt blocks
- multi-turn ORS tool loops
- provider-agnostic tool-calling backends
- live task discovery inside rollout workers

## Minimal `EnvConfig`

If you want to wire `OpenRewardEnv` into another curriculum manually, this is the supported shape:

```python
from marin.rl.environments import EnvConfig

env_config = EnvConfig(
    env_class="marin.rl.environments.openreward_env.OpenRewardEnv",
    env_args={
        "train_manifest_path": "/tmp/openreward-train.json",
        "eval_manifest_path": "/tmp/openreward-eval.json",
        "api_key_env_var": "OPENREWARD_API_KEY",
        "base_url": None,
        "variant": None,
        "secret_env_vars": ["OPENAI_API_KEY"],
    },
)
```

Use prepared manifests for both train and eval, and export the referenced environment variables before launch. That keeps task identity reproducible, avoids worker-side API discovery drift, and keeps live credentials out of executor metadata.
