---
name: dev-tpu
description: Use `scripts/iris/dev_tpu.py` to reserve an Iris-backed TPU worker for fast debugging, testing, and command execution without wiring a full training job. Use when asked for the standard dev TPU workflow.
---

# Skill: Dev TPU

Use this skill when you want the standard fast TPU debugging loop without wiring a full training job each time.

`scripts/iris/dev_tpu.py` reserves a TPU-backed worker through Iris, waits for the worker VM to come up, and lets you SSH into it or run commands directly against it.

It uses `gcloud` SSH and SCP against the worker that Iris assigned to the holder job. There is no persistent `~/.ssh/config` alias.

## Critical concurrency rule

Run at most one TPU job at a time on a given dev TPU VM. Do not launch concurrent TPU commands on the same worker from multiple shells, tmux panes, or background jobs.

## Commands

- `allocate`: submit a TPU holder job, resolve the assigned worker VM or VMs, optionally sync the repo, and block until release
- `status`: show the active local session metadata
- `connect`: open an interactive SSH session to one reserved worker
- `setup_env`: sync the repo by default, then install or refresh the remote `uv` environment on one worker or all reserved workers
- `execute`: sync local files to `~/marin` on the reserved worker or workers, then run one command
- `watch`: sync all reserved workers and rerun a command on the selected worker when local files change
- `release`: terminate the holder job and remove the local session file

## Prerequisites

1. Authenticate to GCP and set up the repo.

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Ensure the Iris controller is already running for your cluster. On the shared Marin clusters this is usually already true; only start it yourself when working against a fresh or local cluster.

3. Use a cluster config that can actually provision the TPU type you want.

## Quick Start

Reserve a single-host TPU and hold it until `Ctrl-C`:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  allocate --tpu-type v5p-8
```

Inspect the active session:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  status
```

Connect to the reserved worker:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  connect
```

After connecting:

```bash
cd ~/marin
source ~/.local/bin/env
```

Run one command with sync:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  execute -- \
  uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

If you allocated with `--no-setup-env`, refresh the remote environment before the first `execute`:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  setup_env
```

### Request a specific zone

Pick a config that exposes the TPU family you want, then pin the holder job with `--zone`:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8-east5" \
  allocate --tpu-type v5p-8 --zone us-east5-b
```

If the zone you want is not available in the config, switch to a config that includes it.

## Worker Selection

Multi-host TPU types reserve more than one worker VM. Use `--worker <index>` with `connect`, `execute`, or `watch` to target a specific worker.

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p16" \
  connect --worker 1
```

`execute` and `watch` default to worker `0`.

## Practical Notes

### No-sync inner loop

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  execute --no-sync -- \
  uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

### Watch local files and rerun

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  watch -- \
  uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

### Observability

Use normal Iris tooling to inspect the backing cluster and holder job:

```bash
uv run iris --config=lib/iris/examples/marin.yaml cluster dashboard
uv run iris --config=lib/iris/examples/marin.yaml cluster vm status
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /$USER/dev-tpu
uv run iris --config=lib/iris/examples/marin.yaml job logs /$USER/dev-tpu-<name>
```

If worker bootstrap fails:

```bash
uv run iris --config=lib/iris/examples/marin.yaml cluster vm logs <worker-id>
```

### Session behavior

- `allocate` requires `--tpu-type`; the Iris config may expose many TPU variants, so the script does not guess.
- Local session state lives under `~/.cache/marin/dev_tpu_iris/`.
- If the `allocate` terminal dies unexpectedly, use `release` to terminate the holder job and clear the stale state file.
- If you skip environment setup during `allocate`, run `setup_env` before `execute` or `watch`.
- `execute` and `watch` already wrap the remote command in `bash -lc`; do not pass your own `bash -c`.

## Agent Usage

Always pass `--tpu-name` to avoid collisions with other agents:

```bash
export TPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
```

Then use that name consistently:

```bash
uv run scripts/iris/dev_tpu.py --config lib/iris/examples/marin.yaml --tpu-name "$TPU_NAME" allocate --tpu-type v5p-8
```

## Cleanup

Normal cleanup is `Ctrl-C` in the `allocate` terminal.

If you need cleanup from another shell:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  release
```
