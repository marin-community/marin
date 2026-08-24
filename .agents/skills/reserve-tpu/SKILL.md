---
name: reserve-tpu
description: Reserve or operate an Iris-backed TPU worker with scripts/iris/dev_tpu.py only when explicitly requested for a dev TPU session.
---

# Dev TPU

`scripts/iris/dev_tpu.py` reserves a TPU-backed worker through Iris, waits for the worker VM to come up, and lets you SSH into it or run commands directly against it. It uses `gcloud` SSH and SCP against the worker Iris assigned to the holder job. There is no persistent `~/.ssh/config` alias.

## Critical concurrency rule

Run at most one TPU job at a time on a given dev TPU VM. Do not launch concurrent TPU commands on the same worker from multiple shells, tmux panes, or background jobs.

## Prerequisites

1. Authenticate to GCP and set up the repo.

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Use a cluster config that provisions the requested TPU. Shared clusters
   normally already have a controller; never start or restart one without
   explicit approval.

## Command pattern

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p8" \
  <subcommand> [flags]
```

Subcommands and distinctive flags:

- `allocate --tpu-type v5p-8` — reserves a single-host TPU and holds it until `Ctrl-C`. `--tpu-type` is required (the config may expose many variants; the script does not guess). Add `--zone us-east5-b` to pin the holder job to a zone (the config must expose that TPU family there). Add `--no-setup-env` to skip remote env setup.
- `status` — show the active session.
- `connect` — interactive SSH to the reserved worker. After connecting: `cd ~/marin && source ~/.local/bin/env`.
- `setup_env` — install/refresh the remote env. Run before the first `execute`/`watch` if you allocated with `--no-setup-env`.
- `execute -- <cmd>` — sync local files, then run `<cmd>`. Add `--no-sync` for a no-sync inner loop. Example: `execute -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`.
- `watch -- <cmd>` — rerun `<cmd>` on local file changes.
- `release` — terminate the holder job and clear the session file.

## Worker Selection

Multi-host TPU types reserve more than one worker VM. Use `--worker <index>` with `connect`, `execute`, or `watch` to target a specific worker (`execute` and `watch` default to worker `0`):

```bash
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$USER-v5p16" connect --worker 1
```

## Observability

Use normal Iris tooling to inspect the backing cluster and holder job:

```bash
uv run iris --config=lib/iris/config/marin.yaml cluster dashboard
uv run iris --config=lib/iris/config/marin.yaml cluster vm status
uv run iris --config=lib/iris/config/marin.yaml job list --prefix /$USER/dev-tpu
uv run iris --config=lib/iris/config/marin.yaml job logs /$USER/dev-tpu-<name>
```

If worker bootstrap fails:

```bash
uv run iris --config=lib/iris/config/marin.yaml cluster vm logs <worker-id>
```

## Session behavior

- Local session state lives under `~/.cache/marin/dev_tpu_iris/`.
- If the `allocate` terminal dies unexpectedly, use `release` to terminate the holder job and clear the stale state file.
- `execute` and `watch` already wrap the remote command in `bash -lc`; do not pass your own `bash -c`.

## Unique session name

Always pass `--tpu-name` to avoid collisions with other agents:

```bash
export TPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml --tpu-name "$TPU_NAME" allocate --tpu-type v5p-8
```

## Cleanup

Normal cleanup is `Ctrl-C` in the `allocate` terminal. To clean up from another shell, run the `release` subcommand.
