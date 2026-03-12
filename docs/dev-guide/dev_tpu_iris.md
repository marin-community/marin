# Using Dev TPUs for Testing with Iris

If you want a fast TPU debugging loop on Iris without wiring a full training job each time, use `scripts/iris/dev_tpu.py`.
It reserves a TPU-backed worker through Iris, waits for the worker VM to come up, and then lets you SSH into it or run commands directly against it.

The Iris version keeps the Ray workflow intact by using a separate script and separate docs.
It does not create a persistent `~/.ssh/config` alias; instead it uses `gcloud` SSH/SCP against the worker that Iris assigned to the holder job.

## Critical concurrency rule

Run at most one TPU job at a time on a given dev TPU VM.
Do not launch concurrent TPU commands on the same worker from multiple shells, tmux panes, or background jobs.

## What it does

- `allocate`: submits a TPU holder job on Iris, resolves the assigned worker VM(s), optionally syncs your repo, and blocks until you release the session
- `status`: shows the active local session metadata
- `connect`: opens an interactive SSH session to one reserved worker
- `setup_env`: syncs the repo by default, then installs or refreshes the remote `uv` environment on one worker or all reserved workers
- `execute`: syncs local files to `~/marin` on the reserved worker(s), then runs one command
- `watch`: syncs all reserved workers + reruns a command on the selected worker when local files change
- `release`: terminates the holder job and removes the local session file

## Prerequisites

1. Authenticate to GCP and set up the repo:

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Ensure the Iris controller is already running for your cluster:

```bash
uv run iris --config=lib/iris/examples/marin.yaml cluster start
```

3. Use a cluster config that can actually provision the TPU type you want.

## Quick start

Reserve a single-host TPU and hold it until `Ctrl-C`:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  allocate --tpu-type v5p-8
```

In another terminal, inspect the active session:

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

After connecting, move into the synced repo and load the shell environment:

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

If you allocated with `--no-setup-env`, set up the remote environment before the first `execute`:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  setup_env
```

Run a no-sync inner loop:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  execute --no-sync -- \
  uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

Watch local files and rerun:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  watch -- \
  uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

## Worker selection

Multi-host TPU types reserve more than one worker VM.
Use `--worker <index>` with `connect`, `execute`, or `watch` to target a specific worker:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p16" \
  connect --worker 1
```

`execute` and `watch` default to worker `0`.
`execute` syncs all reserved workers before each run (unless `--no-sync`), and `watch` does the same before each restart.

## Observability

The script is only for reserving and directly using the worker VM.
Use normal Iris tooling to inspect the backing cluster and holder job:

```bash
uv run iris --config=lib/iris/examples/marin.yaml cluster dashboard
uv run iris --config=lib/iris/examples/marin.yaml cluster vm status
uv run iris --config=lib/iris/examples/marin.yaml job list --prefix /$USER/dev-tpu
uv run iris --config=lib/iris/examples/marin.yaml job logs /$USER/dev-tpu-<name>
```

If worker bootstrap fails, inspect VM logs:

```bash
uv run iris --config=lib/iris/examples/marin.yaml cluster vm logs <worker-id>
```

## Practical notes

- `allocate` requires `--tpu-type`; Iris configs can expose many TPU variants, so the script does not guess.
- The local session state lives under `~/.cache/marin/dev_tpu_iris/`.
- If the `allocate` terminal dies unexpectedly, use `release` to terminate the holder job and clear the stale state file.
- If you skip environment setup during `allocate`, run `setup_env` before `execute` or `watch`.
- `execute` and `watch` already wrap the remote command in `bash -lc`; do not pass your own `bash -c`.

## For agents

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
If you need explicit cleanup from another shell:

```bash
uv run scripts/iris/dev_tpu.py \
  --config lib/iris/examples/marin.yaml \
  --tpu-name "$USER-v5p8" \
  release
```
