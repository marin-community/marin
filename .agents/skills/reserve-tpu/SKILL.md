---
name: reserve-tpu
description: Reserve an Iris-backed TPU worker for fast debugging with dev_tpu.py.
---

# Dev TPU

`scripts/iris/dev_tpu.py` reserves a TPU worker through Iris, uses GCP SSH/SCP,
and can sync the repository or run commands. Run at most one TPU job at a time
on a worker, including across shells, tmux, and background jobs.

Authenticate and set up the repository first:

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

Use a cluster config that provisions the requested TPU. Shared clusters normally
already have a controller; starting one yourself requires a fresh/local cluster.

```bash
export TPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$TPU_NAME" allocate --tpu-type v5p-8
```

The common shape is `... --tpu-name <name> <subcommand> [flags]`. Subcommands:
`allocate` holds until Ctrl-C (`--tpu-type` is required; optional `--zone` and
`--no-setup-env`), `status`, `connect`, `setup_env`, `execute -- <cmd>`,
`watch -- <cmd>`, and `release`. `execute`/`watch` sync to `~/marin` and already
wrap commands in `bash -lc`; use `--no-sync` for an inner loop. Multi-host types
use `--worker <index>` with connect/execute/watch (execute/watch default to 0).

```bash
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name "$TPU_NAME" connect --worker 1
```

Session state is under `~/.cache/marin/dev_tpu_iris/`. If allocation dies, run
`release` to terminate the holder and clear stale state. Inspect the backing job
with the normal Iris dashboard, VM status/logs, and job logs. Never use a
controller restart to repair access without explicit approval.
