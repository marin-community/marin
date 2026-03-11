# Using Dev TPUs for Testing

Using `ray_run` is great for long-running jobs, but if you are trying to debug a TPU test or memory issue, it's often faster to allocate your own TPU.
You can use the `scripts/ray/dev_tpu.py` to allocate a temporary TPU for yourself to SSH into as well as sync your local changes and automatically run commands.
You will need to setup your SSH key in `gcloud` to get started.
It is usually faster than wiring a full Ray job when you want quick iteration. It is less good if you want to run many different commands in parallel,
or if you want to run a long experiment and not worry about the TPU going away.

## Critical concurrency rule

Run at most one TPU job at a time on a given dev TPU VM. Only one process can run TPU code at a time on the same dev TPU.
Do not launch concurrent TPU commands (including in separate shells, tmux panes, or background jobs) against one dev TPU;
queue them and run sequentially instead.

## What it does

- `allocate`: reserves a TPU VM and keeps it alive while the command runs. It also creates an SSH alias for the TPU and writes config to `~/.ssh/config` so you can connect easily.
- `connect`: opens an interactive shell on the TPU.
- `execute`: syncs local files to remote `~/marin/` (unless `--no-sync`), then runs one command.
- `watch`: rsync + restart on local file changes.

## Prerequisites

1. Authenticate to GCP and set up Marin development environment.

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Ensure your SSH public key is in project metadata:
   https://console.cloud.google.com/compute/metadata?resourceTab=sshkeys&project=hai-gcp-models&scopeTab=projectMetadata

## Quick start

### For humans

Set a TPU name once for your session:
Allocate:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  allocate
```

Connect interactively:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  connect
```

Run one command (with sync):

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  execute -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

`dev_tpu.py` creates an alias for a TPU VM monitored by Ray. By default, it uses your username together with the
`cluster_name` from the config to create a name like `dev-<cluster_name>-<user>`.
You can also pass an explicit `--tpu-name` if you want to manage multiple TPUs or prefer a different naming scheme.

You can also connect directly to `dev-tpu-<tpu-name>` from your terminal or VSCode/Cursor Remote-SSH.
It's generally best to ensure the connection with `allocate` is still alive before doing this, to avoid
connecting to a stale VM that might be in use by another agent.

Stop allocation by `Ctrl-C` in the `allocate` terminal.

### For agents

If you are an AI agent, you should always pass `--tpu-name` to avoid collisions with other agents.

```bash
export TPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD)-$(date +%H%M%S)"
# Example: dlwh-moe-ep-benchmark-153012
```

Then just run the commands as above, passing `--tpu-name "$TPU_NAME"` each time. (You don't have to use an env var.)

As an agent you probably won't use `connect` at all, but will rely on `execute` to run commands and sync files,
or just ssh directly to the TPU for a fast inner loop.

## Practical patterns

### 1) Pass extra env vars for one run

Use `-e KEY=VALUE` (repeatable) with `execute`:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py
```

Notes:
- `.levanter.yaml`, `.marin.yaml`, and `.config` env entries are injected automatically.
- `execute` already wraps the command in `bash -c`; do not pass your own `bash -c`.

### 2) Fast inner loop without repeated sync

For iterative profiling/tuning, either skip sync with `--no-sync` or SSH directly:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute --no-sync -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

Or SSH directly:

```bash
ssh "dev-tpu-${TPU_NAME}"
cd ~/marin
source ~/.local/bin/env
```

Then run multiple commands directly on remote.
Run them one at a time when they execute TPU code.

### 3) Pull profiles and traces

```bash
mkdir -p ".profiles/${TPU_NAME}"
scp "dev-tpu-${TPU_NAME}:~/marin/.profiles/<run_name>/plugins/profile/*/*" ".profiles/${TPU_NAME}/"
```

You can also use `scp` for specific `*.trace.json.gz` and `*.xplane.pb` files.

## Multi-TPU usage

When using multiple clusters at once, always pass explicit `--config` and `--tpu-name`.
Use distinct names per cluster to avoid collisions.
Example:

- `infra/marin-us-central1.yaml` with `--tpu-name "${USER}-central1"`
- `infra/marin-us-east5-a.yaml` with `--tpu-name "${USER}-east5"`

This avoids ambiguity in SSH aliases and in your own command history.

## Troubleshooting

### `Could not infer TPU type from config`

Your cluster config does not expose `available_node_types.tpu_worker.node_config.acceleratorType`.
Pass it explicitly:

```bash
uv run scripts/ray/dev_tpu.py --config <config> allocate --tpu-type v5p-8
```

### `SSH configuration ... not found`

Run `allocate` first for that `--tpu-name`, then retry `connect`/`execute`.

### Verify release/cleanup after `allocate`

After finishing work, stop allocation with `Ctrl-C` in the terminal running `allocate`.

Recommended verification:

1. Confirm the allocator exited cleanly (you should see a release message).
2. Confirm no local `allocate` process is still running for that TPU name.
3. Confirm the local alias state is cleaned up by running:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config <config> \
  --tpu-name <name> execute --no-sync -- /bin/bash -lc 'echo ok'
```

Expected result after cleanup: it should fail with
`SSH configuration ... not found` and ask you to run `allocate` first.

### TPU busy / stale lockfile

If TPU init fails due lock contention:

```bash
sudo rm -f /tmp/libtpu_lockfile
sudo lsof -t /dev/vfio/* | xargs -r sudo kill -9
```

Then rerun your command.

### `execute` feels slow

By default it syncs files before each run (using rsync). Use `--no-sync` or direct SSH for repeated runs.

## Reference examples

Run tests:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

Run benchmark:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_mlp_profile.py
```
