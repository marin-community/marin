# Using Dev TPUs for Testing

Use `scripts/ray/dev_tpu.py` for interactive TPU debugging and tight benchmark loops.
It is usually faster than wiring a full Ray job when you want quick iteration.

## What it does

- `allocate`: reserves a TPU VM and keeps it alive while the command runs.
- `connect`: opens an interactive shell on the TPU.
- `execute`: rsyncs local files to remote `~/marin/`, then runs one command.
- `watch`: rsync + restart on local file changes.

The SSH alias created is `dev-tpu-<tpu-name>`.

## Prerequisites

1. Authenticate to GCP and set up Marin development environment.

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Ensure your SSH public key is in project metadata:
   `https://console.cloud.google.com/compute/metadata?resourceTab=sshkeys&project=hai-gcp-models&scopeTab=projectMetadata`

3. The script uses your local username by default (`getpass.getuser()`) and writes SSH config for that user.

## Quick start

Pick a per-user TPU name first:

```bash
export TPU_NAME="${USER}-dev"
```

If you are running as an agent, prefer a nonce TPU name per allocation, ideally tied to your task or branch so concurrent agents do not collide. Reusing a stable name is fine when it is the same agent continuing the same workstream:

```bash
export TPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD)-$(date +%H%M%S)"
# Example: dlwh-moe-ep-benchmark-153012
```

If you are a human and only using one dev TPU per region, you can usually skip `--tpu-name`.
`dev_tpu.py` defaults to `dev-<cluster_name>-<user>`, which is usually reasonable for that workflow.

Allocate:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  allocate
```

Connect interactively:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  connect
```

You can also connect directly to `dev-tpu-<tpu-name>` from your terminal or VSCode/Cursor Remote-SSH.

Run one command (with sync):

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

Stop allocation by `Ctrl-C` in the `allocate` terminal.

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

`execute` always rsyncs first. For iterative profiling/tuning, SSH directly:

```bash
ssh "dev-tpu-${TPU_NAME}"
cd ~/marin
source ~/.local/bin/env
```

Then run multiple commands directly on remote.

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

### TPU busy / stale lockfile

If TPU init fails due lock contention:

```bash
sudo rm -f /tmp/libtpu_lockfile
sudo lsof -t /dev/vfio/* | xargs -r sudo kill -9
```

Then rerun your command.

### `execute` feels slow

It syncs files every time by design. Use direct SSH for repeated runs.

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
