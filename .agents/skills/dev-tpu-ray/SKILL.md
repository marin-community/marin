---
name: dev-tpu-ray
description: Use the legacy `scripts/ray/dev_tpu.py` workflow to allocate a temporary Ray-backed TPU VM for fast debugging, testing, and benchmark iteration. Use only when you specifically need the Ray-backed dev TPU path.
---

# Skill: Legacy Ray Dev TPU

Use this skill only when you specifically need the legacy Ray-backed dev TPU workflow. Prefer `.agents/skills/dev-tpu/SKILL.md` for the current Iris-backed path.

`scripts/ray/dev_tpu.py` can reserve a temporary TPU VM, sync the repo, and run commands remotely. It is good for:

- quick test and benchmark loops,
- memory debugging,
- profiling and trace capture,
- short experiments where you want direct shell access.

It is a bad fit for long unattended experiments or many concurrent TPU commands.

## Critical concurrency rule

Run at most one TPU job at a time on a given dev TPU VM. Do not launch concurrent TPU commands from separate shells, tmux panes, or background jobs against the same dev TPU.

## Commands

- `allocate`: reserve a TPU VM and keep it alive while the command runs. This also writes an SSH alias into `~/.ssh/config`.
- `connect`: open an interactive shell on the TPU.
- `execute`: sync local files to remote `~/marin/` unless `--no-sync`, then run one command.
- `watch`: rsync + restart on local file changes.

## Prerequisites

1. Authenticate to GCP and set up the Marin development environment.

```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Ensure your SSH public key is in project metadata:
   `https://console.cloud.google.com/compute/metadata?resourceTab=sshkeys&project=hai-gcp-models&scopeTab=projectMetadata`

## Quick Start

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

Run one command with sync:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  execute -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

`dev_tpu.py` creates an alias for a TPU VM monitored by Ray. By default it uses your username and the config `cluster_name` to create a name like `dev-<cluster_name>-<user>`.

Stop allocation by pressing `Ctrl-C` in the terminal that is running `allocate`.

## Agent Usage

Always pass `--tpu-name` to avoid collisions with other agents.

```bash
export TPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')-$(date +%H%M%S)"
```

Then reuse that name for `allocate`, `connect`, and `execute`.

## Practical Patterns

### Extra environment variables

Use repeatable `-e KEY=VALUE` with `execute`:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py
```

Notes:

- `.levanter.yaml`, `.marin.yaml`, and `.config` environment entries are injected automatically.
- `execute` already wraps the command in `bash -c`; do not pass your own `bash -c`.

### Fast inner loop

Skip sync with `--no-sync` when the remote checkout is already current:

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

Run remote TPU commands sequentially.

### Copy remote artifacts

```bash
scp "dev-tpu-${TPU_NAME}:~/marin/<remote-path>" "<local-path>"
```

Common examples include profiles, traces, logs, and checkpoints. For example:

```bash
mkdir -p ".profiles/${TPU_NAME}"
scp "dev-tpu-${TPU_NAME}:~/marin/.profiles/<run_name>/plugins/profile/*/*" ".profiles/${TPU_NAME}/"
```

### Multiple clusters

When using multiple clusters at once, always pass explicit `--config` and `--tpu-name`.

Example naming:

- `infra/marin-us-central1.yaml` with `--tpu-name "${USER}-central1"`
- `infra/marin-us-east5-a.yaml` with `--tpu-name "${USER}-east5"`

## Troubleshooting

### `Could not infer TPU type from config`

Pass `--tpu-type` explicitly:

```bash
uv run scripts/ray/dev_tpu.py --config <config> allocate --tpu-type v5p-8
```

### `SSH configuration ... not found`

Run `allocate` first for that `--tpu-name`, then retry `connect` or `execute`.

### Verify cleanup after `allocate`

After finishing work, stop allocation with `Ctrl-C` in the terminal running `allocate`.

Recommended verification:

1. Confirm the allocator exited cleanly.
2. Confirm no local `allocate` process is still running for that TPU name.
3. Confirm the local alias state is cleaned up:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config <config> \
  --tpu-name <name> execute --no-sync -- /bin/bash -lc 'echo ok'
```

Expected result after cleanup: it should fail with `SSH configuration ... not found`.

### TPU busy or stale lockfile

If TPU init fails due to lock contention:

```bash
sudo rm -f /tmp/libtpu_lockfile
sudo lsof -t /dev/vfio/* | xargs -r sudo kill -9
```

Then rerun the command.

### `execute` feels slow

It syncs with `rsync` before each run by default. Use `--no-sync` or direct SSH for repeated runs.

## Reference Examples

Run tests:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -- uv run --package levanter --group test pytest lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py
```

Run a benchmark:

```bash
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/marin-us-east5-a.yaml \
  --tpu-name "$TPU_NAME" \
  execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- \
  uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_mlp_profile.py
```
