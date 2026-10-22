# Maintain the vendored SkyRL runtime

SkyRL lives in `vendor/skyrl` with its own `pyproject.toml` and `uv.lock`.
Source and dependency changes land directly in Marin. SkyRL is excluded from
Marin's workspace and root dependency resolution.

Marin invokes the nested `marinskyrl` executable in an isolated child process:

```bash
uv run --isolated --project vendor/skyrl --frozen --extra cpu --no-group dev marinskyrl --help
```

The launcher clears the parent's Python environment selection. SkyRL installs
Marin libraries and Harbor as editable dependencies from this checkout, using
the paths in `vendor/skyrl/pyproject.toml`. Marin calls SkyRL through the executable.
See [launching RL experiments](../references/rl-launching.md)
for the artifact entry point, launch document, and resource configuration.

The Marin CLI uses SkyRL's `cpu` extra. Set `environment.task_sessions.local: {}`
to enable the local task backend, which probes Linux bubblewrap before task
admission. Omitting this key or setting it to `null` disables the backend.
The launcher forwards `iris.container_profile` to the coordinator and trainer;
local execution requires a successful sandbox capability check.

## Runtime and dependencies

Iris bundles the Marin repository through its standard workspace submission.
Workers run SkyRL from `/app/vendor/skyrl` and install the project in editable
mode using the `megatron`, `vllm`, and `telemetry` extras from the nested lock.
`cloud/iris/bootstrap_runtime.sh` installs these extras for training. Marin artifacts and SkyRL run
records use `rigging.provenance` to record the checkout and launch context
alongside the resolved configuration. Use a fresh RL artifact version after
runtime changes unless checkpoint compatibility has been established.

The runtime supports Megatron training. Marin's curriculum arms use Megatron
for both Qwen and Snowball.

Marin library edits are available to SkyRL without publishing wheels. When
changing dependency declarations, refresh SkyRL's independent lock:

```bash
uv lock --project vendor/skyrl
```

Run the
[train, resume, and evaluation smoke](#train-resume-and-evaluate-through-marin)
to check runtime behavior after dependency changes.

The launcher lock targets Linux x86_64, Linux aarch64, and macOS ARM64.
The pinned Ray release has no Intel macOS wheel.

## CI checks

The `skyrl-wheel` unit lane checks the independent lock and runs the packaging,
launch configuration, checkpoint export, and verifier tests in SkyRL's CPU
environment. Changes to its editable dependencies also select this lane.
The lane discovers `test_*.py` files directly in `infra/tests`, `cloud/iris/tests`,
`skyrl-gym/tests`, and `skyrl-train/tests/cpu/rollouts`; new files matching these
globs need no workflow edit. The pytest configuration excludes the `docker`
marker. Run these tests locally from `vendor/skyrl`:

```bash
uv run --isolated --frozen --extra cpu --no-group dev --with pytest==8.4.2 --with pytest-asyncio==1.4.0 \
  pytest -q infra/tests/test_*.py cloud/iris/tests/test_*.py skyrl-gym/tests/test_*.py \
  skyrl-train/tests/cpu/rollouts/test_*.py
```

Check the lock locally with:

```bash
uv lock --project vendor/skyrl --check
```

`Marin - SkyRL Nightly` runs on-policy distillation, Grug/Megatron training and
serving checks, and asynchronous GPU CatCount at 09:00 UTC from the checked-out
Marin revision. Its Grug CUTLASS kernel check runs weekly or by manual selection;
see [the workflow](https://github.com/marin-community/marin/blob/main/.github/workflows/marin-skyrl-nightly.yaml)
for the selected tests. `Marin - SkyRL CPU Nightly` runs CatCount
learning and checkpoint/resume gates at 06:00 UTC using the nested lock.

## Train, resume, and evaluate through Marin

The artifact smoke uses Qwen2.5-0.5B-Instruct, procedural CatCount training
data, and two colocated H100s on `cw-rno2a`. It performs two Megatron steps,
saves a distributed checkpoint, exports HF weights, then submits a fresh
training job against that checkpoint root with a four-step limit. Its final
artifact evaluates the resumed HF policy on the 128-row GSM8K smoke preset
in Marin's separate serving environment.

Run from the repository root. Build and inspect the graph first:

```bash
uv run python -m experiments.post_training.skyrl_port_smoke \
  --version <new-version> --stage evaluate
```

Submit it with:

```bash
uv run python -m experiments.post_training.skyrl_port_smoke \
  --version <new-version> --stage evaluate --run
```

Choose a new immutable version for each runtime change or independent rerun.
Require successful training and export jobs, evidence that the fresh job
loads step two and advances to step four, a complete final export, and a
successful evaluation artifact. This smoke checks pipeline handoffs. Use
the nightly learning gates to assess learning behavior.
