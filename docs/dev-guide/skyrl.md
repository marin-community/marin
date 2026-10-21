# Maintain the vendored SkyRL runtime

SkyRL lives in `vendor/skyrl` with its own `pyproject.toml` and `uv.lock`.
Source and dependency changes land directly in Marin. SkyRL is excluded from
Marin's workspace and root dependency resolution.

Marin invokes the nested `marinskyrl` executable in an isolated child process:

```bash
uv run --isolated --project vendor/skyrl --frozen --no-group dev marinskyrl --help
```

The launcher clears the parent's Python environment selection. SkyRL uses
published Marin libraries inside its own environment; Marin calls SkyRL
through the executable. See [launching RL experiments](../references/rl-launching.md)
for the artifact entry point, launch document, and resource configuration.

## Runtime and dependencies

The launcher copies the checked-out runtime files, assets, and independent
lock into an Iris workspace. Workers install the bundled project in editable
mode using the `megatron`, `vllm`, and `telemetry` extras from the nested lock.
`cloud/iris/bootstrap_runtime.sh` installs these extras for training. Marin artifacts and SkyRL run
records use `rigging.provenance` to record the checkout and launch context
alongside the resolved configuration. Use a fresh RL artifact version after
runtime changes unless checkpoint compatibility has been established.

The runtime supports Megatron training. Marin's curriculum arms use Megatron
for both Qwen and Snowball.

Local editable Marin packages are not wired into the runtime bundle. To
advance published Marin libraries within SkyRL's declared version constraints:

```bash
uv run config/update-external.py --skyrl-marin-packages
```

The updater resolves the nested lock and checks imported Marin symbols and
explicit constructor keywords in the nested environment. A failed check
restores the previous lock. Edit `vendor/skyrl/pyproject.toml` first when
changing an explicit version constraint. Run the
[train, resume, and evaluation smoke](#train-resume-and-evaluate-through-marin)
to check runtime behavior after dependency changes.

The launcher lock targets Linux x86_64, Linux aarch64, and macOS ARM64.
The pinned Ray release has no Intel macOS wheel.

## CI checks

The `skyrl-wheel` unit lane checks the independent lock, isolated Marin import
contract, and built wheel against the current source:

```bash
uv run --no-project python -m infra.ci.check_skyrl
```

It also exercises source bundles, launch configuration, checkpoint export,
and verifier assets in the isolated CPU environment.

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
