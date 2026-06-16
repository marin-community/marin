# Self-running job examples (issue #6415)

**Status:** implemented (codex-reviewed). See the PR for the final diff.
**Weaver:** #190.

## Problem

Every grug/tutorial example today is launched as a *two-hop* submission:

```bash
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.base.launch
```

`iris job run` submits a **launcher job** (a CPU worker) that runs the example.
The example then calls `current_client()`, which — *because it is now inside an
Iris task* — auto-detects the Iris context and submits the **real** training job
(e.g. `v5p-8`). The cluster/region/launcher-resources all live in the
hand-typed `iris job run` line, not in the script. That hand-typed line is the
error surface the issue calls out:

- forgot the launcher resources / over-requested them (people think the
  launcher needs a TPU; it doesn't);
- wrong region;
- (reservations — **explicitly dropped from scope** per maintainer).

## Goal

Make the examples **self-running**: `uv run python experiments/grug/base/launch.py
--cluster=marin` connects to the cluster *from the dev box*, hoists the Iris
client itself, and submits the training job directly — no launcher job. Expose
the knobs the script actually wants (`--region`, `--tpu-type`) as its own CLI
flags. Warn if the script is accidentally run *inside* `uv run iris` (the old
two-hop), which means the user double-wrapped.

## Key findings from code exploration

### 1. Both submission paths funnel through `current_client()`

The ~13 example scripts use one of two submit paths, **both** of which resolve
the cluster via fray's `current_client()`:

| Path | Scripts | Mechanism |
|---|---|---|
| `executor_main(steps=[...])` | ~10 (most tutorials, grug/moe, grug/modular_opt) | `StepRunner` submits each `ExecutorStep` via `current_client().submit()` (`executor.py:320,541`) |
| `_submit_train_job` / `train` / direct `current_client().submit()` | grug/base, train_tiny_model_tpu, train_tiny_sweep_tpu | `defaults.py:406` `client = current_client(); client.submit(...)` |

`fray.current_client()` (`lib/fray/src/fray/current_client.py:30`) resolution
order: **(1)** explicitly-set client (`set_current_client`) → **(2)** auto-detect
Iris env (`get_iris_ctx()`) → **(3)** `LocalClient()`.

**Consequence:** a single hoist — `set_current_client(<remote FrayIrisClient>)`
— makes *both* paths submit to the cluster. We do not need to touch
`executor_main`, `_submit_train_job`, or any submit internals.

### 2. The connection machinery exists but is 100% click-coupled

There is **no callable SDK** that connects a dev box to a cluster. The full
sequence is spread across the click command tree:

- `iris.cli.main.iris()` group (`main.py:112-160`): resolve `--cluster` →
  config file (`resolve_cluster_config(name, IRIS_CLUSTER_CONFIG_DIRS)`),
  `IrisConfig.load`, `_configure_client_s3(proto)`, token via
  `create_client_token_provider(proto.auth, name)` or `load_token`/`load_any_token`.
- `iris.cli.connect.require_controller_url(ctx)` (`connect.py:83-129`):
  `IrisConfig(config).provider_bundle()` → discover controller address →
  `bundle.controller.tunnel(address)` (a context manager) → tunnel URL.
- each command: `IrisClient.remote(controller_url, workspace=Path.cwd(),
  token_provider=...)` (`job.py:697` etc).

The wrap to a fray client already exists:
`FrayIrisClient.from_iris_client(iris_client)` (`iris_backend.py:561`) — designed
exactly for "wrap an already-connected IrisClient" — and
`fray.set_current_client(client)` is a context manager (`current_client.py:52`).

**`workspace=Path.cwd()` is load-bearing:** `IrisClient.remote` bundles the
workspace and ships it to the workers. A dev-box driver must pass it or the
workers get no code.

### 3. draccus constraints (verified empirically)

- Passing an explicit config to a `@draccus.wrap()` fn **bypasses** argv parsing
  (`executor_main(cfg, steps=...)` does not re-parse) — this is the existing
  `experiments/datakit/testbed/baseline.py` pattern.
- `draccus.parse` **rejects unknown args** (`SystemExit`) — so two independent
  parses over one `argv` is impossible. We need **one** config.
- Embedding a sub-config parses with dotted flags (`--executor.dry_run=True`).

→ The launcher must own a *single* draccus config that **embeds**
`ExecutorMainConfig`, then call `executor_main(config.executor, steps=...)`
explicitly (no re-parse). Executor flags move from `--dry_run` to
`--executor.dry_run`. (Acceptable; examples rarely pass them.)

### 4. `--reserve` is out (dropped) — and would have needed fray plumbing anyway

`IrisClient.submit` accepts `reservation=`, but fray's `JobRequest` /
`ResourceConfig` have **no** reservation field, and `FrayIrisClient.submit`
doesn't pass one. So reservations never flowed through the example submit path.
Maintainer confirmed reservations are not needed → no fray changes.

`--region`/`--zone`/`--tpu-type` *are* native to `ResourceConfig`
(`fray/types.py:314`: `regions`, `zone`, `with_tpu`) — cleanly overridable with
`dataclasses.replace` / `ResourceConfig.with_tpu`.

## Recommendation

Two small pieces, split along the existing dependency boundary
(`iris` → `fray`/`levanter`/`zephyr` → `marin`):

### A. `iris` — a click-free `connect_to_cluster` SDK helper

New module `lib/iris/src/iris/client/connect.py`:

```python
@contextlib.contextmanager
def connect_to_cluster(
    cluster: str, *, workspace: Path | None = None, timeout_ms: int = 30_000
) -> Iterator[IrisClient]:
    """Resolve a named cluster, open a tunnel, yield a connected IrisClient.

    The non-click twin of the `iris` CLI connection path. Resolves `cluster`
    via the same config search dirs, builds the provider bundle, discovers and
    tunnels to the controller, loads the stored `iris login` token, and yields
    `IrisClient.remote(...)`. Closes the client and tunnel on exit.
    """
```

It reuses the *exact* primitives the CLI uses (`resolve_cluster_config`,
`IrisConfig`, `provider_bundle`, `discover_controller`, `tunnel`,
`configure_client_s3`, token resolution, `IrisClient.remote`). To avoid an
`iris.client → iris.cli` import inversion, relocate the three click-free
primitives the CLI currently defines in `iris.cli.*` —
`IRIS_CLUSTER_CONFIG_DIRS`, `resolve_cluster_name`, `create_client_token_provider`
— into this click-free module, and have `iris.cli.main` import them from here.
That is the **dedup win**: one connection-primitive home, used by both the CLI
and the SDK. (We leave `require_controller_url`'s lazy-tunnel-on-`ctx` behavior
in place; it can adopt `connect_to_cluster` in a follow-up. Light touch now.)

Handles the `controller == local` branch (start a `LocalCluster`) the same way
`require_controller_url` does, so `--cluster=local` works for tests.

### B. `experiments` — a tiny shared launcher (`experiments/launch.py`)

```python
@dataclass
class LaunchConfig:
    cluster: str | None = None     # None => local (LocalClient), no connect
    region: str | None = None
    zone: str | None = None
    tpu_type: str | None = None
    local: bool = False            # force local even if --cluster is set
    executor: ExecutorMainConfig = field(default_factory=ExecutorMainConfig)

@contextlib.contextmanager
def launch_session(config: LaunchConfig) -> Iterator[None]:
    """Hoist the Iris client for a self-running example, or run locally.

    - Warn (and do NOT connect) if already inside an Iris job — the user
      double-wrapped via `uv run iris job run`; fall back to the in-cluster
      client so the old flow still works.
    - If --local or no --cluster: yield with no hoist (LocalClient fallback).
    - Else connect_to_cluster(...) -> FrayIrisClient.from_iris_client ->
      set_current_client, then yield.
    """
    if get_job_info() is not None:
        logger.warning(
            "This script is running INSIDE an Iris job. Self-running examples "
            "no longer need `uv run iris job run` — run them directly, e.g. "
            "`uv run python %s --cluster=marin`. Using the in-cluster client.",
            sys.argv[0],
        )
        yield
        return
    if config.local or config.cluster is None:
        yield
        return
    with connect_to_cluster(config.cluster, workspace=Path.cwd()) as iris_client:
        with set_current_client(FrayIrisClient.from_iris_client(iris_client)):
            yield

def override_resources(resources: ResourceConfig, config: LaunchConfig) -> ResourceConfig:
    """Apply --tpu-type / --region / --zone overrides (CLI > script default)."""
    if config.tpu_type:
        resources = ResourceConfig.with_tpu(
            config.tpu_type, cpu=resources.cpu, ram=resources.ram, disk=resources.disk
        )
    repl = {}
    if config.region is not None:
        repl["regions"] = [config.region]
    if config.zone is not None:
        repl["zone"] = config.zone
    return dataclasses.replace(resources, **repl) if repl else resources
```

This is the "small CLI library with draccus-style parsing" the maintainer
blessed: ~60 LOC, no inheritance, embeds the executor config, no magic.

### Per-script conversion

**executor_main scripts** (the majority):

```python
@draccus.wrap()
def main(config: LaunchConfig):
    with launch_session(config):
        executor_main(config.executor, steps=[...], description="...")

if __name__ == "__main__":
    main()
```

When the script wants `--tpu-type`/`--region` to bite, build the step's
`ResourceConfig` through `override_resources(...)` inside `main()` (move the
module-scope `ResourceConfig.with_tpu(...)` into `main`).

**Non-executor scripts** (grug/base, train_tiny_model_tpu, train_tiny_sweep_tpu):

```python
@draccus.wrap()
def main(config: LaunchConfig):
    with launch_session(config):
        resources = override_resources(_GRUG_BASE_RESOURCES, config)
        train_grug(name="grug/base-trial",
                   launch=dataclasses.replace(grug_base_launch, resources=versioned(resources)))
```

### Scripts to convert (verbatim inventory)

grug: `base/launch.py`, `moe/launch.py`, `moe/launch_cw_scale.py`,
`modular_opt/launch.py`.
tutorials: `hello_world.py`, `train_tiny_model_cpu.py`, `train_tiny_model_gpu.py`,
`train_tiny_model_tpu.py`, `train_tiny_sweep_tpu.py`, `train_tiny_sweep_dclm_tpu.py`,
`train_125m_fineweb_edu_gpu.py`, `exp1077_reproduce_dclm_1b1x.py`,
`exp1078_reproduce_dclm_7b1x.py`.

Docs/READMEs to update (drop the `iris job run -- python -m ...` invocation in
favor of `uv run python <script> --cluster=marin --tpu-type=...`):
`experiments/grug/README.md`, `docs/tutorials/train-an-lm.md`,
`docs/explanations/executor.md`, and the `exp10xx` docstrings.

### Misuse-warning semantics

- `get_job_info() is not None` → warn + **do not** hoist (fall back to the
  auto-detected in-cluster client). The old two-hop keeps working; we only nudge.
  Warn-and-continue, not hard-fail (issue says "add a warning").

### Behavior matrix

| Invocation | `get_job_info()` | result |
|---|---|---|
| `python launch.py --cluster=marin` (dev box) | None | connect from dev box, submit job ✅ (new) |
| `python launch.py` / `--local` (dev box) | None | LocalClient (current local behavior) |
| `uv run iris job run -- python launch.py` (old way) | not None | warn + in-cluster client (still works) |

## Post-review revisions (after `codex` adversarial pass)

Codex confirmed the core claim (one `set_current_client` hoist reaches both
submit paths; explicit-config `executor_main` skips only parsing). It surfaced
four real gaps, all rooted in **the driver moving from a regional cluster CPU
worker to a region-less laptop**. Revised decisions:

1. **`MARIN_PREFIX` must be a real regional bucket on the driver (fixes the
   `/tmp/marin` footgun).** `executor_main` computes step output paths from
   `config.prefix or marin_prefix()`, and `marin_prefix()` falls back to
   `/tmp/marin` with no `MARIN_PREFIX`/metadata (`filesystem.py:157`). Same
   `MARIN_PREFIX` drives `resolve_training_env`'s JAX compilation-cache bucket
   via `marin_temp_bucket` (`training.py:424`). On a CPU launcher worker these
   resolve to a regional bucket; on a laptop they don't.
   **Resolution (idiomatic — matches `experiments/datakit/.../exp_full_clusters.py:40`
   and `datakit_tier2_skewed_ferry.py:160`):** `launch_session`, when connecting
   to a cluster, ensures `MARIN_PREFIX` is set: prefer an explicit
   `--executor.prefix` / existing `MARIN_PREFIX`; else derive from `--region`
   (`gs://` + `_REGION_TO_MARIN_BUCKET_OVERRIDES.get(region, "marin-"+region)`);
   else **fail fast** with a message telling the user to pass `--region` or set
   `MARIN_PREFIX`. Never silently use `/tmp/marin` while a cluster is attached.

2. **`override_resources` must preserve all scheduling fields.** The sketch
   rebuilt via `ResourceConfig.with_tpu(tpu_type, cpu=, ram=, disk=)`, which
   drops `replicas`/slice_count, `device_alternatives`, `preemptible`, `image`,
   `regions`, `zone` (`types.py:330,353,384`). **Resolution:** swap only the
   device variant with `dataclasses.replace(resources, device=replace(device,
   variant=tpu_type))`, preserving everything else; **reject** (fail fast) when
   the new variant's `vm_count` differs from the script default or when
   `device_alternatives` is set (multi-slice/flex configs must be edited in the
   script, not flipped on the CLI). `--region`/`--zone` apply via
   `dataclasses.replace`.

3. **Hard-fail `--cluster` inside an Iris job.** `get_job_info() is not None`
   *and* `--cluster` explicitly set is a contradiction (the user double-wrapped
   *and* asked to re-route) → raise. `get_job_info() is not None` with **no**
   `--cluster` is the legacy two-hop (`uv run iris job run -- python …`) → warn
   + fall back to the in-cluster client (back-compat preserved).

4. **`connect_to_cluster` lifecycle is owned explicitly** (codex #6/#7): a
   context manager that, in order, resolves config → loads `IrisConfig` →
   `configure_client_s3` → builds token provider → (local-controller branch:
   start+own a `LocalCluster`) → discovers/falls-back controller address →
   opens the tunnel (closed on exit) → yields `IrisClient.remote(url,
   workspace=<repo root>, token_provider=…)` (closed on exit). `workspace` is
   the **git repo root** (validated to contain `pyproject.toml`), not a blind
   `Path.cwd()`; the bundle is git-tracked files, 25 MB cap (`bundle.py:16`).

5. **Inheritance change is documented, not silently dropped** (codex #2). The
   boot scripts pass explicit `env`/`extras` into their `JobRequest`
   (`defaults.py:403`, `step_runner.py:382`), so child env/extras do not depend
   on parent inheritance. Worker-region pinning that the parent JobInfo used to
   add is replaced by: the executor's own region inference
   (`_regions_for_tpu_variant_from_iris`, which queries the autoscaler over the
   tunnel and still works from a laptop) plus the explicit `--region` flag;
   training path resolution remains deferred to the worker's region. Net: region
   is chosen by `--region` or scheduler placement, and checkpoints still land in
   the worker's region. This is stated in the READMEs.

## Open questions for review

1. **Home for `connect_to_cluster`** — `iris.client.connect` (preferred; keeps
   cluster resolution in iris) vs `fray`. Recommend iris; fray-wrap stays in the
   experiments helper.
2. **Relocating CLI primitives** (`IRIS_CLUSTER_CONFIG_DIRS`,
   `resolve_cluster_name`, `create_client_token_provider`) into the click-free
   module — worth the dedup, or keep `connect_to_cluster` self-contained and
   leave the CLI untouched to minimize blast radius?
3. **Executor flag UX** — embedding `ExecutorMainConfig` makes `--dry_run`
   become `--executor.dry_run`. Accept, or flatten the handful of common
   executor flags onto `LaunchConfig` directly?
4. **Default cluster** — `--cluster` default `None` (explicit, local unless
   asked) vs default `"marin"` (examples target prod by default). Recommend
   `None` for safety; READMEs show `--cluster=marin`.
