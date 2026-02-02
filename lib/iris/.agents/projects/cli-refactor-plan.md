# Iris CLI Refactor Plan

**Issue:** [#2626](https://github.com/marin-community/marin/issues/2626)

## Current State

Four separate CLI modules with scattered, overlapping functionality:

| File | Entry Point | Purpose | Lines |
|------|-------------|---------|-------|
| `src/iris/cli.py` | `iris` | Main CLI: cluster lifecycle, build, submit, autoscaler, slice, VM, RPC commands | 1632 |
| `src/iris/iris_run.py` | `iris-run` | Job submission with env var loading, resource building, tunneling | 497 |
| `src/iris/rpc_cli.py` | (imported by cli.py) | Generic RPC CLI infrastructure: service discovery, dynamic Click commands from protobuf | 426 |
| `scripts/cluster-tools.py` | `uv run python scripts/cluster-tools.py` | Debug/validate: discover, ssh-status, tunnel, health, logs, cleanup, validate | 1030 |

Supporting module:
- `src/iris/cluster/vm/debug.py` — `controller_tunnel()`, `discover_controller_vm()`, log streaming, cleanup helpers

### Problems

1. **`cli.py` is 1632 lines** — cluster lifecycle, build, submit, autoscaler, slice, VM commands all in one file.
2. **`iris-run` vs `iris submit`** — two different job submission CLIs with different capabilities. `iris-run` supports `--tpu`, `--replicas`, `--env-vars`, `--extra`; `iris submit` only takes a Python script with `main()`.
3. **`cluster-tools.py` is a standalone script** — duplicates tunneling logic from `debug.py`, provides debug commands (discover, logs, cleanup, validate) that should be part of the main CLI.
4. **Inconsistent logging initialization** — `cli.py` uses `_configure_logging()`, `iris_run.py` uses `logging.basicConfig()`, `cluster-tools.py` doesn't configure at all.
5. **No `iris cluster dashboard` command** — users must manually use `cluster-tools.py tunnel` or raw `gcloud compute ssh` to access the controller dashboard.

## Proposed Structure

```
src/iris/cli/
├── __init__.py          # Re-export `iris` group for entry point
├── main.py              # Top-level `iris` group + logging setup
├── cluster.py           # cluster start/stop/restart/reload/status/init/dashboard
├── controller.py        # cluster controller start/stop/restart/reload/status/run-local
├── autoscaler.py        # cluster autoscaler status
├── slice.py             # cluster slice create/terminate/list/get
├── vm.py                # cluster vm status/logs/get
├── build.py             # build worker-image/controller-image/push
├── submit.py            # iris submit (merged iris-run capabilities)
├── rpc.py               # ServiceCommands (moved from rpc_cli.py)
├── debug.py             # cluster debug discover/ssh-status/health/logs/bootstrap-logs/cleanup/validate
└── _helpers.py          # Shared helpers: handle_error, format_timestamp, format_status_table, load_autoscaler, etc.
```

Entry points in `pyproject.toml`:
```toml
[project.scripts]
iris = "iris.cli:iris"
iris-run = "iris.cli.submit:iris_run_main"   # backward compat (or deprecate)
```

## Step-by-Step Plan

### Step 1: Create `cli/` package skeleton with `main.py` and `_helpers.py`

- Create `src/iris/cli/__init__.py` that exports the `iris` Click group.
- Move the top-level `iris` group, logging setup, and `handle_error` to `cli/main.py`.
- Extract shared helpers (`_format_timestamp`, `_format_status_table`, `_get_autoscaler_status`, `_load_autoscaler`, `_require_config`, image-related helpers) into `cli/_helpers.py`.
- Update `pyproject.toml` entry point: `iris = "iris.cli:iris"`.
- Delete old `src/iris/cli.py`.
- Test: `uv run iris --help` still works.

### Step 2: Split command groups into separate modules

Move each command group to its own module, importing from `_helpers.py`:

- `cli/cluster.py` — `cluster` group + start/stop/restart/reload/status/init commands
- `cli/controller.py` — `controller` subgroup commands
- `cli/autoscaler.py` — `autoscaler` subgroup commands
- `cli/slice.py` — `slice` subgroup commands
- `cli/vm.py` — `vm` subgroup commands
- `cli/build.py` — `build` group commands
- `cli/submit.py` — `submit` command (current `iris submit`)

Each module defines its Click group/commands and is registered in `cli/main.py` via `iris.add_command(...)`.

- Test: all existing `iris` subcommands still work.

### Step 3: Move `rpc_cli.py` → `cli/rpc.py`

- Move the `ServiceCommands` class and supporting functions from `src/iris/rpc_cli.py` to `cli/rpc.py`.
- Delete `src/iris/rpc_cli.py`.
- Update import in `cli/main.py`.

### Step 4: Merge `iris-run` into `cli/submit.py`

The current `iris submit` is limited (only takes a Python script with `main()`). The `iris-run` CLI is more capable (command passthrough, `--tpu`, `--replicas`, `--env-vars`, `--extra`).

- Move `iris_run.py` core logic (`load_cluster_config`, `load_env_vars`, `add_standard_env_vars`, `build_resources`, `run_iris_job`) into `cli/submit.py`.
- Keep the `iris-run` entry point as a thin wrapper in `cli/submit.py` for backward compatibility.
- Enhance `iris submit` to support the full `iris-run` feature set (or keep both and have them share implementation).
- Delete `src/iris/iris_run.py`.

### Step 5: Absorb `cluster-tools.py` into `cli/debug.py`

Move debug/operational commands from `scripts/cluster-tools.py` into a new `iris cluster debug` subgroup:

```
iris cluster debug discover          # Find controller VM
iris cluster debug ssh-status        # Docker status on controller
iris cluster debug health            # Health check (auto-tunnels)
iris cluster debug logs [--follow]   # Controller docker logs
iris cluster debug bootstrap-logs    # VM startup logs
iris cluster debug cleanup [--no-dry-run]
iris cluster debug validate          # Submit test jobs
iris cluster debug list-workers      # List registered workers
iris cluster debug list-jobs         # List all jobs
```

- Reuse `controller_tunnel()` from `cluster/vm/debug.py` (no duplication).
- Delete `scripts/cluster-tools.py` or make it a thin shim that imports from the CLI.

### Step 6: Add `iris cluster dashboard` command

New command that establishes an SSH tunnel to the controller and prints the dashboard URL:

```bash
# Usage
iris cluster --config=examples/eu-west4.yaml dashboard

# Output
Discovering controller VM in europe-west4-b...
Found controller: iris-controller-abc123
Establishing SSH tunnel (localhost:10000 -> iris-controller-abc123:10000)...
Tunnel ready.

Dashboard: http://localhost:10000
Controller RPC: http://localhost:10000

Press Ctrl+C to close tunnel.
```

Implementation in `cli/cluster.py`:

```python
@cluster.command("dashboard")
@click.option("--port", default=10000, type=int, help="Local port for tunnel")
@click.pass_context
def cluster_dashboard(ctx, port: int):
    """Open SSH tunnel to controller and print dashboard URL."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: --config is required", err=True)
        raise SystemExit(1)

    zone = config.zone
    project = config.project_id
    label_prefix = config.label_prefix or "iris"

    if not zone or not project:
        click.echo("Error: Config must specify zone and project_id", err=True)
        raise SystemExit(1)

    click.echo(f"Discovering controller in {zone}...")

    from iris.cluster.vm.debug import discover_controller_vm, controller_tunnel
    import signal, threading

    vm_name = discover_controller_vm(zone, project, label_prefix)
    if not vm_name:
        click.echo(f"No controller VM found in zone {zone}", err=True)
        raise SystemExit(1)

    click.echo(f"Found controller: {vm_name}")
    click.echo(f"Establishing SSH tunnel (localhost:{port} -> {vm_name}:10000)...")

    # Use controller_tunnel but block until Ctrl+C
    stop = threading.Event()

    def on_signal(sig, frame):
        click.echo("\nClosing tunnel...")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    with controller_tunnel(zone, project, local_port=port, label_prefix=label_prefix) as url:
        click.echo(f"\nDashboard:      {url}")
        click.echo(f"Controller RPC: {url}")
        click.echo("\nPress Ctrl+C to close tunnel.")
        stop.wait()
```

### Step 7: Standardize logging

All CLI modules use the same `configure_logging()` from `iris.logging`, called once in `cli/main.py` based on `--verbose`.

### Step 8: Update documentation

- Update `README.md` CLI Reference section with new command structure.
- Update `AGENTS.md` Key Modules section with new `cli/` directory.

## Migration Notes

- The `iris` entry point signature does not change — all existing commands keep the same invocation syntax.
- `iris-run` continues to work (thin wrapper or alias).
- `scripts/cluster-tools.py` can be kept temporarily as a shim that prints a deprecation notice and delegates to `iris cluster debug ...`.

## New Command Summary

| New Command | Source |
|-------------|--------|
| `iris cluster dashboard` | NEW — SSH tunnel + dashboard URL |
| `iris cluster debug discover` | From `cluster-tools.py` |
| `iris cluster debug ssh-status` | From `cluster-tools.py` |
| `iris cluster debug health` | From `cluster-tools.py` |
| `iris cluster debug logs` | From `cluster-tools.py` |
| `iris cluster debug bootstrap-logs` | From `cluster-tools.py` |
| `iris cluster debug cleanup` | From `cluster-tools.py` |
| `iris cluster debug validate` | From `cluster-tools.py` |
| `iris cluster debug list-workers` | From `cluster-tools.py` |
| `iris cluster debug list-jobs` | From `cluster-tools.py` |
