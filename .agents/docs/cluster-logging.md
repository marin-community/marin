# Consolidating Log Monitoring into ClusterManager

## Analysis

### Inventory of log monitoring code

#### `debug.py` (`lib/iris/src/iris/cluster/vm/debug.py`)

| Function | Lines | Purpose | GCP-specific? |
|----------|-------|---------|---------------|
| `collect_docker_logs()` | 30-90 | One-shot: SSH to VM, run `docker logs`, write to file | Yes (gcloud SSH) |
| `discover_controller_vm()` | 215-251 | Find controller VM by name pattern via `gcloud compute instances list` | Yes |
| `list_iris_tpus()` | 182-212 | List TPU VMs by label via `gcloud compute tpus tpu-vm list` | Yes |
| `cleanup_iris_resources()` | 93-179 | Delete VMs and TPUs (not logging, but used alongside) | Yes |
| `wait_for_port()` | 254-272 | Wait for TCP port (mode-agnostic utility) | No |
| `controller_tunnel()` | 275-352 | SSH tunnel context manager, used by `ClusterManager.connect()` | Yes |

#### `smoke-test.py` (`lib/iris/scripts/smoke-test.py`)

| Class/Function | Lines | Purpose | GCP-specific? |
|----------------|-------|---------|---------------|
| `LogTree` | 86-124 | Manages a directory of log artifacts with metadata | No |
| `SmokeTestLogger` | 175-227 | Dual-output (stdout + markdown file) logger with elapsed time | No |
| `DockerLogStreamer` | 230-335 | Background thread: streams `docker logs -f` from controller/workers via SSH | Yes |
| `TaskSchedulingMonitor` | 338-391 | Background thread: polls controller RPC for task scheduling state, writes JSON | No |

#### `cluster_manager.py` (`lib/iris/src/iris/cluster/vm/cluster_manager.py`)

| Member | Lines | Purpose |
|--------|-------|---------|
| `ClusterManager.start()` | 82-91 | Start controller, return address |
| `ClusterManager.stop()` | 93-98 | Stop controller |
| `ClusterManager.connect()` | 100-120 | Start + tunnel (GCP) or direct URL (local), stop on exit |

ClusterManager has **zero** log monitoring capability today.

### Duplication between `debug.py` and `smoke-test.py`

`DockerLogStreamer._stream_vm_logs()` (lines 295-335) constructs the same `gcloud compute [tpus tpu-vm] ssh ... --command "sudo docker logs ..."` command as `collect_docker_logs()` (lines 55-78), differing only in the `-f` flag. The SSH command construction appears in three places:

1. `collect_docker_logs()` — debug.py line 56-78 (snapshot mode)
2. `DockerLogStreamer._stream_vm_logs()` — smoke-test.py lines 299-322 (streaming mode)
3. `controller_tunnel()` — debug.py lines 312-333 (tunnel, different command but same gcloud SSH pattern)

`DockerLogStreamer._discover_and_stream_workers()` (lines 280-293) calls `list_iris_tpus()` directly, passing zone/project/label_prefix that it received from the caller — the same values `ClusterManager` already holds in `self._config`.

### What's GCP-specific vs mode-agnostic

**GCP-only:** `collect_docker_logs`, `discover_controller_vm`, `list_iris_tpus`, `cleanup_iris_resources`, `controller_tunnel`, `DockerLogStreamer` (all SSH-based).

**Mode-agnostic:** `wait_for_port`, `LogTree`, `SmokeTestLogger`, `TaskSchedulingMonitor` (uses controller URL, works for local and GCP).

---

## Design

### What should move into ClusterManager (and why)

**Docker log streaming** (`DockerLogStreamer`): ClusterManager owns zone, project, label_prefix, and knows whether the cluster is local or GCP. Callers currently must extract these values from the config and manually guard with `if not manager.is_local` (smoke-test.py lines 556-578). Moving streaming into ClusterManager eliminates this boilerplate and ensures streaming lifecycle is tied to cluster lifecycle.

**Worker discovery for streaming**: The loop in `DockerLogStreamer._discover_and_stream_workers()` (lines 280-293) queries `list_iris_tpus()` periodically. ClusterManager is the natural owner of cluster topology knowledge.

### What should stay in callers (and why)

**`TaskSchedulingMonitor`**: Test-specific. It tracks particular job IDs (via `track_job()`), writes JSON snapshots to a caller-chosen directory, and logs to a caller-provided `SmokeTestLogger`. This is observability for a specific test workflow, not a cluster concern.

**`LogTree` and `SmokeTestLogger`**: Presentation-layer. Markdown formatting, artifact tracking, dual-output logging are specific to the smoke test's reporting needs. Other callers (CLI, demos) would want different output formats.

### What should stay in debug.py (and why)

All existing functions stay. They are low-level utilities:

- `collect_docker_logs()`: One-shot debug collection, useful independently of ClusterManager.
- `discover_controller_vm()`, `list_iris_tpus()`: Discovery primitives used by both ClusterManager internals and debug scripts.
- `controller_tunnel()`: Already used by ClusterManager.connect() (line 115).
- `cleanup_iris_resources()`: Resource cleanup, not a logging concern.

A new `stream_docker_logs()` function should be added to debug.py as the streaming counterpart to `collect_docker_logs()`, deduplicating the SSH command construction.

### Proposed interfaces

#### New function in `debug.py`

```python
def stream_docker_logs(
    vm_name: str,
    container_name: str,
    zone: str,
    project: str,
    output_file: Path,
    is_tpu: bool = False,
    stop_event: threading.Event | None = None,
) -> None:
    """Stream docker logs from a VM to a file until stop_event is set.

    Blocking call. Uses `docker logs -f` via gcloud SSH. This is the
    streaming counterpart to collect_docker_logs(), sharing the same
    SSH command construction.
    """
```

This replaces `DockerLogStreamer._stream_vm_logs()` (smoke-test.py lines 295-335) and reuses the SSH command pattern from `collect_docker_logs()` (debug.py lines 55-78).

#### New class in `cluster_manager.py`

```python
class LogStreamHandle:
    """Opaque handle for stopping background log streaming."""

    def stop(self) -> None:
        """Stop all background streaming threads."""

    def discovered_workers(self) -> list[str]:
        """Names of workers discovered so far (informational)."""
```

#### New methods on `ClusterManager`

```python
class ClusterManager:
    # existing: start(), stop(), connect(), is_local, controller

    def start_log_streaming(
        self,
        output_dir: Path,
        controller_container: str = "iris-controller",
        worker_container: str = "iris-worker",
    ) -> LogStreamHandle | None:
        """Start background docker log streaming for all cluster components.

        Returns a handle to stop streaming, or None for local mode
        (local workers log via Python logging; no Docker containers exist).

        GCP mode: streams from controller VM and auto-discovered worker TPUs.
        Uses zone/project/label_prefix from self._config internally.
        """
```

#### Extended `connect()` signature

```python
    @contextmanager
    def connect(
        self,
        log_dir: Path | None = None,
    ) -> Iterator[str]:
        """Start controller, optionally start log streaming, yield URL, cleanup.

        If log_dir is provided, calls start_log_streaming() automatically
        and stops it on exit. This removes ~25 lines of boilerplate from
        callers like smoke-test.py.
        """
```

### How local mode would work for logging

- `start_log_streaming()` returns `None` — there are no VMs or Docker containers.
- Local workers run in-process and log via Python's `logging` module (the `iris.*` logger hierarchy).
- Callers who want local logs to disk add a `logging.FileHandler` to the `iris` logger. No new abstraction needed.
- `TaskSchedulingMonitor` works identically in both modes (it uses the controller URL, not SSH).

---

## Migration Plan

### Step 1: Extract `stream_docker_logs()` into debug.py

**Goal:** Single source of truth for SSH command construction.

**Changes:**
- `debug.py`: Add `stream_docker_logs()` that shares SSH command logic with `collect_docker_logs()`. Extract a helper `_docker_log_ssh_command(vm_name, container_name, zone, project, is_tpu, follow: bool) -> list[str]` used by both functions.
- `smoke-test.py`: `DockerLogStreamer._stream_vm_logs()` (lines 295-335) calls `stream_docker_logs()` instead of building the command inline.

**Files:** `debug.py`, `smoke-test.py`

**Verification:** Run smoke test with `--local` flag (no GCP calls) to confirm no regressions. Manually verify GCP mode if available, or write a unit test that mocks `subprocess.Popen` and asserts the command matches.

### Step 2: Add `LogStreamHandle` and `start_log_streaming()` to ClusterManager

**Goal:** ClusterManager owns streaming lifecycle.

**Changes:**
- `cluster_manager.py`: Add `LogStreamHandle` class and `start_log_streaming()` method. Internally creates threads that call `stream_docker_logs()` (from debug.py) and `list_iris_tpus()` (for worker discovery). Uses `self._config.zone`, `self._config.project_id`, `self._config.label_prefix` — no caller wiring needed.

**Files:** `cluster_manager.py`

**Verification:**
- Unit test: construct `ClusterManager` with a GCP-shaped config, mock `subprocess.Popen` and `list_iris_tpus`, call `start_log_streaming(tmp_path)`, assert threads start and `handle.stop()` joins cleanly.
- Unit test: construct `ClusterManager` with local config, assert `start_log_streaming()` returns `None`.

### Step 3: Add `log_dir` parameter to `connect()`

**Goal:** One-line observability for callers.

**Changes:**
- `cluster_manager.py`: Extend `connect()` to accept optional `log_dir: Path`. When provided, calls `start_log_streaming()` after `start()` and stops the handle in the `finally` block.

**Files:** `cluster_manager.py`

**Verification:**
- Integration test with local mode: `connect(log_dir=tmp_path)` succeeds, no crash, returns `None` handle internally.
- Integration test verifying `connect()` without `log_dir` is unchanged.

### Step 4: Simplify smoke-test.py

**Goal:** Remove `DockerLogStreamer` class, replace with `ClusterManager` API.

**Changes:**
- `smoke-test.py`: Remove `DockerLogStreamer` class (lines 230-335). Replace manual wiring (lines 556-578) with either:
  - `manager.connect(log_dir=self.log_tree.root)` — if the default output structure matches, or
  - Explicit `handle = manager.start_log_streaming(self.log_tree.root)` inside the `with manager.connect()` block, with `handle.stop()` in cleanup.
- Remove `from iris.cluster.vm.debug import list_iris_tpus` import (line 67) if no longer used directly.

**Files:** `smoke-test.py`

**Verification:**
- Run `uv run python lib/iris/scripts/smoke-test.py --config <config> --local` — passes, log directory created.
- Verify log files appear in expected paths under `log_tree.root`.
- `TaskSchedulingMonitor` and `SmokeTestLogger` remain unchanged and functional.

---

## What this does NOT change

- `TaskSchedulingMonitor` stays caller-owned (test-specific concern)
- `SmokeTestLogger` / `LogTree` stay caller-owned (presentation concern)
- `collect_docker_logs()` stays in debug.py (one-shot utility)
- `cleanup_iris_resources()` stays in debug.py (not a logging concern)
- No new config proto fields needed
- No changes to `local_platform.py` or `controller.py`
