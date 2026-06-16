# Smoke Test Review Fix Plan

## Issue 1: `from __future__ import annotations` in `debug.py`

**File:** `lib/iris/src/iris/cluster/vm/debug.py`, line 17

**Problem:** AGENTS.md says "Don't use TYPE_CHECKING. Use the real import." The `from __future__ import annotations` import defers all annotation evaluation, which is the same category of lazy-typing hack. It also breaks runtime type inspection (e.g., `get_type_hints()`). In this file, every annotation is already valid at runtime (`Path | None`, `str | None`, `Iterator[str]`), so the import is unnecessary on Python 3.11+.

**Fix:** Remove line 17 (`from __future__ import annotations`). No other changes needed; all annotations in this file use union syntax (`X | Y`) which is valid at runtime on 3.11+.

```python
# DELETE this line:
from __future__ import annotations
```

---

## Issue 2: Zombie processes in `_stream_vm_logs`

**File:** `lib/iris/scripts/smoke-test.py`, lines 289-295 (`DockerLogStreamer._stream_vm_logs`)

**Problem:** `proc.terminate()` is called but `proc.wait()` is never called. The terminated process becomes a zombie until the parent (Python) process exits. If many workers are discovered, this leaks zombie processes.

**Fix:** Add `proc.wait()` after `proc.terminate()`, with a fallback to `proc.kill()`:

```python
def _stream_vm_logs(self, vm_name: str, log_file: Path, is_tpu: bool):
    """Stream docker logs from a specific VM to file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if is_tpu:
        cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh",
            vm_name, f"--zone={self._zone}", f"--project={self._project}",
            "--command", f"sudo docker logs -f {self._container_name} 2>&1",
        ]
    else:
        cmd = [
            "gcloud", "compute", "ssh",
            vm_name, f"--zone={self._zone}", f"--project={self._project}",
            "--command", f"sudo docker logs -f {self._container_name} 2>&1",
        ]

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while self._running:
            if proc.poll() is not None:
                break
            time.sleep(1.0)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
```

---

## Issue 3: File handle leak in `SmokeTestLogger.__init__`

**File:** `lib/iris/scripts/smoke-test.py`, line 140

**Problem:** `open()` is called directly and assigned to `self._file`. If any exception occurs between construction and `close()`, the file handle leaks. The class is not a context manager, so there's no `with` guard.

**Fix:** Make `SmokeTestLogger` a context manager so `SmokeTestRunner` can use it with `with`:

```python
class SmokeTestLogger:
    """Dual-output logger with timestamps and elapsed time."""

    def __init__(self, log_dir: Path):
        self._start_time = time.monotonic()
        self._start_datetime = datetime.now()
        self._log_dir = log_dir
        self._file: TextIO = open(log_dir / "summary.md", "w")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
```

Then in `SmokeTestRunner.run()`, wrap usage:

```python
# In __init__, don't create logger yet, just store log_dir
# In run():
with SmokeTestLogger(self.config.log_dir) as self.logger:
    ...
```

However, this is a moderate refactor. A simpler fix: just ensure `close()` is called in a `finally` in `run()` -- which it already is (line 588). The real risk is if `__init__` itself raises after `open()`. Since the only thing after `open()` is the assignment, this is extremely unlikely. The current code is acceptable. Mark this as "low risk, defer."

**Simpler fix (recommended):** Just remove the `| None` from `self._file` and remove the None checks throughout. The file is opened unconditionally and closed unconditionally. The nullable pattern adds complexity for no benefit:

```python
def __init__(self, log_dir: Path):
    self._start_time = time.monotonic()
    self._start_datetime = datetime.now()
    self._log_dir = log_dir
    self._file: TextIO = open(log_dir / "summary.md", "w")

def close(self):
    self._file.close()
```

Remove all `if self._file:` / `if not self._file:` guards from `write_header`, `write_artifacts`, `log`, and `section`.

---

## Issue 4: Thread safety for `_discovered_vms` and `_running`

**File:** `lib/iris/scripts/smoke-test.py`, `DockerLogStreamer` class

**Problem:** `_discovered_vms` (a `set`) is read/written from multiple threads (the discovery loop adds, and conceptually could be read). `_running` (a `bool`) is written by the main thread in `stop()` and read by background threads.

**Analysis:** In CPython, `bool` assignment is atomic due to the GIL, and `set.add()` / `in` checks are also GIL-protected. This is safe in practice on CPython. However, `_running` should be a `threading.Event` for clarity and correctness across implementations.

**Fix:** Replace `_running: bool` with `_stop_event: threading.Event`:

```python
def __init__(self, ...):
    ...
    self._stop_event = threading.Event()
    ...

def start(self):
    self._stop_event.clear()
    ...

def stop(self):
    self._stop_event.set()
    if self._thread:
        self._thread.join(timeout=5.0)

def _discover_and_stream_workers(self):
    while not self._stop_event.is_set():
        ...
        self._stop_event.wait(10.0)  # replaces time.sleep(10.0), wakes on stop

def _stream_vm_logs(self, ...):
    ...
    while not self._stop_event.is_set():
        if proc.poll() is not None:
            break
        self._stop_event.wait(1.0)
    proc.terminate()
    ...
```

This also makes `stop()` more responsive -- threads wake up immediately from `wait()` instead of sleeping the full interval.

Apply the same pattern to `TaskSchedulingMonitor`:

```python
# Replace _running with _stop_event in TaskSchedulingMonitor
def __init__(self, ...):
    ...
    self._stop_event = threading.Event()

def stop(self):
    self._stop_event.set()
    if self._thread:
        self._thread.join(timeout=5.0)
    self._client.close()

def _poll_loop(self):
    while not self._stop_event.is_set():
        ...
        self._stop_event.wait(5.0)
```

---

## Issue 5: Duplicate log collection race condition

**File:** `lib/iris/scripts/smoke-test.py`

**Problem:** Two independent mechanisms collect controller logs to `controller-logs.txt`:

1. `DockerLogStreamer` (mode="controller") streams live via `docker logs -f` to `controller-logs.txt` (line 242)
2. `_collect_controller_logs()` does a one-shot `docker logs` (no `-f`) and overwrites `controller-logs.txt` (line 816-817)

The streamer writes continuously. The one-shot collector reads to a temp file then copies over the same path. This creates a race: the streamer may have the file open for writing while the collector overwrites it, potentially corrupting the file or losing data.

**Fix: Remove the one-shot collection entirely.** The streamer already captures everything the one-shot would capture, plus it captures logs in real-time. The one-shot `_collect_controller_logs` and `_collect_worker_logs` methods are redundant.

Delete:
- `_collect_controller_logs` method (lines 791-820)
- `_collect_worker_logs` method (lines 822-855)
- All call sites in `_run_job_test` (lines 736-740, 749-753)
- The call site in `_cleanup` (lines 968-974)

The `collect_docker_logs` function in `debug.py` can remain as a standalone utility, but the smoke test should not use it -- the streamers handle this.

If we want a final log snapshot (e.g., to ensure we got everything after the streamer stops), we can do a final one-shot collection *after* stopping the streamers, writing to a different filename like `controller-logs-final.txt`. But this is unnecessary -- `docker logs -f` already captures everything.

---

## Issue 6: Repeated `self.config.prefix or "iris"` pattern

**File:** `lib/iris/scripts/smoke-test.py`, lines 526, 610, 794, 825, 993

**Problem:** The expression `self.config.prefix or "iris"` appears 5 times. This is the `label_prefix` concept, and it should be computed once.

**Fix:** Add a property or computed field to `SmokeTestConfig`:

```python
@dataclass
class SmokeTestConfig:
    ...
    prefix: str | None = None

    @property
    def label_prefix(self) -> str:
        return self.prefix or "iris"
```

Then replace all `self.config.prefix or "iris"` with `self.config.label_prefix`.

If we remove `_collect_controller_logs` and `_collect_worker_logs` (Issue 5), fewer call sites remain, but it's still worth centralizing.

---

## Issue 7: RPC client leak in `_log_autoscaler_status`

**File:** `lib/iris/scripts/smoke-test.py`, lines 905-928

**Problem:** A new `ControllerServiceClientSync` is created each call. If the RPC call raises, `rpc_client.close()` on line 926 is skipped. Also, creating a new client per call is wasteful.

**Fix:** Use a context manager or `try/finally`:

```python
def _log_autoscaler_status(self, controller_url: str):
    """Log current autoscaler state for observability."""
    rpc_client = ControllerServiceClientSync(controller_url)
    try:
        request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
        response = rpc_client.get_autoscaler_status(request)

        status = response.status
        if status.current_demand:
            demand_str = ", ".join(f"{k}={v}" for k, v in status.current_demand.items())
            self.logger.log(f"  Autoscaler demand: {demand_str}")

        for group in status.groups:
            cfg = group.config
            accel = format_accelerator_display(cfg.accelerator_type, cfg.accelerator_variant)
            self.logger.log(
                f"  Scale group {group.name}: demand={group.current_demand}, "
                f"slices={cfg.min_slices}-{cfg.max_slices}, "
                f"accelerator={accel}"
            )
    except Exception as e:
        self.logger.log(f"  (Could not fetch autoscaler status: {e})")
    finally:
        rpc_client.close()
```

The outer `try/except` is intentional here -- this is observability, not critical path. We log and continue. But the `finally` ensures cleanup.

---

## Issue 8: `timeout_seconds` never enforced

**File:** `lib/iris/scripts/smoke-test.py`, line 383

**Problem:** `SmokeTestConfig.timeout_seconds` defaults to 1800 and is exposed as a CLI flag, but nothing in `run()` checks elapsed time against it. The test can run indefinitely.

**Fix:** Add a deadline check. The simplest approach: record the deadline at the start of `run()` and check it at key points (between phases and between tests):

```python
def run(self) -> bool:
    deadline = time.monotonic() + self.config.timeout_seconds
    ...
    # Before each phase:
    if time.monotonic() > deadline:
        self.logger.log("Global timeout exceeded!", level="ERROR")
        return False
```

Add a helper:

```python
def _check_deadline(self, deadline: float) -> bool:
    """Returns True if deadline has passed. Logs and sets interrupted flag."""
    if time.monotonic() > deadline:
        self.logger.log(
            f"Global timeout ({self.config.timeout_seconds}s) exceeded!",
            level="ERROR",
        )
        self._interrupted = True
        return True
    return False
```

Call `_check_deadline(deadline)` at the same points where `self._interrupted` is already checked.

---

## Issue 9: Unnecessary temp directory indirection

**File:** `lib/iris/scripts/smoke-test.py`, lines 803-817, 837-854

**Problem:** `_collect_controller_logs` writes to `IRIS_ROOT / "logs"` then copies to the structured log dir. This is pointless indirection that also leaves stale files in `IRIS_ROOT/logs/`.

**Fix:** This is mooted by Issue 5 (removing these methods entirely). If we kept them, we'd just pass `self.config.log_dir` directly as `output_dir` to `collect_docker_logs`.

---

## Issue 10: Broad `Exception` catches in cleanup

**File:** `lib/iris/scripts/smoke-test.py`, lines 985-989, 1015-1016

**Problem:** Cleanup catches bare `Exception`. Per AGENTS.md, we should let exceptions propagate unless we're intentionally handling them.

**Analysis:** In cleanup code, swallowing exceptions is actually appropriate -- we want to attempt all cleanup steps even if one fails. The `level="WARN"` logging provides visibility. This is an intentional control flow change (best-effort cleanup). No change needed.

---

## Implementation Order

1. **Issue 1** - Remove `from __future__ import annotations` from `debug.py` (trivial, no dependencies)
2. **Issue 6** - Add `label_prefix` property to `SmokeTestConfig` (enables cleaner code for subsequent changes)
3. **Issue 5** - Remove duplicate log collection methods and call sites (largest change, simplifies the class)
4. **Issue 4** - Replace `_running` booleans with `threading.Event` in `DockerLogStreamer` and `TaskSchedulingMonitor`
5. **Issue 2** - Add `proc.wait()` after `proc.terminate()` in `_stream_vm_logs`
6. **Issue 7** - Add `finally: rpc_client.close()` in `_log_autoscaler_status`
7. **Issue 8** - Add deadline enforcement in `run()`
8. **Issue 3** - Remove nullable pattern from `SmokeTestLogger._file` (optional cleanup)

Issues 1-2 are independent. Issues 4-5 touch `DockerLogStreamer` and should be done together. Issue 6 should precede 5 since removing methods removes some `label_prefix` call sites.
