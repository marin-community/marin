# Iris tmpfs workdir and du monitoring — Research

## 1. ContainerRuntime Protocol

**File:** `lib/iris/src/iris/cluster/runtime/types.py`

`ContainerRuntime` (line 231) is a Protocol with these methods:
- `create_container(config: ContainerConfig) -> ContainerHandle` (line 238)
- `stage_bundle(bundle_id, workdir, workdir_files, bundle_store)` (line 246)
- `list_containers()` (line 262)
- `list_iris_containers(all_states)` (line 266)
- `remove_all_iris_containers()` (line 270)
- `cleanup()` (line 274)

`ContainerHandle` (line 144) has: `container_id`, `build()`, `run()`, `stop()`, `status()`, `log_reader()`, `stats()`, `profile()`, `cleanup()`.

**Existing runtime implementations:**
- `DockerContainerRuntime` in `lib/iris/src/iris/cluster/runtime/docker.py`
- `ProcessRuntime` in `lib/iris/src/iris/cluster/runtime/process.py`
- `KubernetesRuntime` in `lib/iris/src/iris/cluster/runtime/kubernetes.py`

## 2. Task Lifecycle and Workdir

**File:** `lib/iris/src/iris/cluster/worker/task_attempt.py`

### Workdir creation (lines 456–473, in `_setup()`):
```python
self._fast_io_dir = get_fast_io_dir(self._cache_dir)
safe_task_id = self.task_id.to_safe_token()
self.workdir = self._fast_io_dir / "workdirs" / f"{safe_task_id}_attempt_{self.attempt_id}"
self.workdir.mkdir(parents=True, exist_ok=True)
```

### `get_fast_io_dir()` (lines 92–109):
Prefers `/dev/shm/iris` (tmpfs) when available and has ≥1 GB free. Falls back to `cache_dir` (persistent disk).

### Cleanup (lines 821–853, `_cleanup()`):
```python
if self.workdir and self.workdir.exists():
    shutil.rmtree(self.workdir)
```

### `_fast_io_dir` also used for cache mounts (lines 619–624):
```python
uv_cache = self._fast_io_dir / "uv"
cargo_cache = self._fast_io_dir / "cargo"
cargo_target = self._fast_io_dir / "cargo-target"
```
These are bind-mounted into the container at `/uv/cache`, `/root/.cargo/registry`, `/root/.cargo/target`.

## 3. `du` Command Usage — THE KEY FINDING

**File:** `lib/iris/src/iris/cluster/worker/env_probe.py:189–207`

```python
def collect_workdir_size_mb(workdir: Path) -> int:
    """Calculate workdir size in MB using du -sm."""
    if not workdir.exists():
        return 0
    result = subprocess.run(
        ["du", "-sm", str(workdir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    output = result.stdout.strip()
    size_str = output.split("\t")[0]
    return int(size_str)
```

**Called from:** `task_attempt.py:798` in the `_monitor_loop()`:
```python
if self.workdir:
    self.disk_mb = collect_workdir_size_mb(self.workdir)
```

This is called **every poll interval** (default 5s). `du -sm` recursively walks the workdir tree, which is O(files). On tmpfs this is fast but still wasteful when all we need is total filesystem usage.

**This is the only `du` invocation in the iris codebase.**

### `shutil.disk_usage` usages (same file):

1. **`_get_disk_bytes()`** (line 180): Uses `os.statvfs("/")` to get available disk space for worker registration.

2. **`check_worker_health()`** (line 498–538): Uses `shutil.disk_usage(disk_path)` to check if root volume has ≥5% free space.

3. **`HostMetricsCollector._collect_disk()`** (line 583–588): Uses `shutil.disk_usage(self._disk_path)` for worker-level disk snapshot.

## 4. Kubernetes emptyDir Handling

**File:** `lib/iris/src/iris/cluster/runtime/kubernetes.py:219–242`

The K8s runtime skips host-path mounts for `/app` and instead creates a pod-local `emptyDir`:
```python
for i, (host_path, container_path, mode) in enumerate(self.config.mounts):
    if container_path == self.config.workdir:
        continue  # Skip /app — use emptyDir instead
    # ... hostPath volumes for other mounts
mounts.append({"name": "workdir", "mountPath": self.config.workdir, "readOnly": False})
volumes.append({"name": "workdir", "emptyDir": {}})
```

**No `sizeLimit` or `medium: Memory` is set on the emptyDir.** The emptyDir is backed by node ephemeral storage, not tmpfs.

`disk_bytes` from ResourceSpecProto is used for `ephemeral-storage` requests/limits (lines 312–316):
```python
disk_bytes = self.config.get_disk_bytes()
if disk_bytes:
    disk_gi = max(1, disk_bytes // (1024 * 1024 * 1024))
    resources.setdefault("requests", {})["ephemeral-storage"] = f"{disk_gi}Gi"
    resources.setdefault("limits", {})["ephemeral-storage"] = f"{disk_gi}Gi"
```

## 5. Disk Monitoring in Monitor Loop

**File:** `lib/iris/src/iris/cluster/worker/task_attempt.py:786–800`

Inside `_monitor_loop()`, stats are collected every poll cycle:
```python
stats = handle.stats()
if stats.available:
    self.current_memory_mb = stats.memory_mb
    self.current_cpu_percent = stats.cpu_percent
    self.process_count = stats.process_count
    if stats.memory_mb > self.peak_memory_mb:
        self.peak_memory_mb = stats.memory_mb

if self.workdir:
    self.disk_mb = collect_workdir_size_mb(self.workdir)
```

`disk_mb` is reported via `to_proto()` (line 428) as `ResourceUsage.disk_mb`.

The `ContainerStats` dataclass (types.py:101) does **not** include disk — disk is measured separately via `collect_workdir_size_mb()`.

## 6. `_fast_io_dir` Usage Summary

Set in `TaskAttempt._setup()` (line 468):
```python
self._fast_io_dir = get_fast_io_dir(self._cache_dir)
```

Used for:
1. Workdir path: `self._fast_io_dir / "workdirs" / ...` (line 472)
2. uv cache: `self._fast_io_dir / "uv"` (line 619)
3. cargo cache: `self._fast_io_dir / "cargo"` (line 620)
4. cargo target: `self._fast_io_dir / "cargo-target"` (line 621)

The bootstrap script (`bootstrap.py:140–144`) creates `/dev/shm/iris` on GCE workers and bind-mounts it into the worker container at line 187: `-v /dev/shm/iris:/dev/shm/iris`.

## 7. `disk_bytes` Flow Through the System

### Proto definitions:
- `config.proto:165` — `ResourceSpecProto.disk_bytes` (task resource request)
- `vm.proto:47` — `ScaleGroupResources.disk_bytes` (per-VM capacity)
- `cluster.proto:397` — `WorkerMetadata.disk_bytes` (worker's probed disk)
- `cluster.proto:261` — `ResourceUsage.disk_mb` (per-task usage)
- `cluster.proto:495` — `WorkerResourceSnapshot.disk_*` (worker-level snapshot)

### Flow:
1. **Job submission:** User specifies `disk_bytes` in `ResourceSpecProto` (e.g., `resources(disk="100G")`)
2. **Scheduling:** `constraints.py:939` — `ResourceCapacity.disk_bytes` compared against task's `disk_bytes` in `check_resource_fit()` (line 968)
3. **Autoscaler packing:** `autoscaler.py:200` — `AdditiveReq.disk_bytes` used in bin-packing algorithm
4. **Docker runtime:** Not used (no disk limits in Docker)
5. **K8s runtime:** Mapped to `ephemeral-storage` requests/limits (kubernetes.py:312–316)
6. **Monitoring:** `collect_workdir_size_mb()` measures actual usage via `du -sm`, reported as `ResourceUsage.disk_mb`

## Key Problems to Solve

### Problem 1: `du -sm` is wrong for tmpfs workdirs
When workdir is on tmpfs (`/dev/shm/iris/workdirs/...`), `du -sm` measures the size of files in that specific directory tree. But the relevant question for tmpfs is "how much of the tmpfs filesystem is used?" — because all tasks share the same tmpfs mount and it's backed by RAM. `shutil.disk_usage("/dev/shm/iris")` or `os.statvfs("/dev/shm/iris")` would give the filesystem-level answer instantly (O(1)) vs `du`'s O(files).

### Problem 2: `du` is called every 5 seconds
For large workdirs with many files, `du -sm` can be slow. `shutil.disk_usage()` is a single syscall.

### Problem 3: Semantics differ
- `du -sm <workdir>` = "how much disk does THIS task use?"
- `shutil.disk_usage(<mount>)` = "how much of the filesystem is used total?"

For tmpfs, the filesystem-level view is what matters (it's shared RAM). For persistent disk, per-task `du` gives more granular info but is expensive.

### Recommendation
Replace `collect_workdir_size_mb()` with `shutil.disk_usage()` / `os.statvfs()` on the workdir path. This gives:
- O(1) performance (single syscall)
- Correct semantics for tmpfs (shows total tmpfs usage, which is the constraint that matters)
- On persistent disk, shows partition usage (less granular but still useful and much cheaper)

The `disk_mb` field in `ResourceUsage` would shift from "per-task workdir size" to "filesystem usage of workdir's mount". This is arguably more useful for monitoring disk pressure.
