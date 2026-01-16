# DI Cleanup Plan

## Rule of Thumb

Apply dependency injection when:

1. **External environment impact**: Network, filesystem, subprocess, hardware
2. **Simulating failures or time**: Things that are difficult to control in tests

Skip DI for purely computational code with no side effects.

## Current State: Well-Designed

These protocols follow the pattern correctly and should remain:

| Protocol | Implementation | Why DI? |
|----------|---------------|---------|
| `ContainerRuntime` | `DockerRuntime` | Subprocess (docker CLI) |
| `ImageBuilder` | `DockerImageBuilder` | Subprocess (docker build) |
| `BundleProvider` | `BundleCache` | GCS/filesystem I/O |
| `ImageProvider` | `ImageCache` | Wraps ImageBuilder, manages cache |
| `EnvironmentProvider` | `DefaultEnvironmentProvider` | Subprocess (nvidia-smi), filesystem (/proc), network (GCE metadata) |
| `WorkerStubFactory` | `DefaultWorkerStubFactory` | Network RPC |
| `Resolver` | `FixedResolver`, `GcsResolver` | Network (GCE API) |

## Over-Engineered: Consider Removing

### `GcsApi` (actor/resolver.py:89)

**Problem**: Tiny protocol with one method (`list_instances`), only used by `GcsResolver`.

```python
class GcsApi(Protocol):
    def list_instances(self, project: str, zone: str) -> list[dict]: ...
```

**Recommendation**: Remove `GcsApi`. The injection point should be `Resolver` itself, not the GCE client inside `GcsResolver`. Tests that need to control resolution should inject a `FixedResolver` instead of a `GcsResolver` with a mocked `GcsApi`.

**Current**:
```python
resolver = GcsResolver(project, zone, api=MockGcsApi(instances=[...]))
```

**Proposed**:
```python
resolver = FixedResolver({"actor-name": "http://10.0.0.1:8080"})
```

### `JobProvider` (cluster/worker/service.py:30) — Keep for now

**Situation**: This protocol mirrors Worker's public interface.

```python
class JobProvider(Protocol):
    def submit_job(self, request: RunJobRequest) -> str: ...
    def get_job(self, job_id: str) -> Job | None: ...
    def list_jobs(self) -> list[Job]: ...
    def kill_job(self, job_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def get_logs(self, job_id: str, start_line: int = 0) -> list[LogEntry]: ...
```

**Arguments for keeping**:
- Allows testing `WorkerServiceImpl` RPC logic with a minimal fake (no container lifecycle)
- Useful if we want to test validation/error handling in isolation

**Arguments for removing**:
- Current tests use `Worker` with mocked deps — already well-isolated
- No alternate implementation exists

**Verdict**: Low priority. Keep unless it becomes maintenance burden. Worker's dependency injection already provides good test isolation.

## Gaps: Consider Adding DI

### Clock/Time (multiple files)

**Problem**: Many places use `time.time()` and `time.sleep()` directly, making timeout and scheduling logic hard to test.

**Files affected**:
- `cluster/worker/worker.py:501` - timeout checking
- `cluster/controller/controller.py:314` - worker timeout checking
- `cluster/controller/scheduler.py` - scheduling timeout logic
- Various polling loops

**Recommendation**: Consider a `Clock` protocol for critical timeout logic:

```python
class Clock(Protocol):
    def now_ms(self) -> int: ...
    def sleep(self, seconds: float) -> None: ...

class SystemClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
```

**Priority**: Medium. Most timeout logic is tested via integration tests with short timeouts. Add Clock only if we find ourselves needing deterministic time control.

### Git subprocess (cluster/client/bundle.py:32)

**Problem**: `_get_git_non_ignored_files` shells out to `git ls-files` directly.

**Recommendation**: Low priority. This is a utility function for bundle creation, and the fallback behavior (pattern-based exclusion) already handles the "git not available" case. Not worth adding a protocol.

## Naming Conventions

- **Protocol**: `FooProvider` (e.g., `BundleProvider`, `ImageProvider`)
- **Production impl**: Context-specific name (e.g., `BundleCache`, `DockerRuntime`, `GcsResolver`)
- **Test fake**: `MockFoo` or `FakeFoo` with real logic preferred over `Mock(spec=...)`

## Migration Checklist

- [ ] Remove `GcsApi` protocol, update tests to use `FixedResolver`
- [ ] (Optional) Add `Clock` protocol if timeout testing becomes painful
