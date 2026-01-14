# Offline Dashboard Mode Design

> **Status**: Proposal (not yet implemented)

This document outlines a design for adding "offline" mode to fluster's worker and controller dashboards, enabling them to render from serialized state snapshots rather than live APIs.

## Motivation

The codebase has a TODO for this (`cluster_example.py:26-27`):
```python
# TODO, consider having like a post-mortem view of the cluster state
# means cluster state should be serializable, cluster dashboard would always be a mapping over the state
```

**Use cases:**
- Post-mortem debugging (capture state at failure time)
- Testing dashboard rendering without spinning up infrastructure
- Demos with realistic data
- Reproducible bug reports

## Current State

### Worker Dashboard (`cluster/worker/dashboard.py`)
- REST endpoints call `WorkerServiceImpl` RPC methods
- `JobManager._jobs: dict[str, Job]` where `Job.to_proto()` → `cluster_pb2.JobStatus`
- Already has proto-serializable job state

### Controller Dashboard (`cluster/controller/dashboard.py`)
- REST endpoints call `ControllerServiceImpl` and access `ControllerState` directly
- `ControllerState` contains:
  - `_jobs: dict[JobId, ControllerJob]`
  - `_workers: dict[WorkerId, ControllerWorker]`
  - `_endpoints: dict[str, ControllerEndpoint]`
  - `_actions: deque[ActionLogEntry]`

### Gap: Internal Types Not in Protos
- `ControllerJob` (has retry tracking, gang_id not in `JobStatus`)
- `ControllerWorker` (has `running_jobs` set)
- `ActionLogEntry` (dashboard action log)
- No `ClusterSnapshot` message exists

---

## Proposed Design

### 1. Proto Schema: Separate Snapshots

Add to `cluster.proto`:

```protobuf
// Controller job with full tracking info
message ControllerJobSnapshot {
  JobStatus status = 1;
  int32 failure_count = 2;
  int32 preemption_count = 3;
  int32 max_retries_failure = 4;
  int32 max_retries_preemption = 5;
  string gang_id = 6;
  int64 submitted_at_ms = 7;
  LaunchJobRequest request = 8;
}

message ControllerWorkerSnapshot {
  string worker_id = 1;
  string address = 2;
  ResourceSpec resources = 3;
  bool healthy = 4;
  int32 consecutive_failures = 5;
  int64 last_heartbeat_ms = 6;
  repeated string running_job_ids = 7;
}

message ActionLogEntrySnapshot {
  int64 timestamp_ms = 1;
  string action = 2;
  string job_id = 3;
  string worker_id = 4;
  string details = 5;
}

message ControllerSnapshot {
  int64 captured_at_ms = 1;
  repeated ControllerJobSnapshot jobs = 2;
  repeated ControllerWorkerSnapshot workers = 3;
  repeated Endpoint endpoints = 4;
  repeated ActionLogEntrySnapshot actions = 5;
  repeated string queue_order = 6;
}

message WorkerSnapshot {
  int64 captured_at_ms = 1;
  string worker_id = 2;
  repeated JobStatus jobs = 3;
  map<string, LogTail> job_logs = 4;  // job_id -> last N log lines
}

message LogTail {
  repeated LogEntry lines = 1;
  int32 total_lines = 2;  // Original count (for "showing X of Y")
}
```

### 2. Data Source Abstraction

New file: `cluster/dashboard_data.py`

```python
from typing import Protocol

class DashboardDataSource(Protocol):
    """Protocol for dashboard data access.

    Abstracts whether data comes from live APIs or serialized snapshots.
    """

    def get_stats(self) -> dict: ...
    def get_jobs(self) -> list[dict]: ...
    def get_workers(self) -> list[dict]: ...
    def get_endpoints(self) -> list[dict]: ...
    def get_actions(self, limit: int = 50) -> list[dict]: ...
    def get_job_detail(self, job_id: str) -> dict | None: ...


class LiveControllerDataSource:
    """Wraps ControllerServiceImpl + ControllerState."""

    def __init__(self, service: ControllerServiceImpl, state: ControllerState):
        self._service = service
        self._state = state

    # Implements DashboardDataSource using existing code


class SnapshotControllerDataSource:
    """Backed by ControllerSnapshot proto."""

    def __init__(self, snapshot: cluster_pb2.ControllerSnapshot):
        self._snapshot = snapshot
        self._jobs = {j.status.job_id: j for j in snapshot.jobs}
        self._workers = {w.worker_id: w for w in snapshot.workers}

    # Implements DashboardDataSource from snapshot data
```

### 3. Serialization Methods

Add `to_snapshot()` to internal types in `state.py`:

```python
# ControllerJob
def to_snapshot(self) -> cluster_pb2.ControllerJobSnapshot:
    return cluster_pb2.ControllerJobSnapshot(
        status=cluster_pb2.JobStatus(
            job_id=self.job_id,
            state=self.state,
            # ... other fields
        ),
        failure_count=self.failure_count,
        preemption_count=self.preemption_count,
        gang_id=self.gang_id or "",
        request=self.request,
    )

# ControllerState
def to_snapshot(self) -> cluster_pb2.ControllerSnapshot:
    with self._lock:
        return cluster_pb2.ControllerSnapshot(
            captured_at_ms=int(time.time() * 1000),
            jobs=[j.to_snapshot() for j in self._jobs.values()],
            workers=[w.to_snapshot() for w in self._workers.values()],
            # ...
        )
```

### 4. Dashboard Changes

Minimal changes to accept either live or snapshot data:

```python
class ControllerDashboard:
    def __init__(
        self,
        service: ControllerServiceImpl | None = None,
        data_source: DashboardDataSource | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        if data_source:
            self._data_source = data_source
        elif service:
            self._data_source = LiveControllerDataSource(service, service._state)
        else:
            raise ValueError("Either service or data_source required")
```

---

## Style Alignment

Worker and controller dashboards have different primary colors:
- Controller: Blue (`#3498db`)
- Worker: Green (`#4CAF50`)

**Recommendation**: Keep distinct colors (helps users know which dashboard they're viewing), but align structure using CSS variables:

```css
:root {
  /* Shared status colors */
  --status-pending: #f39c12;
  --status-running: #3498db;
  --status-succeeded: #27ae60;
  --status-failed: #e74c3c;
  --status-killed: #95a5a6;
  --status-building: #9b59b6;

  /* Shared layout */
  --max-width: 1400px;
  --spacing-md: 20px;
  --shadow-card: 0 2px 4px rgba(0,0,0,0.1);
}
```

Each dashboard defines its own `--primary-color` while sharing structural styles. No shared code dependencies needed.

---

## Implementation Phases

1. **Proto schema** - Add snapshot messages, regenerate
2. **Serialization** - Add `to_snapshot()` to internal types
3. **Data source abstraction** - Create protocol and implementations
4. **Dashboard integration** - Refactor to use data source
5. **Worker dashboard** - Same pattern
6. **Style alignment** - CSS variable extraction

---

## Design Decisions

- **Snapshot format**: Binary protobuf (compact, type-safe, matches RPC layer)
- **Log inclusion**: Last N lines (configurable, default 1000) with total count

---

## Maintainability Concerns

### 1. Schema Drift
**Risk**: Internal types (`ControllerJob`, etc.) evolve but snapshot protos don't get updated.

**Mitigation**:
- Round-trip tests that fail if fields are missing
- Consider code generation or a single source of truth

### 2. Dual Code Paths
**Risk**: Live and snapshot data sources diverge in behavior.

**Mitigation**:
- Shared test fixtures that run against both implementations
- Property: `live.get_stats() == snapshot_from(live).get_stats()`

### 3. Dashboard Abstraction Overhead
**Risk**: `DashboardDataSource` protocol adds indirection, making debugging harder.

**Mitigation**:
- Keep the protocol simple (6 methods)
- Live implementation is a thin wrapper over existing code

### 4. Proto Message Bloat
**Risk**: Snapshot messages duplicate information already in other protos.

**Mitigation**:
- `ControllerJobSnapshot` embeds `JobStatus` rather than duplicating fields
- Composition over duplication

### 5. CSS Duplication
**Risk**: Style alignment via copy-paste leads to drift.

**Mitigation**:
- CSS variables for shared values
- No shared code = no coordination overhead
- Accept that minor drift is OK (dashboards are internal tools)

---

## Files to Modify

| File | Changes |
|------|---------|
| `proto/cluster.proto` | Add snapshot messages |
| `cluster/controller/state.py` | Add `to_snapshot()` methods |
| `cluster/dashboard_data.py` | New: data source protocol |
| `cluster/controller/dashboard.py` | Use data source |
| `cluster/worker/dashboard.py` | Same pattern |

---

## Alternative: Make Internal Types Protos

Rather than maintaining separate Python dataclasses and proto messages, we could make the internal types protos themselves. This section evaluates that approach.

### Field-by-Field Proto Compatibility

| Type | Proto-able | Notes |
|------|------------|-------|
| **ControllerJob** | 100% | All fields serializable |
| **ControllerWorker** | 100% | `set[JobId]` → `repeated string` |
| **ControllerEndpoint** | 100% | `Endpoint` proto already exists, just add `registered_at_ms` |
| **ActionLogEntry** | 100% | All fields serializable |
| **Job (worker)** | ~80% | `workdir: Path`, `thread: Thread` are local execution state |

### Option A: Keep Dataclasses, Add `to_proto()` (Recommended)

```python
@dataclass
class ControllerJob:
    job_id: JobId
    request: cluster_pb2.LaunchJobRequest
    state: cluster_pb2.JobState = cluster_pb2.JOB_STATE_PENDING
    # ...

    def to_snapshot(self) -> cluster_pb2.ControllerJobSnapshot:
        return cluster_pb2.ControllerJobSnapshot(...)
```

**Pros:**
- Least invasive change
- Best Python ergonomics (methods, default factories, type hints)
- Familiar dataclass patterns

**Cons:**
- Schema drift risk (field added to dataclass but not proto)
- Requires discipline to keep in sync

**Mitigation:** Round-trip tests that fail if fields are missing.

### Option B: Hybrid - Embed Proto in Dataclass

```python
@dataclass
class ControllerJob:
    proto: cluster_pb2.ControllerJobState  # All persistent fields live here

    @property
    def job_id(self) -> JobId:
        return JobId(self.proto.job_id)

    @property
    def state(self) -> cluster_pb2.JobState:
        return self.proto.state

    def to_snapshot(self) -> cluster_pb2.ControllerJobSnapshot:
        return self.proto  # Already a proto!
```

**Pros:**
- Proto is source of truth - no schema drift
- Still get dataclass methods and ergonomics
- Adding a field to proto automatically makes it serializable

**Cons:**
- More boilerplate (property accessors)
- Nested proto assignment is awkward: `job.proto.request.CopyFrom(req)`
- Two layers of indirection

### Option C: Full Proto Types

Make `ControllerJob`, `ControllerWorker`, etc. proto messages directly.

```protobuf
message ControllerJobState {
  string job_id = 1;
  LaunchJobRequest request = 2;
  JobState state = 3;
  string worker_id = 4;
  int32 failure_count = 5;
  int32 preemption_count = 6;
  // ...
}
```

```python
# Usage becomes:
job = cluster_pb2.ControllerJobState()
job.job_id = "abc"
job.request.CopyFrom(req)  # Can't assign nested protos directly
job.failure_count += 1
```

**Pros:**
- Single source of truth
- Automatic serialization
- No conversion code needed

**Cons:**
- **Worse ergonomics:**
  - No default factories (`running_jobs: set = field(default_factory=set)` impossible)
  - Optional fields awkward (`if job.HasField('worker_id')`)
  - Can't assign nested protos directly (must use `CopyFrom`)
  - No methods on proto messages (`transition_to()` becomes standalone function)
- Weaker type hints (generated stubs are stringly-typed)
- Worker `Job` still needs wrapper for `thread`, `workdir`

### Option Comparison

| Aspect | A: Dataclass + to_proto | B: Hybrid | C: Full Proto |
|--------|------------------------|-----------|---------------|
| Schema drift risk | Medium | Low | None |
| Python ergonomics | Best | Good | Poor |
| Boilerplate | Low | Medium | Low |
| Refactoring cost | Low | Medium | High |

### Recommendation

**Option A** for a system this size. The schema drift risk is real but manageable with tests.

**Option B** is worth considering if you find yourself frequently adding fields and forgetting serialization. The embedded proto acts as a forcing function.

**Option C** is overkill - the ergonomic cost outweighs the benefit.

### Note on Existing Protos

The `Endpoint` proto already exists and is nearly identical to `ControllerEndpoint`:

```protobuf
message Endpoint {
  string endpoint_id = 1;
  string name = 2;
  string address = 3;
  string job_id = 4;
  string namespace = 5;
  map<string, string> metadata = 6;
  // Missing: registered_at_ms
}
```

This is a candidate for consolidation - add `registered_at_ms` to `Endpoint` and use it directly instead of `ControllerEndpoint`.

---

## Open Questions

1. Should controller auto-capture snapshots on job failure? (opt-in config)
2. CLI tool for snapshot capture/viewing? (`fluster-snapshot capture`, `fluster-dashboard --snapshot`)
3. Snapshot retention policy? (N most recent, or time-based)
