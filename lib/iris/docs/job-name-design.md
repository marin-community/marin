# JobName Design Document

## Problem Statement

Iris communicates parent/child relationships between jobs via the job name: child jobs start
with their parent name (e.g., `parent-job/child-task`). The current implementation has several issues:

1. **Redundant encoding**: `parent_job_id` is stored as a separate proto field, but it's also encoded in the job name itself
2. **Manual string manipulation**: Job IDs are constructed via f-strings and parsed via `.rsplit("/", 1)` throughout the codebase
3. **Type confusion**: identifiers were modeled as raw strings, leading to scattered string manipulation
4. **Inconsistent format**: Current job IDs lack a leading slash, making it unclear whether names are absolute

## Breaking Change Notice

This design introduces a **new canonical wire format**:
- **Previous format**: `root/child/task-0` (no leading slash, `task-{index}` suffix)
- **Canonical format**: `/root/child/0` (leading slash, numeric task suffix)

This is an intentional breaking change. All clients and controllers must be updated together. There is no backward compatibility period.

## Current State Analysis

### Existing Types

```python
# lib/iris/src/iris/cluster/types.py
JobName = dataclass(...)  # Canonical representation
JobName is the single canonical identifier for both jobs and tasks.
```

All internal code uses `JobName` APIs; only RPC/env boundaries use strings.

### Proto Fields

```protobuf
// lib/iris/src/iris/rpc/cluster.proto

message JobStatus {
  string job_id = 1;
  ...
  reserved 14;  // parent_job_id (derived from job_id)
  string name = 25;           // Original name from request
  ...
}

message LaunchJobRequest {
  string name = 1;            // The user-provided job name
  ...
  reserved 10;                // parent_job_id (derived from name)
  ...
}
```

### Job ID Construction Patterns

Job names are constructed and parsed manually throughout the codebase:

| File | Line | Pattern |
|------|------|---------|
| `client/client.py` | 136 | `job_name.task(index)` |
| `client/client.py` | 284,298 | `job_name.task(index)` |
| `client/client.py` | 604 | `parent.child(name)` |
| `client/client.py` | 748 | `job_id.task(index)` |
| `client/client.py` | 817 | `JobName.parent` |
| `cluster/client/remote_client.py` | 95 | `JobName.parent` |
| `cluster/client/remote_client.py` | 269-272 | `JobName.require_task()` in log fetching |
| `cluster/client/job_info.py` | 30-91 | `JobInfo.task_id` from env vars (job_id derived) |
| `cluster/controller/state.py` | 793 | `job.job_id.task(i)` |
| `cluster/controller/service.py` | 334 | `JobName.from_wire(...).task(...)` |
| `cluster/controller/service.py` | 637 | `job_id.task(...)` in errors |
| `cluster/controller/bundle_store.py` | 58 | `rsplit("/")` for bundle paths |
| `cluster/types.py` | 445 | `JobName.namespace` |

## Proposed Design

### JobName Class

A structured type that encapsulates hierarchical job naming semantics.

```python
from dataclasses import dataclass
from typing import Self

@dataclass(frozen=True, slots=True)
class JobName:
    """Structured hierarchical job name.

    Canonical form: /namespace/parent/child
    Tasks are job names with numeric suffix: /namespace/parent/child/0

    Job names form a tree rooted at the namespace:
        /root-job
        /root-job/child-1
        /root-job/child-1/grandchild
        /root-job/0
    """

    _parts: tuple[str, ...]

    def __post_init__(self):
        if not self._parts:
            raise ValueError("JobName cannot be empty")
        for part in self._parts:
            if "/" in part:
                raise ValueError(f"JobName component cannot contain '/': {part}")
            if not part or not part.strip():
                raise ValueError("JobName component cannot be empty or whitespace")

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Parse a job name string like '/root/child/grandchild'.

        Examples:
            JobName.from_string("/my-job") -> JobName(("my-job",))
            JobName.from_string("/parent/child") -> JobName(("parent", "child"))
            JobName.from_string("/job/0") -> JobName(("job", "0"))
        """
        if not s:
            raise ValueError("Job name cannot be empty")
        if not s.startswith("/"):
            raise ValueError(f"Job name must start with '/': {s}")
        return cls(tuple(s[1:].split("/")))

    @classmethod
    def root(cls, name: str) -> Self:
        """Create a root job name (no parent)."""
        return cls((name,))

    def child(self, name: str) -> Self:
        """Create a child job name."""
        return JobName(self._parts + (name,))

    def task(self, index: int) -> Self:
        """Create a task name for this job.

        Tasks are job names with a numeric suffix.

        Example:
            JobName.from_string("/my-job").task(0) -> JobName(("my-job", "0"))
        """
        return JobName(self._parts + (str(index),))

    @property
    def parent(self) -> Self | None:
        """Get parent job name, or None if this is a root job."""
        if len(self._parts) == 1:
            return None
        return JobName(self._parts[:-1])

    @property
    def namespace(self) -> str:
        """Get the namespace (root component) for actor isolation."""
        return self._parts[0]

    @property
    def name(self) -> str:
        """Get the local name (last component)."""
        return self._parts[-1]

    @property
    def is_root(self) -> bool:
        """True if this is a root job (no parent)."""
        return len(self._parts) == 1

    @property
    def task_index(self) -> int | None:
        """If this is a task (last component is numeric), return the index."""
        try:
            return int(self._parts[-1])
        except ValueError:
            return None

    @property
    def is_task(self) -> bool:
        """True if this is a task (last component is numeric)."""
        return self.task_index is not None

    def __str__(self) -> str:
        """Canonical string representation: '/root/child/grandchild'."""
        return "/" + "/".join(self._parts)

    def __repr__(self) -> str:
        return f"JobName({str(self)!r})"

    def to_wire(self) -> str:
        """Serialize to wire format for RPC/env vars."""
        return str(self)

    @classmethod
    def from_wire(cls, s: str) -> Self:
        """Parse from wire format. Alias for from_string."""
        return cls.from_string(s)
```

### Usage Examples

```python
# Root job
job = JobName.root("my-job")
assert str(job) == "/my-job"
assert job.namespace == "my-job"
assert job.parent is None

# Child job
child = job.child("subtask")
assert str(child) == "/my-job/subtask"
assert child.namespace == "my-job"
assert child.parent == job

# Task
task = job.task(0)
assert str(task) == "/my-job/0"
assert task.parent == job

# Parsing
parsed = JobName.from_string("/my-job/subtask/0")
assert parsed.namespace == "my-job"
assert parsed.parent == JobName.from_string("/my-job/subtask")

# Task detection
task = JobName.from_string("/my-job/0")
assert task.is_task == True
assert task.task_index == 0

job = JobName.from_string("/my-job/child")
assert job.is_task == False
assert job.task_index is None
```

Task identifiers are always carried on the wire as `task_id` (a full `JobName`).
`task_index` is only derived locally from the `JobName` when needed for display
or ordering.

## Namespace Control Flow

Namespaces provide actor isolation - actors in one namespace cannot discover actors in another.
The namespace is the root component of a job name, shared by all jobs in a hierarchy.

### Current Flow (string manipulation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Client submits job                                                      │
│   name = "my-job"                                                       │
│   job_id = f"{parent_job_id}/{name}" if parent else name                │
│            ^^^^ string concatenation                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Namespace derivation                                                    │
│   namespace = Namespace(job_id.split("/")[0])                           │
│                         ^^^^ string parsing                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Actor registration/discovery                                            │
│   NamespacedEndpointRegistry(namespace=namespace)                       │
│   NamespacedResolver(cluster, namespace=namespace)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### New Flow (JobName type)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Client submits job                                                      │
│   ctx = get_iris_ctx()                                                  │
│   if ctx:                                                               │
│       job_name = ctx.job_name.child("my-job")   # /parent/my-job        │
│   else:                                                                 │
│       job_name = JobName.root("my-job")         # /my-job               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Namespace derivation (no parsing needed)                                │
│   namespace = job_name.namespace   # "my-job" or "parent"               │
│               ^^^^ direct property access                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Wire format for RPC                                                     │
│   request.name = job_name.to_wire()   # "/parent/my-job"                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Controller receives job                                                 │
│   job_name = JobName.from_wire(request.name)                            │
│   namespace = job_name.namespace                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Actor registration/discovery (unchanged API)                            │
│   NamespacedEndpointRegistry(namespace=Namespace(job_name.namespace))   │
│   NamespacedResolver(cluster, namespace=Namespace(job_name.namespace))  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Changes

1. **No string parsing for namespace**: `job_name.namespace` directly returns the root component
2. **No string concatenation for hierarchy**: `parent.child("name")` builds the path
3. **Wire format explicit**: `to_wire()` / `from_wire()` at RPC boundaries
4. **Namespace type unchanged**: `Namespace` class remains, constructed from `job_name.namespace`

## Migration Plan

The migration follows a **spiral approach**: each stage is a vertical slice that includes proto changes (if any), server code, client code, and end-to-end tests.

### Stage 1: Introduce JobName Type + Basic Tests

**Goal**: Add JobName class and verify parsing/construction logic.

**Changes**:
- Add `JobName` class to `lib/iris/src/iris/cluster/types.py`
- Add unit tests to `lib/iris/tests/cluster/test_types.py`

**Tests**:
- Parsing: `JobName.from_string("/a/b/c")`
- Construction: `JobName.root("a").child("b").child("c")`
- Parent derivation: `JobName.from_string("/a/b/c").parent == JobName.from_string("/a/b")`
- Namespace: `JobName.from_string("/a/b/c").namespace == "a"`
- Task creation: `JobName.root("job").task(0) == JobName.from_string("/job/0")`
- Validation errors: empty names, names without leading slash, whitespace-only components

**No proto changes. No behavior changes.**

### Stage 2: Use JobName for Namespace Derivation

**Goal**: Replace `Namespace.from_job_id(job_id: str)` with `JobName.namespace`.

**Changes**:
- Update `lib/iris/src/iris/cluster/types.py`:
  - Modify `Namespace.from_job_id` to accept `str | JobName`
  - If `str`, parse to JobName first
- Update one call site (e.g., `client/client.py`) to use JobName
- Add integration test verifying namespace derivation

**Tests**:
- End-to-end: submit job, verify namespace matches job name prefix

**No proto changes.**

### Stage 3: Use JobName for Parent Derivation in Client

**Goal**: Replace `.rsplit("/", 1)` pattern in client layer.

**Changes**:
- Update `lib/iris/src/iris/client/client.py`:
  - Store `JobName` in `IrisContext`
  - `parent_job_id` property uses `job_name.parent`
  - `submit()` uses `JobName.child()` to construct child job names
- Update `lib/iris/src/iris/cluster/client/job_info.py`:
  - Store `JobName` alongside string `job_id`

**Tests**:
- End-to-end: submit parent job, submit child job, verify hierarchy

**No proto changes.**

### Stage 4: Use JobName for Task ID Construction

**Goal**: Replace `f"{job_id}/{i}"` with `job_name.task(i)`.

**Changes**:
- Update `lib/iris/src/iris/cluster/controller/state.py`:
  - Use `JobName.task(i)` instead of f-string
- Update `lib/iris/src/iris/cluster/controller/service.py`:
  - Use `JobName.task(index)` for task ID construction

**Tests**:
- End-to-end: submit job with multiple tasks, verify task IDs match expected format

**No proto changes.**

### Stage 5: Make Controller Derive Parent from job_id

**Goal**: Controller stops using `parent_job_id` proto field, derives it from `job_id` instead.

**Changes**:
- Update `lib/iris/src/iris/cluster/controller/state.py`:
  - When receiving `JobStatus`, parse `job_id` to `JobName`
  - Derive parent via `job_name.parent` (ignore `parent_job_id` field)
- Update `lib/iris/src/iris/cluster/client/remote_client.py`:
  - Stop setting `parent_job_id` in `LaunchJobRequest`

**Tests**:
- End-to-end: submit parent+child jobs, verify parent relationship is correct
- Verify old clients (still sending `parent_job_id`) work with new controller

**No proto changes yet - fields still exist but are ignored.**

### Stage 6: Remove parent_job_id Proto Fields

**Goal**: Remove redundant proto fields, mark them as reserved.

**Changes**:
- Update `lib/iris/src/iris/rpc/cluster.proto`:
  ```protobuf
  message JobStatus {
    string job_id = 1;
    reserved 14;  // was: parent_job_id (now derived from job_id)
    ...
  }

  message LaunchJobRequest {
    string name = 1;
    reserved 10;  // was: parent_job_id (now derived from name)
    ...
  }
  ```
- Run `uv run python lib/iris/scripts/generate_protos.py`
- Remove all references to `parent_job_id` field in codebase

**Tests**:
- Full test suite
- Verify proto compatibility (old messages with field 10/14 are silently ignored)

### Stage 7: Remove JobId/TaskId aliases

**Goal**: Only `JobName` exists. There is no separate `JobId`/`TaskId` alias.

**Changes**:
- Update all type hints to use `JobName` directly
- Run `uv run pyrefly` to verify type checking passes

**Tests**:
- Full test suite
- Static type checking

## Design Decisions

### Why Immutable Dataclass?

- Job names are identifiers that should never change
- Hashable for use as dict keys
- `slots=True` for memory efficiency
- Clear equality semantics

### Why Not Subclass str?

- Can't add methods to NewType
- Subclassing str leads to inheritance gotchas
- Explicit type is clearer than magic methods

### Why No Separate TaskName Class?

Tasks are leaf nodes in the hierarchy - they cannot have children. If a task creates a child job,
the child inherits from the parent *job* name, not the task name:

```
/my-job           <- job
/my-job/0         <- task (leaf)
/my-job/child     <- child job created by task 0 (inherits from /my-job, not /my-job/0)
```

Despite this semantic difference, a separate TaskName class isn't needed because:
- Task names share the same wire format (strings with `/` separator)
- Task names support the same parsing/namespace operations
- The "no children" constraint is enforced by the client logic, not the type
- Using a second representation would reintroduce string parsing/formatting outside `JobName`

### Why Leading Slash in Canonical Form?

Job names are absolute paths from the root namespace, similar to file system paths. The leading slash makes this explicit and prevents ambiguity between relative and absolute names.

### Wire Format

Job names remain strings on the wire. The JobName class is purely a client/server-side construct for safe manipulation. Proto messages continue to use `string job_id`.

### Single Canonical Representation

There is exactly one canonical representation for identifiers: `JobName`.
Tasks are represented as `JobName` values with numeric suffixes, created via
`JobName.task(index)`. No code outside `JobName` should parse or format job/task
IDs. `JobName` is the only identifier type and encodes both job and task names.

### Validation

- Components cannot contain `/` (reserved as separator)
- Components cannot be empty or whitespace-only
- Names must start with `/`

## Open Questions

1. **Should we restrict name components more strictly?**
   - Current: Forbids `/` and whitespace-only
   - Alternative: Restrict to alphanumeric + hyphen

2. **Should Namespace remain a separate type?**
   - Current: Namespace is derived from `JobName.namespace`
   - Alternative: Remove Namespace type entirely, use `str` for namespace

## References

- Issue: #2643
- Current types: `lib/iris/src/iris/cluster/types.py`
- Proto definitions: `lib/iris/src/iris/rpc/cluster.proto`
