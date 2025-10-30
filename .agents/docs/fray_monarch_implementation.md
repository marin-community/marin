# Fray Monarch Backend Implementation Summary

## Status: ✅ IMPLEMENTED (Not Tested - Monarch unavailable on macOS)

The Monarch backend for Fray has been fully implemented following the design document at `.agents/docs/fray_monarch.md`. The implementation is complete and compiles successfully, but cannot be tested yet because `torchmonarch-nightly` is only available for Linux x86_64.

## Implementation Overview

### Files Created

1. **`lib/fray/src/fray/backend/monarch/monarch_helpers.py`**
   - `MonarchObjectRef`: Wraps Monarch Future for Fray compatibility
   - `MonarchActorHandle`: Wraps Monarch Mesh for single-actor API
   - `ObjectStoreActor`: Actor-based implementation of put/get object store
   - Helper functions for object ID generation

2. **`lib/fray/src/fray/backend/monarch/monarch_job.py`**
   - `MonarchJobContext`: Complete JobContext implementation
   - All 5 required methods: `create_task`, `create_actor`, `get`, `wait`, `put`
   - Context propagation via `contextvars.ContextVar`
   - Auto-spawning of Monarch processes from resource config

3. **`lib/fray/src/fray/backend/monarch/monarch_cluster.py`**
   - `MonarchClusterContext`: Complete ClusterContext implementation
   - Subprocess-based job management
   - Job tracking with status updates
   - Log capture to files
   - Methods: `create_job`, `list_jobs`, `delete_job`, `get_job_logs`

4. **`lib/fray/src/fray/backend/monarch/__init__.py`**
   - Exports main classes and MONARCH_AVAILABLE flag

### Configuration Changes

1. **`lib/fray/pyproject.toml`**
   - Added `monarch = ["torchmonarch-nightly"]` to optional dependencies

2. **`lib/fray/tests/test_backend.py`**
   - Updated fixtures to conditionally include "monarch" backend
   - Tests will automatically run against Monarch when available

3. **`lib/fray/tests/conftest.py`**
   - Added Monarch availability check

## Key Design Decisions

### 1. Optional Monarch Imports
```python
try:
    from monarch.actor import Actor, Future, Mesh, endpoint
    MONARCH_AVAILABLE = True
except ImportError:
    MONARCH_AVAILABLE = False
```

This allows the code to compile and import even when Monarch is not available, with tests automatically skipped on unsupported platforms.

### 2. Auto-Spawning Process Pool

MonarchJobContext accepts either:
- `procs`: Pre-spawned Monarch Procs object
- `resource_config`: Dict like `{"gpus": 8, "num_procs": 2}` for auto-spawning

Default: `{"num_procs": 1}` if neither provided.

### 3. Actor-Based Object Store

Since Monarch lacks a built-in object store like Ray:
- Created `ObjectStoreActor` that maintains a `dict[str, Any]`
- `put()` stores objects in this actor
- `get()` retrieves from the actor
- Object IDs generated via UUID

### 4. Task → Actor Mapping

Tasks are implemented as ephemeral actors:
```python
class TaskActor(Actor):
    @endpoint
    def run(self):
        return fn(*args, **kwargs)
```

Each task creates a new actor instance.

### 5. Mesh → Single-Actor Wrapping

Monarch spawns actor "meshes" (one instance per process). We wrap these in `MonarchActorHandle` to provide single-actor semantics:
```python
handle.method()  # Returns MonarchObjectRef with take_first=True
```

Internally: `mesh.method.call()` → returns results from all instances → extract first result.

### 6. Context Propagation

JobContext is propagated to actors via:
1. Capture `current_ctx` in task/actor creation
2. Store in actor's `_fray_ctx` attribute
3. Restore via `_job_context.set(self._fray_ctx)` in endpoints

This allows nested task/actor creation within actors.

## Known Limitations

### 1. wait() Implementation

**Current Implementation**: Sequential blocking check on refs
```python
for ref in not_ready:
    result = self.get(ref)  # Blocks until ready
    ready.append(ref)
```

**Limitation**: Not true wait semantics - doesn't wait for ANY ref to complete, waits for refs in order.

**Fix Required**: When Monarch API is known:
- Check if Future has `is_done()` or `ready()` method
- Use threading to poll multiple futures
- Implement proper FIRST_COMPLETED semantics

### 2. Resource Management

Per-task/per-actor resources (TaskOptions.resources, ActorOptions.resources) are **accepted but ignored**.

**Reason**: Monarch allocates resources at process spawn time, not per-task.

**Behavior**: Resources specified in options have no effect. All tasks/actors use the process pool resources specified at JobContext initialization.

**Compatible with tests**: Tests only verify resources are accepted, not enforced.

### 3. TPU Support

`run_on_tpu()` raises `NotImplementedError`.

**Reason**: TPU support requires complex slice management (see Ray TPU implementation). Deferred to future work.

### 4. Named Actor Lookup

`ActorOptions(name="foo", get_if_exists=True)` **partially implemented**.

**Current behavior**: Name is used, but `get_if_exists` is ignored (always creates new actor).

**Fix Required**: Implement actor registry to track actors by name and return existing instance when `get_if_exists=True`.

### 5. Multi-Host Support

ClusterContext is **single-host only**.

**Current**: Jobs run as subprocesses on local machine.

**Multi-host**: Would require SSH/remote execution or external orchestrator (Kubernetes, Slurm).

### 6. Actor Lifetime

`ActorOptions.lifetime` (EPHEMERAL vs DETACHED) is **accepted but ignored**.

**Reason**: Monarch actors persist until process termination. No equivalent to Ray's detached actors.

## Testing Status

### ✅ Verified
- All files compile successfully
- Imports work correctly
- MONARCH_AVAILABLE flag works
- Test fixtures properly parametrize backends

### ❌ Not Tested (Monarch unavailable)
- Actual execution with Monarch runtime
- Process spawning
- Actor creation and endpoint calls
- Future resolution (get/wait)
- Object store functionality
- Context propagation in actors

### 🔄 Next Steps When Monarch Available

1. **Install Monarch on Linux x86_64**
   ```bash
   pip install torchmonarch-nightly
   ```

2. **Run Test Suite**
   ```bash
   pytest lib/fray/tests/test_backend.py -v -k monarch
   ```

3. **Fix Issues Discovered**
   - Check Future API (get(), is_done(), etc.)
   - Verify actor spawning syntax
   - Test endpoint calling conventions
   - Debug context propagation
   - Validate object store actor
   - Fix wait() implementation

4. **Performance Testing**
   - Compare overhead vs Ray backend
   - Test RDMA transfers (if available)
   - Benchmark actor creation time
   - Measure object store latency

## API Compatibility Matrix

| Feature | In-Memory | Ray | Monarch | Notes |
|---------|-----------|-----|---------|-------|
| create_task | ✅ | ✅ | ✅ | Monarch uses ephemeral actors |
| create_actor | ✅ | ✅ | ✅ | Mesh wrapped for single-actor API |
| get | ✅ | ✅ | ✅ | Handles both futures and object store |
| wait | ✅ | ✅ | ⚠️ | Sequential, needs fix |
| put | ✅ | ✅ | ✅ | Actor-based, slower than Ray |
| Per-task resources | ❌ | ✅ | ❌ | Process-level only |
| Named actors | ✅ | ✅ | ⚠️ | get_if_exists not implemented |
| Detached actors | ❌ | ✅ | ❌ | All actors persist with process |
| Object store | ✅ | ✅ | ⚠️ | Actor-based, different semantics |
| TPU support | ✅ Mock | ✅ | ❌ | Not implemented |
| Multi-host | ❌ | ✅ | ❌ | Single-host only |

Legend:
- ✅ Fully supported
- ⚠️ Partial/suboptimal implementation
- ❌ Not supported

## Code Quality

### Strengths
- Clean separation of concerns (helpers, job, cluster)
- Comprehensive docstrings
- Proper error handling
- Optional imports for cross-platform compatibility
- Follows existing backend patterns

### Potential Issues
- wait() implementation needs improvement
- No actual testing with Monarch runtime
- Resource options silently ignored
- Object store may have performance issues (actor overhead)
- No cleanup of completed task actors (memory leak potential)

## Comparison to Ray Backend

### Advantages of Monarch
- **RDMA support**: Zero-copy GPU transfers
- **Simpler architecture**: No separate cluster daemon
- **Supervision trees**: Hierarchical fault tolerance
- **Mesh operations**: Native SPMD patterns
- **Lower latency**: For fixed-topology workloads

### Advantages of Ray
- **Dynamic allocation**: Tasks scheduled to any worker
- **Object store**: Efficient shared memory for large objects
- **TPU support**: Complete slice management
- **Multi-host**: Built-in cluster management
- **Mature**: Production-tested, extensive ecosystem
- **Resource enforcement**: Per-task CPU/GPU/memory limits

## Recommendations

### For Development
1. Test implementation on Linux x86_64 with Monarch installed
2. Fix wait() implementation based on actual Future API
3. Add cleanup for ephemeral task actors
4. Implement named actor registry for get_if_exists
5. Add integration tests specific to Monarch features (mesh operations, RDMA)

### For Production
1. **Use Ray for**: Dynamic workloads, TPUs, multi-host, per-task resources
2. **Use Monarch for**: SPMD training, fixed worker pools, RDMA transfers
3. **Use In-Memory for**: Testing, single-machine prototyping

### For Documentation
1. Document resource limitation clearly in user docs
2. Provide migration guide from Ray to Monarch
3. Add examples of Monarch-specific patterns (mesh operations)
4. Note platform restrictions (Linux x86_64 only)

## Implementation Checklist

- [x] MonarchObjectRef wrapper
- [x] MonarchActorHandle wrapper
- [x] ObjectStoreActor implementation
- [x] MonarchJobContext with all 5 methods
- [x] MonarchClusterContext with job management
- [x] Context propagation via contextvars
- [x] Optional imports for cross-platform
- [x] Test fixture parametrization
- [x] Compilation verification
- [ ] Actual runtime testing (blocked on Monarch availability)
- [ ] wait() improvement
- [ ] Named actor registry
- [ ] Task actor cleanup
- [ ] Resource management documentation
- [ ] Performance benchmarks

## Conclusion

The Monarch backend implementation is **structurally complete** and follows the design document faithfully. All code compiles and imports correctly. The implementation cannot be fully validated until Monarch is available on a Linux x86_64 system for testing.

Key risks:
1. Monarch API assumptions may be incorrect
2. wait() implementation needs improvement
3. Object store performance may be poor
4. Context propagation may not work as expected

Key strengths:
1. Clean, well-documented code
2. Follows existing backend patterns
3. Graceful degradation (optional imports)
4. Comprehensive error handling

**Ready for Linux testing once Monarch is available.**
