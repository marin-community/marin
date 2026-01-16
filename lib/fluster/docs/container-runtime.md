# ContainerRuntime Refactoring Plan

## Problem

The Worker currently uses `isinstance(self._runtime, DockerRuntime)` checks to determine runtime-specific behavior:

```python
# worker.py lines 491-506
if isinstance(self._runtime, DockerRuntime):
    env["FLUSTER_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(...)
else:
    env["FLUSTER_CONTROLLER_ADDRESS"] = self._config.controller_address

if isinstance(self._runtime, DockerRuntime):
    env["FLUSTER_BIND_HOST"] = "0.0.0.0"
else:
    env["FLUSTER_BIND_HOST"] = "127.0.0.1"
```

This violates the dependency injection pattern - the Worker shouldn't know about specific runtime implementations.

## Current Architecture

```
Worker
  └── builds ContainerConfig (with env vars)
        └── passed to ContainerRuntime.create_container()
              └── DockerRuntime: runs in Docker container
              └── _LocalContainerRuntime: runs in thread, sets JobInfo contextvar
```

The Worker pre-computes all environment variables before passing to the runtime. The runtime just executes with whatever config it receives.

## Runtime-Specific Behaviors

| Behavior | Docker | Local | Why |
|----------|--------|-------|-----|
| Address rewriting | `127.0.0.1` → `host.docker.internal` | No rewriting | Docker containers can't reach host localhost |
| Bind host | `0.0.0.0` | `127.0.0.1` | Docker port mapping requires binding all interfaces |

## Proposed Solution

Add capability properties to the `ContainerRuntime` protocol that the Worker queries:

```python
# docker.py
class ContainerRuntime(Protocol):
    """Protocol for container runtimes."""

    @property
    def requires_address_rewrite(self) -> bool:
        """True if containers cannot reach host localhost directly.

        When True, localhost addresses (127.0.0.1, localhost, 0.0.0.0)
        in FLUSTER_CONTROLLER_ADDRESS will be rewritten to host.docker.internal.
        """
        ...

    @property
    def container_bind_host(self) -> str:
        """Address that containers should bind servers to.

        Returns '0.0.0.0' for Docker (port mapping requires all interfaces),
        '127.0.0.1' for local execution (direct host access).
        """
        ...

    # ... existing methods unchanged
```

## Implementation

### 1. Update ContainerRuntime Protocol

```python
# docker.py
class ContainerRuntime(Protocol):
    @property
    def requires_address_rewrite(self) -> bool:
        ...

    @property
    def container_bind_host(self) -> str:
        ...

    def create_container(self, config: ContainerConfig) -> str:
        ...
    # ... rest unchanged
```

### 2. Implement in DockerRuntime

```python
class DockerRuntime:
    @property
    def requires_address_rewrite(self) -> bool:
        return True

    @property
    def container_bind_host(self) -> str:
        return "0.0.0.0"
```

### 3. Implement in _LocalContainerRuntime

```python
class _LocalContainerRuntime:
    @property
    def requires_address_rewrite(self) -> bool:
        return False

    @property
    def container_bind_host(self) -> str:
        return "127.0.0.1"
```

### 4. Update Worker

```python
# worker.py
if self._config.controller_address:
    if self._runtime.requires_address_rewrite:
        env["FLUSTER_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(...)
    else:
        env["FLUSTER_CONTROLLER_ADDRESS"] = self._config.controller_address

env["FLUSTER_BIND_HOST"] = self._runtime.container_bind_host
```

## Alternative: Runtime Handles Env Setup

A more radical approach would have the runtime handle all JobInfo/env setup:

```python
class ContainerRuntime(Protocol):
    def prepare_job_environment(
        self,
        job_id: str,
        worker_id: str,
        controller_address: str,
        ports: dict[str, int],
    ) -> dict[str, str]:
        """Build environment variables for job execution."""
        ...
```

**Pros:**
- Complete encapsulation of runtime-specific behavior
- Runtime controls its own execution context

**Cons:**
- Duplicates logic across runtimes
- More complex protocol
- The current approach (Worker builds env, runtime executes) is cleaner

**Recommendation:** Use the property-based approach. It's minimal, explicit, and keeps the separation of concerns (Worker builds config, runtime executes).

## Files to Modify

| File | Changes |
|------|---------|
| `src/fluster/cluster/worker/docker.py` | Add properties to Protocol and DockerRuntime |
| `src/fluster/cluster/client/local_client.py` | Add properties to _LocalContainerRuntime |
| `src/fluster/cluster/worker/worker.py` | Replace isinstance checks with property access |

## Testing

Existing E2E tests cover both local and Docker execution paths. No new tests needed - just verify existing tests pass after refactoring.
