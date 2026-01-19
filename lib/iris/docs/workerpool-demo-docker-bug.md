# WorkerPool Docker Demo Bug Report

## Status: RESOLVED

## Summary

The WorkerPool demo failed in Docker mode because the worker pool job entrypoint was not using the allocated port for the ActorServer.

## Root Cause

In `worker_pool.py`, the `worker_job_entrypoint` function was creating an ActorServer without specifying the port:

```python
# BUG: ActorServer picks an arbitrary port that Docker doesn't publish
server = ActorServer(host="0.0.0.0")
```

The job correctly requests `ports=["actor"]` which causes the worker to allocate and publish a port via Docker's `-p` flag. However, the entrypoint wasn't using `ctx.get_port("actor")` to get this allocated port. Instead, it let the ActorServer pick an arbitrary port, which wasn't published by Docker and couldn't be reached from other containers.

## Symptoms

1. The coordinator job (`workerpool-demo`) was dispatched successfully
2. It successfully submitted the pool-worker job with 3 tasks (co-scheduling works)
3. All 3 pool-worker tasks were dispatched and started
4. Pool workers registered their endpoints correctly
5. But when the coordinator tried to connect to workers, it got "Connection refused"

The key insight from the diagnostic logs:
```
[stdout] Pool ready: 2 workers available
[stderr] httpx.ConnectError: [Errno 111] Connection refused
```

The workers were discovered (endpoints registered) but the port they registered wasn't reachable from the coordinator container.

## The Fix

Three changes were made:

### 1. Use allocated port in worker_pool.py (main fix)

```python
# Get the allocated port - this port is published by Docker for container access
port = ctx.get_port("actor")

# Start actor server on the allocated port
server = ActorServer(host="0.0.0.0", port=port)
```

### 2. Add proper error handling to thunk in worker.py

The Python thunk that runs in Docker containers now wraps execution in try/except and uses `-u` for unbuffered output, ensuring any errors are visible in container logs:

```python
thunk = f"""
import cloudpickle
import base64
import sys
import traceback

try:
    fn, args, kwargs = cloudpickle.loads(base64.b64decode('{encoded}'))
    result = fn(*args, **kwargs)
    ...
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
return ["python", "-u", "-c", thunk]
```

### 3. Improve DI boundary for local_client.py

Instead of parsing command strings (fragile, depends on exact command format), the local runtime now receives the entrypoint directly via `ContainerConfig.serialized_entrypoint`. This is a cleaner DI boundary.

In `docker.py`:
```python
@dataclass
class ContainerConfig:
    ...
    # Serialized entrypoint for in-process runtimes (local testing).
    # Docker runtime ignores this and uses the command instead.
    serialized_entrypoint: bytes | None = None
```

In `worker.py`:
```python
# Serialize entrypoint for local runtime (avoids parsing command string)
entrypoint_tuple = (entrypoint.callable, entrypoint.args, entrypoint.kwargs)
serialized_entrypoint = cloudpickle.dumps(entrypoint_tuple)

config = ContainerConfig(
    ...
    serialized_entrypoint=serialized_entrypoint,
)
```

In `local_client.py`:
```python
# Use serialized_entrypoint directly - no command parsing needed
if self.config.serialized_entrypoint is None:
    raise ValueError("LocalContainer requires serialized_entrypoint in ContainerConfig")
fn, args, kwargs = cloudpickle.loads(self.config.serialized_entrypoint)
```

This removed the fragile regex-based command parsing entirely.

## Verification

All tests pass:
- 32 combined E2E and WorkerPool tests
- 101 controller tests
- 18 actor tests
- Full Docker notebook demo

```bash
uv run python examples/demo_cluster.py --test-notebook --docker
# All notebook cells executed successfully!
# Notebook test passed!
```

## Files Changed

- `src/iris/client/worker_pool.py`: Use allocated port from context
- `src/iris/cluster/worker/worker.py`: Improve error handling in thunk, add serialized_entrypoint to ContainerConfig
- `src/iris/cluster/worker/docker.py`: Add serialized_entrypoint field to ContainerConfig
- `src/iris/cluster/client/local_client.py`: Use serialized_entrypoint directly instead of parsing commands
