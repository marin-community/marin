# Fray Monarch Backend Design

## Overview

This document describes the design for a Monarch-based backend for Fray, where:
- **ClusterContext**: Manually implemented in Python (job submission, lifecycle management)
- **JobContext**: Backed by PyTorch Monarch (task/actor execution, distributed communication)

This approach is viable because:
1. Ray also doesn't support TPU pods natively (we handle that separately)
2. We don't use Ray's object store in practice
3. We're moving away from dynamic task allocation toward static worker pools
4. Monarch provides strong actor primitives and RDMA support that align with our needs

## Architecture

### Two-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│ Fray API (unchanged)                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ClusterContext              JobContext                     │
│  (Manual Python)             (Monarch-backed)               │
│                                                             │
│  • Job submission       →    • Actor spawning               │
│  • Job tracking              • Endpoint calls               │
│  • Job deletion              • Future-based results         │
│  • Process management        • Supervision trees            │
│                              • RDMA transfers               │
└─────────────────────────────────────────────────────────────┘
                    ↓                        ↓
           ┌──────────────┐        ┌─────────────────┐
           │  subprocess  │        │ Monarch Runtime │
           │  management  │        │  (actors, mesh) │
           └──────────────┘        └─────────────────┘
```

### Job Lifecycle

1. **Job Submission** (ClusterContext):
   - User calls `cluster.create_job(entrypoint="python train.py", env=...)`
   - ClusterContext launches controller process via subprocess
   - Controller process is tracked as a "job"

2. **Task Execution** (JobContext within job):
   - Controller script initializes Monarch: `procs = this_host().spawn_procs({...})`
   - Fray context is set: `set_job_context(MonarchJobContext(procs))`
   - User code creates tasks/actors via Fray API
   - MonarchJobContext translates to Monarch actor operations

3. **Job Completion** (ClusterContext):
   - Controller process exits
   - ClusterContext detects completion
   - Resources are released

## ClusterContext Implementation (Manual)

### Core Responsibilities

The MonarchClusterContext handles job-level orchestration:
- Launching controller processes
- Tracking job status
- Managing job lifecycle
- Setting up runtime environments

### Implementation Details

```python
import subprocess
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from fray.cluster import ClusterContext, JobInfo, EntryPoint, RuntimeEnv

@dataclass
class MonarchJobInfo:
    job_id: str
    process: subprocess.Popen
    entrypoint: str
    working_dir: Path
    status: str  # "PENDING", "RUNNING", "SUCCEEDED", "FAILED"

class MonarchClusterContext(ClusterContext):
    """Manual implementation of cluster-level job management."""

    def __init__(self, log_dir: Path | None = None):
        self._jobs: dict[str, MonarchJobInfo] = {}
        self._log_dir = log_dir or Path.cwd() / ".fray" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
        """Launch a new job as a subprocess."""
        job_id = f"monarch-{uuid.uuid4().hex[:8]}"

        # Prepare environment
        job_env = os.environ.copy()
        if env.env_vars:
            job_env.update(env.env_vars)

        # Set up working directory
        working_dir = env.working_dir or Path.cwd()

        # Create log files
        stdout_log = self._log_dir / f"{job_id}.stdout"
        stderr_log = self._log_dir / f"{job_id}.stderr"

        # Launch controller process
        with open(stdout_log, "w") as stdout, open(stderr_log, "w") as stderr:
            process = subprocess.Popen(
                entrypoint,
                shell=True,
                env=job_env,
                cwd=working_dir,
                stdout=stdout,
                stderr=stderr,
            )

        # Track job
        self._jobs[job_id] = MonarchJobInfo(
            job_id=job_id,
            process=process,
            entrypoint=entrypoint,
            working_dir=working_dir,
            status="RUNNING",
        )

        return job_id

    def list_jobs(self) -> list[JobInfo]:
        """List all tracked jobs with current status."""
        jobs = []
        for job_id, info in self._jobs.items():
            # Update status based on process state
            if info.status == "RUNNING":
                returncode = info.process.poll()
                if returncode is not None:
                    info.status = "SUCCEEDED" if returncode == 0 else "FAILED"

            jobs.append(JobInfo(
                job_id=job_id,
                status=info.status,
                entrypoint=info.entrypoint,
            ))
        return jobs

    def delete_job(self, job_id: str) -> None:
        """Terminate a running job."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        info = self._jobs[job_id]
        if info.status == "RUNNING":
            info.process.terminate()
            try:
                info.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                info.process.kill()
            info.status = "FAILED"

        del self._jobs[job_id]

    def get_job_logs(self, job_id: str) -> tuple[str, str]:
        """Retrieve stdout/stderr logs for a job."""
        stdout_log = self._log_dir / f"{job_id}.stdout"
        stderr_log = self._log_dir / f"{job_id}.stderr"

        stdout = stdout_log.read_text() if stdout_log.exists() else ""
        stderr = stderr_log.read_text() if stderr_log.exists() else ""

        return stdout, stderr
```

### Key Design Decisions

1. **Subprocess-based**: Each job is a subprocess running a Monarch controller
2. **Local-only**: Initial implementation runs on single host (can extend to multi-host)
3. **Log management**: Stdout/stderr captured to files for debugging
4. **Status tracking**: Poll process status to update job state
5. **No scheduler**: Jobs run immediately (no queue/scheduling)

### Limitations and Future Work

- **Single-host only**: Multi-host would require SSH or similar
- **No resource isolation**: Jobs share host resources (could add cgroups)
- **No scheduling**: First-come-first-served (could add priority queue)
- **No persistent storage**: Jobs lost on restart (could add SQLite)

## JobContext Implementation (Monarch-backed)

### Core Responsibilities

The MonarchJobContext handles task-level execution:
- Creating and managing actors
- Scheduling endpoint calls (tasks)
- Managing futures and results
- Propagating Fray context

### Monarch Mapping

| Fray API | Monarch Equivalent | Notes |
|----------|-------------------|-------|
| `create_task(fn, args)` | Actor with `@endpoint` | Create single-use actor |
| `create_actor(cls, args)` | `procs.spawn(cls)` | Create actor mesh |
| `get(ref)` | `future.get()` | Wait for result |
| `wait(refs, num_returns)` | Multiple `future.get()` | Poll futures |
| `put(obj)` | Not needed | No object store |

### Implementation Details

```python
from typing import Any, Callable, Protocol
import contextvars
from dataclasses import dataclass
from monarch.actor import Actor, endpoint, Procs, Mesh, Future
from fray.job import JobContext, TaskOptions, ActorOptions

# Context variable for propagating JobContext
_job_context: contextvars.ContextVar['MonarchJobContext'] = contextvars.ContextVar('job_context')

class MonarchObjectRef:
    """Wraps Monarch Future to provide Fray-compatible object reference."""

    def __init__(self, future: Future, take_first: bool = False):
        self._future = future
        self._take_first = take_first

    def __repr__(self):
        return f"MonarchObjectRef({self._future})"

class MonarchActorHandle:
    """Wraps Monarch actor mesh to provide single-actor semantics."""

    def __init__(self, mesh: Mesh, actor_index: int = 0):
        self._mesh = mesh
        self._actor_index = actor_index

    def __getattr__(self, name: str):
        """Intercept method calls to create actor method handles."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def method_wrapper(*args, **kwargs):
            # Get the endpoint from the mesh
            endpoint_call = getattr(self._mesh, name)
            future = endpoint_call.call(*args, **kwargs)
            # Return reference that extracts single result
            return MonarchObjectRef(future, take_first=True)

        return method_wrapper

@dataclass
class MonarchJobContext(JobContext):
    """Monarch-backed implementation of Fray JobContext."""

    def __init__(self, procs: Procs):
        self._procs = procs
        self._task_counter = 0
        self._actor_counter = 0

    def create_task(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        options: TaskOptions | None = None
    ) -> Any:
        """
        Create a task by wrapping function in a single-use actor.

        Note: Since we're moving away from dynamic allocation, we spawn
        actors on pre-allocated processes. For true single-task semantics,
        could use a worker pool pattern.
        """
        if kwargs is None:
            kwargs = {}

        # Capture current context
        current_ctx = self

        # Create dynamic actor class for this task
        task_id = self._task_counter
        self._task_counter += 1

        class TaskActor(Actor):
            def __init__(self):
                # Store context in actor
                self._ctx = current_ctx

            @endpoint
            def run(self):
                # Restore context when executing
                _job_context.set(self._ctx)
                return fn(*args, **kwargs)

        # Spawn actor on processes (creates mesh)
        actor_name = f"task_{task_id}"
        mesh = self._procs.spawn(actor_name, TaskActor)

        # Call the run endpoint
        future = mesh.run.call()

        # Return ref that extracts first result (single-task semantics)
        return MonarchObjectRef(future, take_first=True)

    def create_actor(
        self,
        klass: type,
        args: tuple = (),
        kwargs: dict | None = None,
        options: ActorOptions | None = None
    ) -> Any:
        """
        Create a long-lived actor.

        For single-actor semantics, we spawn a mesh but wrap it to provide
        single-actor API. For multi-actor patterns, users can access mesh directly.
        """
        if kwargs is None:
            kwargs = {}

        # Capture current context
        current_ctx = self

        # Wrap actor class to inject context
        class ContextualActor(klass):
            def __init__(self, *init_args, **init_kwargs):
                super().__init__(*init_args, **init_kwargs)
                self._fray_ctx = current_ctx
                # Restore context
                _job_context.set(current_ctx)

        # Spawn actor mesh
        actor_id = self._actor_counter
        self._actor_counter += 1
        actor_name = f"{klass.__name__}_{actor_id}"

        mesh = self._procs.spawn(actor_name, ContextualActor, args=args, kwargs=kwargs)

        # Wrap mesh to provide single-actor semantics
        return MonarchActorHandle(mesh, actor_index=0)

    def get(self, ref: Any) -> Any:
        """Block and retrieve result from object reference."""
        if not isinstance(ref, MonarchObjectRef):
            raise TypeError(f"Expected MonarchObjectRef, got {type(ref)}")

        # Get results from future
        results = ref._future.get()

        # If this is a single-task reference, return first result
        if ref._take_first:
            if isinstance(results, list) and len(results) > 0:
                return results[0]
            return results

        # Otherwise return all results
        return results

    def wait(
        self,
        refs: list[Any],
        num_returns: int = 1,
        timeout: float | None = None
    ) -> tuple[list[Any], list[Any]]:
        """
        Wait for some references to complete.

        Note: Monarch futures don't have built-in wait_any/wait_all semantics,
        so we implement polling-based wait.
        """
        import time

        ready = []
        not_ready = list(refs)
        start_time = time.time()

        while len(ready) < num_returns and len(not_ready) > 0:
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                break

            # Poll futures (check if done without blocking)
            # Note: Monarch futures may not have is_done(), so we attempt get() with timeout
            for ref in not_ready[:]:
                try:
                    # Attempt to get with very short timeout
                    result = ref._future.get()  # This blocks!
                    ready.append(ref)
                    not_ready.remove(ref)

                    if len(ready) >= num_returns:
                        break
                except Exception:
                    # Future not ready yet
                    continue

            # Small sleep to avoid busy waiting
            if len(ready) < num_returns:
                time.sleep(0.01)

        return ready, not_ready

    def put(self, obj: Any) -> Any:
        """
        Store object in distributed store.

        Note: Monarch doesn't have an object store. Options:
        1. Not implement (raise NotImplementedError)
        2. Use in-memory dict (not distributed)
        3. Use actor as storage (heavyweight)

        For now, we don't implement since you mentioned not using object store.
        """
        raise NotImplementedError(
            "Monarch backend does not support object store (put/get). "
            "Pass objects directly or use actors for shared state."
        )
```

### Resource Management

Map Fray's resource specifications to Monarch's process spawning:

```python
def _parse_resource_options(options: TaskOptions | ActorOptions | None) -> dict:
    """
    Convert Fray resource specs to Monarch process requirements.

    Since Monarch allocates resources at process spawn time, we need to
    interpret task/actor resources as hints for which processes to use.
    """
    if options is None or options.resources is None:
        return {}

    monarch_resources = {}

    # GPU resources
    if "GPU" in options.resources:
        num_gpus = int(options.resources["GPU"])
        monarch_resources["gpus"] = num_gpus

    # CPU resources (informational - can't control per-actor)
    if "CPU" in options.resources:
        num_cpus = int(options.resources["CPU"])
        # Monarch doesn't have per-actor CPU limits, but we can use this
        # to inform process spawning strategy
        monarch_resources["cpus"] = num_cpus

    # Memory (informational)
    if "memory" in options.resources:
        memory_bytes = int(options.resources["memory"])
        monarch_resources["memory"] = memory_bytes

    return monarch_resources

# Enhanced JobContext with resource-aware spawning
class MonarchJobContext(JobContext):
    def __init__(self, procs: Procs | None = None, resource_config: dict | None = None):
        """
        Initialize JobContext with optional pre-spawned processes or resource config.

        Args:
            procs: Pre-spawned Monarch processes (if None, will spawn based on config)
            resource_config: Dict like {"gpus": 8, "cpus": 64} for spawning
        """
        if procs is None:
            if resource_config is None:
                raise ValueError("Must provide either procs or resource_config")
            from monarch.actor import this_host
            procs = this_host().spawn_procs(resource_config)

        self._procs = procs
        self._resource_config = resource_config or {}
        self._task_counter = 0
        self._actor_counter = 0
```

### Context Propagation

Ensure Fray context is available within Monarch actors:

```python
from fray.context import get_job_context, set_job_context

# In user code (job entrypoint):
def main():
    from monarch.actor import this_host

    # Spawn worker processes
    procs = this_host().spawn_procs({"gpus": 8})

    # Create and set Fray context
    ctx = MonarchJobContext(procs)
    set_job_context(ctx)

    # Now user code can use Fray API
    def my_task(x):
        # Can call get_job_context() here
        ctx = get_job_context()
        # Create nested tasks if needed
        nested = ctx.create_task(lambda: x * 2)
        return ctx.get(nested)

    ref = ctx.create_task(my_task, args=(5,))
    result = ctx.get(ref)
    print(result)  # 10
```

## Complete Example

### Job Submission (Cluster Level)

```python
# submit_job.py - runs on submission host
from fray.backend.monarch import MonarchClusterContext
from fray.cluster import RuntimeEnv

cluster = MonarchClusterContext()

job_id = cluster.create_job(
    entrypoint="python training_script.py --epochs 100",
    env=RuntimeEnv(
        env_vars={"MONARCH_LOG_LEVEL": "INFO"},
        working_dir="/path/to/project",
    )
)

print(f"Submitted job {job_id}")

# Monitor job
import time
while True:
    jobs = cluster.list_jobs()
    job = next(j for j in jobs if j.job_id == job_id)
    print(f"Job {job_id}: {job.status}")

    if job.status in ["SUCCEEDED", "FAILED"]:
        break

    time.sleep(5)

# Get logs
stdout, stderr = cluster.get_job_logs(job_id)
print("STDOUT:", stdout)
print("STDERR:", stderr)
```

### Task Execution (Job Level)

```python
# training_script.py - runs as Monarch controller
from monarch.actor import this_host, Actor, endpoint
from fray.backend.monarch import MonarchJobContext
from fray.context import set_job_context, get_job_context
import torch

def main():
    # Initialize Monarch worker processes
    procs = this_host().spawn_procs({"gpus": 8})

    # Create Fray context
    ctx = MonarchJobContext(procs)
    set_job_context(ctx)

    # Define trainer actor
    class Trainer(Actor):
        def __init__(self, rank: int):
            self.rank = rank
            self.model = create_model().cuda(rank)

        @endpoint
        def train_step(self, batch_data):
            # Get Fray context if needed
            ctx = get_job_context()

            # Training logic
            loss = self.model.train_step(batch_data)
            return loss

    # Create trainer actors (one per GPU)
    trainers = ctx.create_actor(Trainer, args=(0,))  # Creates mesh

    # Or use direct Monarch API for mesh operations
    trainer_mesh = procs.spawn("trainers", Trainer, args=(0,))

    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            # Option 1: Use Fray API (single-actor semantics)
            loss_ref = trainers.train_step(batch)
            loss = ctx.get(loss_ref)

            # Option 2: Use Monarch mesh API (broadcast to all)
            losses_future = trainer_mesh.train_step.call(batch_data=batch)
            losses = losses_future.get()  # List of losses from all trainers

            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss}")

if __name__ == "__main__":
    main()
```

### Actor-Based Pattern

```python
# actor_example.py - using actors for stateful computation
from monarch.actor import this_host, Actor, endpoint
from fray.backend.monarch import MonarchJobContext
from fray.context import set_job_context

class ParameterServer(Actor):
    """Actor that maintains shared state (parameter server pattern)."""

    def __init__(self, dim: int):
        self.params = torch.randn(dim)
        self.version = 0

    @endpoint
    def get_params(self):
        return self.params, self.version

    @endpoint
    def update_params(self, gradient):
        self.params -= 0.01 * gradient
        self.version += 1
        return self.version

class Worker(Actor):
    """Worker actor that computes gradients."""

    def __init__(self, rank: int, ps_handle):
        self.rank = rank
        self.ps = ps_handle

    @endpoint
    def compute_gradient(self, data):
        # Get current params from PS
        params_ref = self.ps.get_params()
        params, version = get_job_context().get(params_ref)

        # Compute gradient
        gradient = compute_grad(data, params)

        # Update PS
        update_ref = self.ps.update_params(gradient)
        new_version = get_job_context().get(update_ref)

        return new_version

def main():
    procs = this_host().spawn_procs({"gpus": 8})
    ctx = MonarchJobContext(procs)
    set_job_context(ctx)

    # Create parameter server actor
    ps = ctx.create_actor(ParameterServer, args=(1024,))

    # Create worker actors (pass PS handle)
    workers = []
    for i in range(8):
        worker = ctx.create_actor(Worker, args=(i, ps))
        workers.append(worker)

    # Training loop
    for batch in dataloader:
        # Workers compute gradients in parallel
        version_refs = [w.compute_gradient(batch) for w in workers]
        versions = [ctx.get(ref) for ref in version_refs]
        print(f"Updated to version {versions[0]}")

if __name__ == "__main__":
    main()
```

## Comparison to Ray Backend

| Feature | Ray Backend | Monarch Backend |
|---------|-------------|-----------------|
| **Job Submission** | Ray JobSubmissionClient | Manual subprocess |
| **Task Scheduling** | `ray.remote()` functions | Monarch actors + endpoints |
| **Dynamic Allocation** | Yes (task → any worker) | No (task → pre-allocated process) |
| **Object Store** | Yes (`ray.put/get`) | No (not implemented) |
| **Actor Semantics** | Single actor | Mesh (wrapped for single-actor API) |
| **Resource Requests** | Per-task GPU/CPU/memory | Process-level at spawn |
| **TPU Support** | Manual (via run_on_tpu) | Manual (same approach) |
| **Fault Tolerance** | Task retry | Supervision trees |
| **Communication** | Plasma + gRPC | RDMA (libibverbs) |
| **Overhead** | Lower for single tasks | Higher (actor creation) |
| **Best For** | Dynamic workloads | Static SPMD training |

## Migration Considerations

### Code Changes Required

1. **Job Entrypoints**: Must initialize Monarch processes explicitly
   ```python
   # Old (Ray)
   # Just use Fray API directly

   # New (Monarch)
   from monarch.actor import this_host
   procs = this_host().spawn_procs({"gpus": 8})
   ctx = MonarchJobContext(procs)
   set_job_context(ctx)
   ```

2. **Resource Specification**: Move to job level
   ```python
   # Old (Ray)
   ref = ctx.create_task(fn, options=TaskOptions(resources={"GPU": 1}))

   # New (Monarch)
   # Specify resources when spawning processes
   procs = this_host().spawn_procs({"gpus": 8})
   ctx = MonarchJobContext(procs)
   ref = ctx.create_task(fn)  # Runs on one of the 8 GPU processes
   ```

3. **Object Store**: Remove usage
   ```python
   # Old (Ray)
   ref = ctx.put(large_object)
   obj = ctx.get(ref)

   # New (Monarch)
   # Pass objects directly or use actors for shared state
   shared_state = ctx.create_actor(StateActor, args=(large_object,))
   ```

### Advantages of Monarch Backend

1. **RDMA Support**: Zero-copy GPU transfers for high-bandwidth training
2. **Supervision Trees**: Hierarchical fault tolerance
3. **Mesh Operations**: Native support for SPMD patterns
4. **Performance**: Lower overhead for fixed-topology workloads
5. **Simplicity**: No separate cluster daemon (Ray head node)

### Disadvantages of Monarch Backend

1. **No Dynamic Allocation**: Must pre-allocate worker processes
2. **No Object Store**: Can't store/retrieve objects by reference
3. **Manual Cluster Management**: No built-in job scheduler
4. **Higher Per-Task Overhead**: Creating actors for simple tasks is expensive
5. **Mesh Semantics**: Need wrapper layer for single-actor API

## Implementation Checklist

### Phase 1: Core Implementation

- [ ] Implement `MonarchClusterContext`
  - [ ] `create_job()` - subprocess management
  - [ ] `list_jobs()` - job tracking
  - [ ] `delete_job()` - job termination
  - [ ] Log capture and retrieval

- [ ] Implement `MonarchJobContext`
  - [ ] `create_task()` - dynamic actor creation
  - [ ] `create_actor()` - persistent actors
  - [ ] `get()` - future unwrapping
  - [ ] `wait()` - multi-future waiting

- [ ] Implement helper classes
  - [ ] `MonarchObjectRef` - future wrapper
  - [ ] `MonarchActorHandle` - mesh wrapper
  - [ ] Context propagation via `contextvars`

### Phase 2: Resource Management

- [ ] Resource parsing from TaskOptions/ActorOptions
- [ ] Mapping resources to process requirements
- [ ] Process pool management
- [ ] Resource-aware actor placement

### Phase 3: Testing

- [ ] Unit tests for ClusterContext
- [ ] Unit tests for JobContext
- [ ] Integration tests with actual Monarch runtime
- [ ] Performance benchmarks vs Ray backend
- [ ] Fault tolerance tests

### Phase 4: Documentation

- [ ] API documentation
- [ ] Migration guide from Ray backend
- [ ] Example training scripts
- [ ] Troubleshooting guide

## Open Questions

1. **Multi-host Support**: How to handle jobs that span multiple hosts?
   - Option A: Extend subprocess management to use SSH
   - Option B: Require external orchestration (Kubernetes, Slurm)
   - Option C: Use Monarch's built-in multi-host coordination

2. **Resource Isolation**: How to prevent jobs from interfering?
   - Option A: cgroups for CPU/memory limits
   - Option B: Trust users to specify correct resources
   - Option C: Run on separate hosts per job

3. **Worker Pool Strategy**: How to handle task-to-process mapping?
   - Option A: Round-robin assignment
   - Option B: Maintain per-process queues
   - Option C: Create actor per task (current design)

4. **Fault Tolerance**: How to handle actor failures?
   - Option A: Use Monarch's supervision trees
   - Option B: Retry at Fray level
   - Option C: Propagate failures to user

5. **Object Serialization**: How to pass data between actors?
   - Option A: Pickle (standard Python)
   - Option B: CloudPickle (Ray's default)
   - Option C: Monarch's built-in serialization

## Next Steps

1. **Prototype Implementation**: Build minimal MonarchJobContext
2. **Test with Simple Workload**: Single-host, single-actor training
3. **Measure Overhead**: Compare actor creation cost to Ray tasks
4. **Evaluate RDMA**: Test high-bandwidth tensor transfers
5. **Iterate on Design**: Refine based on real usage patterns

## References

- Fray Architecture: `/Users/power/code/marin/lib/fray/architecture.md`
- Ray Backend Implementation: `/Users/power/code/marin/lib/fray/src/fray/backend/ray/`
- Monarch Documentation: https://meta-pytorch.org/monarch/
- Monarch API: https://meta-pytorch.org/monarch/api/index.html
