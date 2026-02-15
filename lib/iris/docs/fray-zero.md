# Fray-Zero Design

As discussed in
[Fray Presentation](https://docs.google.com/presentation/d/1qgPGK7zYmiOSOn70W-rPIrcF7vuuHgg68Qr3e5UIuxw/edit) and
[Fray/RPC Design](https://docs.google.com/document/d/1UteX9nD9obY5ypV2o8KbHF72xwX1JezseRE5uLFBulk/edit?tab=t.0),
we think it's a good time to "grug-ify" our RPC and clustering system while
moving off of the complex and fragile Ray.

## Original Design and Progress

Our original Ray challenges doc
[Ray Infrastructure Challenges](https://docs.google.com/document/d/1gtCz3aN2q72ZF-BNK_nKHCjuS9CT88QSH5vmkTktwWQ/edit?tab=t.0#heading=h.9k9q6db2omrh)
and migration plan
[Ray Migration](https://docs.google.com/document/d/1r-YKwxMxD8dJPKFmQrdJdIsfVvrLnIPo8RnlANbwM5o/edit?tab=t.0)
outlined a mostly "drop-in" approach to replacing Ray with Fray. We would
introduce new wrapper APIs which hid Ray's usage (fray.cluster and fray.job),
move most direct usage into a data library (zephyr), and reduce the complexity
of our dependencies to remove the need for complex venv creation logic.

We've executed most of this plan unaltered:

* We've dramatically reduced the scope of "raw" Ray usage by using Zephyr for our data pipelines
* We've abstracted our Ray usage behind 2 APIs: fray.cluster and fray.job.
* We've dramatically pruned our "extra" Python dependencies, removing the need for on-demand Python environments

We still use Ray for our cluster and job RPC management, but for local tests and
speedruns we now use a local implementation which completely avoids the use of
Ray.

## Refining Our Vision

In the process of implementing the final set of changes to implement our own Ray
compatible backend and clustering system, we started to ask "is this actually
what we want?". That is, while we feel our work up to this point has been
building useful abstractions around Ray, we're now confronted with the
complexity of re-implementing the bulk of Ray, and _only then_ proceeding to
revisit the API decisions and try to incrementally improve them.

While acknowledging the value of incremental changes, we realized we're loathe
to create the same mess we started with. And since we've been so productive at
gradually refactoring our codebase to migrate off of Ray, we want to consider
whether we can push _further_ that direction: instead of recreating Ray, what
simpler primitives can we create that we can use in Marin _in place of Ray_?

## Requirements

We have a few job types in Marin, which in general run independently of each
other (that is, they are launched independently and don't talk to a different
job type).

### 1. Training

Our simplest case, training jobs run on TPU slices, and communicate entirely via
JAX collectives after startup. Moreover training tasks run for a long time, so
startup overhead is not a problem.

### 2. Data Processing

Data processing wants to flexibly be able to use CPU resources. With our latest
Zephyr refactoring, there is minimal communication between tasks, as all work is
staged to object storage.

### 3. Inference/Evaluation

Inference is a combination of training and data processing. We want to start N
inference servers (potentially on slices), and then dispatch work to that pool
via one or more CPU jobs. As we will typically be inference-bound, it's likely a
single CPU task will be sufficient for our initial work.

### 4. RL

RL has 2-3 different jobs which use accelerators:

* Training worker
* Rollout workers (environments)
* Inference workers

Internally these jobs use JAX as expected, but they also need to communicate
metadata about the progress and new checkpoints via actors which are shared
across all processes (CurriculumActor and TrainingActor).

### 5. (Flex) Multi-slice Training

For flex-multi-slice, we have multiple training jobs, each running on a separate
TPU slice. Slices can come and go over time, but we'd like to take advantage of
slices when they are available. The simplest way to express this is to move our
workers and leader into actors, and dispatch work from the leader:

```python
# leader.py
def multislice_train():
  while True:
    slice_workers = build_multi_slice()  # build a cluster based on available slices
    slice_workers.train_step()  # or train_n_steps, or whatever

# worker.py
class Worker:
  def __init__(self):
    self.weights = load_weights()
    self.data = load_data(slice_id)
    self.model = create_model([peer_slices])

  def train_step(self):
    self.model.step(next(self.data))
```

## Design

Our plan is to focus on doing a few simple things well, instead of many things
poorly. We'll explicitly break out goals into 2 parts, a _job management system_
which excels at managing reservations and booting up UV environments on a set of
VMs, and a _RPC system_ which makes it easy for users to setup _their own_
communication between tasks.

### Job Management

In Marin a typical workflow involves starting a controller job which runs
Executor, which then spawns one or more sub-jobs. The job management system is
responsible for managing the lifecycle of these jobs, including:

* Reserving resources
* Booting up UV environments
* Managing auto-scaling
* Managing task failures/restarts/termination
* Providing access to task metrics/logs
* Fairly sharing resources across users

The cluster manager manages workloads for all users and all regions. It manages
a set of VMs as well as auto-scaling VM requests in response to demand. To
request resources from the cluster users create _reservations_ which specify the
minimum and maximum set of resources required for a set of jobs.

For the purposes of locality (we want RL inference workers to run in the same
cluster as the training workers), a user can also define a _job group_ which
specifies a co-located set of job requests.

A job template defines:
 * _entrypoint_ - a Python callable with optional arguments
 * _environment_ - pip packages, extras, and environment variables
 * _resources_ - accelerators, CPU, memory requirements

A user may request multiple instantiations of a job template, and a friendly
name prefix for grouping (e.g. `{user}/rl/rollout/`). Every job receives
a globally unique ID.

**Namespaces**: When a user starts a top-level job, the cluster creates a new
namespace derived from the job name. Sub-jobs launched from within that job
inherit the same namespace. Actor names are scoped to namespaces, so actors in
different namespaces don't collide.

**Environment Variables**: The cluster injects standard environment variables
into every job process:

* `FRAY_JOB_ID` - Unique identifier for this job instance
* `FRAY_JOB_NAME` - User-provided job name
* `FRAY_NAMESPACE` - Namespace for actor registration/lookup
* `FRAY_CLUSTER_ADDRESS` - Address of the cluster controller

**Health Monitoring**: The cluster monitors job health via:

* Process status monitoring (exit codes, crashes)
* Optional health check endpoint - jobs can expose a health route that the
  cluster pings periodically

**Lifecycle Management**: When a parent job terminates, the cluster
automatically terminates all child jobs spawned by that parent. This ensures
cleanup happens automatically without requiring explicit shutdown coordination.

The cluster system provides access to job logs, and handles task failures and
pre-emption recovery. It attempts to minimize the startup time of tasks by
reusing warm environments and re-using warm workers when possible. The cluster
system _will not_ attempt to reuse warm processes, thus the target startup time
for tasks is expected to be <10s for a warm image.

### Metadata Service

The cluster provides a metadata service that maps actor names to endpoints
(job ID + address + port). Because the cluster owns both the jobs and the
metadata service, it can:

* Automatically clean up mappings when jobs terminate
* Atomically update mappings when jobs restart after failure
* Garbage collect stale entries

Multiple actors can register under the same name. The metadata service maintains
a list of all actors for each name. When a job terminates, all actors registered
by that job are automatically removed from any name mappings they participate in.

The metadata service is internal to the cluster. Clients interact with it
through a **Resolver**.

### Resolver and ActorPool

A Resolver provides actor discovery and connection management. It maps actor
names to `ActorPool` instances that handle load balancing, broadcasting, and
failure recovery.

```python
# Create resolver (explicitly, from cluster or environment)
resolver = ClusterResolver(cluster)  # uses cluster's metadata service
resolver = FixedResolver({"inference": "localhost:8080"})  # for testing

# Look up actors - always returns an ActorPool
pool = resolver.lookup("inference_pool")

# Single call (round-robin to one actor)
result = pool.call().predict(x)

# Broadcast to all actors
futures = pool.broadcast().predict(x)
results = [f.result() for f in futures]  # Returns all results, including failures

# Query pool state
print(f"Pool has {pool.size} actors")
pool.wait_for_size(4, timeout=60.0)  # Wait until N actors available
```

The resolver handles:

* **Worker resolution**: Maps actor names to current addresses
* **Fault tolerance**: Re-resolves addresses when workers fail; retries on transient failures
* **Load balancing**: Round-robin distribution via `pool.call()`
* **Fan-out**: Broadcast to all actors via `pool.broadcast()`

### ActorServer

ActorServer lets users expose Python classes as RPC services without writing
protos. Each job runs at most one ActorServer (since it binds to a port).
Serialization uses pickle for arguments and return values.

Actor methods receive an `ActorContext` as their first argument, which provides
access to the cluster, resolver, and job information - enabling actors to call
other actors.

```python
class InferenceActor:
    def __init__(self):
        self.model = load_model()

    def predict(self, ctx: ActorContext, x):
        # ctx.resolver available if this actor needs to call other actors
        return self.model(x)

cluster = current_cluster()
server = ActorServer(cluster)
server.register("inference_pool", InferenceActor())  # Register under pool name
server.serve()  # blocks, handling requests
```

When a job hosting an ActorServer fails and restarts, the cluster automatically
updates the metadata mappings. Clients holding pool references will transparently
reconnect to the new instance on their next call.

### WorkerPool

For task dispatch patterns (like Zephyr), we provide a WorkerPool abstraction
built on top of actors and jobs. WorkerPool manages a set of stateless workers
that can execute arbitrary callables.

```python
# Create a pool of workers
pool = WorkerPool(
    cluster=current_cluster(),
    num_workers=10,
    resources=ResourceConfig.with_cpu(cpu=2, memory="4GB"),
)

# Submit tasks - returns futures
futures = [pool.submit(process_shard, shard) for shard in shards]

# Gather results
results = [f.result() for f in futures]

# Cleanup
pool.shutdown()
```

Internally, WorkerPool:
1. Launches N worker jobs, each running an actor that accepts callables
2. Distributes work via round-robin through the resolver
3. Handles retries on worker failure (stateless workers allow retry on any worker)

### Use Case Implementations

* **Training**: Single cluster job launch, no actors needed
* **Inference**: Launch N server jobs, each registers under same actor name, clients use `pool.call()` for load balancing
* **Zephyr**: Uses WorkerPool to dispatch shard processing tasks
* **RL**: Coordinator job hosts shared actors (curriculum, weights), workers connect via resolver

```python
# rl_controller.py
cluster = current_cluster()
resolver = ClusterResolver(cluster)

# Launch coordinator hosting shared actors
cluster.launch(coordinator_job)

# Launch workers - they connect to coordinator via resolver
cluster.launch(train_worker_job)
for i in range(num_rollout_workers):
    cluster.launch(rollout_worker_job(i))

# Workers internally do:
#   resolver = ClusterResolver(cluster)
#   curriculum = resolver.lookup("curriculum")
#   curriculum.call().get_lesson()
```


## Local Development

For local development, all components have in-process implementations that
preserve the same interfaces. Code written for production works locally
without modification.

* Jobs run as threads in the current process
* Actors are called directly (in-process) but serialization still occurs
* The same code paths execute, catching serialization bugs early

```python
# Works identically in local and production
cluster = current_cluster()  # Returns LocalCluster when FRAY_CLUSTER_SPEC unset
resolver = ClusterResolver(cluster)

server = ActorServer(cluster)
server.register("my_actor", MyActor())
server.serve_background()  # Spawns a thread locally

pool = resolver.lookup("my_actor")
result = pool.call().process(data)  # Serializes args, calls in-process
```

## Detailed Design

This section provides the complete Python interfaces for the Fray system. The
architecture consists of four main components:

1. **Cluster** - Job lifecycle management (launch, monitor, terminate). Owns a
   Metadata service for actor registration.
2. **Resolver** - Actor discovery and connection management. Maps actor names
   to ActorPools and handles reconnection on failure.
3. **ActorServer** - Hosts actor instances, handles RPC calls, registers with
   the cluster's metadata service.
4. **WorkerPool** - High-level task dispatch abstraction built on actors.

### Core Types (should be defined via proto files)

These are just for reference, the actual types should be defined via proto files for objects that will be serialized.
JobState etc types are shared between the cluster controller and worker.

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar
from enum import StrEnum


# Type aliases for clarity
JobName = str
ActorId = str
ReservationId = str
Namespace = str


class JobState(StrEnum):
    """Status of a job in the cluster."""
    PENDING = "pending"      # Waiting for resources
    RUNNING = "running"      # Currently executing
    SUCCEEDED = "succeeded"  # Completed successfully
    FAILED = "failed"        # Terminated with error
    STOPPED = "stopped"      # Manually terminated


@dataclass
class JobInfo:
    """Information about a job's current state."""
    job_id: JobName
    name: str
    state: JobState
    error_message: str | None = None
    start_time: float | None = None
    end_time: float | None = None
```

### Resource Configuration

```python
@dataclass
class CpuConfig:
    """CPU-only resource configuration."""
    kind: str = "cpu"
    variant: str = "default"


@dataclass
class TpuConfig:
    """TPU accelerator configuration."""
    kind: str = "tpu"
    variant: str  # e.g., "v5litepod-4", "v5litepod-16"
    count: int = 1


@dataclass
class GpuConfig:
    """GPU accelerator configuration."""
    kind: str = "gpu"
    variant: str  # e.g., "a100-40gb", "h100"
    count: int = 1


DeviceConfig = CpuConfig | TpuConfig | GpuConfig


@dataclass
class ResourceConfig:
    """Resource requirements for a job."""
    device: DeviceConfig = field(default_factory=CpuConfig)
    cpu: int = 1
    memory: str = "2GB"
    replicas: int = 1

    @classmethod
    def with_cpu(cls, cpu: int = 1, memory: str = "2GB") -> "ResourceConfig":
        """Create CPU-only resource config."""
        return cls(device=CpuConfig(), cpu=cpu, memory=memory)

    @classmethod
    def with_tpu(
        cls,
        tpu_type: str,
        slice_count: int = 1,
    ) -> "ResourceConfig":
        """Create TPU resource config.

        Args:
            tpu_type: TPU variant (e.g., "v5litepod-4", "v5litepod-16")
            slice_count: Number of TPU slices
        """
        return cls(
            device=TpuConfig(variant=tpu_type),
            replicas=slice_count,
        )

    @classmethod
    def with_gpu(
        cls,
        gpu_type: str,
        count: int = 1,
        cpu: int = 4,
        memory: str = "16GB",
    ) -> "ResourceConfig":
        """Create GPU resource config."""
        return cls(
            device=GpuConfig(variant=gpu_type, count=count),
            cpu=cpu,
            memory=memory,
        )
```

### Entrypoint and Environment

```python
@dataclass
class Entrypoint:
    """Job entrypoint specification.

    Jobs are started by invoking a Python callable with the provided arguments.
    The callable and arguments must be picklable.
    """
    callable: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> "Entrypoint":
        """Create entrypoint from a callable.

        Args:
            fn: Python callable to execute
            args: Positional arguments to pass
            kwargs: Keyword arguments to pass
        """
        return cls(callable=fn, args=args, kwargs=kwargs or {})


@dataclass
class EnvironmentConfig:
    """Environment configuration for job execution."""
    extras: list[str] = field(default_factory=list)
    pip_packages: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        extras: list[str] | None = None,
        pip_packages: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> "EnvironmentConfig":
        """Create environment config with optional overrides."""
        return cls(
            extras=extras or [],
            pip_packages=pip_packages or [],
            env_vars=env_vars or {},
        )
```

### Job Request and Group Configuration

```python
@dataclass
class JobRequest:
    """Request to launch a job on the cluster."""
    name: str
    resources: ResourceConfig
    entrypoint: Entrypoint
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


@dataclass
class JobGroupConfig:
    """Configuration for a group of co-located jobs.

    Jobs within a group are scheduled together with network locality
    guarantees. Use this for RL training where inference workers should
    be close to training workers.
    """
    name: str
    namespace: Namespace | None = None
    same_region: bool = True
    same_zone: bool = False  # Stricter locality for low-latency RPCs


@dataclass
class ReservationConfig:
    """Resource reservation for elastic job scheduling."""
    min_resources: ResourceConfig
    max_resources: ResourceConfig
    priority: int = 0
    ttl_seconds: int = 3600
    preemptible: bool = True


@dataclass
class ReservationInfo:
    """Status of an active reservation."""
    reservation_id: ReservationId
    allocated_resources: ResourceConfig
    pending_resources: ResourceConfig
    jobs: list[JobName]
    expires_at: float
```

### Cluster Interface

The Cluster is responsible for job lifecycle management. It does not
provide task execution primitives - jobs are complete processes that run
independently.

```python
class Cluster(Protocol):
    """Abstract interface for cluster job scheduling.

    The Cluster manages job lifecycle: launching, monitoring, and terminating
    jobs. Jobs are independent processes - there is no distributed task
    execution through the Cluster interface.

    Implementations: FrayCluster, LocalCluster
    """

    def launch(self, request: JobRequest) -> JobName:
        """Launch a job on the cluster.

        The job runs as an independent process with its own lifecycle.
        Use the Resolver to discover actors within jobs.
        """
        ...

    def monitor(self, job_id: JobName) -> JobInfo:
        """Stream logs from a running job, blocking until completion."""
        ...

    def poll(self, job_id: JobName) -> JobInfo:
        """Get current status of a job without blocking."""
        ...

    def terminate(self, job_id: JobName) -> None:
        """Terminate a running job.

        Also terminates any child jobs spawned by this job.
        """
        ...

    def list_jobs(self) -> list[JobInfo]:
        """List all jobs managed by this cluster."""
        ...

    def wait(
        self,
        job_ids: JobName | Sequence[JobName],
        raise_on_failure: bool = False,
    ) -> JobInfo | list[JobInfo]:
        """Block until job(s) complete.

        Args:
            job_ids: Single job ID or sequence of job IDs
            raise_on_failure: If True, raise exception if any job fails
        """
        ...

    def create_reservation(self, config: ReservationConfig) -> ReservationId:
        """Create a resource reservation for upcoming jobs."""
        ...

    def release_reservation(self, reservation_id: ReservationId) -> None:
        """Release a reservation, freeing its resources."""
        ...

    def get_reservation(self, reservation_id: ReservationId) -> ReservationInfo:
        """Get current status of a reservation."""
        ...

    def launch_group(
        self,
        requests: Sequence[JobRequest],
        group: JobGroupConfig,
        reservation_id: ReservationId | None = None,
    ) -> list[JobName]:
        """Launch a group of co-located jobs atomically.

        All jobs in the group are scheduled together with locality
        guarantees. If any job cannot be scheduled, none are started.
        """
        ...

    @property
    def namespace(self) -> Namespace:
        """The namespace for this cluster connection."""
        ...


def current_cluster() -> Cluster:
    """Get the current cluster from environment.

    Reads FRAY_CLUSTER_SPEC environment variable:
    - "local" or unset: LocalCluster (in-process execution)
    - "fray://host:port": FrayCluster connecting to controller

    Jobs inherit namespace from FRAY_NAMESPACE environment variable.
    """
    ...
```

### Metadata Service (Internal)

The Metadata service is internal to the Cluster. It maps actor names to
endpoints (job ID + address + port). The Cluster owns it and automatically:

* Cleans up entries when jobs terminate
* Maintains list of all actors registered under each name
* Supports multiple actors per name for load balancing

Clients do not interact with Metadata directly - they use a Resolver.

```python
@dataclass
class ActorEndpoint:
    """Endpoint information for a registered actor."""
    actor_id: ActorId
    name: str
    job_id: JobName
    namespace: Namespace
    metadata: dict[str, str] = field(default_factory=dict)


class Metadata(Protocol):
    """Internal actor registration service owned by the Cluster."""

    def register(
        self,
        name: str,
        job_id: JobName,
        metadata: dict[str, str] | None = None,
    ) -> ActorId:
        """Register an actor endpoint.

        Multiple actors can register under the same name. All registrations
        are tracked and returned by lookup_all().
        """
        ...

    def unregister(self, actor_id: ActorId) -> None:
        """Remove an actor registration."""
        ...

    def lookup(self, name: str) -> ActorEndpoint | None:
        """Look up one actor by name (returns None if not found)."""
        ...

    def lookup_all(self, name: str) -> list[ActorEndpoint]:
        """Look up all actors registered under a name."""
        ...

    def list_actors(self, prefix: str = "") -> list[ActorEndpoint]:
        """List all registered actors, optionally filtered by name prefix."""
        ...
```

### Resolver and ActorPool

A Resolver provides actor discovery and returns ActorPool instances for
managing calls to one or more actors.

```python
T = TypeVar("T")


class ActorPool(Generic[T]):
    """Pool of actors registered under a common name.

    Provides load-balanced calls, broadcasting, and pool state queries.
    All calls go through the resolver for automatic failure handling.
    """

    @property
    def size(self) -> int:
        """Current number of actors in the pool."""
        ...

    @property
    def endpoints(self) -> list[ActorEndpoint]:
        """Current list of actor endpoints (snapshot)."""
        ...

    def wait_for_size(
        self,
        min_size: int,
        timeout: float = 60.0,
    ) -> None:
        """Block until pool has at least min_size actors.

        Useful during startup when waiting for workers to register.

        Raises:
            TimeoutError: If timeout expires before min_size reached
        """
        ...

    def call(self) -> T:
        """Get a handle for single-actor calls (round-robin).

        Returns a proxy that routes method calls to one actor in the pool,
        cycling through actors on successive calls.

        Example:
            pool = resolver.lookup("inference")
            result = pool.call().predict(x)  # Routes to one actor
        """
        ...

    def broadcast(self) -> "BroadcastHandle[T]":
        """Get a handle for broadcasting to all actors.

        Returns a proxy that calls all actors in parallel and collects
        results. Failed calls return exceptions in the results list.

        Example:
            pool = resolver.lookup("workers")
            futures = pool.broadcast().shutdown()
            results = [f.result() for f in futures]  # May contain exceptions
        """
        ...


class BroadcastHandle(Generic[T]):
    """Handle for broadcasting method calls to all actors in a pool."""

    def __getattr__(self, method_name: str) -> Callable[..., list["ActorFuture"]]:
        """Broadcast a method call to all actors.

        Returns a list of futures, one per actor. Each future resolves to
        the result or exception from that actor's call.
        """
        ...


class Resolver(Protocol):
    """Actor discovery and connection management.

    Resolvers map actor names to ActorPools and manage reconnection on failure.
    """

    def lookup(self, name: str) -> ActorPool:
        """Look up actors by name and return a pool.

        Always returns a pool, even if empty. Use pool.wait_for_size()
        to block until actors are available.
        """
        ...


class ClusterResolver(Resolver):
    """Resolver backed by a Cluster's Metadata service.

    This is the standard resolver for production use. It queries the
    Cluster's metadata service and handles reconnection when actors
    restart.
    """

    def __init__(self, cluster: Cluster):
        ...


class FixedResolver(Resolver):
    """Resolver with fixed actor addresses.

    Useful for testing or when connecting to known endpoints.
    """

    def __init__(self, addresses: dict[str, str | list[str]]):
        """Create resolver with fixed addresses.

        Args:
            addresses: Mapping of actor names to addresses.
                       Values can be a single address or list of addresses.

        Example:
            resolver = FixedResolver({
                "inference": "localhost:8080",
                "workers": ["localhost:8081", "localhost:8082"],
            })
        """
        ...
```

### Actor System

The Actor system provides RPC-based communication between jobs:

- **ActorServer**: Hosts actor instances, registers with the cluster's Metadata service
- **ActorContext**: Passed to actor methods, enables actors to call other actors
- **ActorFuture**: Represents an in-flight async call

**Scope notes**: Streaming responses are not supported. Cancellation is not
supported. RPC tracing is not included in the initial implementation.

```python
@dataclass
class ActorContext:
    """Context passed to actor methods as first argument.

    Enables actors to call other actors and access cluster services.
    """
    cluster: Cluster
    resolver: Resolver
    job_id: JobName
    namespace: Namespace

    @classmethod
    def from_environment(cls) -> "ActorContext":
        """Create context from FRAY_* environment variables."""
        ...


class ActorFuture(Protocol[T]):
    """Future representing an in-flight actor method call."""

    def result(self, timeout: float | None = None) -> T:
        """Block until result is available.

        Raises the remote exception if the call failed.
        """
        ...

    def done(self) -> bool:
        """Check if the call has completed."""
        ...

    def exception(self) -> BaseException | None:
        """Get the exception if the call failed, None if succeeded or pending."""
        ...


class ActorServer:
    """Server for hosting actors and handling RPC calls.

    Each job should run at most one ActorServer since it binds to a port.
    The server reads FRAY_JOB_ID from environment to associate registrations
    with the current job.

    Usage:
        cluster = current_cluster()
        server = ActorServer(cluster)
        server.register("my_actor", MyActor(config))
        server.serve()  # Blocks, serving requests
    """

    def __init__(
        self,
        cluster: Cluster,
        host: str = "0.0.0.0",
        port: int = 0,  # 0 = auto-assign
    ):
        """Initialize actor server.

        Args:
            cluster: Cluster for actor registration
            host: Host address to bind
            port: Port to bind (0 for auto-assignment)
        """
        ...

    @property
    def address(self) -> str:
        """The server's bound address (host:port)."""
        ...

    def register(
        self,
        name: str,
        actor: Any,
        metadata: dict[str, str] | None = None,
    ) -> ActorId:
        """Register an actor instance with the server.

        The actor is registered with the cluster's Metadata service.
        Multiple actors (across multiple jobs) can register under the
        same name for load balancing.

        Actor methods should accept ActorContext as their first argument:

            class MyActor:
                def process(self, ctx: ActorContext, data: dict) -> dict:
                    # ctx.resolver allows calling other actors
                    other = ctx.resolver.lookup("other_actor")
                    return other.call().transform(data)

        Args:
            name: Name for lookup (scoped to namespace)
            actor: Actor instance (any object with callable methods)
            metadata: Optional key-value metadata for discovery

        Returns:
            Unique actor ID
        """
        ...

    def unregister(self, name: str) -> None:
        """Unregister an actor."""
        ...

    def serve(self) -> None:
        """Start serving requests (blocks indefinitely)."""
        ...

    def serve_background(self) -> None:
        """Start serving in a background thread."""
        ...

    def shutdown(self, grace_period: float = 5.0) -> None:
        """Gracefully shutdown the server."""
        ...
```

### WorkerPool

WorkerPool provides Ray-like task dispatch for stateless workloads. It manages
a pool of worker jobs that can execute arbitrary callables.

```python
class WorkerPool:
    """Pool of stateless workers for task dispatch.

    Creates worker jobs that can execute arbitrary callables. Workers are
    stateless - if a worker fails, tasks can be retried on any other worker.

    Usage:
        pool = WorkerPool(
            cluster=current_cluster(),
            num_workers=10,
            resources=ResourceConfig.with_cpu(cpu=2, memory="4GB"),
        )

        # Submit tasks
        futures = [pool.submit(process_shard, shard) for shard in shards]

        # Wait for results
        results = [f.result() for f in futures]

        pool.shutdown()
    """

    def __init__(
        self,
        cluster: Cluster,
        num_workers: int,
        resources: ResourceConfig,
        environment: EnvironmentConfig | None = None,
        name_prefix: str = "worker",
    ):
        """Create a worker pool.

        Args:
            cluster: Cluster for launching worker jobs
            num_workers: Number of worker jobs to launch
            resources: Resource requirements per worker
            environment: Optional environment config for workers
            name_prefix: Prefix for worker job names
        """
        ...

    @property
    def size(self) -> int:
        """Number of workers currently available."""
        ...

    def wait_for_workers(
        self,
        min_workers: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Wait for workers to become available.

        Args:
            min_workers: Minimum workers required (default: all workers)
            timeout: Maximum time to wait
        """
        ...

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> ActorFuture[T]:
        """Submit a task for execution.

        The callable and arguments must be picklable. Tasks are distributed
        round-robin across available workers.

        Args:
            fn: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future that resolves to the function's return value
        """
        ...

    def map(
        self,
        fn: Callable[[Any], T],
        items: Sequence[Any],
    ) -> list[ActorFuture[T]]:
        """Map a function over items in parallel.

        Args:
            fn: Function to apply to each item
            items: Items to process

        Returns:
            List of futures, one per item
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        ...
```

### Local Development

For local development, all components have in-process implementations that
preserve the same interfaces. Jobs run as threads, actors are called directly
(but with serialization), and the code paths are identical to production.

```python
class LocalCluster(Cluster):
    """Local cluster for development and testing.

    Runs jobs as threads in the current process. Includes an embedded
    Metadata service. Child jobs are automatically terminated when
    parent jobs exit.
    """

    def __init__(self):
        ...


class LocalActorServer(ActorServer):
    """In-process actor server for local development.

    Actors are called directly (no network) but arguments and return
    values are still serialized/deserialized to catch serialization bugs.
    """
    ...


# Example: local development workflow
cluster = LocalCluster()
resolver = ClusterResolver(cluster)

# Server side (same code as production)
server = ActorServer(cluster)
server.register("my_actor", MyActor())
server.serve_background()

# Client side (same code as production)
pool = resolver.lookup("my_actor")
result = pool.call().process(data)
```

### Zephyr Integration

Zephyr uses WorkerPool internally for distributed execution. The FrayBackend
launches worker jobs and dispatches shard processing tasks to them.

```python
class FrayBackend(Backend):
    """Zephyr backend using Fray for distributed execution.

    Launches a WorkerPool and dispatches shard processing tasks.
    Workers are stateless - all coordination happens through object storage.
    """

    def __init__(
        self,
        cluster: Cluster | None = None,  # None = current_cluster()
        max_parallelism: int = 100,
        memory_per_worker: str = "2GB",
    ):
        ...

    def execute(self, dataset: Dataset) -> list[Any]:
        """Execute a dataset pipeline on Fray workers.

        1. Plans the dataset into shards
        2. Launches a WorkerPool with max_parallelism workers
        3. Submits each shard for processing
        4. Collects and returns results
        """
        ...
```

## Example Usage



### 1. Training (Simple Job)

```python
def train_main():
    # ... training logic ...
    pass

# Launch a single job
cluster = current_cluster()
job_id = cluster.launch(
    JobRequest(
        name="llama-training",
        resources=ResourceConfig.with_tpu("v5litepod-16"),
        entrypoint=Entrypoint.from_callable(train_main),
        environment=EnvironmentConfig.create(extras=["tpu"])
    )
)

# Wait for completion
cluster.wait(job_id, raise_on_failure=True)
```

### 2. Inference (Client/Server)

```python
# Server: Host an actor
def server_main():
    server = ActorServer(current_cluster())
    server.register("inference", InferenceModel()) # Registers with cluster metadata
    server.serve()

# Client: Dispatch requests
def run_client():
    resolver = ClusterResolver(current_cluster())
    pool = resolver.lookup("inference") # Discovers all actors named "inference"

    # Load balance requests across the pool
    results = pool.call().predict(batch_of_prompts)
```

### 3. Zephyr (Task Dispatch)

```python
# Create a worker pool for map/reduce style tasks
pool = WorkerPool(current_cluster(), num_workers=100, resources=ResourceConfig.with_cpu())

# Dispatch tasks - pool handles distribution
futures = [pool.submit(process_shard, shard) for shard in shards]
results = [f.result() for f in futures]
pool.shutdown()
```

### 4. RL (Coordinated Workers)

```python
# Coordinator: Hosts shared state actors
def coordinator_main():
    server = ActorServer(current_cluster())
    server.register("curriculum", CurriculumActor())
    server.register("weights", WeightStore())
    server.serve()

# Worker: Connects to shared actors
def worker_main():
    resolver = ClusterResolver(current_cluster())
    curriculum = resolver.lookup("curriculum")
    weights = resolver.lookup("weights")

    while True:
        # Fetch latest state and data
        w = weights.call().get_weights()
        task = curriculum.call().get_next_task()

        # ... perform rollout ...
        curriculum.call().report_result(result)

# Launch as a co-located group to ensure low latency
cluster.launch_group(
    [
        JobRequest(name="coordinator", entrypoint=Entrypoint.from_callable(coordinator_main), ...),
        JobRequest(name="worker-1", entrypoint=Entrypoint.from_callable(worker_main), ...),
        # ...
    ],
    group=JobGroupConfig(same_region=True)
)
```

### 5. Multi-Slice (Elastic Coordination)

```python
class BarrierActor:
    def wait_for_barrier(self, ctx: ActorContext, slice_id, step):
        # ... verify all slices have reached step ...
        pass

def slice_worker(slice_id):
    coordinator = ClusterResolver(current_cluster()).lookup("coordinator")
    while True:
        # Sync then train
        coordinator.call().wait_for_barrier(slice_id, step)
        train_step()

# Elastic Scheduling
reservation = cluster.create_reservation(
    ReservationConfig(min_resources=..., max_resources=...)
)

# Launch coordinator and initial slices...
# Cluster auto-scales within reservation as resources become available
```

### 6. Local Development

Code behaves identically locally. `current_cluster()` returns `LocalCluster` when no cluster address is configured.

```python
# No code changes needed.
cluster = current_cluster() # -> LocalCluster
server = ActorServer(cluster) # -> Spawns threads
```
