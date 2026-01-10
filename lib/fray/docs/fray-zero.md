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

## Implementation Plan

We propose we first roll out the actor system and integrate it into Zephyr
underneath our current Ray cluster. This will give us some more experience with
the RPC infrastructure and minimize user disruption.

We'd then move on to building out the cluster management system, gradually
adding features and complexity:

* Single region, static cluster, single VMs
* Single region, static cluster, TPU slices
* Single region, auto-scaling cluster, single VMs
* Single region, auto-scaling cluster, TPU slices
* Multi-region

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

### Core Types

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar
from enum import StrEnum


# Type aliases for clarity
JobId = str
ActorId = str
ReservationId = str
Namespace = str


class JobStatus(StrEnum):
    """Status of a job in the cluster."""
    PENDING = "pending"      # Waiting for resources
    RUNNING = "running"      # Currently executing
    SUCCEEDED = "succeeded"  # Completed successfully
    FAILED = "failed"        # Terminated with error
    STOPPED = "stopped"      # Manually terminated


@dataclass
class JobInfo:
    """Information about a job's current state."""
    job_id: JobId
    name: str
    status: JobStatus
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
    jobs: list[JobId]
    expires_at: float
```

### Cluster Interface

The Cluster is responsible solely for job lifecycle management. It does not
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

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job on the cluster.

        The job runs as an independent process with its own lifecycle.
        Use the Resolver to discover actors within jobs.
        """
        ...

    def monitor(self, job_id: JobId) -> JobInfo:
        """Stream logs from a running job, blocking until completion."""
        ...

    def poll(self, job_id: JobId) -> JobInfo:
        """Get current status of a job without blocking."""
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job.

        Also terminates any child jobs spawned by this job.
        """
        ...

    def list_jobs(self) -> list[JobInfo]:
        """List all jobs managed by this cluster."""
        ...

    def wait(
        self,
        job_ids: JobId | Sequence[JobId],
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
    ) -> list[JobId]:
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
    address: str  # host:port for RPC connection
    job_id: JobId
    namespace: Namespace
    metadata: dict[str, str] = field(default_factory=dict)


class Metadata(Protocol):
    """Internal actor registration service owned by the Cluster."""

    def register(
        self,
        name: str,
        address: str,
        job_id: JobId,
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
    job_id: JobId
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

This section provides complete code examples for each use case outlined in the
Requirements section.

### 1. Training

Training is the simplest case - a single job running on a TPU slice.

```python
# experiments/training_example.py
from fray.cluster import (
    current_cluster,
    Entrypoint,
    EnvironmentConfig,
    JobRequest,
    ResourceConfig,
)


def train_model():
    """Training entrypoint - runs on TPU pod."""
    import jax
    from levanter.main.train_lm import main as train_lm

    print(f"Running on {jax.device_count()} devices")
    train_lm()


def launch_training(
    name: str,
    tpu_type: str = "v5litepod-16",
    num_slices: int = 1,
):
    """Launch a training job on the cluster."""
    cluster = current_cluster()

    job_id = cluster.launch(
        JobRequest(
            name=f"train-{name}",
            resources=ResourceConfig.with_tpu(tpu_type, slice_count=num_slices),
            entrypoint=Entrypoint.from_callable(train_model),
            environment=EnvironmentConfig.create(
                extras=["tpu"],
                env_vars={
                    "WANDB_PROJECT": "marin",
                    "WANDB_RUN_NAME": name,
                },
            ),
        )
    )

    result = cluster.wait(job_id, raise_on_failure=True)
    print(f"Training completed with status: {result.status}")
    return result


if __name__ == "__main__":
    launch_training("llama-8b-baseline", tpu_type="v5litepod-64")
```

### 2. Data Processing (Zephyr)

Data processing uses Zephyr pipelines executed on Fray workers.

```python
# marin/transform/example_pipeline.py
from dataclasses import dataclass
from zephyr import Backend, Dataset


@dataclass
class TransformConfig:
    input_pattern: str
    output_path: str
    max_parallelism: int = 100
    memory_per_worker: str = "4GB"


def transform_document(doc: dict) -> dict:
    """Transform a single document."""
    return {
        "id": doc["id"],
        "text": doc["text"].lower(),
        "token_count": len(doc["text"].split()),
    }


def run_transform(config: TransformConfig) -> list[str]:
    """Run a data transformation pipeline."""
    pipeline = (
        Dataset.from_files(config.input_pattern)
        .load_jsonl()
        .map(transform_document)
        .filter(lambda doc: doc["token_count"] > 10)
        .write_jsonl(f"{config.output_path}/output-{{shard:05d}}.jsonl.gz")
    )

    # Backend auto-selects FrayBackend on cluster, ThreadpoolBackend locally
    output_files = Backend.execute(
        pipeline,
        max_parallelism=config.max_parallelism,
        memory=config.memory_per_worker,
    )

    return output_files
```

### 3. Inference/Evaluation

Inference requires launching inference servers and dispatching work to them.

```python
# marin/inference/inference_pool.py
from dataclasses import dataclass
from fray.cluster import (
    current_cluster,
    ClusterResolver,
    Entrypoint,
    EnvironmentConfig,
    JobGroupConfig,
    JobRequest,
    ResourceConfig,
)
from fray.rpc import ActorContext, ActorServer


@dataclass
class InferenceConfig:
    model_path: str
    tpu_type: str = "v5litepod-4"
    num_servers: int = 4
    batch_size: int = 32


class InferenceActor:
    """Actor that hosts an inference model."""

    def __init__(self, model_path: str):
        import jax
        from levanter.inference import load_model

        self.model = load_model(model_path)
        self.device_count = jax.device_count()

    def predict(self, ctx: ActorContext, prompts: list[str]) -> list[str]:
        return self.model.generate(prompts)


def inference_server_main(config: InferenceConfig, server_id: int):
    """Entrypoint for inference server jobs."""
    cluster = current_cluster()
    server = ActorServer(cluster)

    actor = InferenceActor(config.model_path)
    server.register("inference_pool", actor)
    server.serve()


class InferencePool:
    """Client for dispatching work to inference servers."""

    def __init__(self, cluster, job_ids: list, config: InferenceConfig):
        self.cluster = cluster
        self.resolver = ClusterResolver(cluster)
        self.job_ids = job_ids
        self.config = config

    def predict(self, prompts: list[str]) -> list[str]:
        """Run inference on prompts, load-balancing across servers."""
        pool = self.resolver.lookup("inference_pool")
        results = []

        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            results.extend(pool.call().predict(batch))

        return results

    def shutdown(self):
        for job_id in self.job_ids:
            self.cluster.terminate(job_id)


def launch_inference_pool(config: InferenceConfig) -> InferencePool:
    """Launch a pool of inference servers."""
    cluster = current_cluster()
    resolver = ClusterResolver(cluster)

    requests = [
        JobRequest(
            name=f"inference-{i}",
            resources=ResourceConfig.with_tpu(config.tpu_type),
            entrypoint=Entrypoint.from_callable(
                inference_server_main, args=(config, i)
            ),
            environment=EnvironmentConfig.create(extras=["tpu"]),
        )
        for i in range(config.num_servers)
    ]

    job_ids = cluster.launch_group(
        requests,
        group=JobGroupConfig(name="inference-pool", same_region=True),
    )

    # Wait for all servers to register
    pool = resolver.lookup("inference_pool")
    pool.wait_for_size(config.num_servers)

    return InferencePool(cluster, job_ids, config)


if __name__ == "__main__":
    config = InferenceConfig(
        model_path="gs://bucket/checkpoints/llama-8b",
        num_servers=4,
    )

    pool = launch_inference_pool(config)

    prompts = ["Hello, world!", "What is the capital of France?"]
    results = pool.predict(prompts)
    print(results)

    pool.shutdown()
```

### 4. Reinforcement Learning

RL requires coordinating training workers, rollout workers, and shared actors.

```python
# marin/rl/fray_rl_job.py
from dataclasses import dataclass
import os

from fray.cluster import (
    current_cluster,
    ClusterResolver,
    Entrypoint,
    EnvironmentConfig,
    JobGroupConfig,
    JobRequest,
    ResourceConfig,
)
from fray.rpc import ActorContext, ActorServer


@dataclass
class RLConfig:
    model_path: str
    train_tpu_type: str = "v5litepod-16"
    inference_tpu_type: str = "v5litepod-4"
    num_rollout_workers: int = 4
    checkpoint_path: str = "gs://bucket/checkpoints"


# === Shared Actors ===

class CurriculumActor:
    """Manages training curriculum - shared across all workers."""

    def __init__(self, lessons: dict):
        self.lessons = lessons
        self.step = 0

    def get_lesson(self, ctx: ActorContext) -> dict:
        lesson_name = list(self.lessons.keys())[self.step % len(self.lessons)]
        return self.lessons[lesson_name]

    def report_result(self, ctx: ActorContext, lesson_name: str, reward: float):
        self.step += 1


class WeightCoordinator:
    """Coordinates weight sync between train and rollout workers."""

    def __init__(self):
        self.current_step = 0
        self.checkpoint_path: str | None = None

    def publish_checkpoint(self, ctx: ActorContext, step: int, path: str):
        self.current_step = step
        self.checkpoint_path = path

    def get_latest_checkpoint(self, ctx: ActorContext) -> tuple[int, str | None]:
        return self.current_step, self.checkpoint_path


# === Worker Entrypoints ===

def coordinator_main(config: RLConfig):
    """Coordinator job hosting shared actors."""
    cluster = current_cluster()
    server = ActorServer(cluster)

    server.register("curriculum", CurriculumActor({"math": {}, "code": {}}))
    server.register("weight_coordinator", WeightCoordinator())
    server.serve()


def train_worker_main(config: RLConfig):
    """Training worker - connects to shared actors."""
    cluster = current_cluster()
    resolver = ClusterResolver(cluster)

    curriculum_pool = resolver.lookup("curriculum")
    curriculum_pool.wait_for_size(1)

    weight_pool = resolver.lookup("weight_coordinator")
    weight_pool.wait_for_size(1)

    from marin.rl.train_worker import TrainWorker
    worker = TrainWorker(config)

    for step in worker.train_loop():
        if step % 100 == 0:
            ckpt_path = f"{config.checkpoint_path}/step-{step}"
            worker.save_checkpoint(ckpt_path)
            weight_pool.call().publish_checkpoint(step, ckpt_path)


def rollout_worker_main(config: RLConfig, worker_id: int):
    """Rollout worker - generates training data."""
    cluster = current_cluster()
    resolver = ClusterResolver(cluster)

    curriculum_pool = resolver.lookup("curriculum")
    curriculum_pool.wait_for_size(1)

    weight_pool = resolver.lookup("weight_coordinator")
    weight_pool.wait_for_size(1)

    from marin.rl.rollout_worker import RolloutWorker
    worker = RolloutWorker(config, worker_id)

    while True:
        step, ckpt = weight_pool.call().get_latest_checkpoint()
        if ckpt:
            worker.load_checkpoint(ckpt)

        lesson = curriculum_pool.call().get_lesson()
        rollouts, reward = worker.generate_rollout(lesson)
        curriculum_pool.call().report_result(lesson["name"], reward)
        worker.write_rollouts(rollouts)


# === Job Orchestration ===

class RLJob:
    """Orchestrates an RL training job."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.cluster = current_cluster()
        self.resolver = ClusterResolver(self.cluster)
        self.run_id = f"rl-{os.urandom(4).hex()}"

    def run(self) -> dict:
        group = JobGroupConfig(name=f"rl-{self.run_id}", same_region=True)

        requests = [
            # Coordinator
            JobRequest(
                name=f"rl-coordinator-{self.run_id}",
                resources=ResourceConfig.with_cpu(cpu=4, memory="8GB"),
                entrypoint=Entrypoint.from_callable(
                    coordinator_main, args=(self.config,)
                ),
            ),
            # Training worker
            JobRequest(
                name=f"rl-train-{self.run_id}",
                resources=ResourceConfig.with_tpu(self.config.train_tpu_type),
                entrypoint=Entrypoint.from_callable(
                    train_worker_main, args=(self.config,)
                ),
                environment=EnvironmentConfig.create(extras=["tpu"]),
            ),
        ]

        # Rollout workers
        for i in range(self.config.num_rollout_workers):
            requests.append(
                JobRequest(
                    name=f"rl-rollout-{self.run_id}-{i}",
                    resources=ResourceConfig.with_tpu(self.config.inference_tpu_type),
                    entrypoint=Entrypoint.from_callable(
                        rollout_worker_main, args=(self.config, i)
                    ),
                    environment=EnvironmentConfig.create(extras=["tpu"]),
                )
            )

        self.job_ids = self.cluster.launch_group(requests, group=group)

        # Wait for coordinator actors
        self.resolver.lookup("curriculum").wait_for_size(1)
        self.resolver.lookup("weight_coordinator").wait_for_size(1)

        # Monitor training worker (job_ids[1] is the train worker)
        return self.cluster.monitor(self.job_ids[1])

    def shutdown(self):
        for job_id in self.job_ids:
            self.cluster.terminate(job_id)
```

### 5. Multi-Slice Training (Flex)

Multi-slice training coordinates multiple TPU slices that can dynamically
join and leave.

```python
# marin/training/multislice_training.py
from dataclasses import dataclass

from fray.cluster import (
    current_cluster,
    ClusterResolver,
    Entrypoint,
    EnvironmentConfig,
    JobId,
    JobRequest,
    JobStatus,
    ReservationConfig,
    ResourceConfig,
)
from fray.rpc import ActorContext, ActorServer


@dataclass
class MultiSliceConfig:
    model_path: str
    tpu_type: str = "v5litepod-16"
    min_slices: int = 2
    max_slices: int = 8
    checkpoint_path: str = "gs://bucket/checkpoints"


class SliceCoordinator:
    """Coordinates slice membership and synchronization barriers."""

    def __init__(self, min_slices: int):
        self.min_slices = min_slices
        self.slices: dict[str, str] = {}  # slice_id -> address
        self.current_step = 0
        self.barrier_count = 0

    def register_slice(
        self, ctx: ActorContext, slice_id: str, address: str
    ) -> dict:
        self.slices[slice_id] = address
        return {"step": self.current_step, "peers": list(self.slices.keys())}

    def unregister_slice(self, ctx: ActorContext, slice_id: str):
        self.slices.pop(slice_id, None)

    def heartbeat(self, ctx: ActorContext, slice_id: str) -> dict:
        return {"step": self.current_step, "slice_count": len(self.slices)}

    def barrier_enter(
        self, ctx: ActorContext, slice_id: str, step: int
    ) -> bool:
        if step != self.current_step:
            return False
        self.barrier_count += 1
        return self.barrier_count >= len(self.slices)

    def barrier_exit(self, ctx: ActorContext, slice_id: str, step: int):
        if self.barrier_count >= len(self.slices):
            self.current_step += 1
            self.barrier_count = 0


def coordinator_main(config: MultiSliceConfig):
    """Coordinator hosting the SliceCoordinator actor."""
    cluster = current_cluster()
    server = ActorServer(cluster)

    server.register("slice_coordinator", SliceCoordinator(config.min_slices))
    server.serve()


def slice_worker_main(config: MultiSliceConfig, slice_id: str):
    """Training worker for a single slice."""
    cluster = current_cluster()
    resolver = ClusterResolver(cluster)

    coord_pool = resolver.lookup("slice_coordinator")
    coord_pool.wait_for_size(1)
    coordinator = coord_pool.call()

    import jax
    address = f"{jax.process_index()}@{jax.local_devices()[0]}"
    coordinator.register_slice(slice_id, address)

    from marin.training.trainer import Trainer
    trainer = Trainer(config.model_path)

    try:
        while True:
            state = coordinator.heartbeat(slice_id)
            if state["slice_count"] < config.min_slices:
                continue  # Wait for more slices

            step = state["step"]
            coordinator.barrier_enter(slice_id, step)
            trainer.train_step()
            coordinator.barrier_exit(slice_id, step)

            if step % 100 == 0:
                trainer.save_checkpoint(f"{config.checkpoint_path}/step-{step}")
    finally:
        coordinator.unregister_slice(slice_id)


class MultiSliceJob:
    """Manages elastic multi-slice training."""

    def __init__(self, config: MultiSliceConfig):
        self.config = config
        self.cluster = current_cluster()
        self.resolver = ClusterResolver(self.cluster)
        self.slice_jobs: dict[str, JobId] = {}

    def run(self):
        # Create elastic reservation
        reservation = self.cluster.create_reservation(
            ReservationConfig(
                min_resources=ResourceConfig.with_tpu(
                    self.config.tpu_type, slice_count=self.config.min_slices
                ),
                max_resources=ResourceConfig.with_tpu(
                    self.config.tpu_type, slice_count=self.config.max_slices
                ),
                preemptible=True,
            )
        )

        try:
            # Launch coordinator
            self.cluster.launch(
                JobRequest(
                    name="multislice-coordinator",
                    resources=ResourceConfig.with_cpu(cpu=4, memory="8GB"),
                    entrypoint=Entrypoint.from_callable(
                        coordinator_main, args=(self.config,)
                    ),
                )
            )

            coord_pool = self.resolver.lookup("slice_coordinator")
            coord_pool.wait_for_size(1)

            # Launch initial slices
            for i in range(self.config.min_slices):
                self._launch_slice(i)

            # Monitor and scale
            while True:
                self._check_health_and_scale(reservation)
        finally:
            self.cluster.release_reservation(reservation)

    def _launch_slice(self, idx: int):
        job_id = self.cluster.launch(
            JobRequest(
                name=f"multislice-worker-{idx}",
                resources=ResourceConfig.with_tpu(self.config.tpu_type),
                entrypoint=Entrypoint.from_callable(
                    slice_worker_main, args=(self.config, f"slice-{idx}")
                ),
                environment=EnvironmentConfig.create(extras=["tpu"]),
            )
        )
        self.slice_jobs[f"slice-{idx}"] = job_id

    def _check_health_and_scale(self, reservation):
        # Restart failed slices
        for slice_id, job_id in list(self.slice_jobs.items()):
            status = self.cluster.poll(job_id).status
            if status in (JobStatus.FAILED, JobStatus.STOPPED):
                self._launch_slice(int(slice_id.split("-")[1]))

        # Scale based on reservation
        reservation_info = self.cluster.get_reservation(reservation)
        available = reservation_info.allocated_resources.replicas
        current = len(self.slice_jobs)
        if available > current < self.config.max_slices:
            for i in range(current, min(available, self.config.max_slices)):
                self._launch_slice(i)


if __name__ == "__main__":
    config = MultiSliceConfig(
        model_path="gs://bucket/models/llama-70b",
        tpu_type="v5litepod-64",
        min_slices=2,
        max_slices=8,
    )

    job = MultiSliceJob(config)
    job.run()
```

### 6. Local Development

All examples work locally with the same code - `current_cluster()` returns
a LocalCluster when `FRAY_CLUSTER_SPEC` is unset.

```python
# tests/test_local_development.py
"""Local development uses identical patterns to production."""

from fray.cluster import LocalCluster, ClusterResolver
from fray.rpc import ActorServer, ActorContext


def test_local_inference():
    """Test inference locally - same code pattern as production."""
    cluster = LocalCluster()
    resolver = ClusterResolver(cluster)

    class MockInferenceActor:
        def predict(self, ctx: ActorContext, prompts: list[str]) -> list[str]:
            return [f"Response to: {p}" for p in prompts]

    server = ActorServer(cluster)
    server.register("inference", MockInferenceActor())
    server.serve_background()

    pool = resolver.lookup("inference")
    pool.wait_for_size(1)
    result = pool.call().predict(["Hello", "World"])

    assert result == ["Response to: Hello", "Response to: World"]
    server.shutdown()


def test_worker_pool():
    """Test WorkerPool locally."""
    from fray.pool import WorkerPool

    cluster = LocalCluster()

    def process_item(x: int) -> int:
        return x * 2

    pool = WorkerPool(
        cluster=cluster,
        num_workers=4,
        resources=ResourceConfig.with_cpu(),
    )
    pool.wait_for_workers()

    futures = pool.map(process_item, [1, 2, 3, 4, 5])
    results = [f.result() for f in futures]

    assert results == [2, 4, 6, 8, 10]
    pool.shutdown()


def test_broadcast():
    """Test broadcast to multiple actors."""
    cluster = LocalCluster()
    resolver = ClusterResolver(cluster)

    class CounterActor:
        def __init__(self, actor_id: int):
            self.actor_id = actor_id
            self.count = 0

        def increment(self, ctx: ActorContext) -> int:
            self.count += 1
            return self.actor_id

    # Register multiple actors under same name
    server = ActorServer(cluster)
    server.register("counters", CounterActor(1))
    server.serve_background()

    server2 = ActorServer(cluster)
    server2.register("counters", CounterActor(2))
    server2.serve_background()

    pool = resolver.lookup("counters")
    pool.wait_for_size(2)

    # Broadcast returns futures for all actors
    futures = pool.broadcast().increment()
    results = sorted([f.result() for f in futures])

    assert results == [1, 2]

    server.shutdown()
    server2.shutdown()
```
