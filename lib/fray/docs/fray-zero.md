# Fray-Fluster Zero Design

As discussed in
[https://docs.google.com/presentation/d/1qgPGK7zYmiOSOn70W-rPIrcF7vuuHgg68Qr3e5UIuxw/edit](Fray
Presentation) and
[https://docs.google.com/document/d/1UteX9nD9obY5ypV2o8KbHF72xwX1JezseRE5uLFBulk/edit?tab=t.0](Fray/RPC
Design), we think it's a good time to "grug-ify" our RPC and clustering system
while moving off of the complex and fragile Ray.

## Original Design and Progress

Our original Ray challenges doc
[https://docs.google.com/document/d/1gtCz3aN2q72ZF-BNK_nKHCjuS9CT88QSH5vmkTktwWQ/edit?tab=t.0#heading=h.9k9q6db2omrh](Ray
Infrastructure Challenges) and migration plan
[https://docs.google.com/document/d/1r-YKwxMxD8dJPKFmQrdJdIsfVvrLnIPo8RnlANbwM5o/edit?tab=t.0](Ray
Migration) outlined a mostly "drop-in" approach to replacing Ray with Fray. We
would introduce new wrapper APIs which hid Ray's usage (fray.cluster and
fray.job), move most direct usage into a data library (zephyr), and reduce the
complexity of our dependencies to remove the need for complex venv creation
logic.

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

1. Training

Our simplest case, training jobs run on TPU slices, and communicate entirely via
JAX collectives after startup. Moreover training tasks run for a long time, so
startup overhead is not a problem.

2. Data Processing

Data processing wants to flexibly be able to use CPU resources. With our latest
Zephyr refactoring, there is minimal communication between tasks, as all work is
staged to object storage.

3. Inference/Evaluation

Inference is a combination of training and data processing. We want to start N
inference servers (potentially on slices), and then dispatch work to that pool
via one or more CPU jobs. As we will typically be inference-bound, it's likely a
single CPU task will be sufficient for our initial work.

4. RL

RL has 2-3 different jobs which use accelerators:

* Training worker
* Rollout workers (environments)
* Inference workers

Internally these jobs use JAX as expected, but they also need to communicate
metadata about the progress and new checkpoints via actors which are shared
across all processes (CurriculumActor and TrainingActor).

5. (Flex) Multi-slice Training

For flex-multi-slice, we have multiple training jobs, each running on a separate
TPU slice. Slices can come and go over time, but we'd like to take advantage of
slices when they are available. The simplest way to express this is to move our
workers and leader into actors, and dispatch work from the leader:

```python

# leader.py
def multislice_train():
  while True:
    slice_workers = build_multi_slice() # build a multi-slice cluster based on available slices
    slice_workers.train_step() # or train_n_steps, or whatever

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
 * _entrypoint_ (either a script or Python function)
 * _environment_ (a Docker reference + project directory), and a set of
 * _resources_ (a set of accelerators, memory, ports, etc)

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

The metadata service is internal to the cluster. Clients interact with it
through a **Resolver**.

### Resolver

A Resolver provides actor discovery and connection management. It maps actor
names to live endpoints and handles reconnection on failure.

```python
# Create resolver (explicitly, from cluster or environment)
resolver = ClusterResolver(cluster)  # uses cluster's metadata service
resolver = FixedResolver("localhost:8080")  # for testing, fixed target

# Look up actors
actor = resolver.lookup("inference_0")  # single actor
actors = resolver.lookup_all("inference_pool")  # multiple for load balancing

# Call methods
result = actor.predict(x)
```

Resolvers are independent from clusters - `ClusterResolver` is one
implementation backed by a cluster's metadata service, but other
implementations (e.g., `FixedResolver` for testing) can resolve to fixed
targets.

The resolver handles:

* **Worker resolution**: Maps actor names to current addresses
* **Fault tolerance**: Re-resolves addresses when workers fail and restart
* **Load balancing**: Distributes calls across multiple actors registered under the same name

### ActorServer

ActorServer lets users expose Python classes as RPC services without writing
protos. It takes the cluster for registration (uses the cluster's metadata
service) and handles serialization automatically.

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
server.register("inference_0", InferenceActor())
server.serve()  # blocks, handling requests
```

When a job hosting an ActorServer fails and restarts, the cluster automatically
updates the metadata mappings. Clients holding actor handles will transparently
reconnect to the new instance on their next call.

### Back to Ray

With a ClusterResolver backed by the metadata service, we can recover Ray-like
dynamic task dispatch. Start a pool of worker actors, dispatch tasks to
available workers, and rely on resolution to always connect to a valid worker.

```python
class WorkerPool:
    def __init__(self, cluster: Cluster, num_workers: int):
        self.cluster = cluster
        self.resolver = ClusterResolver(cluster)
        # Launch worker jobs that register as "worker_pool"
        for i in range(num_workers):
            cluster.launch(worker_job_request(i))

    def run(self, fn, *args, **kwargs):
        # lookup_all returns all workers, round-robin or pick idle
        workers = self.resolver.lookup_all("worker_pool")
        worker = pick_idle(workers)
        return worker.execute(fn, *args, **kwargs)
```

### Use Case Implementations

* **Training**: Single cluster job launch, no actors needed
* **Inference**: Launch N server jobs, each registers with same actor name, clients use `lookup_all()` for load balancing
* **Zephyr**: Launches worker jobs via cluster, coordinates via object storage (no actors)
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
#   curriculum.get_lesson()
```


## Concerns

### Local runs

What about running locally? Great we've got this whole cluster system, but does
this "scale down" to local runs? How do I bootup the RPC stuff on a local host?

My tentative proposal would be that we'd provide a stub local cluster which
simply runs everything in the users current process/venv.

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
architecture consists of three components:

1. **Cluster** - Job lifecycle management (launch, monitor, terminate). Owns a
   Metadata service for actor registration.
2. **Resolver** - Actor discovery and connection management. Maps actor names
   to endpoints and handles reconnection on failure. Independent from Cluster.
3. **ActorServer** - Hosts actor instances, handles RPC calls, registers with
   a Resolver.

There is no Ray-like `JobContext` with `run()`/`get()`/`wait()`. Jobs are
processes; actors are services within those jobs; resolvers provide the
discovery glue.

### Core Types

```python
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, TypeVar, NewType
from enum import StrEnum


JobId = NewType("JobId", str)
ActorId = NewType("ActorId", str)
ReservationId = NewType("ReservationId", str)
Namespace = NewType("Namespace", str)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
```

### Job Management Types

```python
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
        Use the Metadata service to discover actors within jobs.
        """
        ...

    def monitor(self, job_id: JobId) -> JobInfo:
        """Stream logs from a running job, blocking until completion."""
        ...

    def poll(self, job_id: JobId) -> JobInfo:
        """Get current status of a job without blocking."""
        ...

    def terminate(self, job_id: JobId) -> None:
        """Terminate a running job."""
        ...

    def list_jobs(self) -> list[JobInfo]:
        """List all jobs managed by this cluster."""
        ...

    def wait(
        self,
        job_id: JobId | Sequence[JobId],
        raise_on_failure: bool = False
    ) -> JobInfo | list[JobInfo]:
        """Block until job(s) complete."""
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

    @property
    def metadata(self) -> "Metadata":
        """The Metadata service for this cluster."""
        ...


def current_cluster() -> Cluster:
    """Get the current cluster from environment.

    Reads FRAY_CLUSTER_SPEC environment variable:
    - "local" or unset: LocalCluster
    - "fray://host:port": FrayCluster connecting to controller

    Jobs inherit namespace from FRAY_NAMESPACE environment variable.
    """
    ...
```

### Metadata Service (Internal)

The Metadata service is internal to the Cluster. It maps actor names to
endpoints (job ID + address + port). The Cluster owns it and automatically:

* Cleans up entries when jobs terminate
* Updates entries atomically when jobs restart after failure
* Garbage collects stale entries based on TTL

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
        """Register an actor endpoint (called by ActorServer)."""
        ...

    def unregister(self, actor_id: ActorId) -> None:
        """Remove an actor registration."""
        ...

    def lookup(self, name: str) -> ActorEndpoint:
        """Look up an actor by exact name."""
        ...

    def lookup_all(self, name: str) -> list[ActorEndpoint]:
        """Look up all actors registered under a name."""
        ...

    def list_actors(self, prefix: str = "") -> list[ActorEndpoint]:
        """List all registered actors, optionally filtered by name prefix."""
        ...
```

### Resolver Interface

A Resolver provides actor discovery and connection management. It's the
client-facing interface for finding and connecting to actors.

Resolvers are independent from Clusters. `ClusterResolver` is one implementation
that uses a Cluster's Metadata service; other implementations exist for testing
or special use cases.

```python
class Resolver(Protocol):
    """Actor discovery and connection management.

    Resolvers map actor names to handles and manage reconnection on failure.
    """

    def lookup(self, name: str) -> "ActorHandle":
        """Look up an actor by name and return a handle.

        The handle automatically reconnects if the actor restarts.

        Raises:
            ActorNotFoundError: If no actor with this name exists
        """
        ...

    def lookup_all(self, name: str) -> list["ActorHandle"]:
        """Look up all actors registered under a name.

        Use for load-balanced actor pools where multiple actors register
        under the same logical name.
        """
        ...

    def wait_for_actor(
        self,
        name: str,
        timeout: float = 60.0,
    ) -> "ActorHandle":
        """Wait for an actor to become available.

        Useful during startup when actors may not be registered yet.
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
    """Resolver that connects to a fixed address.

    Useful for testing or when connecting to a known endpoint.
    """

    def __init__(self, address: str):
        ...
```

### Actor System

The Actor system provides RPC-based communication between jobs:

- **ActorServer**: Hosts actor instances, registers with the cluster's Metadata service
- **ActorHandle**: Client-side proxy returned by `resolver.lookup()`
- **ActorContext**: Passed to actor methods, enables actors to call other actors

**Scope notes**: Streaming responses are not supported. RPC tracing is not
included in the initial implementation.

```python
T = TypeVar("T")


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


class ActorHandle(Protocol[T]):
    """Client-side handle for calling methods on a remote actor.

    Obtained via resolver.lookup(). Method calls are made via attribute access:

        resolver = ClusterResolver(cluster)
        handle = resolver.lookup("inference_0")
        result = handle.predict(x)  # Synchronous call

    For async calls:

        future = handle.predict.remote(x)
        result = future.result()
    """

    @property
    def actor_id(self) -> ActorId:
        """Unique identifier for this actor."""
        ...

    @property
    def address(self) -> str:
        """RPC address of the actor."""
        ...

    def __getattr__(self, method_name: str) -> "ActorMethod":
        """Get a method wrapper for calling remote methods."""
        ...


class ActorMethod(Protocol):
    """Wrapper for calling a specific method on an actor."""

    def remote(self, *args: Any, **kwargs: Any) -> "ActorFuture":
        """Call method asynchronously, returning a future."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call method synchronously, blocking until complete."""
        ...


class ActorFuture(Protocol[T]):
    """Future representing an in-flight actor method call."""

    def result(self, timeout: float | None = None) -> T:
        """Block until result is available."""
        ...

    def done(self) -> bool:
        """Check if the call has completed."""
        ...

    def exception(self) -> Exception | None:
        """Get the exception if the call failed."""
        ...


class ActorServer:
    """Server for hosting actors and handling RPC calls.

    Takes a Cluster (or Metadata service) for registration. The server reads
    FRAY_JOB_ID from environment to associate registrations with the current job.

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
            cluster: Cluster for actor registration (uses cluster.metadata)
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

        The actor is registered with the cluster's Metadata service using
        the job_id from FRAY_JOB_ID environment variable.

        Actor methods should accept ActorContext as their first argument:

            class MyActor:
                def process(self, ctx: ActorContext, data: dict) -> dict:
                    # ctx.resolver allows calling other actors
                    other = ctx.resolver.lookup("other_actor")
                    return other.transform(data)

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

### Local Development

For local development, all components have in-process implementations that
preserve the same interfaces. Code written for production works locally
without modification.

```python
class LocalCluster(Cluster):
    """Local cluster for development and testing.

    Runs jobs as threads or subprocesses in the current environment.
    Includes an embedded Metadata service.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
    ):
        ...


class LocalResolver(Resolver):
    """In-process resolver for local development.

    Can be backed by a LocalCluster's metadata or operate standalone
    for unit tests.
    """

    def __init__(self, cluster: LocalCluster | None = None):
        ...


# Example: local development workflow
cluster = LocalCluster()
resolver = LocalResolver(cluster)

# Server side (same code as production)
server = ActorServer(cluster)  # Takes cluster for registration
server.register("my_actor", MyActor())
server.serve_background()

# Client side (same code as production)
handle = resolver.lookup("my_actor")
result = handle.process(data)
```

### Zephyr Integration

Zephyr continues to work unchanged. It uses its own backend abstraction
which can launch Fray jobs internally.

```python
class FrayBackend(Backend):
    """Zephyr backend using Fray for distributed execution.

    Launches worker jobs via the Cluster and dispatches work to them.
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
        """Execute a dataset pipeline on Fray workers."""
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

    # JAX will automatically discover TPU devices
    print(f"Running on {jax.device_count()} devices")
    train_lm()  # Uses config from environment/args


def launch_training(
    name: str,
    tpu_type: str = "v5litepod-16",
    num_slices: int = 1,
):
    """Launch a training job on the cluster.

    Args:
        name: Job name for tracking
        tpu_type: TPU accelerator type
        num_slices: Number of TPU slices for multi-slice training
    """
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

    # Wait for completion and get final status
    result = cluster.wait(job_id, raise_on_failure=True)
    print(f"Training completed with status: {result.status}")
    return result


# Usage from executor
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
    """Run a data transformation pipeline.

    This example shows Zephyr usage - the backend is selected automatically
    based on environment. On a Fray cluster, this uses FrayBackend; locally
    it uses ThreadpoolBackend.
    """
    pipeline = (
        Dataset.from_files(config.input_pattern)
        .load_jsonl()
        .map(transform_document)
        .filter(lambda doc: doc["token_count"] > 10)
        .write_jsonl(f"{config.output_path}/output-{{shard:05d}}.jsonl.gz")
    )

    # Execute with auto-detected backend
    output_files = Backend.execute(
        pipeline,
        max_parallelism=config.max_parallelism,
        memory=config.memory_per_worker,
    )

    return output_files


# For explicit Fray backend usage:
def run_transform_on_fray(config: TransformConfig) -> list[str]:
    """Run transformation explicitly on Fray cluster."""
    from fray.cluster import current_cluster
    from zephyr.backends import FrayBackend

    pipeline = (
        Dataset.from_files(config.input_pattern)
        .load_jsonl()
        .map(transform_document)
        .write_jsonl(f"{config.output_path}/output-{{shard:05d}}.jsonl.gz")
    )

    backend = FrayBackend(
        cluster=current_cluster(),
        max_parallelism=config.max_parallelism,
        memory_per_worker=config.memory_per_worker,
    )

    return backend.execute(pipeline)
```

### 3. Inference/Evaluation

Inference requires launching inference servers and dispatching work to them.

```python
# marin/inference/inference_pool.py
from dataclasses import dataclass
from typing import Iterator

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
        # ctx available if this actor needs to call other actors
        return self.model.generate(prompts)


def inference_server_main(config: InferenceConfig, server_id: int):
    """Entrypoint for inference server jobs."""
    cluster = current_cluster()
    server = ActorServer(cluster)

    actor = InferenceActor(config.model_path)
    server.register(f"inference_{server_id}", actor)
    server.register("inference_pool", actor)  # Pool name for load balancing
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
        handles = self.resolver.lookup_all("inference_pool")
        results = []

        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            handle = handles[i // self.config.batch_size % len(handles)]
            results.extend(handle.predict(batch))

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
            entrypoint=Entrypoint.from_callable(inference_server_main, args=(config, i)),
            environment=EnvironmentConfig.create(extras=["tpu"]),
        )
        for i in range(config.num_servers)
    ]

    job_ids = cluster.launch_group(
        requests,
        group=JobGroupConfig(name="inference-pool", same_region=True),
    )

    # Wait for all servers to register
    for i in range(config.num_servers):
        resolver.wait_for_actor(f"inference_{i}")

    return InferencePool(cluster, job_ids, config)


# Example usage
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

    curriculum = resolver.wait_for_actor("curriculum")
    weight_coord = resolver.wait_for_actor("weight_coordinator")

    # Training loop
    from marin.rl.train_worker import TrainWorker
    worker = TrainWorker(config)

    for step in worker.train_loop():
        if step % 100 == 0:
            ckpt_path = f"{config.checkpoint_path}/step-{step}"
            worker.save_checkpoint(ckpt_path)
            weight_coord.publish_checkpoint(step, ckpt_path)


def rollout_worker_main(config: RLConfig, worker_id: int):
    """Rollout worker - generates training data."""
    cluster = current_cluster()
    resolver = ClusterResolver(cluster)

    curriculum = resolver.wait_for_actor("curriculum")
    weight_coord = resolver.wait_for_actor("weight_coordinator")

    from marin.rl.rollout_worker import RolloutWorker
    worker = RolloutWorker(config, worker_id)

    while True:
        step, ckpt = weight_coord.get_latest_checkpoint()
        if ckpt:
            worker.load_checkpoint(ckpt)

        lesson = curriculum.get_lesson()
        rollouts, reward = worker.generate_rollout(lesson)
        curriculum.report_result(lesson["name"], reward)
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
                resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
                entrypoint=Entrypoint.from_callable(coordinator_main, args=(self.config,)),
            ),
            # Training worker
            JobRequest(
                name=f"rl-train-{self.run_id}",
                resources=ResourceConfig.with_tpu(self.config.train_tpu_type),
                entrypoint=Entrypoint.from_callable(train_worker_main, args=(self.config,)),
                environment=EnvironmentConfig.create(extras=["tpu"]),
            ),
        ]

        # Rollout workers
        for i in range(self.config.num_rollout_workers):
            requests.append(
                JobRequest(
                    name=f"rl-rollout-{self.run_id}-{i}",
                    resources=ResourceConfig.with_tpu(self.config.inference_tpu_type),
                    entrypoint=Entrypoint.from_callable(rollout_worker_main, args=(self.config, i)),
                    environment=EnvironmentConfig.create(extras=["tpu"]),
                )
            )

        self.job_ids = self.cluster.launch_group(requests, group=group)

        # Wait for coordinator actors
        self.resolver.wait_for_actor("curriculum")
        self.resolver.wait_for_actor("weight_coordinator")

        # Monitor training worker
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

    def register_slice(self, ctx: ActorContext, slice_id: str, address: str) -> dict:
        self.slices[slice_id] = address
        return {"step": self.current_step, "peers": list(self.slices.keys())}

    def unregister_slice(self, ctx: ActorContext, slice_id: str):
        self.slices.pop(slice_id, None)

    def heartbeat(self, ctx: ActorContext, slice_id: str) -> dict:
        return {"step": self.current_step, "slice_count": len(self.slices)}

    def barrier_enter(self, ctx: ActorContext, slice_id: str, step: int) -> bool:
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

    coordinator = resolver.wait_for_actor("slice_coordinator")

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
                min_resources=ResourceConfig.with_tpu(self.config.tpu_type, slice_count=self.config.min_slices),
                max_resources=ResourceConfig.with_tpu(self.config.tpu_type, slice_count=self.config.max_slices),
                preemptible=True,
            )
        )

        try:
            # Launch coordinator
            self.cluster.launch(
                JobRequest(
                    name="multislice-coordinator",
                    resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
                    entrypoint=Entrypoint.from_callable(coordinator_main, args=(self.config,)),
                )
            )
            self.resolver.wait_for_actor("slice_coordinator")

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
                entrypoint=Entrypoint.from_callable(slice_worker_main, args=(self.config, f"slice-{idx}")),
                environment=EnvironmentConfig.create(extras=["tpu"]),
            )
        )
        self.slice_jobs[f"slice-{idx}"] = job_id

    def _check_health_and_scale(self, reservation):
        # Restart failed slices, scale up if resources available
        for slice_id, job_id in list(self.slice_jobs.items()):
            if self.cluster.poll(job_id).status in (JobStatus.FAILED, JobStatus.STOPPED):
                self._launch_slice(int(slice_id.split("-")[1]))

        # Scale based on reservation
        reservation_info = self.cluster.get_reservation(reservation)
        available = reservation_info.allocated_resources.replicas
        if available > len(self.slice_jobs) < self.config.max_slices:
            for i in range(len(self.slice_jobs), available):
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

All examples work locally with the same code - just use `LocalCluster` and
`LocalResolver` instead of production implementations.

```python
# tests/test_local_development.py
"""Local development uses identical patterns to production."""

from fray.cluster import LocalCluster, LocalResolver
from fray.rpc import ActorServer, ActorContext


def test_local_inference():
    """Test inference locally - same code pattern as production."""
    cluster = LocalCluster()
    resolver = LocalResolver(cluster)

    # Server side (identical to production)
    class MockInferenceActor:
        def predict(self, ctx: ActorContext, prompts: list[str]) -> list[str]:
            return [f"Response to: {p}" for p in prompts]

    server = ActorServer(cluster)
    server.register("inference", MockInferenceActor())
    server.serve_background()

    # Client side (identical to production)
    handle = resolver.lookup("inference")
    result = handle.predict(["Hello", "World"])

    assert result == ["Response to: Hello", "Response to: World"]
    server.shutdown()


def test_fixed_resolver():
    """FixedResolver for testing against known endpoints."""
    from fray.cluster import FixedResolver

    # Connect directly to a known address (useful for debugging)
    resolver = FixedResolver("localhost:8080")
    handle = resolver.lookup("my_actor")
    # handle.some_method()
```
