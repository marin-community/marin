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
name prefix for grouping (e.g. `{user}/rl/rollout/`). Job instances are provided
the expected environment variables (`FRAY_JOB_NAME`, `FRAY_JOB_ID`,
`FRAY_CLUSTER_ADDRESS`, etc) to bootstrap their environment. Every job receives
a globally unique ID.

The cluster system provides access to job logs, and handles task failures and
pre-emption recovery. It attempts to minimize the startup time of tasks by
reusing warm environments and re-using warm workers when possible. The cluster
system _will not_ attempt to reuse warm processes, thus the target startup time
for tasks is expected to be <10s for a warm image.

### Job/Actor Discovery

In addition to the expected ability to list jobs and their information, we also
need a mechanism to map actors (described below) to the supporting job. To do
this, we'll provide a metadata store on the cluster which specifically maps
against job IDs. This will look something like:

```
create_mapping(job_id, port_name, namespace, actor_name)
```

Because the cluster manager owns the jobs and the mapping service, it can
automatically clean up mappings which correpond to terminated jobs, and update
mappings in the case of task failures.

### Actor System

Great, so now we have reservations, and jobs, and tasks within those jobs, and
we can see the logs: this is a good start -- we're trying to provide a
simplified version of the VM environment from a cloud provider. Now we need to
make it _simple_ to setup communication within our jobs as needed.

Recall that Ray provides 3 mechanisms for communication: the object store
(`.put/get`), actors, and remote task calls. In our experience, the actor model
has the most consistent behavior around failures: you can reconnect to actors
but it's the user's responsibility to ensure that the actor recovers to a
consistent state.  To this end, we propose an actor system on top of Connect/RPC
(or gRPC) which simplifies managing tasks in a Fray environment.

Since users are understandably hesistant to write protos, we'll provide a
simpler class based API on top of the raw RPC system. We'll make a simplifying
assumption that the number of actors is small, and rely on our cluster manager
to store indirection information.

```python
# client
resolver = fray_resolver(env["FRAY_NAMESPACE"])
actor = resolver.lookup(f"inference_0")
actor.predict(x)
```

```python
# server
class InferenceActor:
  def __init__(self):
    self.model = load_model()

  def predict(self, x):
    return self.model(x)

resolver = fray_resolver()
server = FrayActorServer(resolver)
server.register(f"inference_{task_id}", InferenceActor)
```

The resolver neatly handles a few common challenges:

_Worker resolution_: We can consistently identify the worker address for a given actor name.
_Fault tolerance_: As we use an indirect address for our actors, we can re-resolve the owning worker automatically when tasks fail.
_Load balancing_: If we allow registering multiple targets for a given name, we can automatically load-balance between backends in our RPC interface.

The resolver backend needn't be the same process as the cluster manager, but for
convenience we assume they are co-located for the moment.

### Back to Ray

Note that if we can build a resolver which is _backed_ by our job and metadata
service, we can recover the dynamic task dispatch behavior of Ray. For example,
we can startup a pool of Worker actors, dispatching tasks to available workers,
and relying on our metadata resolution to ensure we always connect to a valid
worker.

```python
class DynamicResolver:
    cluster: Cluster
    workers: Dict[str, Worker]
    resource_config: ResourceConfig

    def run(self, fn, *args, **kw):
        # find idle worker
        # if not available, ask cluster for new worker
        # run fn on worker
        # return future
```

### Use Case Implementations

* *Training* is straightforward, just a cluster job launch
* *Inference* requires multiple jobs, with either a proxy for round-robin or dispatch via the resolver
* *Zephyr* may either start a fixed worker pool and manually dispatch work  _or_ starting a new job for each work chunk. Alternatively Zephyr could leverage the dynamic resolver above should it prove useful.
* *RL*

```
# controller.py
# by default, Fray sets up a new namespace for each top-level job
curriculum = cluster.launch(curriculum_actor, ports={"actor_service"})
training_actor = cluster.launch(training_actor, ports={"actor_service"})

# start worker jobs
worker_jobs = cluster.launch(rl.worker)

resolver = fray_resolver()
rl_worker = RLWorker(resolver.lookup(curriculum.id))
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

This section provides the complete Python interfaces for the Fray system. These
interfaces extend and refine the existing `fray.cluster` and `fray.job` APIs
while maintaining backward compatibility where possible.

### Core Types

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence, TypeVar
from enum import StrEnum


# Existing types from fray.cluster.base (unchanged)
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


class ActorStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
```

### Reservation & Job Group APIs

These new types extend the existing `JobRequest` to support co-location and
resource pooling.

```python
@dataclass
class JobGroupConfig:
    """Configuration for a group of co-located jobs.

    Jobs within a group are scheduled together and have network locality
    guarantees. Use this for RL training where inference workers should
    be close to training workers.
    """
    name: str
    namespace: Namespace | None = None

    # Locality constraints
    same_region: bool = True
    same_zone: bool = False  # Stricter locality for low-latency RPCs


@dataclass
class ReservationConfig:
    """Resource reservation for elastic job scheduling.

    Reservations allow the cluster to pre-allocate resources for a set of
    jobs, improving startup latency and ensuring availability.
    """
    min_resources: ResourceConfig
    max_resources: ResourceConfig
    priority: int = 0  # Higher priority reservations are filled first
    ttl_seconds: int = 3600  # Auto-release after this duration
    preemptible: bool = True


@dataclass
class ReservationInfo:
    """Status of an active reservation."""
    reservation_id: ReservationId
    allocated_resources: ResourceConfig
    pending_resources: ResourceConfig
    jobs: list[JobId]
    expires_at: float  # Unix timestamp
```

### Extended Cluster Interface

```python
class Cluster(Protocol):
    """Abstract interface for cluster job scheduling.

    Implementations: RayCluster, FrayCluster, LocalCluster
    """

    # === Existing methods (unchanged) ===

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job on the cluster."""
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

    # === New methods for Fray Zero ===

    def create_reservation(self, config: ReservationConfig) -> ReservationId:
        """Create a resource reservation for upcoming jobs.

        The cluster will attempt to allocate resources matching the
        reservation config. Jobs launched with this reservation will
        use pre-allocated resources for faster startup.
        """
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

        Args:
            requests: Job specifications to launch
            group: Co-location configuration
            reservation_id: Optional reservation to draw resources from

        Returns:
            List of job IDs in the same order as requests
        """
        ...

    def get_namespace(self) -> Namespace:
        """Get the current namespace for this cluster connection.

        Namespaces isolate actors and job metadata between different
        top-level jobs or users.
        """
        ...


def current_cluster() -> Cluster:
    """Get the current cluster from environment.

    Reads FRAY_CLUSTER_SPEC environment variable to determine which
    cluster implementation to use. Falls back to LocalCluster if unset.
    """
    ...
```

### Resolver Interface

The resolver provides actor discovery and fault-tolerant reconnection.

```python
@dataclass
class ActorEndpoint:
    """Endpoint information for a registered actor."""
    actor_id: ActorId
    address: str  # host:port
    job_id: JobId
    metadata: dict[str, str] = field(default_factory=dict)


class Resolver(Protocol):
    """Actor discovery and resolution service.

    The resolver maintains a mapping from actor names to endpoints,
    enabling clients to find and connect to actors by name. It handles
    automatic re-resolution when actors move or restart.
    """

    def lookup(self, name: str) -> ActorHandle:
        """Look up an actor by name and return a handle.

        The returned handle automatically reconnects if the actor
        restarts or moves to a different worker.

        Args:
            name: Actor name (unique within namespace)

        Returns:
            ActorHandle for making method calls

        Raises:
            ActorNotFoundError: If no actor with this name exists
        """
        ...

    def lookup_all(self, name: str) -> list[ActorHandle]:
        """Look up all actors matching a name pattern.

        Use this for load-balanced actor pools where multiple actors
        are registered under the same logical name.

        Args:
            name: Actor name pattern (may match multiple actors)

        Returns:
            List of ActorHandles (may be empty)
        """
        ...

    def register(
        self,
        name: str,
        endpoint: ActorEndpoint,
        ttl_seconds: int = 60,
    ) -> None:
        """Register an actor endpoint with the resolver.

        Registration must be renewed before ttl_seconds expires.
        The ActorServer handles this automatically.

        Args:
            name: Actor name for lookup
            endpoint: Actor endpoint information
            ttl_seconds: Registration time-to-live
        """
        ...

    def unregister(self, name: str, actor_id: ActorId) -> None:
        """Remove an actor registration."""
        ...

    def list_actors(self, prefix: str = "") -> list[ActorEndpoint]:
        """List all registered actors, optionally filtered by name prefix."""
        ...


def fray_resolver(namespace: Namespace | None = None) -> Resolver:
    """Create a resolver for the given namespace.

    If namespace is None, uses FRAY_NAMESPACE from environment or
    creates a new namespace scoped to the current job.
    """
    ...
```

### Actor System

```python
T = TypeVar("T")


class ActorHandle(Protocol[T]):
    """Handle for calling methods on a remote actor.

    Method calls are made via attribute access, returning futures:

        handle = resolver.lookup("my_actor")
        future = handle.predict.remote(x)
        result = future.result()

    Or with the synchronous shortcut:

        result = handle.predict(x)  # Blocks until complete
    """

    @property
    def actor_id(self) -> ActorId:
        """Unique identifier for this actor."""
        ...

    def __getattr__(self, method_name: str) -> ActorMethod:
        """Get a method wrapper for calling remote methods."""
        ...


class ActorMethod(Protocol):
    """Wrapper for calling a specific method on an actor."""

    def remote(self, *args: Any, **kwargs: Any) -> ActorFuture:
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


class FrayActorServer:
    """Server for hosting actors and handling RPC calls.

    The actor server manages the lifecycle of actors, handles incoming
    RPC requests, and maintains registration with the resolver.

    Usage:
        server = FrayActorServer(resolver)
        server.register("my_actor", MyActor(config))
        server.serve()  # Blocks, serving requests
    """

    def __init__(
        self,
        resolver: Resolver,
        host: str = "0.0.0.0",
        port: int = 0,  # 0 = auto-assign
    ):
        """Initialize actor server.

        Args:
            resolver: Resolver for actor registration
            host: Host address to bind
            port: Port to bind (0 for auto-assignment)
        """
        ...

    @property
    def address(self) -> str:
        """Get the server's bound address (host:port)."""
        ...

    def register(
        self,
        name: str,
        actor: Any,
        metadata: dict[str, str] | None = None,
    ) -> ActorId:
        """Register an actor instance with the server.

        Args:
            name: Name for resolver lookup
            actor: Actor instance (any object with callable methods)
            metadata: Optional metadata for discovery

        Returns:
            Unique actor ID
        """
        ...

    def unregister(self, name: str) -> None:
        """Unregister and stop an actor."""
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


# Decorator for defining actor classes
def actor(
    name: str | None = None,
    num_cpus: float = 1,
    num_gpus: float = 0,
    memory: str | None = None,
):
    """Decorator to mark a class as an actor.

    Usage:
        @actor(name="inference", num_gpus=1)
        class InferenceActor:
            def __init__(self, model_path: str):
                self.model = load_model(model_path)

            def predict(self, x):
                return self.model(x)
    """
    ...
```

### JobContext Interface

The existing `JobContext` protocol is extended to support actor creation:

```python
class JobContext(Protocol):
    """Protocol for execution contexts within a job.

    Implementations: RayContext, ThreadContext, SyncContext, FrayContext
    """

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference."""
        ...

    def run(self, fn: Callable, *args: Any) -> Any:
        """Execute a function with arguments and return a future."""
        ...

    def wait(
        self,
        futures: list,
        num_returns: int = 1
    ) -> tuple[list, list]:
        """Wait for futures to complete."""
        ...

    def create_actor(
        self,
        actor_class: type[T],
        *args: Any,
        name: str | None = None,
        get_if_exists: bool = False,
        lifetime: Literal["non_detached", "detached"] = "non_detached",
        **kwargs: Any,
    ) -> ActorHandle[T]:
        """Create an actor within this execution context.

        Args:
            actor_class: Class to instantiate as actor
            *args: Constructor positional arguments
            name: Optional name for the actor
            get_if_exists: Return existing actor with same name
            lifetime: "detached" actors survive job completion
            **kwargs: Constructor keyword arguments

        Returns:
            Handle for calling actor methods
        """
        ...


class FrayContext(JobContext):
    """RPC-based execution context for Fray clusters.

    Connects to a FrayController to execute tasks and manage actors
    across distributed workers.
    """

    def __init__(self, controller_address: str):
        """Connect to a Fray controller.

        Args:
            controller_address: Controller URL (e.g., "http://localhost:50051")
        """
        ...


def create_job_ctx(
    context_type: Literal["ray", "threadpool", "sync", "fray", "auto"] = "auto",
    max_workers: int = 1,
    controller_address: str | None = None,
    **options: Any,
) -> JobContext:
    """Create a job context for the specified backend.

    Args:
        context_type: Backend type ("auto" selects based on environment)
        max_workers: Worker count for threadpool backend
        controller_address: Required for "fray" backend
        **options: Backend-specific options

    Returns:
        JobContext implementation
    """
    ...


def get_default_job_ctx() -> JobContext:
    """Get the current default job context."""
    ...


@contextmanager
def fray_default_job_ctx(ctx: JobContext):
    """Set the default job context for a scope."""
    ...
```

### Local Cluster (Development)

```python
class LocalCluster(Cluster):
    """Local cluster for development and testing.

    Runs all jobs in the current process using threading or multiprocessing.
    Provides the same interface as remote clusters for seamless development.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
    ):
        """Initialize local cluster.

        Args:
            max_workers: Maximum concurrent jobs
            use_processes: Use processes instead of threads
        """
        ...


class LocalResolver(Resolver):
    """In-process resolver for local development.

    Actors are hosted in the same process, enabling debugging with
    standard Python tools.
    """
    ...


class LocalActorServer(FrayActorServer):
    """Actor server for local development.

    Handles actor calls synchronously in the same process.
    """
    ...
```

### Zephyr Backend Integration

```python
# In zephyr/backends.py

class FrayBackend(Backend):
    """Zephyr backend using Fray for distributed execution.

    Uses Fray's job management for worker orchestration instead of Ray.
    """

    def __init__(
        self,
        cluster: Cluster,
        max_parallelism: int = 100,
        memory_per_worker: str = "2GB",
    ):
        ...

    def execute(self, dataset: Dataset) -> list[Any]:
        """Execute a dataset pipeline on Fray workers."""
        ...


# Factory function
def get_backend(
    backend_type: Literal["ray", "fray", "threadpool", "auto"] = "auto",
    **kwargs: Any,
) -> Backend:
    """Get a Zephyr backend.

    Args:
        backend_type: Backend type ("auto" selects based on environment)
        **kwargs: Backend-specific configuration

    Returns:
        Backend implementation
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
    Entrypoint,
    EnvironmentConfig,
    JobGroupConfig,
    JobRequest,
    ResourceConfig,
)
from fray.rpc import FrayActorServer, fray_resolver


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
        print(f"Loaded model on {self.device_count} devices")

    def predict(self, prompts: list[str]) -> list[str]:
        """Generate completions for a batch of prompts."""
        return self.model.generate(prompts)

    def health_check(self) -> dict:
        """Return server health status."""
        return {"status": "healthy", "devices": self.device_count}


def inference_server_main(config: InferenceConfig, server_id: int):
    """Entrypoint for inference server jobs."""
    import os

    resolver = fray_resolver()
    server = FrayActorServer(resolver, port=0)

    # Register actor with a unique name
    actor = InferenceActor(config.model_path)
    server.register(f"inference_{server_id}", actor)

    # Also register under a pool name for load balancing
    server.register("inference_pool", actor)

    print(f"Inference server {server_id} ready at {server.address}")
    server.serve()  # Blocks


def launch_inference_pool(config: InferenceConfig) -> "InferencePool":
    """Launch a pool of inference servers.

    Returns an InferencePool object that can be used to dispatch work.
    """
    cluster = current_cluster()

    # Launch all inference servers as a co-located group
    requests = [
        JobRequest(
            name=f"inference-{i}",
            resources=ResourceConfig.with_tpu(config.tpu_type),
            entrypoint=Entrypoint.from_callable(
                inference_server_main,
                args=(config, i),
            ),
            environment=EnvironmentConfig.create(extras=["tpu"]),
        )
        for i in range(config.num_servers)
    ]

    job_ids = cluster.launch_group(
        requests,
        group=JobGroupConfig(name="inference-pool", same_region=True),
    )

    # Wait for servers to be ready (poll until actors are registered)
    resolver = fray_resolver()
    import time
    for _ in range(60):  # 60 second timeout
        actors = resolver.list_actors(prefix="inference_pool")
        if len(actors) == config.num_servers:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Inference servers failed to start")

    return InferencePool(job_ids, resolver, config)


class InferencePool:
    """Client for dispatching work to inference servers."""

    def __init__(self, job_ids: list, resolver, config: InferenceConfig):
        self.job_ids = job_ids
        self.resolver = resolver
        self.config = config

    def predict(self, prompts: list[str]) -> list[str]:
        """Run inference on prompts, load-balancing across servers."""
        # Get handles to all inference actors
        handles = self.resolver.lookup_all("inference_pool")

        # Batch prompts across servers
        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            # Round-robin to servers
            handle = handles[i // batch_size % len(handles)]
            results.extend(handle.predict(batch))

        return results

    def predict_async(
        self,
        prompts: Iterator[list[str]]
    ) -> Iterator[list[str]]:
        """Stream inference results for batches of prompts."""
        handles = self.resolver.lookup_all("inference_pool")
        pending_futures = []

        for i, batch in enumerate(prompts):
            handle = handles[i % len(handles)]
            future = handle.predict.remote(batch)
            pending_futures.append(future)

            # Yield completed results as they become available
            while pending_futures and pending_futures[0].done():
                yield pending_futures.pop(0).result()

        # Drain remaining futures
        for future in pending_futures:
            yield future.result()

    def shutdown(self):
        """Terminate all inference servers."""
        cluster = current_cluster()
        for job_id in self.job_ids:
            cluster.terminate(job_id)


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
from dataclasses import dataclass, field
import os
from typing import Iterator

from fray.cluster import (
    current_cluster,
    Entrypoint,
    EnvironmentConfig,
    JobGroupConfig,
    JobRequest,
    ResourceConfig,
)
from fray.rpc import FrayActorServer, fray_resolver, ActorHandle


@dataclass
class RLConfig:
    model_path: str
    train_tpu_type: str = "v5litepod-16"
    inference_tpu_type: str = "v5litepod-4"
    num_rollout_workers: int = 4
    rollout_storage_path: str = "gs://bucket/rollouts"
    checkpoint_path: str = "gs://bucket/checkpoints"


# === Shared Actors ===

class CurriculumActor:
    """Manages training curriculum and lesson selection.

    Shared across all workers to coordinate which lessons to sample.
    """

    def __init__(self, curriculum_config: dict):
        self.config = curriculum_config
        self.step = 0
        self.lesson_stats: dict[str, dict] = {}

    def get_lesson(self) -> dict:
        """Get the next lesson to run."""
        # Select lesson based on curriculum strategy
        lesson_name = self._select_lesson()
        return self.config["lessons"][lesson_name]

    def report_result(self, lesson_name: str, reward: float):
        """Report results from a completed rollout."""
        if lesson_name not in self.lesson_stats:
            self.lesson_stats[lesson_name] = {"count": 0, "total_reward": 0}

        self.lesson_stats[lesson_name]["count"] += 1
        self.lesson_stats[lesson_name]["total_reward"] += reward

    def get_stats(self) -> dict:
        """Get curriculum statistics."""
        return {"step": self.step, "lessons": self.lesson_stats}

    def _select_lesson(self) -> str:
        # Implement lesson selection logic
        lessons = list(self.config["lessons"].keys())
        return lessons[self.step % len(lessons)]


class WeightCoordinator:
    """Coordinates weight synchronization between train and rollout workers.

    The training worker publishes new checkpoints; rollout workers poll
    for updates.
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.current_step = 0
        self.current_checkpoint: str | None = None

    def publish_checkpoint(self, step: int, path: str):
        """Called by training worker when new weights are ready."""
        self.current_step = step
        self.current_checkpoint = path

    def get_latest_checkpoint(self) -> tuple[int, str | None]:
        """Get the latest checkpoint (step, path)."""
        return self.current_step, self.current_checkpoint


# === Worker Implementations ===

def train_worker_main(config: RLConfig, run_id: str):
    """Training worker entrypoint."""
    import logging
    logging.basicConfig(level=logging.INFO)

    resolver = fray_resolver()

    # Connect to shared actors
    curriculum: ActorHandle = resolver.lookup("curriculum")
    weight_coord: ActorHandle = resolver.lookup("weight_coordinator")

    # Initialize model and optimizer
    from marin.rl.train_worker import TrainWorker
    worker = TrainWorker(config)

    step = 0
    while True:
        # Fetch rollouts from storage
        rollouts = worker.fetch_rollouts(config.rollout_storage_path)

        if not rollouts:
            import time
            time.sleep(1)
            continue

        # Train on rollouts
        metrics = worker.train_step(rollouts)
        step += 1

        # Periodically save checkpoint and notify coordinator
        if step % 100 == 0:
            ckpt_path = f"{config.checkpoint_path}/step-{step}"
            worker.save_checkpoint(ckpt_path)
            weight_coord.publish_checkpoint(step, ckpt_path)

        # Log metrics
        if step % 10 == 0:
            stats = curriculum.get_stats()
            print(f"Step {step}: loss={metrics['loss']:.4f}, curriculum={stats}")


def rollout_worker_main(config: RLConfig, worker_id: int, run_id: str):
    """Rollout worker entrypoint."""
    import logging
    logging.basicConfig(level=logging.INFO)

    resolver = fray_resolver()

    # Connect to shared actors
    curriculum: ActorHandle = resolver.lookup("curriculum")
    weight_coord: ActorHandle = resolver.lookup("weight_coordinator")

    # Initialize inference model
    from marin.rl.rollout_worker import RolloutWorker
    worker = RolloutWorker(config, worker_id)

    last_checkpoint_step = 0

    while True:
        # Check for new weights
        step, checkpoint_path = weight_coord.get_latest_checkpoint()
        if step > last_checkpoint_step and checkpoint_path:
            worker.load_checkpoint(checkpoint_path)
            last_checkpoint_step = step

        # Get lesson from curriculum
        lesson = curriculum.get_lesson()

        # Generate rollouts
        rollouts, reward = worker.generate_rollout(lesson)

        # Report results to curriculum
        curriculum.report_result(lesson["name"], reward)

        # Write rollouts to storage
        worker.write_rollouts(rollouts, config.rollout_storage_path)


def coordinator_main(config: RLConfig, run_id: str):
    """Coordinator job that hosts shared actors."""
    import logging
    logging.basicConfig(level=logging.INFO)

    resolver = fray_resolver()
    server = FrayActorServer(resolver, port=0)

    # Create and register shared actors
    curriculum_config = {"lessons": {"math": {}, "code": {}, "reasoning": {}}}
    curriculum = CurriculumActor(curriculum_config)
    server.register("curriculum", curriculum)

    weight_coord = WeightCoordinator(config.checkpoint_path)
    server.register("weight_coordinator", weight_coord)

    print(f"Coordinator ready at {server.address}")
    server.serve()  # Blocks


# === Job Orchestration ===

class RLJob:
    """Orchestrates an RL training job."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.cluster = current_cluster()
        self.run_id = f"rl-{os.urandom(4).hex()}"

    def run(self) -> dict:
        """Launch all workers and wait for completion."""
        # Create job group for co-location
        group = JobGroupConfig(
            name=f"rl-{self.run_id}",
            same_region=True,
        )

        # Build job requests
        requests = []

        # 1. Coordinator (hosts shared actors)
        requests.append(
            JobRequest(
                name=f"rl-coordinator-{self.run_id}",
                resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
                entrypoint=Entrypoint.from_callable(
                    coordinator_main,
                    args=(self.config, self.run_id),
                ),
                environment=EnvironmentConfig.create(),
            )
        )

        # 2. Training worker
        requests.append(
            JobRequest(
                name=f"rl-train-{self.run_id}",
                resources=ResourceConfig.with_tpu(self.config.train_tpu_type),
                entrypoint=Entrypoint.from_callable(
                    train_worker_main,
                    args=(self.config, self.run_id),
                ),
                environment=EnvironmentConfig.create(extras=["tpu"]),
            )
        )

        # 3. Rollout workers
        for i in range(self.config.num_rollout_workers):
            requests.append(
                JobRequest(
                    name=f"rl-rollout-{self.run_id}-{i}",
                    resources=ResourceConfig.with_tpu(self.config.inference_tpu_type),
                    entrypoint=Entrypoint.from_callable(
                        rollout_worker_main,
                        args=(self.config, i, self.run_id),
                    ),
                    environment=EnvironmentConfig.create(extras=["tpu"]),
                )
            )

        # Launch all jobs as a group
        self.job_ids = self.cluster.launch_group(requests, group=group)

        # Wait for coordinator to be ready
        resolver = fray_resolver()
        import time
        for _ in range(60):
            try:
                resolver.lookup("curriculum")
                resolver.lookup("weight_coordinator")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Coordinator actors failed to start")

        print(f"RL job {self.run_id} started with {len(self.job_ids)} workers")

        # Monitor training worker (it's the primary job)
        train_job_id = self.job_ids[1]  # Second job is training
        result = self.cluster.monitor(train_job_id)

        return {"status": result.status, "run_id": self.run_id}

    def shutdown(self):
        """Terminate all workers."""
        for job_id in self.job_ids:
            self.cluster.terminate(job_id)


# Example usage
if __name__ == "__main__":
    config = RLConfig(
        model_path="gs://bucket/models/llama-8b",
        train_tpu_type="v5litepod-16",
        inference_tpu_type="v5litepod-4",
        num_rollout_workers=4,
    )

    job = RLJob(config)
    result = job.run()
    print(f"RL training completed: {result}")
```

### 5. Multi-Slice Training (Flex)

Multi-slice training coordinates multiple TPU slices that can dynamically
join and leave.

```python
# marin/training/multislice_training.py
from dataclasses import dataclass
from typing import Iterator
import time

from fray.cluster import (
    current_cluster,
    Entrypoint,
    EnvironmentConfig,
    JobGroupConfig,
    JobId,
    JobRequest,
    JobStatus,
    ReservationConfig,
    ResourceConfig,
)
from fray.rpc import FrayActorServer, fray_resolver, ActorHandle


@dataclass
class MultiSliceConfig:
    model_path: str
    tpu_type: str = "v5litepod-16"
    min_slices: int = 2
    max_slices: int = 8
    checkpoint_path: str = "gs://bucket/checkpoints"


class SliceCoordinator:
    """Coordinates multiple training slices.

    Manages slice membership, synchronization barriers, and gradient
    aggregation across a dynamic set of slices.
    """

    def __init__(self, config: MultiSliceConfig):
        self.config = config
        self.registered_slices: dict[str, dict] = {}
        self.current_step = 0
        self.barrier_count = 0
        self.gradient_buffers: dict[str, bytes] = {}

    def register_slice(self, slice_id: str, address: str) -> dict:
        """Register a new slice with the coordinator."""
        self.registered_slices[slice_id] = {
            "address": address,
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
        }
        return {
            "slice_id": slice_id,
            "current_step": self.current_step,
            "peer_slices": list(self.registered_slices.keys()),
        }

    def unregister_slice(self, slice_id: str):
        """Remove a slice from the coordinator."""
        self.registered_slices.pop(slice_id, None)

    def heartbeat(self, slice_id: str) -> dict:
        """Update heartbeat and return current cluster state."""
        if slice_id in self.registered_slices:
            self.registered_slices[slice_id]["last_heartbeat"] = time.time()

        return {
            "current_step": self.current_step,
            "peer_slices": list(self.registered_slices.keys()),
            "slice_count": len(self.registered_slices),
        }

    def barrier_enter(self, slice_id: str, step: int) -> bool:
        """Enter synchronization barrier. Returns True when all slices ready."""
        if step != self.current_step:
            return False

        self.barrier_count += 1
        return self.barrier_count >= len(self.registered_slices)

    def barrier_exit(self, slice_id: str, step: int):
        """Exit synchronization barrier."""
        if self.barrier_count >= len(self.registered_slices):
            self.current_step += 1
            self.barrier_count = 0

    def submit_gradients(self, slice_id: str, gradients: bytes):
        """Submit gradients for aggregation."""
        self.gradient_buffers[slice_id] = gradients

    def get_aggregated_gradients(self) -> bytes | None:
        """Get aggregated gradients if all slices have submitted."""
        if len(self.gradient_buffers) < len(self.registered_slices):
            return None

        # In practice, aggregation happens via JAX collectives
        # This is just for metadata coordination
        result = list(self.gradient_buffers.values())[0]
        self.gradient_buffers.clear()
        return result

    def get_cluster_state(self) -> dict:
        """Get current cluster state."""
        # Prune stale slices
        now = time.time()
        stale_threshold = 60  # seconds
        stale = [
            sid for sid, info in self.registered_slices.items()
            if now - info["last_heartbeat"] > stale_threshold
        ]
        for sid in stale:
            self.unregister_slice(sid)

        return {
            "step": self.current_step,
            "slice_count": len(self.registered_slices),
            "slices": list(self.registered_slices.keys()),
        }


def slice_worker_main(config: MultiSliceConfig, slice_id: str):
    """Training worker for a single slice."""
    import jax
    import logging
    logging.basicConfig(level=logging.INFO)

    resolver = fray_resolver()
    coordinator: ActorHandle = resolver.lookup("slice_coordinator")

    # Register with coordinator
    my_address = f"{jax.process_index()}@{os.environ.get('HOSTNAME', 'localhost')}"
    registration = coordinator.register_slice(slice_id, my_address)
    print(f"Slice {slice_id} registered, peers: {registration['peer_slices']}")

    # Initialize model
    from marin.training.trainer import Trainer
    trainer = Trainer(config.model_path)

    try:
        while True:
            # Heartbeat and get cluster state
            state = coordinator.heartbeat(slice_id)
            peer_count = state["slice_count"]

            if peer_count < config.min_slices:
                print(f"Waiting for more slices ({peer_count}/{config.min_slices})")
                time.sleep(5)
                continue

            # Synchronization barrier
            step = state["current_step"]
            while not coordinator.barrier_enter(slice_id, step):
                time.sleep(0.1)

            # Run training step with JAX collectives across slices
            # JAX handles the actual gradient synchronization
            metrics = trainer.train_step()

            # Exit barrier
            coordinator.barrier_exit(slice_id, step)

            if step % 100 == 0:
                trainer.save_checkpoint(f"{config.checkpoint_path}/step-{step}")

            print(f"Slice {slice_id} step {step}: loss={metrics['loss']:.4f}")

    finally:
        coordinator.unregister_slice(slice_id)


def coordinator_main(config: MultiSliceConfig):
    """Coordinator for multi-slice training."""
    import logging
    logging.basicConfig(level=logging.INFO)

    resolver = fray_resolver()
    server = FrayActorServer(resolver, port=0)

    coordinator = SliceCoordinator(config)
    server.register("slice_coordinator", coordinator)

    print(f"Multi-slice coordinator ready at {server.address}")
    server.serve()


class MultiSliceJob:
    """Manages a flexible multi-slice training job."""

    def __init__(self, config: MultiSliceConfig):
        self.config = config
        self.cluster = current_cluster()
        self.slice_jobs: dict[str, JobId] = {}
        self.coordinator_job: JobId | None = None

    def run(self):
        """Run multi-slice training with dynamic scaling."""
        # Create reservation for elastic scaling
        reservation = self.cluster.create_reservation(
            ReservationConfig(
                min_resources=ResourceConfig.with_tpu(
                    self.config.tpu_type,
                    slice_count=self.config.min_slices
                ),
                max_resources=ResourceConfig.with_tpu(
                    self.config.tpu_type,
                    slice_count=self.config.max_slices
                ),
                priority=1,
                preemptible=True,
            )
        )

        try:
            # Launch coordinator
            self.coordinator_job = self.cluster.launch(
                JobRequest(
                    name="multislice-coordinator",
                    resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
                    entrypoint=Entrypoint.from_callable(
                        coordinator_main,
                        args=(self.config,),
                    ),
                    environment=EnvironmentConfig.create(),
                )
            )

            # Wait for coordinator
            resolver = fray_resolver()
            for _ in range(30):
                try:
                    resolver.lookup("slice_coordinator")
                    break
                except Exception:
                    time.sleep(1)

            # Launch initial slices
            for i in range(self.config.min_slices):
                self._launch_slice(i, reservation)

            # Main loop: monitor and scale
            while True:
                self._check_slice_health()
                self._maybe_scale(reservation)
                time.sleep(10)

        finally:
            self.cluster.release_reservation(reservation)

    def _launch_slice(self, slice_idx: int, reservation_id):
        """Launch a new slice worker."""
        slice_id = f"slice-{slice_idx}"

        job_id = self.cluster.launch(
            JobRequest(
                name=f"multislice-worker-{slice_idx}",
                resources=ResourceConfig.with_tpu(self.config.tpu_type),
                entrypoint=Entrypoint.from_callable(
                    slice_worker_main,
                    args=(self.config, slice_id),
                ),
                environment=EnvironmentConfig.create(extras=["tpu"]),
            )
        )

        self.slice_jobs[slice_id] = job_id

    def _check_slice_health(self):
        """Check health of slice workers and restart failed ones."""
        for slice_id, job_id in list(self.slice_jobs.items()):
            info = self.cluster.poll(job_id)

            if info.status in (JobStatus.FAILED, JobStatus.STOPPED):
                print(f"Slice {slice_id} failed, restarting...")
                slice_idx = int(slice_id.split("-")[1])
                self._launch_slice(slice_idx, None)

    def _maybe_scale(self, reservation_id):
        """Scale slices based on reservation availability."""
        reservation = self.cluster.get_reservation(reservation_id)

        # If we have more resources available, add slices
        current_slices = len(self.slice_jobs)
        available_slices = reservation.allocated_resources.replicas

        if available_slices > current_slices < self.config.max_slices:
            print(f"Scaling up from {current_slices} to {available_slices} slices")
            for i in range(current_slices, available_slices):
                self._launch_slice(i, reservation_id)


# Example usage
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

All examples above work locally with minimal changes thanks to the LocalCluster.

```python
# tests/test_local_development.py
"""Example showing local development workflow."""

from fray.cluster import LocalCluster, current_cluster
from fray.rpc import LocalResolver, LocalActorServer
import os


def test_local_inference():
    """Test inference locally without a cluster."""
    # Set up local cluster
    os.environ["FRAY_CLUSTER_SPEC"] = "local"

    cluster = current_cluster()
    assert isinstance(cluster, LocalCluster)

    # Create a simple actor
    class MockInferenceActor:
        def predict(self, prompts):
            return [f"Response to: {p}" for p in prompts]

    # Register actor locally
    resolver = LocalResolver()
    server = LocalActorServer(resolver)
    server.register("inference", MockInferenceActor())

    # Look up and call actor (synchronous in local mode)
    handle = resolver.lookup("inference")
    result = handle.predict(["Hello", "World"])

    assert result == ["Response to: Hello", "Response to: World"]


def test_local_zephyr_pipeline():
    """Test Zephyr pipeline locally."""
    from zephyr import Backend, Dataset

    # Pipeline runs locally with threadpool backend
    pipeline = (
        Dataset.from_list([{"x": i} for i in range(100)])
        .map(lambda d: {"x": d["x"] * 2})
        .filter(lambda d: d["x"] > 50)
    )

    # Execute locally
    results = Backend.execute(pipeline, max_parallelism=4)
    assert len(results) == 75  # 26-50 doubled = 52-100, filter keeps > 50


def test_switch_to_cluster():
    """Show how code switches between local and cluster."""
    # This function works identically in both modes
    def run_pipeline(data: list[dict]) -> list[dict]:
        from zephyr import Backend, Dataset

        pipeline = (
            Dataset.from_list(data)
            .map(lambda d: {**d, "processed": True})
        )

        return Backend.execute(pipeline)

    # Locally
    os.environ["FRAY_CLUSTER_SPEC"] = "local"
    result = run_pipeline([{"x": 1}])
    assert result == [{"x": 1, "processed": True}]

    # On cluster (would connect to real cluster)
    # os.environ["FRAY_CLUSTER_SPEC"] = "fray://controller:50051"
    # result = run_pipeline([{"x": 1}])
```
