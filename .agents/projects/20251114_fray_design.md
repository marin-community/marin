# "Fray" -- distributed processing abstraction layer

_ (Better names wanted!)_

Fray provides an abstraction layer for the core distributed primitives needed
for Marin ML tasks. We developed Fray to give us optionality for working with
Ray versus other frameworks such as Monarch for task management. Below we
outline a long-term design for Fray, as well at a set of _baby steps_ we hope to
take along the way to get there. As no design survives contact with reality for
long, we're hoping to achieve some incremental improvements in our usability
while we learn more about what we actually need.

> **Note**: Some forward-looking concepts from this document (credit-based scheduling,
> WorkerPool, Resolver/ActorPool pattern) have evolved and are being implemented in
> `lib/iris/`. See `lib/iris/docs/fray-zero.md` for the next-generation design.

# Baby Steps

**Status**: The following has been **implemented**:
- Cluster interface in `fray.cluster` with `LocalCluster` and `RayCluster` backends
- Job Context API in `fray.job` with `SyncContext`, `ThreadContext`, `RayContext` backends
- Queue system with lease semantics (`MemoryQueue`, `FileQueue`, `HttpQueue`)
- Actor support for distributed coordination
- TPU orchestration migrated from Levanter (`run_on_pod`, gang scheduling, multislice)
- Resource configuration for CPU, GPU, and TPU
- Zephyr fully migrated to fray Job context

See the main README.md for usage examples.

Let's walk through what we're trying to accomplish. We're struggling with the
Ray cluster management system, as documented elsewhere, and we'd like
_optionality_ to from Ray to something like Monarch for our internal execution
primitives in the future.

Our job execution today is somewhat haphazard, with `ray_run` and `ray_deps` etc
and special cases for various Ray workarounds littering our codebase. We also
have our special purpose TPU-actor based system to work around Rays limitations
on gang scheduling.

Some short-ish term things we'd like to have access to:

* Cross cluster scheduling
* Better job isolation
* Worker pool and auto-scaling support

How can we build _towards_ our long-term design in an incremental way that gives
us some of these features in the short-term?

We have a few ways we use Ray now:

* Data processing (cpu) — **migrated to Zephyr**
* Launching TPUs (slices) — **migrated to fray.cluster.ray.tpu**
* Running inference (pools)
* RL (actors) — **supported via fray.job actors**

Our data processing has almost entirely moved over to `zephyr` at this point,
providing a clean Ray-free boundary for that work.

For launching TPUs, we propose lifting the existing Levanter TPU launching code
into fray, as a TpuJobRequest, and then providing a simple "cluster" API to
interact with jobs. This will serve as the kernel for our longer-term design
while still building on top of our Ray system. **Status**: This has been implemented
in `fray.cluster.ray.tpu` with `run_on_pod`, `run_on_pod_resumable`, and multislice
variants.

## Example job request

```python
from fray.cluster import current_cluster, JobRequest, ResourceConfig, TpuConfig

cluster = current_cluster()

job_id = cluster.launch(JobRequest(
    name="my-training-job",
    resources=ResourceConfig(
        device=TpuConfig(tpu_type="v5litepod-16"),
        ram="128g",
    ),
    entrypoint=train_fn,
))

cluster.monitor(job_id)  # streams logs, blocks until complete
```

## Cluster interface

Our v0 cluster interface will have our standard contextvar access pattern with
methods to launch and check the status of jobs, and wait for job success:

```python
class Cluster:
    def launch(job_request) -> JobId
    def monitor(job_id) -> JobInfo  # wait for completion, spool stdout to waiter
    def poll(job_id) -> JobInfo
    def terminate(job_id)
    def list_jobs() -> list[JobInfo]
    def wait(job_id, fail_on_error=False) -> JobInfo
```

We define Ray and local backends in `fray.cluster`:
- `LocalCluster`: Runs jobs as subprocesses, useful for development/testing
- `RayCluster`: Submits jobs via Ray's JobSubmissionClient

The cluster launches a job and starts it running from a provided script
entrypoint/main function. It uses environment variables as needed to ensure that
the `job_context.py` code can auto-detect and use the correct job environment.
e.g. we might set `FRAY_ENVIRONMENT=ray:...` to indicate we're in a Ray
execution mode.

## Queues

**Status**: The Queue interface described below has been **implemented** in `fray.queue`.
Available backends: `MemoryQueue` (in-process), `FileQueue` (fsspec-compatible),
`HttpQueue` (client for `HttpQueueServer`).

Our thorniest Ray dependency is LLM inference, where we'd like to be able to
support scaling inference via pools of workers. To support this, we define a
Queue abstraction for distributing work to a pool of workers.

The Queue supports a standard distributed queue lease/pop interface:

```python
class Queue[T]:
    def push(item: T)
    def peek() -> T | None
    def pop(lease_timeout: int) -> Lease[T]
    def done(lease: Lease[T])
    def release(lease: Lease[T])  # return to queue on failure
    def pending() -> int
```

The lease semantics ensure at-least-once delivery: if a worker fails to call
`done()` before the lease expires, the item is automatically returned to the
queue for another worker to process.

```python
from fray.queue import MemoryQueue, FileQueue, HttpQueue

# In-memory (for testing)
queue = MemoryQueue()

# File-based (works with GCS, S3 via fsspec)
queue = FileQueue("gs://bucket/queue-dir")

# HTTP client connecting to HttpQueueServer
queue = HttpQueue("http://localhost:8080", "my-queue")
```

The controller creates a queue, then issues create_job requests to the cluster,
providing the queue name as e.g. a command line flag to the users
worker_pool.py. The user code will typically listen on the queue for requests,
take a lease, apply e.g. inference, then push the inference result on a result
queue for retrieval.

> **Note**: The WorkerPool abstraction that manages pools of workers has evolved
> and is being implemented in `lib/iris/`. See `iris.client.worker_pool`.

## Actors (Job API)

Actors are stateful services that maintain state across multiple method calls.
They are used in a few places in Marin for distributed coordination. We may
phase them out but they are useful for compatibility with existing Ray code.

```python
from fray.job import create_job_ctx

ctx = create_job_ctx()

# Create an actor
actor = ctx.create_actor(
    MyActorClass,
    constructor_arg1,
    name="my-actor",
    get_if_exists=True,
    lifetime="detached",
    num_cpus=0  # Important: use 0 for actors on head node to avoid resource contention
)

future = actor.my_method.remote(arg1, arg2)
result = ctx.get(future)
```

Named actors enable workers to share the same actor instance:

```python
# Worker 1: Create
curriculum = ctx.create_actor(
    Curriculum,
    config,
    name="curriculum",
    get_if_exists=True
)

# Worker 2: Get same instance
curriculum = ctx.create_actor(
    Curriculum,
    config,  # Ignored if actor exists
    name="curriculum",
    get_if_exists=True
)
```

### Integration with Fray Primitives

Actor method results are compatible with Fray's put/get/wait:

```python
future = actor.compute.remote(data)
result = ctx.get(future)

futures = [actor.process.remote(i) for i in range(10)]
ready, pending = ctx.wait(futures, num_returns=5)
results = [ctx.get(f) for f in ready]

actor_ref = ctx.put(actor)
actor = ctx.get(actor_ref)
```

### CPU Allocation for Actors

For RL training, set `num_cpus=0` on actors (CurriculumActor, ArrowFlightCoordinator)
to avoid blocking job scheduling on the head node.

# Design

Fray provides 2 related interfaces for _cluster_ vs _job_ level APIs.

The `Cluster` API provides the ability to launch and manipulate jobs and monitor cluster status.
The `Job` API is used to manage _tasks_ and _objects_ inside of a job.

Jobs are isolated from each other, objects referenced within a given job cannot
be shared to another Job. To share information between Jobs, users must use an
explicit `Queue` or mirror data to an external source

## Clusters

A cluster is typically organized around a set of physical machines or VMs in a
region. The underlying backend may support re-use of VMs, but jobs are always
run in an isolated environment - they should never assume they have access to
previous VM state.

A job request consists of a device type, a set of resources, and an execution
environment to run. "Slices" provide gang-scheduling support for e.g. GPU or TPU
clusters, where all workers must run simultaneously.

```python
DeviceConfig = CpuConfig | GpuConfig | TpuConfig

class TpuConfig:
    tpu_type: str  # e.g., "v5litepod-16", "v6e-256"

class GpuConfig:
    gpu_type: str  # e.g., "a100", "h100"
    count: int

class ResourceConfig:
    """Job resource worker configuration."""
    device: DeviceConfig = CpuConfig()
    ram: str   # e.g., "8g"
    disk: str  # e.g., "10g"
    cpu: int   # measured in cores
    count: int = 1  # number of replicas

class EnvironmentConfig:
    """Environment configuration for the job."""
    extras: list[str]           # pip extras (e.g., ["tpu"])
    pip_packages: list[str]
    env_vars: dict[str, str]

class JobRequest:
    name: str
    resources: ResourceConfig
    environment: EnvironmentConfig
    entrypoint: Callable | Entrypoint

class Cluster:
    def launch(request: JobRequest) -> JobId
    def monitor(job_id: JobId) -> JobInfo
    def poll(job_id: JobId) -> JobInfo
    def list_jobs() -> list[JobInfo]
    def terminate(job_id: JobId)
    def wait(job_id: JobId, fail_on_error: bool = False) -> JobInfo
```

### Job Scheduling

> **Note**: The credit-based scheduling system described below has not been
> implemented in Fray. Advanced scheduling features are being developed in
> `lib/iris/`. The current implementation uses Ray's native scheduling.

A cluster manages a set of underlying VMs and makes global scheduling decisions
based on a credit system. Users are assigned an initial credit amount which is
used as a "bid" for their job to schedule. Jobs with higher bids are
preferentially scheduled to resources. As a user consumes more resources, future
job requests are made with lower bids, automatically deprioritizing large sweeps
over individual runs.

The cluster only assigns jobs to running pools of VMs, it will not start a new
slice of VMs for a specific job.  If the requested set of jobs exceeds the
cluster capacity it use the underlying VM manager, e.g.  GCP, to request more
VMs, up to a pre-configured maximum.


## Walking through a typical experiment

How do these abstractions work together in the course of a typical Marin
experiment? Let's walk through an example where we want to train our model on a
new dataset. We'll need to filter and tokenize our dataset, and then train on
the resulting tokenized output.


### Steps -> Jobs
We express this pipline of operations as a set
of `ExecutorSteps` stemming from our initial job:

```python
download = download_dataset("cool_fray"),
filtered = filter_bad_data(download)
validation_data, training_data = tokenize(filtered)
trained_model = train(training_data, validation_data)
evaluation = evaluate(trained_model)

steps = [download, filtered, tokenize, trained_model, evaluation]
```

If we dig into these steps, we'll typically express each as a distinct
`JobRequest` which will be sent to our cluster. For example, our
`download_dataset` task will construct a job request like:

```python
download_req = JobRequest(
  resources=ResourceConfig(ram="1g", disk="1g", cpu=16, count=1),
  environment=EnvironmentConfig(
    workspace=local_workspace(),
    entry_point="marin.datasets.download.download_hf",
    entry_point_args=["cool_fray"],
  )
)
```

Our download job doesn't require a lot of hosts or CPU, and streams the output
to GCS, so we can keep our request small. Once our download completes, we'll
want to schedule our filter and tokenizer steps. These are similar, they need
more resources but don't require an accelerator:

```python
filter_req = JobRequest(
    resources=ResourceConfig(ram="8g", disk="16g", cpu=1, count=128),
    environment=...
)
tokenize_req = JobRequest(
    resources=ResourceConfig(ram="8g", disk="1g", cpu=1, count=128),
    environment=...
)
```

Finally our accelerator job requires a more complex configuration to specify the
acceptable device types and slice size:

```python
JobRequest(
  resources=ResourceConfig(
    device=TpuConfig(tpu_type="v5litepod-16"),
    ram="128g",
    disk="64g",
    cpu=64,
    count=1  # one slice
  ),
  environment=EnvironmentConfig(extras=["tpu"]),
  entrypoint=train_fn,
)
```

### Jobs -> Execution

To run our jobs, we first need to allocate a controller job which will execute
the individual steps and monitor the progress. The controller does not perform
any direct work itself other than to dispatch sub-jobs. We launch the controller
with a zero CPU request to ensure it can always schedule so long as ram is
available on a non-preemtible machine.

```python
controller = cluster.launch(JobRequest(
    name="executor",
    resources=ResourceConfig(cpu=0, ram="512m"),
    entrypoint=executor_main,
))
```

The controller assembles individual job requests and submits them to the cluster
as their dependencies are available:

```python
for step in ready_steps:
    job_id = cluster.launch(step.job_request)
    cluster.wait(job_id, fail_on_error=True)
```

As a future enhancement, we may allow execution steps to be submitted to the
cluster simultaneously as a DAG, allowing the user to "fire-and-forget" job
requests.

### Execution -> Job Environment

The cluster hands off to us by running our entry point script on all tasks
requested by our job. We now need to boot up our local environment and hand off
control to the user program. Let's walk through this for our `Ray` job backend.

```python
# jobs/ray_backend.py

def boot_ray(user_entrypoint: str, user_args: list[Any]):
  workers = os.environ["FRAY_WORKERS"].split(";")
  ray.init(controller="ray://{workers[0]}")
  fray.set_job_backend(ray_backend(workers))

  # if we're the primary worker, call into the users entrypoint to start ray processing
  if os.environ["FRAY_INDEX"] == "0":
    user_module = importlib.import(user_entrypoint)
    user_module(*user_args)
  else:
    time.sleep(86400)
```

The Fray controller sets environment variables to tell us about our cluster
environment. We initialize Ray by assuming the first worker will be the
controller, and then trampoline to call the underlying user entrypoint. A user
entrypoint typically uses Fray to do some processing:

```python
def tokenize():
  backend = zephyr.current_flow()
  ds = Dataset.from_files(...).map().flat_map().writer_jsonl()
  backend.execute(ds)
```

## TPU Orchestration

**Status**: Implemented in `fray.cluster.ray.tpu`.

For TPU jobs requiring gang scheduling:

```python
from fray.cluster.ray.tpu import run_on_pod_resumable, run_on_pod_multislice_resumable

# Single slice with automatic retry on preemption
result = run_on_pod_resumable(
    my_tpu_function,
    tpu_type="v5litepod-16",
)

# Multislice training
result = run_on_pod_multislice_resumable(
    my_multislice_function,
    tpu_type="v5litepod-16",
    num_slices=4,
)
```

The TPU orchestration layer handles:
- Gang scheduling (all hosts in a slice must run simultaneously)
- Preemption detection and automatic retry
- Multislice coordination via MEGASCALE environment variables
- `OwnerDiedError` classified as preemption for retry purposes

## Questions

~~With things like Ray for scheduling, we want the Ray scheduler to run on a
persistent host, and the rest of the cluster to run on the pre-emptible part.
How do we specify that?~~

**Resolved**: Launch the Ray head node on a non-preemptible worker. For actors
that need to persist (like CurriculumActor), use `num_cpus=0` to avoid resource
contention on the head node.

### ~~Multi-part jobs?~~

**Resolved**: This concept has evolved into Iris's co-scheduling with failure
domains. See `lib/iris/docs/iris-coscheduling.md`.

### ~~RL and Actors~~

**Resolved**: RL training uses Fray actors with `num_cpus=0` for CurriculumActor
and ArrowFlightCoordinator. This prevents actors from blocking job scheduling.
StatusActor was replaced with a file-based semaphore pattern. RobustActor was
replaced by launching critical processes on non-preemptible workers.

---

## Change Log

**Jan 2026 Update:**
- Updated status to reflect implemented features
- Added notes where concepts have evolved into Iris
- Documented Queue system implementation (MemoryQueue, FileQueue, HttpQueue)
- Documented TPU orchestration (run_on_pod, multislice support)
- Added `num_cpus=0` guidance for actors
- Marked resolved questions

For detailed history, see PRs #1985, #2014, #2121, #2145, #2176, #2258.
