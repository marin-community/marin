# "Fray" -- distributed processing abstraction layer

_ (Better names wanted!)_

Fray provides an abstraction layer for the core distributed primitives needed
for Marin ML tasks. We developed Fray to give us optionality for working with
Ray versus other frameworks such as Monarch for task management. Below we
outline a long-term design for Fray, as well at a set of _baby steps_ we hope to
take along the way to get there. As no design survives contact with reality for
long, we're hoping to achieve some incremental improvements in our usability
while we learn more about what we actually need.

# Baby Steps

**Status**: The Cluster interface described below has been **implemented** in `fray.cluster`.
Available backends: `LocalCluster` (subprocess-based) and `RayCluster` (Ray job submission).
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

* Data processing (cpu)
* Launching TPUs (slices)
* Running inference (pools)
* RL (actors)

Our data processing has almost entirely moved over to `zephyr` at this point,
providing a clean Ray-free boundary for that work.

For launching TPUs, we propose lifting the existing Levanter TPU launching code
into fray, as a TpuJobRequest, and then providing a simple "cluster" API to
interact with jobs. This will serve as the kernel for our longer-term design
while still building on top of our Ray system.

## Example job request

```
job_request = {
  "slice": ...
  "environment": ...
}
# cluster translates this into a ray job request internally
cluster.run_job(job_request)
```

## Cluster interface

Our v0 cluster interface will have our standard contextvar access pattern with
methods to launch and check the status of jobs, and wait for job success:

class Cluster:
  def launch(job_request) -> id
  def monitor(job_id) -> status # wait for completion, spool stdout to waiter
  def poll(job_id) -> status

We'll define Ray and multiprocess backends, e.g. cluster/{ray,multiprocess.py}

The cluster launches a job and starts it running from a provided script
entrypoint/main function. It uses environment variables as needed to ensure that
the `job_context.py` code can auto-detect and use the correct job environment.
e.g. we might set `FRAY_ENVIRONMENT=ray:...` to indicate we're in a Ray
execution mode.

## Pools

Our thorniest Ray dependency is LLM inference, where we'd like to be able to
support scaling inference via pools of workers. To support this, we'll define a
WorkerPool abstraction that manages a set of individual jobs via the cluster
scheduler. A worker pool can manage jobs of any type, but we're mostly concerned
with TPU jobs.

The pool _controller_ creates a distributed queue by requesting it from the cluster:

cluster.create_queue(name: str) -> Queue

The Queue supports a standard distributed queue lease/pop interface:

Queue:
  push()
  peek()
  pop() -> Lease[T]
  done(Lease[T])
  pending() -> int

The Ray queue will use a Ray actor, the multiprocessing queue can use a helper
process.

The controller creates a queue, then issues create_job requests to the cluster,
providing the queue name as e.g. a command line flag to the users
worker_pool.py. The user code will typically listen on the queue for requests,
take a lease, apply e.g. inference, then push the inference result on a result
queue for retrieval.

## Actors (Job API)

Actors are stateful services that maintain state across multiple method calls.
They are used in a few places in Marin for distributed coordination. We may
phase them out but they are useful for compatibility with existing Ray code.

```python
from fray import fray_job_ctx

ctx = fray_job_ctx()

# Create an actor
actor = ctx.create_actor(
    MyActorClass,
    constructor_arg1,
    name="my-actor",
    get_if_exists=True,
    lifetime="detached",
    num_cpus=0
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

```
DeviceConfig = CpuConfig | GpuConfig | TpuConfig

class TpuConfig:
  types: list[TpuType] # list of acceptable TPUs to run on
  size: int # number of TPU chips required


class ResourceConfig:
  """Job resource worker configuration."""
  device: DeviceConfig = CpuConfig()
  ram: int
  disk: int
  cpu: int # measured in cores

  # how many instances of this resource to schedule
  # for accelerators, an instance may involve multiple hosts
  count: int
  min_count: int
  max_count: int

  # which regions is this job okay scheduling on, or anywhere if blank
  regions: list[str] | None

  # filters the target hosts must satisfy. for example, NON_PREEMPTIBLE
  # ensures your job will run on a persistent host
  constraints: dict[str, str]



class EnvironmentConfig:
  """An environment is either a workspace containing a pyproject.toml, or a docker image."""
  workspace: Url
  docker_image: str

class JobRequest:
  user: str
  name: str
  resources: ResourceConfig
  environment: EnvironmentConfig

class Cluster:
  schedule(request: JobRequest) -> JobId
  list() -> JobInfo
  status(id: JobId) -> JobInfo
  terminate(id: JobId)
```

### Job Scheduling

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

```
JobRequest(
  resources=ResourceConfig(
    device=TpuConfig(types=["v5e", "v6e", "v5p"], size="4x4"),
    ram="128g",
    disk="64g",
    cpu=64,
    count=1 # measure in device units, so one _slice_
  ),
  environment=...levanter.train_lm
)
```

### Jobs -> Execution

To run our jobs, we first need to allocate a controller job which will execute
the individual steps and monitor the progress. The controller does not perform
any direct work itself other than to dispatch sub-jobs. We launch the controller
with a zero CPU request to ensure it can always schedule so long as ram is
available on a non-preemtible machine.

```
controller = cluster.launch_job("marin.executor", job_list, cpu=0, ram="512m", constraints={fray.NON_PREEMPTIBLE})
```

The controller assembles individual job requests and submits them to the cluster
as their dependencies are available:

```
job = cluster.launch_job(stepN)
while True:
  cluster.status(job)
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

```
def tokenize():
  backend = zephyr.current_flow()
  ds = Dataset.from_files(...).map().flat_map().writer_jsonl()
  backend.execute(ds)
```

## Questions

With things like Ray for scheduling, we want the Ray scheduler to run on a
persistent host, and the rest of the cluster to run on the pre-emptible part.
How do we specify that? Ray doesn't make this easy? Or certainly not as easy as
just running ray.init() on all of the machines.

### Multi-part jobs?

One thought would be that jobs could have multiple simultaneous resource
requests with independent entry points, so you could have a Job that requests 2
separate sets of workers:

```
ray_controller = WorkerRequest(..., entrypoint="ray.main")
ray_worker = WorkerRequest(..., entrypoint="run_filter.main")
```

If any sub-worker failed, then the cluster would abort all of the workers and
cleanup.

Workers should be able to query the cluster to find the IP & port of running
jobs and find their controller and register with it, for example. So I guess could have a stub function which just registers the worker with Ray:

```
def ray_boot():
  ray_controller = find_controller(my_environment)
  ray.init(controller_address=ray_controller)
  time.sleep(forever)
```

How much do we care/need the resumability _within_ a Ray job, can we just
continue by resuming the whole job? This feels over-complicated, since most of
our tasks can be resumable if we checkpoint in the middle.

### RL and Actors

What about situations like RL, what's the scheduling setup? Curriculum is
currently shared across all of the RL tasks via a Ray actor, so how do we keep
that working?

We save our curriculum state with the training checkpoint, but that's
unnecessary and can be replaced by having the curriculum checkpoint itself.

Rollout workers and the trainer coordinate via a Ray actor for how to get
checkpoints, but this can likely be replaced with some kind of queue mechanism:
the trainer would pop from the queue, and rollout workers would peek.


