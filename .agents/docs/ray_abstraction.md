[Russell Power](mailto:russell.power@openathena.ai)
Sep 29, 2025

*I could not think of a tidy quote here, but think of a gentle light shining upon the darkness. And there you are.*

# Background

Previous rants:

[Ray Infrastructure Challenges](https://docs.google.com/document/d/1gtCz3aN2q72ZF-BNK_nKHCjuS9CT88QSH5vmkTktwWQ/edit?tab=t.0#heading=h.9k9q6db2omrh)

We currently rely on [Ray](https://github.com/ray-project/ray) for distributed execution in Marin. And to be honest, we’re struggling with it. Ray is deceptively simple to get started with, with its global namespace, automatic cluster management and auto-pickling of functions. But the interface and implementation for Ray are causing us to lose hours of time daily.

Some of our current issues include:

* Ray doesn’t properly isolate jobs from each other. Instead packages reuse the same virtual environment & inherit from the system environment. This means that a job might work one day (because someone else installed the dependency ahead of time), and not the next. Or a job might fail because an incompatible version of a dependency was installed.
  * We are seeing issues with this *multiple times per day*. Unlike say, a Docker environment where a user can get a version working and have it be consistent across runs, we are finding jobs failing seemingly at random.
* Ray’s Python API is exposed via globals, e.g. ray.get. This results in all sorts of contortions in the Ray code (and our code) to handle initialization, and it’s painfully easy to accidentally connect, for example, to your production cluster when trying to run a unit-test. It also makes mocking out the Ray interface for testing almost impossible.
* Ray doesn’t support the gang-scheduling required by TPUs and as a result we have had to develop fragile and [elaborate workarounds](https://github.com/marin-community/levanter/blob/main/src/levanter/infra/ray_tpu.py) to schedule multi-host TPU workloads. These workarounds obscure scheduling decisions and job tracking and are a frequent source of failures.
* Ray’s job scheduling is deeply inconsistent and opaque. A job requesting 1 TPU on a system with 8 free may schedule immediately, or it may hang indefinitely. We have no visibility into the scheduler to understand why the scheduling failed or how it’s making a decision. We often just reboot clusters when they become “stuck”, but that’s generally only after losing a day of productivity.
  * We’ve even written our own job restoration logic to avoid losing jobs across restarts.
* Ray’s object store relies on distributed reference counting, and the implementation appears to be lazy or broken. References easily accumulate over time. This may be acceptable for jobs with a short-lifespan or small numbers of objects, but it makes it useless for storing larger amounts of data.
* Ray exposes 3 separate services (“dashboard”, “API”, and “GCS”). It is unable to decide itself which to use. For instance, ray job list uses the API address, while ray list jobs uses the dashboard. Some limitations are bizarre: you can’t create actors and list them using the same connection, for instance. These limitations don’t bite you all the time, but it’s easy to accidentally create an actor and not be able to destroy it.
* Ray doesn’t isolate worker or client processes effectively. This means that client processes can do things like [hold onto the TPU lock](https://github.com/marin-community/levanter/blob/8d457ce7944b383454989bf943268063f7c9073c/src/levanter/infra/ray_tpu.py#L1227) between invocations, resulting in awkward hacks on our side to try to clean up whenever possible.
* Ray is developed using C++ and Bazel which makes forking deeply unappealing.

Before we continue let’s acknowledge what Ray is good at:

* Ray is radically *easy to get started with*. You don’t need to write a complex cluster configuration or learn a new language, or cobble together a bunch of different tools. You don’t need to manage a lot of application state or context.

  You get a common set of functionality that users want out-of-the-box, primarily a distributed storage layer and the ability to launch tasks in parallel. While the fault-tolerance story for Ray might not be amazing, *that’s not what most people need*. Python users in particular have lacked tools to write these types of systems.

* Ray has *reasonable* performance for certain types of operations. The core of Ray is in C++ and it relies on well-known tooling like gRPC, making the core of the system (reasonably) robust.
* Ray has bolted on a number of extra features to this core set which are appealing. These include a web dashboard, log management, cluster auto-scaling and data & serving libraries. These extras often don’t work very well: Ray can’t figure out where logs come from, the web UI for jobs is terrible, auto-scaling is slow, etc. But they’re better than nothing.

Fundamentally we’ve hit the limits of Ray’s implementation and the design doesn’t give us many easy exit points: we can’t simply replace one broken piece, as the API is global and interconnected. We’re losing person-days, every day, due to issues with isolation, job scheduling, health checking and more. We need to fix these issues somehow.

# Whereto

*Repair, or replace?*

We outlined some options in [Ray Infrastructure Challenges](https://docs.google.com/document/u/0/d/1gtCz3aN2q72ZF-BNK_nKHCjuS9CT88QSH5vmkTktwWQ/edit), but let’s go through them again here. Roughly we can try to fix the issues in Ray ourselves, or we can try to replace it with our own infrastructure or by building on existing work. We don’t expect these issues to be fixed by upstream in a timeframe we care about – our usage is outside of their norm and they have other priorities.

## Repair

The most direct approach to our issue is to directly fix some of the underlying implementation issues. While any ability to change the Ray *API* – to address issues like global state or implicit references/initialization –is unlikely, we could try to improve some of our most pressing issues. Ray is actively developed and while it took some time to get feedback on a simple PR like [\#57109 \- Loosen Ray Self-Dependency](https://github.com/ray-project/ray/pull/57019), we could likely move faster as they became comfortable with us.

A few things we could try to do ourselves

* Implement package isolation in the Ray worker runtime
* Fix logging to properly track to worker processes and add useful filtering
* Figure out the distributed reference counting bug or add manual management
* Fix the cluster provisioning to properly dispose of preempted workers
* Fix workers to automatically recycle in the case of disk space or hardware failures (health checks)

Less likely to be feasible:

* Add context API to avoid accidentally triggering Ray instantiation
* Fix port management
* Cross-region management

There’s an obvious upside here: we’d be directly and incrementally helping other users along the way, and any improvements would in theory benefit us immediately without requiring us to change existing Marin code. Going this route minimizes the amount of churn for Marin.

There are challenges:

* Ray is a pain to develop in. The build process is complex enough that their own getting started guide provides shortcuts to avoid dealing with it when working with certain types of code: [Building Ray from Source — Ray 2.49.2](https://docs.ray.io/en/master/ray-contribute/development.html#building-ray-python-only)
* The velocity with working with the Ray developers is unclear.
* Ray is now commercialized in the form of their AnyScale platform, and it’s unclear how receptive the team would be to adding features that duplicate their commercial work (e.g. improved cluster management or multi-clustering)
* Given the scope of Ray, our ability to make deep changes would be severely limited.

**Investigate AnyScale as an alternative**

* Requires self-hosting GKE

## Replace

If we can’t fix Ray, we can try to replace it, or at least parts of it. We can first try to reduce the scope we use it into situations where we know it works better. For instance, once you have a cluster, Ray’s actor references work acceptably. In effect, we want to define a “Ray-shaped box”, migrate our code to use that box, and then see if we can replace the implementation with something that works better for us.

### Ray Usage Analysis

Let’s analyze how we use Ray today. Marin uses 2 different “modes” of Ray today (Ray doesn’t cleanly divide this functionality in its API, but they are distinct functionally and in the implementation \- e.g. you don’t use the same port for job scheduling as you do for creating objects).

* Cluster management (job scheduling, autoscaling)
* In-task utilities
  * Object storage
  * Actor creation & reference
  * Task creation

Almost all usage of Ray in Marin first launches one or more jobs, and then uses the task level interfaces within Ray to schedule work within the “job context”. We don’t tend to launch new jobs at the “leafs” of our computations. Instead, a typical experiment launches jobs at the “top-level”, and then everything inside of that simply uses the Ray task level features:

```
def experiment_entry_point(config):
  trainer = create_tpu_cluster(config.tpu_type, config.tpu_slice_size)
```

(A notable exception is tokenization & data processing, which we discuss in more detail below). As mentioned above, Ray’s global heavy API makes direct migration difficult: matching the exact API and behavior for all usages at the same time would be frustrating and confining. Instead, as a first step, we could try to “de-Ray” our code base by introducing an intermediate abstraction layer which provides interfaces we want to supply (but still backed by Ray), and then migrate our code base to it.

A quick breakdown of how we use Ray in Marin:

| API / Feature | Count | Notes |
| :---- | :---- | :---- |
| @ray.remote decorators | 160 | Task & actor definitions |
| .remote() calls | 18 | Task/actor invocations |
| ray.get() | 227 | Retrieving task results |
| ray.wait() | 37 | Waiting for tasks with progress |
| ray.put() | 2 | Object store writes |
| ray.init() | 3 |  |
| Ray actors | \~15 | Long-lived stateful services |

Our dominant API usage is “launch a remote function and wait for the result” \- we have limited use of the ray object store directly or any other parts of the API. Let’s say our initial goal was to mock out the Ray API for testing. We’d also like to clearly separate out the “resource scheduling” from “task execution” to facilitate using alternative backends (e.g. Slurm or our own manager). This goal is complicated by Ray’s task design.

Ray fudges the job/task boundary \- tasks can actually schedule anywhere in the cluster. This is convenient: you can think of tasks almost like “threads”. A task isn’t limited to using resources from its job \- it can request any resource it wants, and Ray will attempt to satisfy it from the cluster.

This is easy to use and works fine for single-purpose clusters, but is harder to manage with multiple users (as we are increasingly finding out), and doesn’t scale down well (data-processing on your laptop will OOM or run out of resources for no good reason).

Our main/only use of these free-floating tasks is in the [Levanter data processing pipeline](https://github.com/marin-community/levanter/blob/main/docs/design/Data-Loader-Design.md). There, tasks are kicked off to effectively fill the data cache on-demand. This pipeline works fine, so rewriting it as a prerequisite for porting is unattractive. One caveat is that the pipeline was designed with the assumption that users would want to be able to “fast start” \- start training without having to wait for the data processing to stop. We can likely relax this constraint in practice:

* We can process data fast enough that this isn’t likely a big issue
* Users can always start trial runs on a smaller part of the dataset
* *Not looking at the data you’re training on seems nuts*

The fast-start design also critically limits the data processing pipeline to be effective a map-only implementation: any type of preprocessing that requires a shuffle (for example deduplication) is incompatible with fast-start anyway, and ends up needing to be run out of band.

All that said any alternative we provide needs to:

* Provide similar auto-scaling functionality
* Be easily implemented on top of existing Ray functionality (during the porting window)
* Give us more implementation flexibility than the Ray: we don’t want to be compelled to re-implement Ray’s API exactly.

According to [David Hall](mailto:david.hall@openathena.ai), we should be able to hide most of this functionality behind a straightforward `map` call which runs a task across a range of inputs.

### API Design and Migration

So to recap, we want an API that:

* Effectively encapsulates existing Ray usage
  * Supports enough Ray-like functionality that we can easily port our existing work to it without major modifications
  * But not a carbon-copy of Ray, locking us into the same patterns
* Splits the demands of cluster management, which Ray is bad at, from task management, which Ray is merely mediocre at.
* Ideally allows us to address Ray scaling & management pain points early

As part of migrating to this new API, we will also likely want to have incremental gains. It’s all well and good to have our fancy new system but if it all lands 16 months from now we’ve missed the window where it can actually be useful. There’s a famous 2-pager from Luis Barroso (RIP) from inside of Google, with the best snapshot I can find here (page 4):

* [https://fontoura.org/papers/barroso.pdf](https://fontoura.org/papers/barroso.pdf)

And of course, from Joel Spolsky:

* [Things You Should Never Do, Part I](https://www.joelonsoftware.com/2000/04/06/things-you-should-never-do-part-i/)

So rather than search for perfection with our API work, we want to first get something integrated \- *even if it just looks like a clone of Ray* – and then iterate on it as we go. Python doesn’t facilitate “fearless refactoring”, but our move to a mono-repo will dramatically ease our ability to make cross-cutting changes as we go, and each change can potentially improve the QOL for everyone using Marin.

So below we sketch out our initial “Ray-clone” \- Lux, for lack of a better term \- followed by some potential short-term improvements that we can make.

Our initial API will look just like Ray, with some mechanical changes to better reflect our sensibilities: we view resource allocation as a separate behavior from task management, so we want to show that in the API. This won’t affect our initial usage or implementation.

So our first design will:

* Define a Ray-like set of behaviors accessible via “context” objects
* Splits the job and task behavior into separate classes

By switching to a context object (which would be accessed via e.g. Python’s `ContextVar` paradigm), we gain the ability to swap out the implementation without changing user code.

Our implementation plan would then ping-pong between making API changes and then adapting our implementation to reduce or replace the use of Ray.

1. Design and test out the fake-ray (“fray”) API
   1. Write local-only and Ray based backend
2. Mechanically adopt the new API in the Marin codebase
   1. Manual updates may be required for a few complex areas (ray-tpu, ray-run)
3. **Switchover to the testing backend for unittests**
4. Define new “map” API to replace fork-join task launching
   1. Build on top of Ray with either ray-data or load-balancing implementation
   2. Implement a new `map` backend which can launch in a separate auto-scaling job to request additional resources
5. **Switch to an alternative job manager**
   1. Investigate Slurm, or write our own to reflect our needs (e.g. multi-cluster, pre-emption aware, etc)
   2. [Xmanager is dead](https://github.com/google-deepmind/xmanager), but we can [steal the ideas](https://storage.googleapis.com/gresearch/xmanager/deepmind_xmanager_slides.pdf). (Not mentioned in that deck is the credit mechanism to ensure fair sharing across users).

If we can get to (5), we’d have addressed our most pressing issues with Ray. Beyond that, there’s many opportunities to further tune the API or replace the Ray task implementation to get better performance.

1. Update object storage with better checkpointing & persistence ideas
   1. Many usages of actors could be replaced with a KV-table and CRDTs
   2. Ray Actors lose state on restart, we could provide stateful actors to address this
2. Better data abstractions (some of these are available from [Ray Data](https://docs.ray.io/en/latest/data/data.html) but they inherit all of the usual Ray problems)

**I won’t pretend this is the most *fun* way to approach our problem.** I’d much rather go off and build something new and shiny and then figure out how to integrate it. But there isn’t anything new and shiny that we can use, off the shelf, that we can easily drop-in to replace our usage of Ray. And going off and building our new thing would take engineering-months that we desperately need to work on all of the existing Marin infrastructure.

A little more detail on the first step, as we’ll likely adjust our thinking as we go beyond there:

### Phase 0: Fray – Fake-Ray

```py
# Describes the environment we want to run our _job_ in.
# All tasks share the same environment and resources from their job.
class RuntimeEnv:
  package_requirements: list[str]
  minimum_resources: list[Resource]
  # for autoscaling
  maximum_resources: list[Resource]
  env: dict[str, str]

# A job context provides access to the object store and task scheduling facilities
class JobContext:
  def lookup(service: str) -> Service
  def new_task(fn, runtime_env) # create a task in this job
  def create_actor(klass, env)
  def put_object(obj) -> str
  def get_object(id) -> obj

# entrypoint for a job
EntryPoint: Callable[[JobContext], []]

# The cluster context allows requesting resources and launching jobs.
# A job is a set of resources and an isolated execution context.
# Tasks within a job are launched with the JobContext::remote_call
class ClusterContext:
  def list_jobs
  def delete_job
  def create_job(entry_point, env)

cluster = make_cluster()
```

With this API, we can semi-mechanically replace all usages of Ray with equivalent “fake-Ray” API calls. For example, `ray.remote` translates to `fray.get_job_ctx().remote` , mutatis mutandis for the other APIs. Once we have Levanter landed into our mono-repo, this is a task we can easily handle with an agent and adequate testing.

With this new API, we can capture some initial benefit by simply defining a purely local/testing implementation of our API:

```
class LocalClusterContext
class LocalJobContext

# do the obvious thing with launching threads and storing to a locally managed table
```

A test context is obviously useful for unittesting, but it would also avoid a common failure when running scripts on a node which happens to have an existing Ray worker on it (this occurs when we allocate a [development TPU](https://github.com/marin-community/marin/blob/main/scripts/ray/dev_tpu.py) now). We have many parts of our code base which might use Ray “just for a minute” \- e.g. they might run something in a task context just in case they are running a distributed job, but don’t need the distribution by default. With our current setup, this results in a weird and hard to debug error.

If we by default instantiate a local context, and only boot up a Ray/distributed context when explicitly running in a Ray job, we can eliminate this frequent source of annoyance.

## Appendix

Notes from discussion with David:

We have jobs today that do something like:

* Run training on tpu-v4-8
* Want to get data from some dataset
* The dataset requests { CPU: 100000 }
* Ray will find CPUs from the cluster to accommodate this
* Or autoscale
* How do we balance this if we have a job/task separation?

**David**

I have a chunk of work that needs 8 CPUs and 1 GPU, and I need to run 400 of these
That model is useful
How does Beam handle this?
YOu would typically run the GPUs on a server and talk to them

**Reserved nodes**

We have reserved nodes now
Zak wants them back
How do we deal with Sam wanting some TPUs and other person X?
How do we deal with prioritization
Only a certain amount of reserved compute
Hand-roll xmanager
Credit allocation & bid system

**Fundamental units we need**

* Gang jobs for TPUs jobs, ideally with flex multislice
* Map-reducing type things for scheduling elastic work with trivial reduction
  * More complicated now because we wanted eager dataset usage
* RL with async trainer/rollout

**Questions**

How to deal with @ray.remote , switch out with our own decorator?

* Almost all already inside of a function
* Global but the only user is a wrapper function that hides the fact it’s a ray.remote
* Might be able to just replace all of these with inline ctx.remote calls

How to thread context through existing code?

* [ContextVar](https://docs.python.org/3/library/contextvars.html) ?
  * distributed.get\_active\_context().remote
  * Automatically injected by the runtime before entrypoint for the job
  * Thread-local etc

Handling CPU \+ TPU jobs?

* Define map-reduce like interface
* Or define Service abstraction & lookup
* Turn existing Levanter task/actor into separate job \+ service
* Autoscale that service and fetch results
* map\_files\_in\_directory
