# Migration to Fray

Fray is our new library -- see @lib/fray/docs/design.md -- for abstracting job and task scheduling.

We'd like to replace our usage of Ray with Fray throughout the Marin code base.
We'll first work on drop-in replacement and cleanup of Ray references before
moving on to more serious refactoring.

## Plan

Our usage of Ray/Fray has 2 components as mentioned in the design. We launch
"jobs" which create new environments and request resources, and we also launch
"tasks" inside of a job which consume job resources and share job containers
etc.

We'll migrate these 2 types of functionality independently.

### Migrate resource configuration

We'll first move our metadata - our resource specifications - to use Fray. This
is now completed in https://github.com/marin-community/marin/pull/2154 . This
uses the Fray system of specifying resources consistently throughout our code base.

### Migrating job launches

Now that we've migrated our resources, we can switch from using
explicit ray.remote / ray_run calls for running jobs to instead running jobs
with Fray. We'll do this system by system, testing and committing our work as we go.

## Migrating Tips

When migrating, consider options in priority order:

### Remove the target code entirely

Is this dead code? Research the code base to see if it has active users. If not,
remove it entirely.

### Remove the Ray dependency entirely.

Some Ray dependencies are _unused_ in the codebase, or the code doesn't actually
need to use Ray.  For example, if a function is decorated with a plain `@ray.remote`
but all of it's callers are _also_ marked `ray.remote`, then this call is typically
a no-op, and can be removed.


### Convert the Ray code to use Zephyr

If the code in question is data processing, e.g. loading or manipulating files,
it should be converted to use Zephyr for the top-level concurrency instead.

See examples in transform_dclm_hq.py or dedupe.py for how to use Zephyr, as well
as @.agents/docs/zephyr-migration.md.

### Replace ray.remote tag with equivalent Fray code.

Fray doesn't have a "implicit" remote execution model like Ray does.  Instead,
you must request the current job context and use it to schedule tasks. We thus
replace `@ray.remote` tags with calls from the call site:

```python
ctx = fray_job_ctx()
future = ctx.run(remote_function, arg1, arg2)
assert ctx.get(future) == 10
```

The simplest cases involve bare ray.remote calls and can be handled as above.

More complex cases require analysis.

#### Runtime Environment

If a runtime_environment is specified in order to install packages or set environment variables,
launch a new Job with the appropriate JobRequest:

```python
@ray.remote(runtime_env=build_runtime_env_for_packages(...))
def foo():

```

Becomes:

```python
from fray import JobRequest

# foo just uses packages as needed
def foo():
    import package_x
    import package_y

def foo_caller():
    request = JobRequest(
        name="foo",
        entrypoint=Entrypoint(callable=foo),
        resources=ResourceConfig(replicas=1),
        environment=create_environment(pip_packages=["package_x", "package_y"]),
    )
    ctx = current_cluster()
    job_id = ctx.launch(request)
    ctx.wait(job_id)

### Resource requirements

Fray breaks up job and task scheduling into separate concerns. We have 2 places this impacts our changes:

#### Head node scheduling

Scheduling on "head" node. This is typically used to place a controller process
which should not be preempted. In Fray, this is expressed by putting a
requirement for: "non-preemptible" in the ResourceConfig for a JobRequest.

#### TPU scheduling

TPUs should be scheduled like any other job, using a JobRequest. The TPU type
and slice size are specified in the TpuConfig section of the ResourceConfig. For
"multi-slice" operation, specify replicas > 1:

```python
 resource_config = ResourceConfig(
        cpu=1,
        ram="16",
        disk="10g",
        device=TpuConfig(type="v5litepod-4"),
        replicas=1,
        regions=["eu-west4"],
    )

    job_request = JobRequest(
        name="vllm-inference-pool",
        entrypoint=Entrypoint(
            callable=vllm_server_worker,
            function_args={
                "model": self.config.model_config,
                "request_queue": self.request_queue,
                "response_queue": self.response_queue,
            },
        ),
        resources=resource_config,
        environment=create_environment(),
    )

    ctx = current_cluster()
    job_id = ctx.launch(request)
    ctx.wait(job_id)
```

### Ray actors

TBD

### Inference tasks

TBD


## Package Migration

We'll migrate one package at a time:

* generation
* execution
* rl
* datashop
* evaluation
* resources
* training
