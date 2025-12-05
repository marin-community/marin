# Migration to Fray

Fray is our new library -- see @lib/fray/docs/design.md -- for abstracting job and task scheduling.

We'd like to replace our usage of Ray with Fray throughout the Marin code base.
We'll first work on drop-in replacement and cleanup of Ray references before
moving on to more serious refactoring.

## Plan

We'll proceed task by task:

* generation
* execution
* rl
* datashop
* evaluation
* resources
* training

When migrating, consider options in priority order:

### Remove the Ray dependency entirely.

Some Ray dependencies are _unused_ in the codebase, or the code doesn't actually
need to use Ray.  For example, if a function is decorated with a plain `@ray.remote`
but all of it's callers are _also_ marked `ray.remote`, then this call is typically
a no-op, and can be removed.

If a section of code is entirely unused, you may remove it entirely.

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

```

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

## Files Requiring Migration

This section catalogs all files in `lib/marin/` that use Ray, organized by migration strategy.

### Convert to Zephyr

#### `lib/marin/src/marin/generation/inference.py`
Text generation inference using Ray Data pipelines for vLLM inference on TPUs.
**Approach:** Convert Ray Data pipeline to Zephyr for concurrent data processing.

#### `lib/marin/src/marin/datashop/pipeline.py`
MEDU data generation pipeline using Ray Data for concurrent vLLM inference with placement group scheduling.
**Approach:** Delete MEDU related files, they are not used.

### Replace ray.remote with Fray Job Context

#### `lib/marin/src/marin/classifiers/bert/training.py`
BERT model training on TPU pods using `@ray.remote()` with TPU resource requirements.
**Approach:** Replace `@ray.remote` with Fray `ctx.run()` within a Job context, specify TPU resources in `JobRequest`.

#### `lib/marin/src/marin/classifiers/fasttext/training.py`
fastText model training using `@ray.remote()` for remote function execution.
**Approach:** Replace `@ray.remote` with Fray task execution using `ctx.run()`.

#### `lib/marin/src/marin/evaluation/log_probs.py`
Language model evaluation using Levanter with minimal Ray usage.
**Approach:** Remove unused Ray import; ensure Levanter configuration works with Fray context.

#### `lib/marin/src/marin/rl/rl_job.py`
High-level RL job coordinator orchestrating training and rollout workers on TPU pods.
**Approach:** Already partially migrated to Fray via `run_on_pod_ray`; complete by replacing Ray actor management with Fray equivalents.

#### `lib/marin/src/marin/rl/evaluate_environment.py`
Environment evaluation for RL running inference server on TPU.
**Approach:** Already uses `run_on_pod_ray`; ensure full Fray integration.

#### `lib/marin/src/marin/training/training.py`
Training coordination using Fray's `run_on_pod` for TPU pod execution.
**Approach:** Already using Fray; verify full migration is complete.

### Runtime Environment Handling

#### `lib/marin/src/marin/run/ray_run.py`
Ray job submission CLI tool submitting jobs to Ray cluster with custom runtime environment and resources.
**Approach:** Replace `JobSubmissionClient.submit_job()` with Fray `ctx.launch(JobRequest)`.

#### `lib/marin/src/marin/evaluation/evaluators/levanter_tpu_evaluator.py`
Launches LM evaluation on TPUs using `@ray.remote()` with runtime environment and TPU resource specs.
**Approach:** Convert to Fray `JobRequest` with TPU `ResourceConfig` and environment configuration.

#### `lib/marin/src/marin/evaluation/evaluators/vllm_tpu_evaluator.py`
Launches vLLM-based evaluation on TPUs with runtime environment and resource requirements.
**Approach:** Convert to Fray `JobRequest` with TPU `ResourceConfig`.

#### `lib/marin/src/marin/classifiers/hf/launch_ray_training.py`
Hugging Face model training on TPUs using `@ray.remote()` with runtime environment.
**Approach:** Convert to Fray `JobRequest` with TPU resources and environment configuration.

### Resource Requirements Handling

#### `lib/marin/src/marin/resources.py`
Abstract `ResourceConfig` protocol defining hardware resource specifications with `as_decorator()` returning ray.remote decorators.
**Approach:** Update protocol to work with Fray `JobRequest` resource specifications instead of Ray decorators.

#### `lib/marin/src/marin/generation/ray_utils.py`
Utility for TPU tensor parallel scheduling creating placement groups for distributed TPU execution.
**Approach:** Move to Fray resource scheduling configuration.

#### `lib/marin/src/marin/utilities/ray_utils.py`
Utilities for Ray cluster introspection and head node scheduling strategy creation.
**Approach:** Replace with Fray cluster API equivalents (list nodes, scheduling strategies).

### Ray Actors

#### `lib/marin/src/marin/execution/executor.py`
Distributed execution framework for DAGs of ExecutorSteps managing task scheduling and status tracking with StatusActor.
**Approach:** Major refactoring required—replace Ray task submission with Fray job submission; implement equivalent Fray mechanism for status tracking.

#### `lib/marin/src/marin/execution/status_actor.py`
Ray actor for tracking pipeline step status across cluster failures.
**Approach:** Use Fray actors.

#### `lib/marin/src/marin/rl/weight_transfer/jax.py`
JAX-based weight transfer between training and inference workers using Ray actors.
**Approach:** Use Fray actors.

#### `lib/marin/src/marin/processing/classification/classifier.py`
Ray actor-based classifier implementations (BERT, fastText, dummy).
**Approach:** Convert to Fray-compatible distributed workers.

#### `lib/marin/src/marin/processing/classification/autoscaler.py`
Autoscaling actor pool for classification tasks managing actor lifecycle and task distribution.
**Approach:** Convert Ray actors to Fray distributed workers with autoscaling support.

#### `lib/marin/src/marin/processing/classification/inference.py`
Quality classifier inference using autoscaling actor pools and Ray queues.
**Approach:** Convert Ray actors and queues to Fray distributed workers and queue abstractions.
