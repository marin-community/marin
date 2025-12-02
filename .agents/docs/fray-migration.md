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
_need_ to use Ray.  For example, if a function is decorated with `@ray.remote`
but all of it's callers are _also_ marked `ray.remote`, then you can simply remove the annotation.

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

If a runtime_environment is specified, e.g. to specify environment variables or
pip packages, it should be replaced with an equivalent Fray IsolatedVirtualEnv:

```python
ray.remote(runtime_env=build_runtime_env_for_packages(...))
def foo():

```

Becomes:

```python
def foo():
    with IsolatedVirtualEnv(packages=["x", "y"], env={"z"}) as venv:
        venv.run(...)
        

#### Resource requirements

Fray breaks up job and task scheduling into separate concerns.