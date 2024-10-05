"""
The `Executor` framework provides a way to specify a DAG of `ExecutorStep`s that
are executed in a topological order using Ray.  Beyond that:

1. The key distinguishing feature of the framework is allowing the user to
   flexibly control what steps are "new".

2. A secondary feature of the framework is that it creates sensible output paths
   for each step to free the user from having to come up with interpretable
   names that don't clash.

As an example, suppose you have a two-step pipeline:

    transform(method) -> tokenize(method)

which can be instantiated as:

    [A] transform(trafilatura) -> tokenize(llama2)
    [B] transform(resiliparse) -> tokenize(llama2)
    [C] transform(trafilatura) -> tokenize(llama3)
    [D] transform(resiliparse) -> tokenize(llama3)

If you have already run a particular instantiation, running it again
should be a no-op (assume idempotence).  If you run [A], then running [C] should
reuse `transform(trafilatura)`.

## Versioning

But the big question is: when is a step `transform(trafilatura)` "new"?
In the extreme, you have to hash the code of `transform` and the precise
configuration passed into it, but this is too strict: Semantics-preserving
changes to the code or config (e.g., adding logging) should not trigger a rerun.

We want to compute a *version* for each step.  Here's what the user supplies:
1. a `name` (that characterizes the code and also is useful for interpretability).
2. which fields of a `config` should be included in the version (things like the
   "method", not default thresholds that don't change).

The version of a step is identified by the name, versioned fields, and the
versions of all the dependencies. This version is represented as a hash (e.g.,
8ce902).

## Output paths

Having established the version, the question is what the output path should be.
One extreme is to let the framework automatically specify all the paths, but
then the paths are opaque and you can't easily find where things are stored.

Solution: based on the name and version, the output path of a step is computed.
For example, if name is "documents/fineweb-resiliparse", then the full path
might be:

    gs://marin-us-central2/documents/fineweb-resiliparse-8c2f3a

## Final remarks

- If you prefer to manage the output paths yourself, you can not use `versioned`
  fields and specify everything you want in the name.  Note the version will
  still depend on upstream dependencies.

- The pipeline might get too big and unwieldy, in which case we can cut it up by
  specifying a hard-coded path as the input to a step.  Or perhaps we can have
  our cake and eat it to by putting in an "assert" statement to ensure the input
  path that's computed from upstream dependencies is what we expect.

- If we decide to rename fields, we can extend `versioned` to take a string of
  the old field name to preserve backward compatibility.
"""

import hashlib
import json
import logging
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import draccus
import ray
import ray.remote_function

from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAITING,
    append_status,
    get_current_status,
    get_status_path,
    read_events,
)

logger = logging.getLogger("ray")

ExecutorFunction = Callable | ray.remote_function.RemoteFunction | None


@dataclass(frozen=True)
class ExecutorStep:
    """
    An `ExecutorStep` represents a single step of a larger pipeline (e.g.,
    transforming HTML to text).  It is specified by:
     - a name (str), which is used to determine the `output_path`.
     - a function `fn` (Ray remote), and
     - a configuration `config` which gets passed into `fn`.

    When a step is run, we compute the following two things for each step:
    - `version`: represents all the upstream dependencies of the step
    - `output_path`: the path where the output of the step are stored, based on
    the name and a hash of the version.

    The `config` is a dataclass object that recursively might have special
    values of the following form:
    - `InputName(step, name)`: a dependency on another `step`, resolve to the step.output_path / name
    - `OutputName(name)`: resolves to the output_path / name
    - `VersionedValue(value)`: a value that should be part of the version
    The `config` is instantiated by replacing these special values with the
    actual paths during execution.
    """

    name: str
    fn: ExecutorFunction
    config: dataclass

    override_output_path: str | None = None
    """Specifies the `output_path` that should be used.  Print warning if it
    doesn't match the automatically computed one."""

    def __hash__(self):
        """Hash based on the ID (every object is different)."""
        return hash(id(self))


@dataclass(frozen=True)
class InputName:
    """To be interpreted as a previous `step`'s output_path joined with `name`."""

    step: ExecutorStep
    name: str = ""


def output_path_of(step: ExecutorStep, name: str = ""):
    return InputName(step=step, name=name)


@dataclass(frozen=True)
class OutputName:
    """To be interpreted as part of this step's output_path joined with `name`."""

    name: str = ""


def this_output_path(name: str = ""):
    return OutputName(name=name)


@dataclass(frozen=True)
class VersionedValue:
    """Wraps a value, to signal that this value (part of a config) should be part of the version."""

    value: Any


def versioned(value: Any):
    return VersionedValue(value)


############################################################


def dependency_index_str(i: int) -> str:
    return f"DEP[{i}]"


def collect_dependencies_and_version(obj: Any, dependencies: list[ExecutorStep], version: dict[str, Any]):
    """Recurse through `obj` to find all the versioned values, and return them
    as a dict where the key is the sequence of fields identifying where the
    value resides in obj.  Example:

        get_version(Foo(a=versioned(1), b=Bar(c=versioned(2)))

           should return

        {"a": 1, "b.c": 2}

    Along the way, compute the list of dependencies.
    """

    def recurse(obj: Any, prefix: str):
        new_prefix = prefix + "." if prefix else ""
        if isinstance(obj, VersionedValue):
            # Just extract the value
            version[prefix] = obj.value
        elif isinstance(obj, InputName):
            # Put string i for the i-th dependency
            index = len(dependencies)
            dependencies.append(obj.step)
            version[prefix] = dependency_index_str(index) + ("/" + obj.name if obj.name else "")
        elif is_dataclass(obj):
            # Recurse through dataclasses
            for field in fields(obj):
                value = getattr(obj, field.name)
                recurse(value, new_prefix + field.name)
        elif isinstance(obj, list):
            # Recurse through lists
            for i, x in enumerate(obj):
                recurse(x, new_prefix + f"[{i}]")
        elif isinstance(obj, dict):
            # Recurse through dicts
            for i, x in obj.items():
                recurse(x, new_prefix + i)

    recurse(obj, "")


def instantiate_config(config: dataclass, output_path: str, output_paths: dict[ExecutorStep, str]) -> dataclass:
    """
    Return a "real" config where all the special values (e.g., `InputName`,
    `OutputName`, and `VersionedValue`) have been replaced with
    the actual paths that they represent.
    `output_path`: represents the output path of the current step.
    `output_paths`: a dict from `ExecutorStep` to their output paths.
    """

    def recurse(obj: Any):
        if obj is None:
            return None
        if isinstance(obj, InputName):
            return os.path.join(output_paths[obj.step], obj.name)
        elif isinstance(obj, OutputName):
            return os.path.join(output_path, obj.name)
        elif isinstance(obj, VersionedValue):
            return obj.value
        elif is_dataclass(obj):
            # Recurse through dataclasses
            result = {}
            for field in fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = recurse(value)
            return replace(obj, **result)
        elif isinstance(obj, list):
            # Recurse through lists
            return [recurse(x) for x in obj]
        elif isinstance(obj, dict):
            # Recurse through dicts
            return dict((i, recurse(x)) for i, x in obj.items())
        else:
            return obj

    return recurse(config)


class Executor:
    """ "
    Performs the execution of a pipeline of `ExecutorStep`s.
    1. Instantiate all the `output_path`s for each `ExecutorStep` based on `prefix`, names, and versions of everything.
    2. Run each `ExecutorStep` in a proper topological sort order.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.dependencies: dict[ExecutorStep, list[ExecutorStep]] = {}
        self.versions: dict[ExecutorStep, dict[str, Any]] = {}
        self.output_paths: dict[ExecutorStep, str] = {}
        self.steps: list[ExecutorStep] = []
        self.refs: dict[ExecutorStep, ray.ObjectRef] = {}

    def run(self, steps: list[ExecutorStep], dry_run: bool = False):
        # Gather all the steps, compute versions and output paths for all of them.
        for step in steps:
            self.compute_version(step)

        # Run each step
        for step in self.steps:
            self.run_step(step, dry_run=dry_run)
        ray.get(list(self.refs.values()))

    def compute_version(self, step: ExecutorStep) -> dict[str, Any]:
        if step in self.versions:
            return self.versions[step]

        # Collect dependencies and the config version
        dependencies: list[ExecutorStep] = []
        config_version: dict[str, Any] = {}
        collect_dependencies_and_version(obj=step.config, dependencies=dependencies, version=config_version)

        # Recurse on dependencies
        for dep in dependencies:
            self.compute_version(dep)

        # The version specifies precisely all the information that uniquely
        # identifies this step.  Note that the fn name is not part of the
        # version.
        version = {
            "name": step.name,
            "config": config_version,
            "dependencies": [self.versions[dep] for dep in dependencies],
        }

        # Compute output path
        version_str = json.dumps(version, sort_keys=True)
        hashed_version = hashlib.md5(version_str.encode()).hexdigest()[:6]
        output_path = os.path.join(self.prefix, step.name + "-" + hashed_version)

        # Override output path if specified
        if step.override_output_path is not None:
            if output_path != step.override_output_path:
                logger.warning(f"Output path {output_path} doesn't match {step.override_output_path}, using the latter.")
                output_path = step.override_output_path

        # Record everything
        self.steps.append(step)
        self.dependencies[step] = dependencies
        self.versions[step] = version
        self.output_paths[step] = output_path

        return version

    def run_step(self, step: ExecutorStep, dry_run: bool):
        """
        Return a Ray object reference to the result of running the `step`.
        If `dry_run`, only print out what needs to be done.
        """
        config = instantiate_config(
            config=step.config,
            output_path=self.output_paths[step],
            output_paths=self.output_paths,
        )

        config_version = self.versions[step]["config"]
        output_path = self.output_paths[step]

        # Figure out the status of this step
        status_path = get_status_path(output_path)
        statuses = read_events(status_path)
        status = get_current_status(statuses)

        # Print information about this step
        logger.info(f"[{status}] {step.name}: {get_fn_name(step.fn)}")
        logger.info(f"  output_path = {output_path}")
        logger.info(f"  config = {json.dumps(config_version)}")
        for i, dep in enumerate(self.dependencies[step]):
            logger.info(f"  {dependency_index_str(i)} = {self.output_paths[dep]}")
        logger.info("")

        # Only start if there's no status
        should_run = not dry_run and status is None
        dependencies = [self.refs[dep] for dep in self.dependencies[step]]
        name = f"execute_after_dependencies({get_fn_name(step.fn, short=True)})::{step.name})"
        self.refs[step] = execute_after_dependencies.options(name=name).remote(
            step.fn, config, dependencies, output_path, should_run
        )


@ray.remote
def execute_after_dependencies(
    fn: ExecutorFunction, config: dataclass, dependencies: list[ray.ObjectRef], output_path: str, should_run: bool
):
    """
    Run a function `fn` with the given `config`, after all the `dependencies` have finished.
    Only do stuff if `should_run` is True.
    """
    status_path = get_status_path(output_path)

    # Ensure that dependencies are all run first
    if should_run:
        append_status(status_path, STATUS_WAITING)
    ray.get(dependencies)

    # Call fn(config)
    if should_run:
        append_status(status_path, STATUS_RUNNING)
    try:
        if isinstance(fn, ray.remote_function.RemoteFunction):
            if should_run:
                ray.get(fn.remote(config))
        elif isinstance(fn, Callable):
            if should_run:
                fn(config)
        else:
            raise ValueError(f"Expected a Callable or Ray function, but got {fn}")
    except Exception as e:
        # Failed due to some exception
        message = traceback.format_exc()
        if should_run:
            append_status(status_path, STATUS_FAILED, message=message)
        raise e

    # Success!
    if should_run:
        append_status(status_path, STATUS_SUCCESS)


def get_fn_name(fn: Callable | ray.remote_function.RemoteFunction, short: bool = False):
    """Just for debugging: get the name of the function."""
    if fn is None:
        return "None"
    if isinstance(fn, ray.remote_function.RemoteFunction):
        if short:
            return f"{fn._function.__name__}"
        else:
            return f"{fn._function.__module__}.{fn._function.__qualname__}"
    else:
        if short:
            return f"{fn.__name__}"
        else:
            return f"{fn.__module__}.{fn.__qualname__}"


############################################################


@dataclass(frozen=True)
class ExecutorMainConfig:
    prefix: str = "gs://marin-us-central2"
    dry_run: bool = False


@draccus.wrap()
def executor_main(config: ExecutorMainConfig, steps: list[ExecutorStep]):
    """Main entry point for experiments (to standardize)"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    executor = Executor(prefix=config.prefix)
    executor.run(steps=steps, dry_run=config.dry_run)
