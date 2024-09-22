import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import draccus
import ray
import ray.remote_function

from marin.utils import fsspec_exists

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
    """
    name: str
    fn: ExecutorFunction
    config: dataclass

    def __hash__(self):
        """Hash based on the ID (every object is different)."""
        return hash(id(self))


@dataclass(frozen=True)
class InputName:
    """To be interpreted as a previous `step`'s output_path joined with `name`."""
    step: ExecutorStep
    name: str = ""

def get_input(step: ExecutorStep, name: str = ""):
    return InputName(step=step, name=name)


@dataclass(frozen=True)
class OutputName:
    """To be interpreted as part of this step's output_path joined with `name`."""
    name: str = ""

def get_output(name: str = ""):
    return OutputName(name=name)


@dataclass(frozen=True)
class VersionedValue:
    """Wraps a value, to signal that this value (part of a config) should be part of the version."""
    value: Any

def versioned(value: Any):
    return VersionedValue(value)


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
            # Recurse through sub-dataclasses
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
    if config is None:
        return None
    updates = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, InputName):
            updates[field.name] = os.path.join(output_paths[value.step], value.name)
        elif isinstance(value, OutputName):
            updates[field.name] = os.path.join(output_path, value.name)
        elif isinstance(value, VersionedValue):
            updates[field.name] = value.value
        elif is_dataclass(value):
            updates[field.name] = instantiate_config(value, output_path, output_paths)
        # Note unversioned primitives don't need to be updated.
    return replace(config, **updates)


class Executor:
    """"
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
            self.refs[step] = self.run_step(step, dry_run=dry_run)
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

        # Record everything
        self.steps.append(step)
        self.dependencies[step] = dependencies
        self.versions[step] = version
        self.output_paths[step] = output_path

        return version


    def run_step(self, step: ExecutorStep, dry_run: bool) -> ray.ObjectRef:
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

        # Completed = whether the output path exists (note this is optimistic)
        completed = fsspec_exists(output_path)
        completed_str = "COMPLETED" if completed else "PENDING"

        # Print information
        logger.info(f"[{completed_str}] {step.name}: {get_fn_name(step.fn)}")
        logger.info(f"  output_path = {output_path}")
        logger.info(f"  config = {json.dumps(config_version)}")
        for i, dep_version in enumerate(self.versions[step]["dependencies"]):
            logger.info(f"  {dependency_index_str(i)} = {dep_version['name']}")

        # Call `fn` if it hasn't been done yet
        fn = step.fn if not dry_run and not completed else None

        logger.info("")

        dependencies = [self.refs[dep] for dep in self.dependencies[step]]
        return execute_after_dependencies.remote(fn, config, dependencies)

@ray.remote
def execute_after_dependencies(fn: ExecutorFunction, config: dataclass, dependencies: list[ray.ObjectRef]):
    """Run a function with the given config."""
    # Ensure that dependencies are all run first
    ray.get(dependencies)

    # Call fn(config)
    if fn is None:
        pass
    elif isinstance(fn, ray.remote_function.RemoteFunction):
        ray.get(fn.remote(config))
    elif isinstance(fn, Callable):
        fn(config)
    else:
        raise ValueError(f"Expected a Callable or Ray function, but got {fn}")


def get_fn_name(fn: Callable | ray.remote_function.RemoteFunction):
    """Just for debugging: get the name of the function."""
    if fn is None:
        return "None"
    if isinstance(fn, ray.remote_function.RemoteFunction):
        return f"{fn._function.__module__}.{fn._function.__name__}"
    else:
        return f"{fn.__module__}.{fn.__name__}"

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
