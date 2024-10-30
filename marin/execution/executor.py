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
import inspect
import json
import logging
import os
import subprocess
import traceback
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from typing import Any, Generic, TypeVar

import draccus
import fsspec
import ray
import ray.remote_function

from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_WAITING,
    append_status,
    get_current_status,
    get_status_path,
    read_events,
)
from marin.utilities.json_encoder import CustomJsonEncoder

logger = logging.getLogger("ray")

ConfigT = TypeVar("ConfigT", covariant=True, bound=dataclass)
T_co = TypeVar("T_co", covariant=True)

ExecutorFunction = Callable | ray.remote_function.RemoteFunction | None


@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT]):
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

    Note: `step: ExecutorStep` is interpreted as `InputName(step, None)`.
    """

    name: str
    fn: ExecutorFunction
    config: ConfigT
    description: str | None = None

    override_output_path: str | None = None
    """Specifies the `output_path` that should be used.  Print warning if it
    doesn't match the automatically computed one."""

    def cd(self, name: str) -> "InputName":
        """Refer to the `name` under `self`'s output_path."""
        return InputName(self, name=name)

    def __hash__(self):
        """Hash based on the ID (every object is different)."""
        return hash(id(self))


@dataclass(frozen=True)
class InputName:
    """To be interpreted as a previous `step`'s output_path joined with `name`."""

    step: ExecutorStep
    name: str | None

    def cd(self, name: str) -> "InputName":
        return InputName(self.step, name=os.path.join(self.name, name) if self.name else name)


def output_path_of(step: ExecutorStep, name: str | None = None):
    return InputName(step=step, name=name)


@dataclass(frozen=True)
class OutputName:
    """To be interpreted as part of this step's output_path joined with `name`."""

    name: str | None


def this_output_path(name: str | None = None):
    return OutputName(name=name)


@dataclass(frozen=True)
class VersionedValue(Generic[T_co]):
    """Wraps a value, to signal that this value (part of a config) should be part of the version."""

    value: T_co


def versioned(value: T_co) -> VersionedValue[T_co]:
    if isinstance(value, VersionedValue):
        raise ValueError("Can't nest VersionedValue")
    return VersionedValue(value)


############################################################


@dataclass(frozen=True)
class ExecutorStepInfo:
    """
    Contains the information about an `ExecutorStep` that can be serialized into JSON.
    Note that this conversion is not reversible.
    """

    name: str
    """`step.name`."""

    fn_name: str
    """Rendered string of `step.fn`."""

    config: dataclass
    """`step.config`, but concretized (no more `InputName`, `OutputName`, or `VersionedValue`)."""

    description: str | None
    """`step.description`."""

    override_output_path: str | None
    """`step.override_output_path`."""

    version: dict[str, Any]
    """`executor.versions[step]`."""

    dependencies: list[str]
    """Fully realized output_paths of the dependencies."""

    output_path: str
    """`executor.output_paths[step]`."""


@dataclass(frozen=True)
class ExecutorInfo:
    """Contains information about an execution."""

    # Metadata related to the launch
    ray_job_id: str
    git_commit: str | None
    caller_path: str
    created_date: str
    user: str | None

    # Information taken from `Executor`
    prefix: str
    description: str | None
    steps: list[ExecutorStepInfo]


def get_info_path(output_path: str) -> str:
    """Return the `path` of the info file associated with `output_path`."""
    return os.path.join(output_path, ".executor_info")


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

        if isinstance(obj, ExecutorStep):
            obj = output_path_of(obj, None)

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
                if not isinstance(i, str):
                    raise ValueError(f"dict keys must be strs, but got {i} (type: {type(i)})")
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

    def join_path(output_path: str, name: str | None) -> str:
        return os.path.join(output_path, name) if name else output_path

    def recurse(obj: Any):
        if obj is None:
            return None

        if isinstance(obj, ExecutorStep):
            obj = output_path_of(obj)

        if isinstance(obj, InputName):
            return join_path(output_paths[obj.step], obj.name)
        elif isinstance(obj, OutputName):
            return join_path(output_path, obj.name)
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

    def __init__(
        self,
        prefix: str,
        executor_info_base_path: str,
        description: str | None = None,
    ):
        self.prefix = prefix
        self.executor_info_base_path = executor_info_base_path
        self.description = description

        self.configs: dict[ExecutorStep, dataclass] = {}
        self.dependencies: dict[ExecutorStep, list[ExecutorStep]] = {}
        self.versions: dict[ExecutorStep, dict[str, Any]] = {}
        self.version_strs: dict[ExecutorStep, str] = {}
        self.version_str_to_step: dict[str, ExecutorStep] = {}
        self.output_paths: dict[ExecutorStep, str] = {}
        self.steps: list[ExecutorStep] = []
        self.statuses: dict[ExecutorStep, str] = {}
        self.refs: dict[ExecutorStep, ray.ObjectRef] = {}

    def run(
        self,
        steps: list[ExecutorStep | InputName],
        dry_run: bool = False,
        force_run: list[str] | None = None,
        force_run_failed: bool = False,
    ):
        # Gather all the steps, compute versions and output paths for all of them.
        logger.info(f"### Inspecting the {len(steps)} provided steps ###")
        for step in steps:
            if isinstance(step, InputName):  # Interpret InputName as the underlying step
                step = step.step
            self.compute_version(step)

        logger.info(f"### Reading {len(self.steps)} statuses ###")
        self.read_statuses()

        logger.info(f"### Launching {len(self.steps)} steps ###")
        for step in self.steps:
            self.run_step(step, dry_run=dry_run, force_run=force_run, force_run_failed=force_run_failed)

        logger.info("### Writing metadata ###")
        self.write_infos()

        logger.info("### Waiting for all steps to finish ###")
        ray.get(list(self.refs.values()))

    def compute_version(self, step: ExecutorStep):
        if step in self.versions:
            return

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
        version_str = json.dumps(version, sort_keys=True, cls=CustomJsonEncoder)
        hashed_version = hashlib.md5(version_str.encode()).hexdigest()[:6]
        output_path = os.path.join(self.prefix, step.name + "-" + hashed_version)

        # Override output path if specified
        if step.override_output_path is not None:
            if output_path != step.override_output_path:
                logger.warning(
                    f"Output path {output_path} doesn't match given "
                    "override {step.override_output_path}, using the latter."
                )
                output_path = step.override_output_path

        # Record everything
        # Multiple `ExecutorStep`s can have the same version, so only keep one
        # of them.  Note that some `ExecutorStep`s might have depenedencies that
        # are not part of `self.steps`, but there will be some step with the
        # same version.
        if version_str not in self.version_str_to_step:
            self.steps.append(step)
            self.version_str_to_step[version_str] = step
        else:
            logger.warning(
                f"Multiple `ExecutorStep`s (named {step.name}) have the same version; try to instantiate only once."
            )
        self.configs[step] = instantiate_config(
            config=step.config,
            output_path=output_path,
            output_paths=self.output_paths,
        )
        self.dependencies[step] = list(map(self.canonicalize, dependencies))
        self.versions[step] = version
        self.version_strs[step] = version_str
        self.output_paths[step] = output_path

    def canonicalize(self, step: ExecutorStep) -> ExecutorStep:
        """Multiple instances of `ExecutorStep` might have the same version."""
        return self.version_str_to_step[self.version_strs[step]]

    def write_infos(self):
        """Output JSON files (one for the entire execution, one for each step)."""

        # Compute info for each step
        step_infos: list[ExecutorStepInfo] = []
        for step in self.steps:
            step_infos.append(
                ExecutorStepInfo(
                    name=step.name,
                    fn_name=get_fn_name(step.fn),
                    config=self.configs[step],
                    description=step.description,
                    override_output_path=step.override_output_path,
                    version=self.versions[step],
                    dependencies=[self.output_paths[dep] for dep in self.dependencies[step]],
                    output_path=self.output_paths[step],
                )
            )

        # Compute info for the entire execution
        executor_info = ExecutorInfo(
            git_commit=get_git_commit(),
            caller_path=get_caller_path(),
            created_date=datetime.now().isoformat(),
            user=get_user(),
            ray_job_id=ray.get_runtime_context().get_job_id(),
            prefix=self.prefix,
            description=self.description,
            steps=step_infos,
        )

        # Set executor_info_path based on hash and caller path name (e.g., 72_baselines-8c2f3a.json)
        executor_version_str = json.dumps(
            list(map(asdict_without_description, step_infos)), sort_keys=True, cls=CustomJsonEncoder
        )
        executor_version_hash = hashlib.md5(executor_version_str.encode()).hexdigest()[:6]
        name = os.path.basename(executor_info.caller_path).replace(".py", "")
        self.executor_info_path = os.path.join(
            self.executor_info_base_path,
            f"{name}-{executor_version_hash}.json",
        )

        # Print where to find the executor info (experiments JSON)
        logger.info(f"Writing executor info to {self.executor_info_path}")
        # TODO: don't hardcode this webserver later
        experimentUrl = "https://marlin-subtle-barnacle.ngrok-free.app/experiment?path=" + urllib.parse.quote(
            self.executor_info_path
        )
        logger.info("To view the experiment page, go to:")
        logger.info("")
        logger.info(experimentUrl)
        logger.info("")

        # Write out info for each step
        for step, info in zip(self.steps, step_infos, strict=True):
            info_path = get_info_path(self.output_paths[step])
            with fsspec.open(info_path, "w") as f:
                print(json.dumps(asdict(info), indent=2, cls=CustomJsonEncoder), file=f)

        # Write out info for the entire execution
        with fsspec.open(self.executor_info_path, "w") as f:
            print(json.dumps(asdict(executor_info), indent=2, cls=CustomJsonEncoder), file=f)

    def read_statuses(self):
        """Read the statuses of all the steps in parallel."""

        def get_status(step: ExecutorStep):
            status_path = get_status_path(self.output_paths[step])
            statuses = read_events(status_path)
            status = get_current_status(statuses)
            self.statuses[step] = status

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(get_status, self.steps)

    def run_step(self, step: ExecutorStep, dry_run: bool, force_run: list[str] | None, force_run_failed: bool):
        """
        Return a Ray object reference to the result of running the `step`.
        If `dry_run`, only print out what needs to be done.
        """
        config = self.configs[step]
        config_version = self.versions[step]["config"]
        output_path = self.output_paths[step]
        status = self.statuses[step]

        # Print information about this step
        logger.info(f"[{status}] {step.name}: {get_fn_name(step.fn)}")
        logger.info(f"  output_path = {output_path}")
        logger.info(f"  config = {json.dumps(config_version, cls=CustomJsonEncoder)}")
        for i, dep in enumerate(self.dependencies[step]):
            logger.info(f"  {dependency_index_str(i)} = {self.output_paths[dep]}")
        logger.info("")
        should_force_run = (force_run and step.name in force_run) or (force_run_failed and status == STATUS_FAILED)
        if should_force_run:
            logger.info(f"Force running {step.name}, previous status: {status}")

        # Only start if there's no status
        should_run = not dry_run and (status is None or should_force_run)
        dependencies = [self.refs[dep] for dep in self.dependencies[step]]
        name = f"execute_after_dependencies({get_fn_name(step.fn, short=True)})::{step.name})"
        self.refs[step] = execute_after_dependencies.options(name=name).remote(
            step.fn, config, dependencies, output_path, should_run
        )


def asdict_without_description(obj: dataclass) -> dict[str, Any]:
    """Return the `asdict` of an object, but remove the `description` field, because it doesn't affect the semantics."""
    d = asdict(obj)
    d.pop("description", None)
    return d


@ray.remote
def execute_after_dependencies(
    fn: ExecutorFunction, config: dataclass, dependencies: list[ray.ObjectRef], output_path: str, should_run: bool
):
    """
    Run a function `fn` with the given `config`, after all the `dependencies` have finished.
    Only do stuff if `should_run` is True.
    """
    status_path = get_status_path(output_path)
    ray_task_id = ray.get_runtime_context().get_task_id()

    # Ensure that dependencies are all run first
    if should_run:
        append_status(status_path, STATUS_WAITING, ray_task_id=ray_task_id)
    ray.get(dependencies)

    # Call fn(config)
    if should_run:
        append_status(status_path, STATUS_WAITING, ray_task_id=ray_task_id)
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
            append_status(status_path, STATUS_FAILED, message=message, ray_task_id=ray_task_id)
        raise e

    # Success!
    if should_run:
        append_status(status_path, STATUS_SUCCESS, ray_task_id=ray_task_id)


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


def get_git_commit() -> str | None:
    """Return the git commit of the current branch (if it can be found)"""
    if os.path.exists(".git"):
        return os.popen("git rev-parse HEAD").read().strip()
    else:
        return None


def get_caller_path() -> str:
    """Return the path of the file that called this function."""
    return inspect.stack()[-1].filename


def get_user() -> str | None:
    return subprocess.check_output("whoami", shell=True).strip().decode("utf-8")


############################################################


@dataclass(frozen=True)
class ExecutorMainConfig:
    prefix: str = "gs://marin-us-central2"
    """Attached to every output path that's constructed (e.g., the GCS bucket)."""

    executor_info_base_path: str = "gs://marin-us-central2/experiments"
    """Where the executor info should be stored under a file determined by a hash."""

    dry_run: bool = False
    force_run: list[str] = field(default_factory=list)  # <list of steps name>: run list of steps (names)
    force_run_failed: bool = False  # Force run failed steps


@draccus.wrap()
def executor_main(config: ExecutorMainConfig, steps: list[ExecutorStep], description: str | None = None):
    """Main entry point for experiments (to standardize)"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    executor = Executor(
        prefix=config.prefix,
        executor_info_base_path=config.executor_info_base_path,
        description=description,
    )
    executor.run(
        steps=steps,
        dry_run=config.dry_run,
        force_run=config.force_run,
        force_run_failed=config.force_run_failed,
    )
