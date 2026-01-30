# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The `Executor` framework provides a way to specify a DAG of `ExecutorStep`s that
are executed in a topological order using Fray.  Beyond that:

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
  still depend on upstream dependencies and "pseudo-dependencies."

- The pipeline might get too big and unwieldy, in which case we can cut it up by
  specifying a hard-coded path as the input to a step.  Or perhaps we can have
  our cake and eat it to by putting in an "assert" statement to ensure the input
  path that's computed from upstream dependencies is what we expect.

- If we decide to rename fields, we can extend `versioned` to take a string of
  the old field name to preserve backward compatibility.

- "Pseudo-dependencies" are dependencies that do not block the execution of
  the step, but are still included in the version.  This is useful for depending
   on checkpoints of in-progress training runs, for example. When you run a step
  that has a pseudo-dependency, it will not wait for the pseudo-dependency to
  finish executing (or even check if it is executing or failed) before running.
"""

import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import re
import subprocess
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from urllib.parse import urlparse

import draccus
import fsspec
import levanter.utils.fsspec_utils as fsspec_utils

from fray.v2 import (
    Client,
    Entrypoint,
    EnvironmentConfig,
    JobHandle,
    JobRequest,
    JobStatus,
    ResourceConfig,
    current_client,
)

from marin.execution.executor_step_status import (
    HEARTBEAT_INTERVAL,
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    StatusFile,
)
from marin.utilities.json_encoder import CustomJsonEncoder

logger = logging.getLogger("ray")

_LOCAL_DATA_BROWSER_PORT_RE = re.compile(r"^\s*port\s*:\s*(\d+)\s*(?:#.*)?$")
_LOCAL_DATA_BROWSER_CONFIG_REL = Path("data_browser") / "conf" / "local.conf"


def _find_data_browser_local_conf(max_parents: int = 6) -> Path | None:
    here = Path.cwd().resolve()
    for _ in range(max_parents + 1):
        candidate = here / _LOCAL_DATA_BROWSER_CONFIG_REL
        if candidate.exists():
            return candidate
        parent = here.parent
        if parent == here:
            break
        here = parent
    return None


def _get_local_data_browser_port(default: int = 5000) -> int:
    # looks for the port in the local data browser config file
    config_path = _find_data_browser_local_conf()
    if config_path is None:
        return default

    try:
        with config_path.open() as fp:
            for line in fp:
                match = _LOCAL_DATA_BROWSER_PORT_RE.match(line)
                if match:
                    return int(match.group(1))
    except OSError:
        return default

    return default


ConfigT = TypeVar("ConfigT", covariant=True, bound=dataclass)
T_co = TypeVar("T_co", covariant=True)

ExecutorFunction = Callable | None


class StepRunner:
    """Manages execution of a single step.

    To prevent duplicate execution, Executor uses a per-step lease directory to track
    the owner of a given step. Leases must be continually refreshed to prevent other
    executors from running the same step.
    """

    def __init__(self, client: Client, status_file: StatusFile):
        self.client = client
        self._status_file = status_file
        self._job_handle: JobHandle | None = None
        self._heartbeat_thread: Thread | None = None
        self._stop_event = Event()

    @property
    def job_id(self) -> str | None:
        """Return the job ID from the handle, if available."""
        return self._job_handle.job_id if self._job_handle else None

    def launch(self, job_request: JobRequest) -> None:
        """Launch job and start heartbeat thread."""
        self._status_file.write_status(STATUS_RUNNING)
        self._job_handle = self.client.submit(job_request)
        self._start_heartbeat()

    def wait(self) -> None:
        """Wait for job to complete, stop heartbeat, write final status."""
        try:
            result = self._job_handle.wait(raise_on_failure=False)
            if result == JobStatus.FAILED:
                self._status_file.write_status(STATUS_FAILED)
                raise RuntimeError(f"Job {self.job_id} failed")
            self._status_file.write_status(STATUS_SUCCESS)
        except Exception:
            if self._status_file.status != STATUS_FAILED:
                self._status_file.write_status(STATUS_FAILED)
            raise
        finally:
            self._stop_heartbeat()
            self._status_file.release_lock()

    def poll(self) -> bool:
        """Return True if job is finished."""
        if self._job_handle is None:
            return True
        return JobStatus.finished(self._job_handle.status())

    def _start_heartbeat(self) -> None:
        """Start background thread that periodically refreshes the lease."""

        def heartbeat_loop():
            while not self._stop_event.wait(HEARTBEAT_INTERVAL):
                self._status_file.refresh_lock()

        self._heartbeat_thread = Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)


def worker_id() -> str:
    import threading

    return f"{os.uname()[1]}-{threading.get_ident()}"


def asdict_without_description(obj: dataclass) -> dict[str, Any]:
    """Return the dict form of a dataclass, but remove the `description` field."""

    def recurse(value: Any):
        if is_dataclass(value):
            return {f.name: recurse(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*(recurse(v) for v in value))
        if isinstance(value, (list, tuple)):
            return type(value)(recurse(v) for v in value)
        if isinstance(value, dict):
            # RuntimeEnv (and other dict subclasses) require keyword-only init,
            # so we normalize to a plain dict to avoid construction errors.
            return {recurse(k): recurse(v) for k, v in value.items()}
        return copy.deepcopy(value)

    d = recurse(obj)
    assert isinstance(d, dict)
    d.pop("description", None)
    assert isinstance(d, dict)
    return d


@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT]):
    """
    An `ExecutorStep` represents a single step of a larger pipeline (e.g.,
    transforming HTML to text).  It is specified by:
     - a name (str), which is used to determine the `output_path`.
     - a function `fn` (Callable), and
     - a configuration `config` which gets passed into `fn`.
     - a pip dependencies list (Optional[list[str]]) which are the pip dependencies required for the step.
     These can be keys of project.optional-dependencies in the project's pyproject.toml file or any other pip package.

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

    pip_dependency_groups: list[str] | None = None
    """List of `extra` dependencies from pyproject.toml to include with this step."""

    resources: ResourceConfig | None = None
    """Resource requirements for this step (GPU, TPU, CPU). If None, defaults to CPU."""

    def cd(self, name: str) -> "InputName":
        """Refer to the `name` under `self`'s output_path."""
        return InputName(self, name=name)

    def __truediv__(self, other: str) -> "InputName":
        """Alias for `cd`. That looks more Pythonic."""
        return InputName(self, name=other)

    def __hash__(self):
        """Hash based on the ID (every object is different)."""
        return hash(id(self))

    def with_output_path(self, output_path: str) -> "ExecutorStep":
        """Return a copy of the step with the given output_path."""
        return replace(self, override_output_path=output_path)

    def as_input_name(self) -> "InputName":
        return InputName(step=self, name=None)


@dataclass(frozen=True)
class InputName:
    """To be interpreted as a previous `step`'s output_path joined with `name`."""

    step: ExecutorStep | None
    name: str | None
    block_on_step: bool = True
    """
    If False, the step that uses this InputName
    will not block (or attempt to execute) `step`. We use this for
    documenting dependencies in the config, but where that step might not have technically finished...

    For instance, we sometimes use training checkpoints before the training step has finished.

    These "pseudo-dependencies" still impact the hash of the step, but they don't block execution.
    """

    def cd(self, name: str) -> "InputName":
        return InputName(self.step, name=os.path.join(self.name, name) if self.name else name)

    def __truediv__(self, other: str) -> "InputName":
        """Alias for `cd` that looks more Pythonic."""
        return self.cd(other)

    @staticmethod
    def hardcoded(path: str) -> "InputName":
        """
        Sometimes we want to specify a path that is not part of the pipeline but is still relative to the prefix.
        Try to use this sparingly.
        """
        return InputName(None, name=path)

    def nonblocking(self) -> "InputName":
        """
        the step will not block on (or attempt to execute) the parent step.

         (Note that if another step depends on the parent step, it will still block on it.)
        """
        return dataclasses.replace(self, block_on_step=False)


def get_executor_step(run: ExecutorStep | InputName) -> ExecutorStep:
    """
    Helper function to extract the ExecutorStep from an InputName or ExecutorStep.

    Args:
        run (ExecutorStep | InputName): The input to extract the step from.

    Returns:
        ExecutorStep: The extracted step.
    """
    if isinstance(run, ExecutorStep):
        return run
    elif isinstance(run, InputName):
        step = run.step
        if step is None:
            raise ValueError(f"Hardcoded path {run.name} is not part of the pipeline")
        return step
    else:
        raise ValueError(f"Unexpected type {type(run)} for run: {run}")


def output_path_of(step: ExecutorStep, name: str | None = None) -> InputName:
    return InputName(step=step, name=name)


if TYPE_CHECKING:

    class OutputName(str):
        """Type-checking stub treated as a string so defaults like THIS_OUTPUT_PATH fit `str`."""

        name: str | None

else:

    @dataclass(frozen=True)
    class OutputName:
        """To be interpreted as part of this step's output_path joined with `name`."""

        name: str | None


def this_output_path(name: str | None = None):
    return OutputName(name=name)


# constant so we can use it in fields of dataclasses
THIS_OUTPUT_PATH = OutputName(None)


@dataclass(frozen=True)
class VersionedValue(Generic[T_co]):
    """Wraps a value, to signal that this value (part of a config) should be part of the version."""

    value: T_co


def versioned(value: T_co) -> VersionedValue[T_co]:
    if isinstance(value, VersionedValue):
        raise ValueError("Can't nest VersionedValue")
    elif isinstance(value, InputName):
        # TODO: We have also run into Versioned([InputName(...), ...])
        raise ValueError("Can't version an InputName")

    return VersionedValue(value)


def ensure_versioned(value: VersionedValue[T_co] | T_co) -> VersionedValue[T_co]:
    """
    Ensure that the value is wrapped in a VersionedValue. If it is already wrapped, return it as is.
    """
    return value if isinstance(value, VersionedValue) else VersionedValue(value)


def unwrap_versioned_value(value: VersionedValue[T_co] | T_co) -> T_co:
    """
    Unwrap the value if it is a VersionedValue, otherwise return the value as is.

    Recurses into dataclasses, dicts and lists to unwrap any nested VersionedValue instances.
    This method cannot handle InputName, OutputName, or ExecutorStep instances inside VersionedValue as
    their values depend on execution results.
    """

    def recurse(obj: Any):
        if isinstance(obj, VersionedValue):
            return recurse(obj.value)
        if isinstance(obj, OutputName | InputName | ExecutorStep):
            raise ValueError(f"Cannot unwrap VersionedValue containing {type(obj)}: {obj}")
        if is_dataclass(obj):
            result = {}
            for field in fields(obj):
                val = getattr(obj, field.name)
                result[field.name] = recurse(val)
            return replace(obj, **result)
        if isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [recurse(x) for x in obj]
        return obj

    return recurse(value)  # type: ignore


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
    worker_id: str
    git_commit: str | None
    caller_path: str
    created_date: str
    user: str | None

    # Information taken from `Executor`
    prefix: str
    description: str | None
    steps: list[ExecutorStepInfo]


def _get_info_path(output_path: str) -> str:
    """Return the `path` of the info file associated with `output_path`."""
    return os.path.join(output_path, ".executor_info")


############################################################


def dependency_index_str(i: int) -> str:
    return f"DEP[{i}]"


@dataclass(frozen=True)
class _Dependencies:
    """
    Contains the dependencies of a step, the pseudo-dependencies, and the version of the dependencies.
    Internal use.
    """

    dependencies: list[ExecutorStep]
    """List of dependencies."""
    pseudo_dependencies: list[ExecutorStep]
    """List of pseudo-dependencies."""
    version: dict[str, Any]
    """Version of the dependencies."""


def collect_dependencies_and_version(obj: Any) -> _Dependencies:
    """Recurse through `obj` to find all the versioned values, and return them
    as a dict where the key is the sequence of fields identifying where the
    value resides in obj.  Example:

        get_version(Foo(a=versioned(1), b=Bar(c=versioned(2)))

           should return

        {"a": 1, "b.c": 2}

    Along the way, compute the list of dependencies.

    Returns:
        - dependencies: list of `ExecutorStep`s that are dependencies of the
          current step.
        - version: dict of versioned values, where the key is the sequence of
          fields identifying where the value resides in obj.
        - pseudo_dependencies: list of `ExecutorStep`s that are dependencies of the step but that we won't
            actually block on
    """
    pseudo_dependencies: list[ExecutorStep] = []
    dependencies: list[ExecutorStep] = []
    version: dict[str, Any] = {}

    def recurse(obj: Any, prefix: str):
        new_prefix = prefix + "." if prefix else ""

        if isinstance(obj, ExecutorStep):
            obj = output_path_of(obj, None)

        if isinstance(obj, VersionedValue):
            version[prefix] = obj.value
        elif isinstance(obj, InputName):
            # Put string i for the i-th dependency
            if obj.step is not None:
                index = len(dependencies) + len(pseudo_dependencies)
                if not obj.block_on_step:
                    pseudo_dependencies.append(obj.step)
                else:
                    dependencies.append(obj.step)
                version[prefix] = dependency_index_str(index) + ("/" + obj.name if obj.name else "")
            else:
                version[prefix] = obj.name
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

    return _Dependencies(dependencies, pseudo_dependencies, version)


def instantiate_config(
    config: dataclass, output_path: str, output_paths: dict[ExecutorStep, str], prefix: str
) -> dataclass:
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
            if obj.step is None:
                return _make_prefix_absolute_path(prefix, obj.name)
            else:
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
    """
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
        self.client = current_client()
        self.prefix = prefix
        self.executor_info_base_path = executor_info_base_path
        self.description = description

        self.configs: dict[ExecutorStep, dataclass] = {}
        self.dependencies: dict[ExecutorStep, list[ExecutorStep]] = {}
        self.versions: dict[ExecutorStep, dict[str, Any]] = {}
        # pseudo-dependencies only impact version but don't block execution of descendants
        # this dict contains is True for steps that are only used as pseudo-dependencies
        self.is_pseudo_dep: dict[ExecutorStep, bool] = {}
        self.version_strs: dict[ExecutorStep, str] = {}
        self.version_str_to_step: dict[str, ExecutorStep] = {}
        self.output_paths: dict[ExecutorStep, str] = {}
        self.steps: list[ExecutorStep] = []
        self.step_runners: dict[ExecutorStep, StepRunner] = {}
        self.step_infos: list[ExecutorStepInfo] = []
        self.executor_info: ExecutorInfo | None = None
        # Populated during dry runs to summarize which steps would be executed.
        # Tuple is (step_name, action, reason, output_path)
        self._dry_run_plan: list[tuple[str, str, str, str]] = []

    def run(
        self,
        steps: list[ExecutorStep | InputName],
        *,
        dry_run: bool = False,
        run_only: list[str] | None = None,
        force_run_failed: bool = True,
        max_concurrent: int | None = None,
    ):
        """
        Run the pipeline of `ExecutorStep`s.

        Args:
            steps: The steps to run.
            dry_run: If True, only print out what needs to be done. Reads existing
                statuses to report which steps would actually be executed.
            run_only: If not None, only run the steps in the list and their dependencies. Matches steps' names as regex
            force_run_failed: If True, run steps even if they have already been run (including if they failed)
            max_concurrent: Maximum number of steps to run concurrently. If None, run all ready steps in parallel.
        """
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be a positive integer, got {max_concurrent}")

        # Gather all the steps, compute versions and output paths for all of them.
        logger.info(f"### Inspecting the {len(steps)} provided steps ###")
        for step in steps:
            if isinstance(step, InputName):  # Interpret InputName as the underlying step
                step = step.step
            if step is not None:
                self.compute_version(step, is_pseudo_dep=False)

        self.get_infos()
        logger.info(f"### Reading {len(self.steps)} statuses ###")

        if run_only is not None:
            steps_to_run = self._compute_transitive_deps(self.steps, run_only)
        else:
            steps_to_run = [step for step in self.steps if not self.is_pseudo_dep[step]]

        if steps_to_run != self.steps:
            logger.info(f"### Running {len(steps_to_run)} steps out of {len(self.steps)} ###")

        logger.info("### Writing metadata ###")
        self.write_infos()

        logger.info(f"### Launching {len(steps_to_run)} steps ###")
        if max_concurrent is not None:
            logger.info(f"### Max concurrent steps: {max_concurrent} ###")
        if dry_run:
            self._dry_run_plan = []
        self._run_steps(steps_to_run, dry_run=dry_run, force_run_failed=force_run_failed, max_concurrent=max_concurrent)

        if dry_run:
            self._log_dry_run_summary()

        logger.info("### Waiting for all steps to finish ###")
        for runner in self.step_runners.values():
            runner.wait()

    def _run_steps(
        self,
        steps_to_run: list[ExecutorStep],
        *,
        dry_run: bool,
        force_run_failed: bool,
        max_concurrent: int | None = None,
    ) -> None:
        remaining_deps: dict[ExecutorStep, set[ExecutorStep]] = {
            step: set(dep for dep in self.dependencies[step] if dep in steps_to_run) for step in steps_to_run
        }
        dependents: dict[ExecutorStep, list[ExecutorStep]] = {step: [] for step in steps_to_run}
        for step, deps in remaining_deps.items():
            for dep in deps:
                dependents[dep].append(step)

        ready = [step for step, deps in remaining_deps.items() if not deps]
        running: dict[ExecutorStep, StepRunner | None] = {}

        while ready or running:
            # Launch ready steps, respecting max_concurrent limit if set
            # Use pop(0) for FIFO ordering
            while ready and (max_concurrent is None or len(running) < max_concurrent):
                step = ready.pop(0)
                runner = self._launch_step(step, dry_run=dry_run, force_run_failed=force_run_failed)
                if runner is not None:
                    self.step_runners[step] = runner
                running[step] = runner

            if not running:
                break

            finished_steps = []
            for step, runner in running.items():
                if runner is None:
                    # Dry run or already completed - immediately finished
                    finished_steps.append(step)
                elif runner.poll():
                    finished_steps.append(step)

            if not finished_steps:
                time.sleep(1)
                continue

            for finished_step in finished_steps:
                runner = running[finished_step]
                if runner is not None:
                    logger.info("Waiting for %s to finish for step %s", runner.job_id, finished_step.name)
                    runner.wait()

                running.pop(finished_step)
                for child in dependents.get(finished_step, []):
                    remaining_deps[child].remove(finished_step)
                    if not remaining_deps[child]:
                        ready.append(child)

    def _launch_step(self, step: ExecutorStep, *, dry_run: bool, force_run_failed: bool) -> StepRunner | None:
        config = self.configs[step]
        config_version = self.versions[step]["config"]
        output_path = self.output_paths[step]

        logger.info("%s: %s", step.name, get_fn_name(step.fn))
        logger.info("  output_path = %s", output_path)
        logger.info("  config = %s", json.dumps(config_version, cls=CustomJsonEncoder))
        for i, dep in enumerate(self.dependencies[step]):
            logger.info("  %s = %s", dependency_index_str(i), self.output_paths[dep])

        if dry_run:
            action, reason = self._plan_dry_run(step, force_run_failed=force_run_failed)
            self._record_dry_run(step, action, reason, output_path)
            if action == "run":
                logger.info("[DRY RUN] Would run %s -> %s: %s", step.name, output_path, reason)
            elif action == "wait":
                logger.info("[DRY RUN] %s currently running elsewhere: %s", step.name, reason)
            else:
                logger.info("[DRY RUN] Skip %s: %s", step.name, reason)
            return None

        step_name = f"{step.name}: {get_fn_name(step.fn)}"
        status_file = StatusFile(output_path, worker_id())

        if not should_run(status_file, step_name, force_run_failed):
            return None

        # need this hack for now to make ray remote functions work, as they aren't directly callable.
        import ray

        step_fn = step.fn
        if isinstance(step.fn, ray.remote_function.RemoteFunction):

            def _call_remote(*args, **kw):
                return ray.get(step.fn.remote(*args, **kw))

            step_fn = _call_remote

        # Use the step's resources if specified, otherwise default to CPU
        step_resources = step.resources if step.resources is not None else ResourceConfig.with_cpu(preemptible=False)
        fray_job = JobRequest(
            name=f"{get_fn_name(step.fn, short=True)}:{step.name}",
            entrypoint=Entrypoint.from_callable(step_fn, args=[config]),
            resources=step_resources,
            environment=EnvironmentConfig.create(extras=step.pip_dependency_groups or []),
        )

        runner = StepRunner(self.client, status_file)
        runner.launch(fray_job)
        return runner

    def _plan_dry_run(self, step: ExecutorStep, *, force_run_failed: bool) -> tuple[str, str]:
        """Return a dry-run action and reason without acquiring locks or writing."""
        status_file = StatusFile(self.output_paths[step], worker_id="dry-run")
        status = status_file.status

        if status == STATUS_SUCCESS:
            return ("skip", "already succeeded")

        if status in [STATUS_FAILED, STATUS_DEP_FAILED]:
            if force_run_failed:
                return ("run", f"previous status {status}; would rerun because force_run_failed=True")
            raise PreviousTaskFailedError(
                f"Step {step.name} failed previously with status {status}. "
                "Rerun with force_run_failed=True to execute again."
            )

        if status == STATUS_RUNNING:
            if status_file.has_active_lock():
                return ("wait", "currently running with an active lock")
            return ("run", "RUNNING with no active lock; would take over execution")

        return ("run", "no status recorded")

    def _record_dry_run(self, step: ExecutorStep, action: str, reason: str, output_path: str) -> None:
        """Track dry-run decisions for summary output."""
        self._dry_run_plan.append((step.name, action, reason, output_path))

    def _log_dry_run_summary(self) -> None:
        """Log a concise summary of dry-run decisions."""
        if not self._dry_run_plan:
            logger.info("### Dry run summary: no steps inspected ###")
            return

        to_run = [(name, path) for name, action, _, path in self._dry_run_plan if action == "run"]
        waiting = [name for name, action, _, _ in self._dry_run_plan if action == "wait"]
        skipped = [name for name, action, _, _ in self._dry_run_plan if action == "skip"]

        logger.info("### Dry run summary ###")
        if skipped:
            formatted = "\n".join(f"- {name}" for name in skipped)
            logger.info("Already succeeded:\n%s", formatted)

        if to_run:
            formatted = "\n".join(f"- {name} -> {path}" for name, path in to_run)
            logger.info("Would run %d step(s):\n%s", len(to_run), formatted)
        else:
            logger.info("No steps need to be launched.")

        if waiting:
            formatted = "\n".join(f"- {name}" for name in waiting)
            logger.info("Currently running (not relaunched):\n%s", formatted)

    def _compute_transitive_deps(self, steps: list[ExecutorStep], run_steps: list[str]) -> list[ExecutorStep]:
        """
        Compute the transitive dependencies of the steps that match the run_steps list.

        Returns steps in topological order.

        Args:
            steps: The list of all steps.
            run_steps: The list of step names to run. The names are matched as regex.
        """
        regexes = [re.compile(run_step) for run_step in run_steps]
        used_regexes: set[int] = set()

        def matches(step: ExecutorStep) -> bool:
            # track which regexes have been used
            for i, regex in enumerate(regexes):
                if regex.search(step.name):
                    used_regexes.add(i)
                    return True

            return False

        # Compute the transitive dependencies of the steps that match the run_steps list
        to_run: list[ExecutorStep] = []
        visited: set[ExecutorStep] = set()
        in_stack: set[ExecutorStep] = set()  # cycle detection

        def dfs(step: ExecutorStep):
            if step in in_stack:
                raise ValueError(f"Cycle detected in {step.name}")

            if step in visited:
                return

            visited.add(step)
            in_stack.add(step)

            info = self.step_infos[self.steps.index(step)]

            # only run if the step hasn't already been run
            status_file = StatusFile(info.output_path, worker_id="check")
            if status_file.status != STATUS_SUCCESS:
                for dep in self.dependencies[step]:
                    dfs(dep)
                to_run.append(step)
            else:
                logger.info(f"Skipping {step.name}'s dependencies as it has already been run")
            in_stack.remove(step)

        for step in steps:
            if matches(step):
                dfs(step)

        if used_regexes != set(range(len(regexes))):
            unused_regexes = [regexes[i].pattern for i in set(range(len(regexes))) - used_regexes]
            logger.warning(f"Regexes {unused_regexes} did not match any steps")

        return to_run

    def compute_version(self, step: ExecutorStep, is_pseudo_dep: bool):
        if step in self.versions:
            if not is_pseudo_dep and self.is_pseudo_dep[step]:
                logger.info(f"Step {step.name} was previously marked as skippable, but is not anymore.")
                self.is_pseudo_dep[step] = False

            return

        # Collect dependencies and the config version
        computed_deps = collect_dependencies_and_version(obj=step.config)
        # Recurse on dependencies
        for dep in computed_deps.dependencies:
            self.compute_version(dep, is_pseudo_dep=is_pseudo_dep)

        for dep in computed_deps.pseudo_dependencies:
            self.compute_version(dep, is_pseudo_dep=True)

        # The version specifies precisely all the information that uniquely
        # identifies this step.  Note that the fn name is not part of the
        # version.
        version = {
            "name": step.name,
            "config": computed_deps.version,
            "dependencies": [self.versions[dep] for dep in computed_deps.dependencies],
        }

        if computed_deps.pseudo_dependencies:
            # don't put this in the literal to avoid changing the hash for runs without pseudo-deps
            version["pseudo_dependencies"] = [self.versions[dep] for dep in computed_deps.pseudo_dependencies]

        # Compute output path
        version_str = json.dumps(version, sort_keys=True, cls=CustomJsonEncoder)
        hashed_version = hashlib.md5(version_str.encode()).hexdigest()[:6]
        output_path = os.path.join(self.prefix, step.name + "-" + hashed_version)

        # Override output path if specified
        override_path = step.override_output_path
        if override_path is not None:
            override_path = _make_prefix_absolute_path(self.prefix, override_path)

            if output_path != override_path:
                logger.warning(
                    f"Output path {output_path} doesn't match given "
                    f"override {step.override_output_path}, using the latter."
                )
                output_path = override_path

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
            prefix=self.prefix,
        )
        self.dependencies[step] = list(map(self.canonicalize, computed_deps.dependencies))
        self.versions[step] = version
        self.version_strs[step] = version_str
        self.output_paths[step] = output_path
        self.is_pseudo_dep[step] = is_pseudo_dep

    def canonicalize(self, step: ExecutorStep) -> ExecutorStep:
        """Multiple instances of `ExecutorStep` might have the same version."""
        return self.version_str_to_step[self.version_strs[step]]

    def get_infos(self):
        """Calculates info files for each step and also entire execution"""
        # Compute info for each step
        for step in self.steps:
            self.step_infos.append(
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
        path = get_caller_path()
        self.executor_info = ExecutorInfo(
            git_commit=get_git_commit(),
            caller_path=path,
            created_date=datetime.now().isoformat(),
            user=get_user(),
            worker_id=worker_id(),
            prefix=self.prefix,
            description=self.description,
            steps=self.step_infos,
        )

    def get_experiment_url(self) -> str:
        """Return the URL where the experiment can be viewed."""
        if self.prefix.startswith("gs://"):
            host = "https://marin.community/data-browser"
        else:
            host = f"http://localhost:{_get_local_data_browser_port()}"

        return host + "/experiment?path=" + urllib.parse.quote(self.executor_info_path)

    def write_infos(self):
        """Output JSON files (one for the entire execution, one for each step)."""

        # Set executor_info_path based on hash and caller path name (e.g., 72_baselines-8c2f3a.json)
        # import pdb; pdb.set_trace()

        # we pre-compute the asdict as it can be expensive.
        executor_info_dict = asdict_without_description(self.executor_info)
        step_infos = executor_info_dict["steps"]
        for s in step_infos:
            s.pop("description", None)

        executor_version_str = json.dumps(step_infos, sort_keys=True, cls=CustomJsonEncoder)
        executor_version_hash = hashlib.md5(executor_version_str.encode()).hexdigest()[:6]
        name = os.path.basename(self.executor_info.caller_path).replace(".py", "")
        self.executor_info_path = os.path.join(
            self.executor_info_base_path,
            f"{name}-{executor_version_hash}.json",
        )

        # Print where to find the executor info (experiments JSON)
        logger.info(f"Writing executor info to {self.executor_info_path}")
        if not self.prefix.startswith("gs://"):
            logger.info("Start data browser: cd data_browser && uv run python run-dev.py --config conf/local.conf")
        logger.info("To view the experiment page, go to:")
        logger.info("")
        logger.info(self.get_experiment_url())
        logger.info("")
        # Write out info for each step
        for step, info in zip(self.steps, executor_info_dict["steps"], strict=True):
            info_path = _get_info_path(self.output_paths[step])
            fsspec_utils.mkdirs(os.path.dirname(info_path))
            with fsspec.open(info_path, "w") as f:
                print(json.dumps(info, indent=2, cls=CustomJsonEncoder), file=f)

        # Write out info for the entire execution
        fsspec_utils.mkdirs(os.path.dirname(self.executor_info_path))
        with fsspec.open(self.executor_info_path, "w") as f:
            print(json.dumps(executor_info_dict, indent=2, cls=CustomJsonEncoder), file=f)


class PreviousTaskFailedError(Exception):
    """Raised when a step failed previously and force_run_failed is False."""

    pass


def should_run(status_file: StatusFile, step_name: str, force_run_failed: bool = True) -> bool:
    """Check if the step should run based on lease-based distributed locking.

    Uses lease files for distributed locking and status file for final state.
    """
    worker_id = status_file.worker_id
    log_once = True

    while True:
        status = status_file.status

        if log_once:
            logger.info(f"[{worker_id}] Status {step_name}: {status}")
            log_once = False

        if status == STATUS_SUCCESS:
            logger.info(f"[{worker_id}] Step {step_name} has already succeeded.")
            return False

        if status in [STATUS_FAILED, STATUS_DEP_FAILED]:
            if force_run_failed:
                logger.info(f"[{worker_id}] Force running {step_name}, previous status: {status}")
                # Fall through to acquire lock
            else:
                raise PreviousTaskFailedError(f"Step {step_name} failed previously. Status: {status}")
        elif status == STATUS_RUNNING and status_file.has_active_lock():
            # Another worker is actively running with current lease
            logger.debug(f"[{worker_id}] Step {step_name} has active lock, waiting...")
            time.sleep(5)
            continue
        elif status == STATUS_RUNNING:
            logger.info(f"[{worker_id}] Step {step_name} has no active lock, taking over.")

        logger.info(f"[{worker_id}] Attempting to acquire lock for {step_name}")
        if status_file.try_acquire_lock():
            status_file.write_status(STATUS_RUNNING)
            logger.info(f"[{worker_id}] Acquired lock for {step_name}")
            return True

        logger.info(f"[{worker_id}] Lost lock race for {step_name}, retrying...")
        time.sleep(1)


def get_fn_name(fn: ExecutorFunction, short: bool = False):
    """Just for debugging: get the name of the function."""
    if fn is None:
        return "None"
    import ray

    if isinstance(fn, ray.remote_function.RemoteFunction):
        return fn._function.__name__
    if short:
        return f"{fn.__name__}"
    else:
        return str(fn)


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
    prefix: str | None = None
    """Attached to every output path that's constructed (e.g., the GCS bucket)."""

    executor_info_base_path: str | None = None
    """Where the executor info should be stored under a file determined by a hash."""

    dry_run: bool = False
    force_run_failed: bool = True  # Force run failed steps
    run_only: list[str] | None = None
    """Run these steps (matched by regex.search) and their dependencies only. If None, run all steps."""

    max_concurrent: int | None = None
    """Maximum number of steps to run concurrently. If None, run all ready steps in parallel (default)."""


@draccus.wrap()
def executor_main(config: ExecutorMainConfig, steps: list[ExecutorStep], description: str | None = None):
    """Main entry point for experiments (to standardize)"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    time_in = time.time()

    prefix = config.prefix
    if prefix is None:
        # infer from the environment
        if "MARIN_PREFIX" in os.environ:
            prefix = os.environ["MARIN_PREFIX"]
        else:
            raise ValueError("Must specify a prefix or set the MARIN_PREFIX environment variable")
    elif "MARIN_PREFIX" in os.environ:
        if prefix != os.environ["MARIN_PREFIX"]:
            logger.warning(
                f"MARIN_PREFIX environment variable ({os.environ['MARIN_PREFIX']}) is different from the "
                f"specified prefix ({prefix})"
            )

    executor_info_base_path = config.executor_info_base_path
    if executor_info_base_path is None:
        # infer from prefix
        executor_info_base_path = os.path.join(prefix, "experiments")

    executor = Executor(
        prefix=prefix,
        executor_info_base_path=executor_info_base_path,
        description=description,
    )

    executor.run(
        steps=steps,
        dry_run=config.dry_run,
        run_only=config.run_only,
        force_run_failed=config.force_run_failed,
        max_concurrent=config.max_concurrent,
    )
    time_out = time.time()
    logger.info(f"Executor run took {time_out - time_in:.2f}s")
    # print json path again so it's easy to copy
    logger.info(f"Executor info written to {executor.executor_info_path}")
    if not executor.prefix.startswith("gs://"):
        logger.info("Start data browser: cd data_browser && uv run python run-dev.py --config conf/local.conf")
    logger.info(f"View the experiment at {executor.get_experiment_url()}")


def _is_relative_path(url_or_path):
    # if it's a url, it's not a relative path
    parsed_url = urlparse(url_or_path)

    if parsed_url.scheme:
        return False

    # otherwise if it starts with a slash, it's not a relative path
    return not url_or_path.startswith("/")


def _make_prefix_absolute_path(prefix, override_path):
    if _is_relative_path(override_path):
        override_path = os.path.join(prefix, override_path)
    return override_path
