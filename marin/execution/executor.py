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
import re
import subprocess
import time
import traceback
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime
from typing import Any, Generic, TypeVar
from urllib.parse import urlparse

import draccus
import fsspec
import ray
import ray.remote_function
from ray.runtime_env import RuntimeEnv
from ray.util import state  # noqa
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from marin.execution.executor_step_status import (
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAITING,
    is_failure,
)
from marin.execution.status_actor import StatusActor
from marin.utilities.executor_utils import compare_dicts, get_pip_dependencies
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


def output_path_of(step: ExecutorStep, name: str | None = None) -> InputName:
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


def unwrap_versioned_value(value: VersionedValue[T_co] | T_co) -> T_co:
    """
    Unwrap the value if it is a VersionedValue, otherwise return the value as is.

    Sometimes we need to actually use a value that is wrapped in a VersionedValue before it is used in a config.
    """
    return value.value if isinstance(value, VersionedValue) else value


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
        self.step_infos: list[ExecutorStepInfo] = []
        self.executor_info: ExecutorInfo | None = None
        self.status_actor: StatusActor = StatusActor.options(
            name="status_actor",
            get_if_exists=True,
            lifetime="detached",
            # This is to ensure that the status actor is only schduled on the headnode
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote()
        # TODO: Add a design docstring of how status_actor works

    def run(
        self,
        steps: list[ExecutorStep | InputName],
        *,
        dry_run: bool = False,
        run_only: list[str] | None = None,
        force_run_failed: bool = False,
    ):
        """Run the pipeline of `ExecutorStep`s."""
        self._run(
            steps, dry_run=dry_run, run_only=run_only, force_run_failed=force_run_failed, wait_for_status_actor=False
        )

    def _run(
        self,
        steps: list[ExecutorStep | InputName],
        *,
        dry_run: bool = False,
        run_only: list[str] | None = None,
        force_run_failed: bool = False,
        wait_for_status_actor: bool = False,
    ):
        """
        Run the pipeline of `ExecutorStep`s.

        Args:
            step: The step to run.
            dry_run: If True, only print out what needs to be done.
            run_only: If not None, only run the steps in the list and their dependencies. Matches steps' names as regex
            force_run_failed: If True, run steps even if they have already been run (including if they failed)
            wait_for_status_actor: If True, wait for the status actor updates to be written before returning. This can
            be used in unit testing or to make sure that the status updates are written before moving further.
        """

        # Gather all the steps, compute versions and output paths for all of them.
        logger.info(f"### Inspecting the {len(steps)} provided steps ###")
        for step in steps:
            if isinstance(step, InputName):  # Interpret InputName as the underlying step
                step = step.step
            self.compute_version(step)

        self.get_infos()
        logger.info(f"### Reading {len(self.steps)} statuses ###")
        self.read_statuses()

        if run_only is not None:
            steps_to_run = self._compute_transitive_deps(self.steps, run_only)
        else:
            steps_to_run = self.steps

        if steps_to_run != self.steps:
            logger.info(f"### Running {len(steps_to_run)} steps out of {len(self.steps)} ###")

        logger.info(f"### Launching {len(steps_to_run)} steps ###")
        for step in steps_to_run:
            self.run_step(
                step, dry_run=dry_run, force_run_failed=force_run_failed, wait_for_status_actor=wait_for_status_actor
            )

        logger.info("### Writing metadata ###")
        self.write_infos()

        logger.info("### Waiting for all steps to finish ###")
        ray.get(list(self.refs.values()))

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

            # only run if the step hasn't already been run
            if self.statuses.get(step) not in [STATUS_SUCCESS]:
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

    def compute_hashed_version_str(self, step: ExecutorStep) -> str:
        version = {
            "name": step.name,
            "config": self.versions[step]["config"],
            "dependencies": [self.versions[dep] for dep in self.dependencies[step]],
        }
        return json.dumps(version, sort_keys=True, cls=CustomJsonEncoder)

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
        override_path = step.override_output_path
        if override_path is not None:
            if _is_relative_path(override_path):
                override_path = os.path.join(self.prefix, override_path)

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
        )
        self.dependencies[step] = list(map(self.canonicalize, dependencies))
        self.versions[step] = version
        self.version_strs[step] = version_str
        self.output_paths[step] = output_path

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
        self.executor_info = ExecutorInfo(
            git_commit=get_git_commit(),
            caller_path=get_caller_path(),
            created_date=datetime.now().isoformat(),
            user=get_user(),
            ray_job_id=ray.get_runtime_context().get_job_id(),
            prefix=self.prefix,
            description=self.description,
            steps=self.step_infos,
        )

    def get_experiment_url(self) -> str:
        """Return the URL where the experiment can be viewed."""
        # TODO: remove hardcoding
        if self.prefix.startswith("gs://"):
            host = "https://marlin-subtle-barnacle.ngrok-free.app"
        else:
            host = "http://localhost:5000"

        return host + "/experiment?path=" + urllib.parse.quote(self.executor_info_path)

    def write_infos(self):
        """Output JSON files (one for the entire execution, one for each step)."""

        # Set executor_info_path based on hash and caller path name (e.g., 72_baselines-8c2f3a.json)
        # import pdb; pdb.set_trace()
        executor_version_str = json.dumps(
            list(map(asdict_without_description, self.step_infos)), sort_keys=True, cls=CustomJsonEncoder
        )
        executor_version_hash = hashlib.md5(executor_version_str.encode()).hexdigest()[:6]
        name = os.path.basename(self.executor_info.caller_path).replace(".py", "")
        self.executor_info_path = os.path.join(
            self.executor_info_base_path,
            f"{name}-{executor_version_hash}.json",
        )

        # Print where to find the executor info (experiments JSON)
        logger.info(f"Writing executor info to {self.executor_info_path}")
        logger.info("To view the experiment page, go to:")
        logger.info("")
        logger.info(self.get_experiment_url())
        logger.info("")

        # Write out info for each step
        for step, info in zip(self.steps, self.step_infos, strict=True):
            info_path = get_info_path(self.output_paths[step])
            with fsspec.open(info_path, "w") as f:
                print(json.dumps(asdict(info), indent=2, cls=CustomJsonEncoder), file=f)

        # Write out info for the entire execution
        with fsspec.open(self.executor_info_path, "w") as f:
            print(json.dumps(asdict(self.executor_info), indent=2, cls=CustomJsonEncoder), file=f)

    def read_statuses(self):
        """Read the statuses of all the steps."""

        output_paths_list = list(self.output_paths.values())

        status_values = ray.get(self.status_actor.get_statuses.remote(output_paths_list))
        self.statuses = {key: status for key, status in zip(self.output_paths.keys(), status_values, strict=False)}

    def run_step(self, step: ExecutorStep, dry_run: bool, force_run_failed: bool, wait_for_status_actor: bool) -> None:
        """
        Return a Ray object reference to the result of running the `step`.

        If the step has already been run, returns the result of the previous run.

        Args:
            step: The step to run.
            dry_run: If True, only print out what needs to be done.
            force_run_failed: If True, run step even if is already ran (including if it failed)
            wait_for_status_actor: If True, execute_after_dependencies waits for the status actor updates to be
             written before returning. This is mainly used in unit testing.
        """
        config = self.configs[step]
        config_version = self.versions[step]["config"]
        output_path = self.output_paths[step]

        # Note that this status might be stale, since it might have been overwritten by another experiment since
        # we read it.
        # Nonetheless, we will either decide to run this step or not based on the read status.
        # If we do not run this step, it's not a problem. If we do decide to run it,
        # there is a final check in execute_after_dependencies function to ensure that we do not execute the
        # same step concurrently.
        # TODO(abhinav): There is a small chance that we might run same step twice due to stale status.
        # In particular, if the stale status is None, but the step actually succeed and lock was released.
        # We can fix this by making status opaque to executor and keeping the status only inside the actor.
        status = self.statuses[step]

        # Print information about this step
        logger.info(f"[{status}] {step.name}: {get_fn_name(step.fn)}")
        logger.info(f"  output_path = {output_path}")
        logger.info(f"  config = {json.dumps(config_version, cls=CustomJsonEncoder)}")
        for i, dep in enumerate(self.dependencies[step]):
            logger.info(f"  {dependency_index_str(i)} = {self.output_paths[dep]}")

        should_run = status is None
        if force_run_failed and is_failure(status):
            logger.info(f"Force running {step.name}, previous status: {status}")
            should_run = True

        if not should_run:
            # We skip running a step if we find a SUCCESS file for a step, but here we compare the complete info
            # to show if previous info and current info match so we aren't accidentally using the wrong version.
            # This is important since we aren't versioning everything
            # Compare the info files too and print the diff
            info_path = get_info_path(output_path)
            with fsspec.open(info_path, "r") as f:
                previous_info = json.load(f)
            step_idx = self.steps.index(step)
            current_info = json.loads(json.dumps(asdict(self.step_infos[step_idx]), indent=2, cls=CustomJsonEncoder))
            if not compare_dicts(previous_info, current_info):
                logger.warning(
                    f"The current and previous info files are not same for {step.name} "
                    f"and executor will override the previous info-file."
                )

        dependencies = self._get_refs_for_active_deps(step)
        name = f"execute_after_dependencies({get_fn_name(step.fn, short=True)})::{step.name})"

        if step.pip_dependency_groups is not None:
            pip_dependencies = get_pip_dependencies(step.pip_dependency_groups)
        else:
            pip_dependencies = None

        if self.statuses[step] in [STATUS_SUCCESS]:
            logger.info(f"Skipping {step.name} as it has already been run")

        if should_run and not dry_run and self.statuses[step] not in [STATUS_SUCCESS]:
            self.refs[step] = execute_after_dependencies.options(
                name=name,
                runtime_env=RuntimeEnv(
                    pip=pip_dependencies,
                ),
            ).remote(step.fn, config, dependencies, output_path, self.status_actor, wait_for_status_actor)
        else:
            self.refs[step] = ray.put(None)  # Necessary as we call ray.get on all the deps in execute_after_dependencies

    def _get_refs_for_active_deps(self, step):
        """
        Get the references for the active dependencies of the step. Active means dependencies
        that are not already in SUCCESS state.
        """
        out = []
        for dep in self.dependencies[step]:
            if self.statuses[dep] not in [STATUS_SUCCESS]:
                out.append(self.refs[dep])

        return out


def asdict_without_description(obj: dataclass) -> dict[str, Any]:
    """Return the `asdict` of an object, but remove the `description` field, because it doesn't affect the semantics."""
    d = asdict(obj)
    d.pop("description", None)
    return d


def _release_lock_and_wait_for_status_actor(
    status_actor: StatusActor, output_path: str, actor_refs: list[ray.ObjectRef], wait_for_status_actor: bool
):
    """
    Release the lock and wait for the status actor to finish writing the updates if wait_for_status_actor is True.
    """
    # Release the lock
    actor_refs.append(status_actor.release_lock.remote(output_path))
    if wait_for_status_actor:
        # Wait for the status actor to finish writing the updates
        ray.get(actor_refs)


def _get_lock_or_wait_for_step_with_lock(
    output_path: str, status_actor: StatusActor, ray_task_id: str
) -> list[bool, bool]:
    """This function tries to get the lock on a particular output path. Incase it can't get the lock, it waits
    for the step that has the lock on the particular output path, to finish or fail."""

    # Try and get the lock from the status actor
    actor_lock_task_id = ray.get(status_actor.get_lock.remote(output_path, ray_task_id=ray_task_id))

    if actor_lock_task_id != ray_task_id:
        # Lock is with some other step. Wait for the other step to finish or fail, and propagate accordingly
        while True:
            actor_task_state = ray.util.state.get_task(actor_lock_task_id)

            if type(actor_task_state) is list:  # Due to retires in ray, task_state can be a list of states
                actor_task_state = actor_task_state[-1]

            # Sometimes the actor_state is not ready
            if actor_task_state is None:
                logger.warning(
                    f"Status for step/task {actor_lock_task_id} is not ready. "
                    f"{actor_lock_task_id} has the lock"
                    f"for {output_path}. Waiting for the status to be ready."
                )
            elif actor_task_state.state == "FINISHED":  # The other step with the lock has finished successfully
                logger.info(
                    f"The step with task id: {actor_lock_task_id} has succeeded. Since {actor_lock_task_id} "
                    f"had the lock for {output_path}, we are returning the current step with task id "
                    f"{ray_task_id} without running but considering success."
                )
                return False, True
            elif actor_task_state.state == "FAILED":  # The other step with the lock has failed, raise this exception
                raise Exception(
                    f"The step with task id: {actor_lock_task_id} has failed. Since {actor_lock_task_id} "
                    f"had the lock for {output_path}, we are failing the current step with task id "
                    f"{ray_task_id}"
                )
            else:
                logger.info(
                    f"Task {actor_lock_task_id} is {actor_task_state.state}. {actor_lock_task_id} has the"
                    f"lock for {output_path}. Waiting for it to finish or fail."
                )

            time.sleep(5)  # Wait for 5 seconds and check again
    else:
        # Lock is with this step
        return True, False


@ray.remote
def execute_after_dependencies(
    fn: ExecutorFunction,
    config: dataclass,
    dependencies: list[ray.ObjectRef],
    output_path: str,
    status_actor: StatusActor,
    wait_for_status_actor: bool = False,
):
    """
    Run a function `fn` with the given `config`, after all the `dependencies` have finished.
    Only do stuff if `should_run` is True.
    """
    ray_task_id = ray.get_runtime_context().get_task_id()
    try:
        has_lock, finished = _get_lock_or_wait_for_step_with_lock(output_path, status_actor, ray_task_id)
        if finished:  # Step with lock finished successfully
            return
    except Exception as e:
        raise e

    assert has_lock, f"Must have the lock on {output_path} before proceeding"
    # Lock is with this process, we can proceed

    actor_refs: list[ray.ObjectRef] = []  # Holds references to ray remote calls made by this functions

    actor_refs.append(status_actor.update_status.remote(output_path, STATUS_WAITING, ray_task_id=ray_task_id))

    # Get all the dependencies
    try:
        ray.get(dependencies)
    except Exception as e:
        # Failed due to some exception
        message = traceback.format_exc()
        actor_refs.append(
            status_actor.update_status.remote(output_path, STATUS_DEP_FAILED, message=message, ray_task_id=ray_task_id)
        )
        # Release the lock
        _release_lock_and_wait_for_status_actor(status_actor, output_path, actor_refs, wait_for_status_actor)

        raise e

    # Call fn(config)
    actor_refs.append(status_actor.update_status.remote(output_path, STATUS_RUNNING, ray_task_id=ray_task_id))
    try:
        if isinstance(fn, ray.remote_function.RemoteFunction):
            ray.get(fn.remote(config))
        elif isinstance(fn, Callable):
            fn(config)
        else:
            raise ValueError(f"Expected a Callable or Ray function, but got {fn}")
    except Exception as e:
        # Failed due to some exception
        message = traceback.format_exc()
        actor_refs.append(
            status_actor.update_status.remote(output_path, STATUS_FAILED, message=message, ray_task_id=ray_task_id)
        )
        # Release the lock
        _release_lock_and_wait_for_status_actor(status_actor, output_path, actor_refs, wait_for_status_actor)
        raise e

    # Success
    actor_refs.append(status_actor.update_status.remote(output_path, STATUS_SUCCESS, ray_task_id=ray_task_id))

    # Release the lock
    _release_lock_and_wait_for_status_actor(status_actor, output_path, actor_refs, wait_for_status_actor)


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
    prefix: str | None = None
    """Attached to every output path that's constructed (e.g., the GCS bucket)."""

    executor_info_base_path: str | None = None
    """Where the executor info should be stored under a file determined by a hash."""

    dry_run: bool = False
    force_run_failed: bool = False  # Force run failed steps
    run_only: list[str] | None = None
    """Run these steps (matched by regex.search) and their dependencies only. If None, run all steps."""


@draccus.wrap()
def executor_main(config: ExecutorMainConfig, steps: list[ExecutorStep], description: str | None = None):
    """Main entry point for experiments (to standardize)"""
    ray.init(namespace="marin")  # We need to init ray here to make sure we have the correct namespace for actors
    # (status_actor in particular)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

    executor.run(steps=steps, dry_run=config.dry_run, run_only=config.run_only, force_run_failed=config.force_run_failed)


def _is_relative_path(url_or_path):
    # if it's a url, it's not a relative path
    parsed_url = urlparse(url_or_path)

    if parsed_url.scheme:
        return False

    # otherwise if it starts with a slash, it's not a relative path
    return not url_or_path.startswith("/")
