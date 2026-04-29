# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data shapes and pure-substitution helpers for the executor pipeline.

This module owns the configuration-graph primitives:

  - ``ExecutorStep`` — a single step in the pipeline.
  - ``InputName`` / ``OutputName`` / ``VersionedValue`` / ``MirroredValue`` —
    placeholder values that appear inside a step's ``config`` and get
    resolved at instantiation time.
  - ``walk_config`` and ``upstream_steps`` — deterministic traversal of a
    config graph, emitting typed events at every concrete reference.
  - ``instantiate_config`` and ``resolve_local_placeholders`` — pure
    substitution passes that turn placeholder configs into concrete ones.

This module **does not** import from :mod:`marin.execution.executor`; the
dependency direction is one-way (``executor → dag``). Helpers that need
to drive an :class:`Executor` instance (``materialize``,
``compute_output_path``) live in ``executor.py``.

Walker event semantics
----------------------

  - ``InputNameEvent`` — at every concrete reference to another step. Carries
    the dotted path prefix (e.g. ``"data.cache_dir"``) and the underlying
    `InputName`. Bare `ExecutorStep` references and `InputName.hardcoded(...)`
    paths both surface as `InputNameEvent`s — the consumer decides what to do
    with them based on whether `.step` is set.
  - ``VersionedEvent`` — at every concrete `VersionedValue`. Carries the
    prefix and the wrapped value. `VersionedValue` is a traversal leaf: its
    `.value` cannot contain `InputName` or nested `VersionedValue`
    (validated by `versioned()`), so we do not descend into it.

`MirroredValue` is transparently unwrapped (no event emitted; events surface
from inside `.value`). Dataclasses, dicts, lists, tuples, sets, and
frozensets recurse element-wise; primitives are leaves.
"""

import dataclasses
import os
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from urllib.parse import urlparse

from fray.types import ResourceConfig

ConfigT = TypeVar("ConfigT")
ConfigT_co = TypeVar("ConfigT_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)

ExecutorFunction = Any  # Callable | None — kept loose to avoid pulling in remote types here.


############################################################
# Placeholder data shapes
############################################################


@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT_co]):
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
    config: ConfigT_co
    description: str | None = None

    override_output_path: str | None = None
    """Specifies the `output_path` that should be used.  Print warning if it
    doesn't match the automatically computed one."""

    resources: ResourceConfig | None = None
    """If set, this step is submitted as its own Fray job using these
    resources. ``fn`` is invoked inside the submitted job.

    If ``None``, behavior is determined by ``fn``: a ``RemoteCallable``
    submits as a Fray job; a plain callable runs inline in-process.
    """

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
        if isinstance(obj, MirroredValue):
            return recurse(obj.value)
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


@dataclass(frozen=True)
class MirroredValue(Generic[T_co]):
    """Wraps a path value to signal that it should be mirrored from any marin regional bucket.

    At config instantiation time, the path is resolved to the local marin prefix.
    Before step execution, the executor copies the data from whichever region has it.
    """

    value: T_co
    budget_gb: float = 10


def mirrored(value: str | VersionedValue[str], budget_gb: float = 10) -> MirroredValue:
    """Mark a path for cross-region mirroring with a transfer budget.

    Usage: input_path=mirrored(versioned("documents/stackexchange/..."), budget_gb=50)
    """
    if isinstance(value, MirroredValue):
        raise ValueError("Can't nest MirroredValue")
    return MirroredValue(value=value, budget_gb=budget_gb)


############################################################
# Walker
############################################################


@dataclass(frozen=True)
class InputNameEvent:
    """Emitted when the walker encounters an `InputName` (or a bare
    `ExecutorStep`, which is canonicalized to `InputName(step=step, name=None)`
    before yielding).

    `prefix` is the dotted path from the root config to this reference,
    matching the version-string keys used by
    `collect_dependencies_and_version` (e.g. `"data.cache_dir"`).
    """

    prefix: str
    input_name: InputName


@dataclass(frozen=True)
class VersionedEvent:
    """Emitted at a concrete `VersionedValue`. `prefix` is the dotted path."""

    prefix: str
    value: Any


_Event = InputNameEvent | VersionedEvent


def walk_config(obj: Any) -> Iterator[_Event]:
    """Walk `obj` in deterministic order, yielding `InputNameEvent` and
    `VersionedEvent` instances.

    Order:
      - Dataclasses: `dataclasses.fields` declaration order.
      - Dicts: insertion order (values only; keys must be `str`).
      - Lists, tuples, sets, frozensets: iteration order.

    `MirroredValue` is unwrapped transparently. `None` and primitives are
    leaves.
    """
    yield from _walk(obj, "")


def _walk(obj: Any, prefix: str) -> Iterator[_Event]:
    new_prefix = prefix + "." if prefix else ""

    if obj is None:
        return

    if isinstance(obj, ExecutorStep):
        # Canonicalize a bare ExecutorStep to InputName(step=step, name=None)
        # so consumers only need to handle one shape.
        yield InputNameEvent(prefix, output_path_of(obj, None))
        return

    if isinstance(obj, InputName):
        yield InputNameEvent(prefix, obj)
        return

    if isinstance(obj, MirroredValue):
        yield from _walk(obj.value, prefix)
        return

    if isinstance(obj, VersionedValue):
        yield VersionedEvent(prefix, obj.value)
        return

    if is_dataclass(obj) and not isinstance(obj, type):
        for field in fields(obj):
            yield from _walk(getattr(obj, field.name), new_prefix + field.name)
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(f"dict keys must be strs, but got {key} (type: {type(key)})")
            yield from _walk(value, new_prefix + key)
        return

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk(item, new_prefix + f"[{i}]")
        return

    if isinstance(obj, (tuple, set, frozenset)):
        # sets/frozensets are unordered; callers must not put hashing-relevant
        # types inside them since iteration order would affect emitted prefixes.
        for i, item in enumerate(obj):
            yield from _walk(item, new_prefix + f"[{i}]")
        return

    # Primitives and unrecognized types are leaves.


def upstream_steps(obj: Any) -> list[ExecutorStep]:
    """Recursively walk `obj` and return every `ExecutorStep` referenced from it.

    Walks dataclasses (via `dataclasses.fields`), dicts (values),
    lists/tuples/sets (elements), and `ExecutorStep` instances themselves. The
    same step appearing multiple times in the object graph is returned exactly
    once. Order is deterministic (depth-first, fields/keys/elements in
    declaration order).

    Does NOT walk into the returned steps' configs — it returns the steps the
    caller's `obj` references directly. Transitive dependencies are discovered
    by `Executor.run()` itself, which already walks step configs to build its
    dependency graph.

    Args:
        obj: Any object — typically a config dataclass like
            `GrugBaseLaunchConfig`, but accepts any value.

    Returns:
        Deterministically ordered list of unique `ExecutorStep` instances.
    """
    seen: set[int] = set()
    result: list[ExecutorStep] = []
    for event in walk_config(obj):
        if not isinstance(event, InputNameEvent):
            continue
        step = event.input_name.step
        if step is None:
            # `InputName.hardcoded(path)` — no upstream step, nothing to track.
            continue
        key = id(step)
        if key in seen:
            continue
        seen.add(key)
        result.append(step)
    return result


############################################################
# Substitution
############################################################


def _is_relative_path(url_or_path: str) -> bool:
    parsed_url = urlparse(url_or_path)
    if parsed_url.scheme:
        return False
    return not url_or_path.startswith("/")


def _make_prefix_absolute_path(prefix: str, override_path: str) -> str:
    if _is_relative_path(override_path):
        override_path = os.path.join(prefix, override_path)
    return override_path


def instantiate_config(config: Any, output_path: str, output_paths: dict[ExecutorStep, str], prefix: str) -> Any:
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

        if isinstance(obj, MirroredValue):
            inner = recurse(obj.value)
            # Resolve to mirror:// protocol — MirrorFileSystem handles cross-region copying
            if isinstance(inner, str) and not inner.startswith("mirror://"):
                return f"mirror://{inner}"
            return inner

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


def resolve_local_placeholders(config: ConfigT, output_path: str) -> ConfigT:
    """Resolve every placeholder that the *caller* can resolve locally:
    ``OutputName`` substitutions and ``VersionedValue`` unwrapping.

    ``InputName(step=…)`` and bare ``ExecutorStep`` references are deferred
    for the worker's ``materialize`` call (which resolves them under the
    worker's region). ``MirroredValue`` is preserved (rebuilt around its
    recursed inner value); its meaning is region-aware so resolution belongs
    on the worker.

    Use this when the caller needs a config it can read concrete values out
    of (e.g. ``launch.mp``, ``launch.batch_size``) to assemble a downstream
    config tree before submission.
    """

    def join_path(name: str | None) -> str:
        return os.path.join(output_path, name) if name else output_path

    def recurse(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, OutputName):
            return join_path(obj.name)
        if isinstance(obj, MirroredValue):
            return replace(obj, value=recurse(obj.value))
        if isinstance(obj, VersionedValue):
            # Version tracking only matters for hash assembly, which the
            # caller has already performed via the executor's version pass;
            # unwrap fully here.
            return recurse(obj.value)
        if isinstance(obj, (InputName, ExecutorStep)):
            return obj
        if is_dataclass(obj) and not isinstance(obj, type):
            updates = {field.name: recurse(getattr(obj, field.name)) for field in fields(obj)}
            return replace(obj, **updates)
        if isinstance(obj, list):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items()}
        return obj

    return recurse(config)
