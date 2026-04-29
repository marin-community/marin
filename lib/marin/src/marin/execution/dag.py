# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-graph helpers for the executor.

The walker emits typed events that callers consume:

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

import os
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, TypeVar

from fray import client as fray_client
from fray.types import Entrypoint, JobRequest, create_environment
from rigging.filesystem import marin_prefix

from marin.execution.executor import (
    Executor,
    ExecutorStep,
    InputName,
    MirroredValue,
    OutputName,
    VersionedValue,
    instantiate_config,
    output_path_of,
)
from marin.execution.remote import RemoteCallable, _sanitize_job_name

ConfigT = TypeVar("ConfigT")


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


def materialize(
    config: ConfigT,
    *,
    prefix: str | None = None,
    output_path: str | None = None,
) -> ConfigT:
    """Run any `ExecutorStep`s embedded in `config`, then return a copy of
    `config` with all placeholder paths substituted.

    Composes three pieces:

      1. ``upstream_steps(config)`` — find embedded ``ExecutorStep``s.
      2. ``Executor(prefix=...).run(steps)`` — submit them as sub-jobs and
         block on completion. The distributed ``step_lock`` keeps concurrent
         callers (sweep members, multi-VM TPU tasks) coordinated; only one
         actually runs each step, the rest read ``STATUS_SUCCESS`` and skip.
      3. ``instantiate_config(config, output_path=<resolved>,
         output_paths=executor.output_paths, prefix=prefix)`` — substitute
         ``InputName`` / ``OutputName`` / ``VersionedValue`` / ``ExecutorStep``
         placeholders using the just-computed paths.

    Args:
        config: A launcher config dataclass that may embed ``ExecutorStep``s
            and placeholder values.
        prefix: Storage prefix for newly-submitted sub-jobs. Defaults to
            ``marin_prefix()`` (the worker's regional ``gs://marin-{R}``
            bucket), so upstream data is co-located with training.
        output_path: Concrete output path for the current step, used to
            resolve ``OutputName(name=...)`` placeholders inside ``config``.
            If ``None``, ``materialize`` reads ``config.output_path``. For
            callers whose config type does not expose ``output_path``, pass
            it explicitly.

    Returns:
        A copy of ``config`` with all placeholders substituted to concrete
        paths. A config containing no placeholders round-trips unchanged
        (idempotent — no sub-jobs submitted).
    """
    resolved_prefix = prefix if prefix is not None else marin_prefix()

    steps = upstream_steps(config)

    # Idempotence guard: if no sub-steps reference the config, skip the
    # `Executor.run` path entirely. `Executor.run([])` would otherwise still
    # write out an executor-info JSON to GCS (`write_infos` runs unconditionally
    # in `executor.run`), which is both pointless and an unwanted I/O side
    # effect for a placeholder-free config.
    if steps:
        executor_info_base_path = os.path.join(resolved_prefix, "experiments")
        executor = Executor(
            prefix=resolved_prefix,
            executor_info_base_path=executor_info_base_path,
        )
        output_paths: dict[ExecutorStep, str] = executor.run(steps=steps)
    else:
        output_paths = {}

    if output_path is None:
        current_output_path = config.output_path  # type: ignore[attr-defined]
    else:
        current_output_path = output_path

    if isinstance(current_output_path, OutputName):
        raise TypeError(
            "materialize(config): output_path is still an OutputName "
            "placeholder. The launcher / job-submission layer must resolve the "
            "current step's output_path to a concrete string before calling "
            "the worker function. Got: "
            f"{current_output_path!r}"
        )

    return instantiate_config(
        config=config,
        output_path=current_output_path,
        output_paths=output_paths,
        prefix=resolved_prefix,
    )


def _resolve_step_output_path(step: ExecutorStep, *, prefix: str | None = None) -> str:
    """Compute the concrete output path for ``step`` without running anything.

    Drives ``Executor.compute_version`` (which only walks the dependency graph
    and computes hashes — no GCS I/O, no job submission) far enough to populate
    ``Executor.output_paths[step]``.
    """
    resolved_prefix = prefix if prefix is not None else marin_prefix()
    executor_info_base_path = os.path.join(resolved_prefix, "experiments")
    executor = Executor(
        prefix=resolved_prefix,
        executor_info_base_path=executor_info_base_path,
    )
    executor.compute_version(step, is_pseudo_dep=False)
    return executor.output_paths[step]


def _substitute_output_paths_only(config: ConfigT, output_path: str) -> ConfigT:
    """Replace ``OutputName`` placeholders in ``config`` with paths under
    ``output_path``. Other placeholders (``InputName``, ``ExecutorStep``,
    ``VersionedValue``, ``MirroredValue``) are left intact for the worker's
    ``materialize`` call to resolve in the worker's own region.

    Recurses *into* ``MirroredValue`` and ``VersionedValue`` to surface any
    nested ``OutputName``, but rebuilds the wrapper rather than resolving it —
    the worker still needs to see the wrapper to apply its region-aware
    semantics.
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
            return replace(obj, value=recurse(obj.value))
        # InputName / ExecutorStep are deferred to the worker's materialize().
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
            # Version tracking only matters for hash assembly, which was done
            # by `_resolve_step_output_path`; unwrap fully here.
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


def submit_step_to_iris(step: ExecutorStep) -> None:
    """Submit ``step`` as a top-level Iris job and block until it terminates.

    The submitted job runs ``step.fn(resolved_config)`` on a worker chosen by
    ``step.resources``. ``resolved_config`` is ``step.config`` with the
    current step's ``OutputName`` placeholders pre-substituted to a concrete
    path; upstream ``InputName(step=…)`` / ``ExecutorStep`` references are
    left for the worker to resolve via ``materialize`` in its own region.

    Args:
        step: The ExecutorStep to submit. Must have ``resources`` set.

    Raises:
        ValueError: if ``step.resources`` is ``None``.
        Any exception raised by the submitted job.
    """
    if step.resources is None:
        raise ValueError(
            f"submit_step_to_iris requires step.resources to be set on {step.name!r}; "
            f"got None. Set ExecutorStep(resources=ResourceConfig.with_tpu(...))."
        )

    step_output_path = _resolve_step_output_path(step)
    resolved_config = _substitute_output_paths_only(step.config, step_output_path)

    # Unwrap RemoteCallable so the inner fn runs directly in the submitted
    # job; re-submitting from inside would double-launch.
    inner_fn = step.fn.fn if isinstance(step.fn, RemoteCallable) else step.fn

    job_request = JobRequest(
        name=_sanitize_job_name(step.name),
        entrypoint=Entrypoint.from_callable(inner_fn, args=[resolved_config]),
        resources=step.resources,
        environment=create_environment(),
    )

    client = fray_client.current_client()
    handle = client.submit(job_request)
    handle.wait(raise_on_failure=True)
