# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact steps addressed by an explicit ``name@version``.

The model:

- An :class:`ArtifactStep` is a content-free handle to a versioned artifact:
  ``(name, version, artifact_type)`` plus how to build it — ``run``, ``build_config``, ``deps``.
  Constructing one runs nothing; :func:`run` / :meth:`ArtifactStep.resolve` execute it.
- ``build_config(ctx)`` is a pure function of a :class:`StepContext`, which draws the line between
  identity and execution: values written as literals (model, hyperparameters, dep *versions*)
  bear identity, while values *pulled* from the context — ``ctx.output_path``,
  ``ctx.artifact_path(dep)``, ``ctx.prefix``, ``ctx.region``, ``ctx.runtime_arg(key)`` — are
  where/how the step runs and never do.
- ``run(config)`` is the step function: a plain callable. It returns the produced
  :class:`~marin.execution.artifact.Artifact` (a data ref or a value), or ``None`` for a data
  step that writes its bytes to ``ctx.output_path``. To run a step on Fray, wrap your own function
  with :func:`~marin.execution.remote.remote` and pass that as ``run`` — the step layer does no
  remote dispatch and knows nothing about resources.
- Identity is the explicit ``{prefix}/{name}/{version}`` path — no content hash. The recipe
  *fingerprint* (the config built with dep versions in place of paths and placeholders for
  everything pulled from ``ctx``) is recorded for an advisory drift check, never in the path.

Pre-existing data is brought in with :func:`adopt`. :func:`run` runs handles for their side
effects; :meth:`ArtifactStep.resolve` runs one and loads its typed artifact. A custom step is
just a function that constructs and returns an ``ArtifactStep`` (see ``marin.experiment.data`` /
``marin.experiment.train`` for the dataset and training builders) — there is no generic wrapper to
learn.
"""

import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Generic, TypeVar
from urllib.parse import urlparse

from rigging.filesystem import marin_prefix, marin_region, url_to_fs

from marin.execution.artifact import (
    EXPECTED_FINGERPRINT_KEY,
    FINGERPRINT_KEY,
    RESULT_TYPE_KEY,
    VERSION_KEY,
    Artifact,
    ArtifactRecord,
    ArtifactTypeMismatchError,
    FingerprintMismatchError,
    is_mutable_version,
    read_record,
    write_record,
)
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.provenance import Provenance
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec, _is_relative_path

T = TypeVar("T", bound=Artifact)

# A calendar version: YYYY.MM.DD, optionally .N for two immutable revisions on the same day.
_CALVER_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}(\.\d+)?$")


def _artifact_path(name: str, version: str, prefix: str) -> str:
    """The explicit, hash-free address of an artifact under a storage prefix.

    A prefix may already end in ``/`` — an empty-authority scheme like ``mirror://``, or a
    ``MARIN_PREFIX`` written with a trailing slash — so the separator is not doubled.
    """
    base = prefix if prefix.endswith("/") else f"{prefix}/"
    return f"{base}{name}/{version}"


def _validate_segment(label: str, value: str) -> None:
    """A ``name``/``version`` is a path segment: non-empty, no ``..``, no leading/trailing
    slash, no URL scheme. A malformed one is a caller bug, not a silent malformed path."""
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if "://" in value or urlparse(value).scheme:
        raise ValueError(f"{label} {value!r} must not contain a URL scheme")
    if ".." in value:
        raise ValueError(f"{label} {value!r} must not contain '..'")
    if value.startswith("/") or value.endswith("/"):
        raise ValueError(f"{label} {value!r} must not start or end with '/'")


def _validate_version(version: str) -> None:
    """A version is a calendar version ``YYYY.MM.DD[.N]`` or a mutable ``dev``/``<label>-dev``.

    ``v1``-style and other ad-hoc strings are rejected: an artifact's version is the author's
    explicit statement of "when this recipe was frozen", not an opaque tag.
    """
    _validate_segment("version", version)
    if is_mutable_version(version):
        return
    if not _CALVER_RE.match(version):
        raise ValueError(
            f"version {version!r} must be a calendar version YYYY.MM.DD (optionally YYYY.MM.DD.N) "
            "or a mutable 'dev'/'<label>-dev'"
        )


@dataclass(frozen=True)
class StepContext:
    """What a step resolves its config against — and the dividing line for identity.

    Values written as literals in ``build_config`` (the model, hyperparameters, dep *versions*)
    bear identity and enter the fingerprint; values pulled from the context never do.
    :meth:`for_run` binds the live environment; :meth:`for_fingerprint` substitutes placeholders,
    which is what realizes the exclusion.

    Pull points: ``ctx.output_path`` (this step's output dir), ``ctx.artifact_path(dep)`` (a dep's
    resolved path), ``ctx.prefix`` (the live storage prefix), ``ctx.region`` (the live region),
    and ``ctx.runtime_arg(key)`` (a step-declared runtime arg, e.g. the TPU a dispatched job runs
    on). ``is_fingerprint`` is ``True`` only under :meth:`for_fingerprint`, for a ``build_config``
    that must emit an identity summary (rather than read records that do not exist yet).
    """

    output_path: str
    prefix: str
    region: str | None
    is_fingerprint: bool
    _dep_ref: Callable[["ArtifactStep"], str]
    _runtime_args: Mapping[str, Any]

    def artifact_path(self, dep: "ArtifactStep") -> str:
        """The resolved output path of a dependency this step builds on."""
        return self._dep_ref(dep)

    def runtime_arg(self, key: str) -> Any:
        """A step-declared runtime arg: its real value at run time, a ``<key>`` placeholder at
        fingerprint time. Pull execution choices through here so they reach the step but never
        bear on its identity."""
        try:
            return self._runtime_args[key]
        except KeyError:
            raise KeyError(
                f"runtime arg {key!r} is not declared in the step's runtime_args {sorted(self._runtime_args)}"
            ) from None

    @staticmethod
    def for_run(
        output_path: str, prefix: str, *, region: str | None = None, runtime_args: Mapping[str, Any] | None = None
    ) -> "StepContext":
        return StepContext(
            output_path=output_path,
            prefix=prefix,
            region=region,
            is_fingerprint=False,
            _dep_ref=lambda d: d.path(prefix),
            _runtime_args=runtime_args or {},
        )

    @staticmethod
    def for_fingerprint(runtime_arg_keys: Iterable[str] = ()) -> "StepContext":
        return StepContext(
            output_path="<output_path>",
            prefix="<prefix>",
            region="<region>",
            is_fingerprint=True,
            _dep_ref=lambda d: f"{d.name}@{d.version}",
            _runtime_args={key: f"<{key}>" for key in runtime_arg_keys},
        )


# eq=False gives handles object identity (hashable despite unhashable values in their config),
# so a handle can key a mixture dict. Value-equality lives in name@version and the fingerprint.
@dataclass(frozen=True, eq=False)
class ArtifactStep(Generic[T]):
    """A lazy, content-free handle to a versioned artifact and how to build it.

    ``run(build_config(ctx))`` produces the artifact: ``build_config`` assembles the config from
    the :class:`StepContext` (literals bear identity; pulled values do not), and ``run`` is the
    plain callable that does the work, returning the produced ``artifact_type`` (or ``None`` for a
    data step that writes to ``ctx.output_path``). ``deps`` are the handles this step builds on;
    pass the same handles ``build_config`` reads via ``ctx.artifact_path`` so they materialize
    first.
    """

    name: str
    version: str
    artifact_type: type[T]
    run: Callable[[Any], "Artifact | None"]
    build_config: Callable[[StepContext], Any]
    deps: tuple["ArtifactStep", ...] = ()
    runtime_args: Mapping[str, Any] = field(default_factory=dict)
    """Execution choices the config pulls via ``ctx.runtime_arg(key)``. Excluded from the
    fingerprint, so changing one never forks identity."""
    override_path: str | None = None
    """Pin the artifact to an existing location instead of ``{name}/{version}``. A relative
    pin is resolved against the prefix; an absolute one is used as-is. Writes no record.
    Mutually exclusive with ``adopt_source``."""
    adopt_source: str | None = None
    """Adopt pre-existing data at this location as a managed ``name@version`` (set via
    :func:`adopt`). Consumers resolve to ``adopt_source``; running it writes a provenance
    record at the canonical address. Mutually exclusive with ``override_path``."""
    adopt_config: dict | None = None
    """For an adopted artifact, the synthetic ``config`` to record (e.g. a tokenized cache's
    tokenizer/format), so consumers read metadata the same way as for a produced artifact."""
    expected_fingerprint: str | None = None
    """Opt-in hard pin: when set, :meth:`lower` raises :class:`FingerprintMismatchError` if the
    config now fingerprints to something else, and the drift check raises rather than warns. Leave
    ``None`` to opt out."""

    def __post_init__(self) -> None:
        _validate_segment("name", self.name)
        _validate_version(self.version)
        if self.adopt_source is not None and self.override_path is not None:
            raise ValueError(f"{self.name}@{self.version}: an artifact cannot be both adopted and pinned")

    def fingerprint_payload(self) -> str:
        """The canonical config bytes this handle's fingerprint hashes. For a computed
        artifact, the config built with dep versions for paths and placeholders for pulled
        values; for an adopted one, its source location and recorded config."""
        if self.adopt_source is not None:
            return json.dumps({"adopt_source": self.adopt_source, "config": self.adopt_config}, sort_keys=True)
        config = self.build_config(StepContext.for_fingerprint(self.runtime_args.keys()))
        return canonical_json(config)

    def fingerprint(self) -> str:
        """Stable id of *how* this artifact is built. Changes iff a config value or a dep
        version changes; independent of the storage prefix/region."""
        return fingerprint_hash(self.fingerprint_payload())

    def path(self, prefix: str | None = None) -> str:
        resolved_prefix = prefix or marin_prefix()
        # An adopted artifact resolves to its source; a pin to its location; else the
        # canonical name@version address.
        location = self.adopt_source if self.adopt_source is not None else self.override_path
        if location is not None:
            if _is_relative_path(location):
                return f"{resolved_prefix}/{location}"
            return location
        return _artifact_path(self.name, self.version, resolved_prefix)

    def lower(self) -> StepSpec:
        """Lower this handle graph into a ``StepSpec`` graph the ``StepRunner`` can run.

        Captures provenance once and threads it through the graph. See :func:`_lower`.
        """
        return _lower(self, Provenance.capture())

    def resolve(self: "ArtifactStep[T]", *, max_concurrent: int = 8) -> T:
        """Run this handle then load its realized, typed :class:`Artifact` via ``artifact_type.load``.

        Checks the served record's ``result_type`` matches and raises
        :class:`ArtifactTypeMismatchError` on a mismatch (a value artifact whose schema changed
        under a reused version) before loading. No build on a cache hit.
        """
        run(self, max_concurrent=max_concurrent)
        path = self.path()
        record = read_record(path)
        expected = _result_type_name(self.artifact_type)
        if record is not None and record.result_type and record.result_type != expected:
            raise ArtifactTypeMismatchError(
                f"{self.name}@{self.version}: recorded result_type is {record.result_type}, "
                f"but the handle requests {expected}. The value type changed under a reused version — "
                "bump the version."
            )
        return self.artifact_type.load(path)


def _result_type_name(artifact_type: type[Artifact]) -> str:
    return f"{artifact_type.__module__}.{artifact_type.__qualname__}"


def _adopt_noop(_config: Any) -> None:
    raise AssertionError("adopted artifacts are registered, not computed")


def adopt(
    name: str,
    version: str,
    source: str,
    *,
    kind: type[T] = Artifact,  # pyrefly: ignore[bad-function-definition]
    config: dict | None = None,
) -> "ArtifactStep[T]":
    """Register pre-existing data at ``source`` as a managed ``name@version``.

    The data is neither moved nor recomputed: a consumer that depends on the returned handle
    resolves to ``source``. Running it writes a provenance record at the canonical
    ``{prefix}/{name}/{version}`` (with ``source`` recorded). ``kind`` selects the handle's
    ``artifact_type`` for consumer routing (:class:`Artifact` by default). ``config`` records
    synthetic metadata (e.g. a tokenized cache's tokenizer/format) so consumers read it the same
    way as for a produced artifact.
    """
    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=kind,
        run=_adopt_noop,
        build_config=lambda _ctx: None,
        adopt_source=source,
        adopt_config=config,
    )


def materialized_config(handle: "ArtifactStep", prefix: str) -> Any:
    """The concrete config a run would receive, for inspection/golden tests.

    Resolves ``ctx.output_path`` / ``ctx.artifact_path(dep)`` against ``prefix`` and passes the
    handle's real ``runtime_args`` (so a ``build_config`` that reads ``ctx.runtime_arg(...)`` does
    not ``KeyError``) without running anything. Region is ``None``, so this is not identical to the
    run-time config, which also binds a real region.
    """
    return handle.build_config(
        StepContext.for_run(output_path=handle.path(prefix), prefix=prefix, runtime_args=handle.runtime_args)
    )


def _output_path_spec(handle: "ArtifactStep") -> str:
    """The relative output path StepSpec should use: the pin if set, else ``{name}/{version}``."""
    return handle.override_path or f"{handle.name}/{handle.version}"


def _lower(handle: "ArtifactStep", provenance: Provenance) -> StepSpec:
    """Lower a handle graph into a ``StepSpec`` graph, recording ``provenance`` on every step.

    Each handle becomes a step addressed by its explicit ``{name}/{version}`` (or its pin); the
    fingerprint, version, artifact type, and dep identities travel in ``hash_attrs`` so the runner
    applies the drift check before serving a cached output, and the step fn writes a full
    :class:`ArtifactRecord` on success (unless pinned). Pure transform of *structure* — it never
    inspects ``handle.run``.

    Raises :class:`FingerprintMismatchError` if ``expected_fingerprint`` is set and differs;
    :class:`ValueError` if a fixed (non-``dev``) handle has a ``dev`` dep.
    """
    dep_specs = [_lower(dep, provenance) for dep in handle.deps]
    payload = handle.fingerprint_payload()
    fingerprint = fingerprint_hash(payload)
    if handle.expected_fingerprint is not None and fingerprint != handle.expected_fingerprint:
        raise FingerprintMismatchError(
            f"{handle.name}@{handle.version}: config fingerprint is {fingerprint}, but "
            f"expected_fingerprint pins {handle.expected_fingerprint}. The recipe changed — update "
            f"the pin, and bump the version if this is meant to be a different artifact."
        )

    if not is_mutable_version(handle.version):
        for dep in handle.deps:
            if is_mutable_version(dep.version):
                raise ValueError(
                    f"{handle.name}@{handle.version} is a fixed version but depends on mutable "
                    f"{dep.name}@{dep.version}; a mutable dep would rebuild while the fixed parent stays "
                    "cached. Make the parent dev, or pin the dep."
                )

    dep_refs = [f"{d.name}@{d.version}" for d in handle.deps]
    result_type_name = _result_type_name(handle.artifact_type)
    # A pin references existing data; it writes no record. Everything else records provenance.
    record_provenance = handle.override_path is None

    def fn(output_path: str, _handle: "ArtifactStep" = handle) -> Any:
        if _handle.adopt_source is not None:
            # Adoption registers pre-existing data: no compute, the record points at it.
            source = _handle.path(marin_prefix())
            fs = url_to_fs(source, use_listings_cache=False)[0]
            if not fs.exists(source):
                raise FileNotFoundError(f"cannot adopt {_handle.name}@{_handle.version}: no data at {source}")
            result = None
            config_json: dict[str, Any] | None = _handle.adopt_config
            result_json: dict[str, Any] | None = None
        else:
            ctx = StepContext.for_run(
                output_path=output_path,
                prefix=marin_prefix(),
                region=marin_region(),
                runtime_args=_handle.runtime_args,
            )
            config = _handle.build_config(ctx)
            result = _handle.run(config)
            source = None
            config_json = json.loads(canonical_json(config))
            result_json = result.result_payload() if isinstance(result, Artifact) else None
        if record_provenance:
            write_record(
                ArtifactRecord(
                    name=_handle.name,
                    version=_handle.version,
                    fingerprint=fingerprint,
                    result_type=result_type_name,
                    output_path=output_path,
                    deps=dep_refs,
                    config=config_json,
                    source=source,
                    result=result_json,
                    fingerprint_payload=payload,
                    provenance=provenance,
                )
            )
        return result

    hash_attrs: dict[str, Any] = {
        FINGERPRINT_KEY: fingerprint,
        VERSION_KEY: handle.version,
        RESULT_TYPE_KEY: result_type_name,
        "deps": dep_refs,
    }
    if handle.expected_fingerprint is not None:
        hash_attrs[EXPECTED_FINGERPRINT_KEY] = handle.expected_fingerprint

    return StepSpec(
        name=handle.name,
        override_output_path=_output_path_spec(handle),
        deps=dep_specs,
        hash_attrs=hash_attrs,
        fingerprint_payload=payload,
        fn=fn,
        writes_record=True,
    )


def run(
    *handles: "ArtifactStep",
    max_concurrent: int = 8,
    dry_run: bool = False,
    force_run_failed: bool = True,
) -> None:
    """Lower and run ``handles`` for side effects — the entry point for one or several handles.

    Provenance is captured once for the whole invocation, so every artifact built records the
    same launch. ``dry_run`` logs without touching remote status; ``force_run_failed`` reruns a
    previously-FAILED step instead of raising.
    """
    provenance = Provenance.capture()
    StepRunner().run(
        [_lower(h, provenance) for h in handles],
        dry_run=dry_run,
        force_run_failed=force_run_failed,
        max_concurrent=max_concurrent,
    )


def lower(handle: "ArtifactStep") -> StepSpec:
    """Lower ``handle``'s graph into a ``StepSpec`` the ``StepRunner`` can run.

    The free-function form of :meth:`ArtifactStep.lower`, for the
    ``StepRunner().run([lower(step) for step in steps])`` idiom.
    """
    return handle.lower()


def resolve(handle: "ArtifactStep[T]", *, max_concurrent: int = 8) -> T:
    """Run ``handle`` then load its realized, typed :class:`Artifact`.

    The free-function form of :meth:`ArtifactStep.resolve`.
    """
    return handle.resolve(max_concurrent=max_concurrent)


class _OutSentinel:
    """The :data:`OUT` sentinel for :func:`apply`: resolves to ``ctx.output_path`` at run time."""

    def __repr__(self) -> str:
        return "OUT"


OUT: Final = _OutSentinel()


def _collect_deps(inputs: Mapping[str, Any]) -> list["ArtifactStep"]:
    """The ``ArtifactStep`` handles in ``inputs`` (recursing list/tuple/dict), deduped by
    identity, in first-seen order."""
    deps: list[ArtifactStep] = []
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, ArtifactStep):
            if id(value) not in seen:
                seen.add(id(value))
                deps.append(value)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for value in inputs.values():
        walk(value)
    return deps


def _resolve_input(value: Any, ctx: StepContext) -> Any:
    """Resolve one ``apply`` input against ``ctx``: an ``ArtifactStep`` to its path, ``OUT`` to
    ``ctx.output_path``, recursing builtin containers; anything else is a literal."""
    if isinstance(value, ArtifactStep):
        return ctx.artifact_path(value)
    if value is OUT:
        return ctx.output_path
    if isinstance(value, list):
        return [_resolve_input(item, ctx) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_input(item, ctx) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_input(item, ctx) for key, item in value.items()}
    return value


def apply(
    name: str,
    fn: Callable[..., Any],
    *,
    version: str,
    artifact_type: type[T] = Artifact,  # pyrefly: ignore[bad-function-definition]
    pin: str | None = None,
    **inputs: Any,
) -> "ArtifactStep[T]":
    """A single-step builder that calls ``fn(**inputs)`` directly — for a step whose function
    takes keyword arguments rather than a config object.

    Each value in ``inputs`` is classified, recursing into ``list``/``tuple``/``dict``: an
    :class:`ArtifactStep` handle becomes a dep and resolves to ``ctx.artifact_path(handle)`` at run
    time (its ``name@version`` enters identity); the :data:`OUT` sentinel resolves to
    ``ctx.output_path``; anything else is a literal that bears identity. To dispatch on Fray, pass
    an already-wrapped ``remote(fn, resources=…)`` as ``fn``. ``artifact_type`` selects the
    produced :class:`Artifact` type; ``pin`` references existing data.

    Raises :class:`TypeError` if ``fn``'s signature cannot bind the inputs.
    """
    try:
        inspect.signature(fn).bind(**inputs)
    except TypeError as e:
        fn_name = getattr(fn, "__name__", repr(fn))
        raise TypeError(f"apply({name!r}): {fn_name} cannot bind the given inputs: {e}") from e

    deps = _collect_deps(inputs)

    def build_config(ctx: StepContext) -> dict[str, Any]:
        return {key: _resolve_input(value, ctx) for key, value in inputs.items()}

    def step_fn(config: dict[str, Any], _fn: Callable[..., Any] = fn) -> Any:
        return _fn(**config)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=artifact_type,
        run=step_fn,
        build_config=build_config,
        deps=tuple(deps),
        override_path=pin,
    )
