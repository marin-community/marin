# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy artifacts addressed by an explicit ``name@version``.

Prototype of the executor redesign (see
``.agents/projects/executor-lazy-artifact-extraction.md``).

The model:

- An :class:`Artifact` is a lazy handle: ``(name, version, recipe)``. Building one
  runs nothing.
- A :class:`Recipe` is ``(fn, build_config, deps, run_args)``. ``build_config`` is a
  pure function of a :class:`RunContext`, which draws the line between identity and
  execution: values written as literals (model, hyperparameters) bear identity,
  while values *pulled* from the context — ``ctx.out``, ``ctx.path(dep)``,
  ``ctx.prefix``, ``ctx.region``, ``ctx.run_arg(key)`` — are where/how the step runs
  and never do. This replaces ``THIS_OUTPUT_PATH`` and ``InputName``/``.cd()`` with
  plain typed calls.
- ``fn(config) -> result`` is the step function. The config already carries its
  output path (``ctx.out``), matching the existing ``ExecutorStep`` fn convention. To
  run the step on an accelerator, ``fn`` is a
  :class:`~marin.execution.remote.RemoteCallable` (``remote(fn, resources=…)``): the
  compute rides with the function, so it never touches the graph node or the
  fingerprint — there is no resources field on the step.
- Identity is the explicit ``{prefix}/{name}/{version}`` path — no hash.
- The *recipe fingerprint* is the config built with dependency **versions** in
  place of paths and context placeholders for everything pulled from ``ctx`` (so it
  captures hyperparameters + dep identities, but not the region, prefix, or the TPU
  a job runs on). It is recorded for the build-once-immutability guard, never in the
  path.

Pre-existing data is brought in with :func:`adopt`: it registers a ``name@version``
that points at data already on disk (no move, no recompute) while still recording
provenance and obeying the build-once guard.

:func:`lower` turns a pure handle graph into the existing :class:`StepSpec` graph,
which the existing ``StepRunner`` runs (cache → lock → run) — no content-addressing.
"""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rigging.filesystem import marin_prefix, marin_region, url_to_fs

from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.provenance import created_now, get_git_commit, get_user
from marin.execution.registry import (
    FINGERPRINT_KEY,
    VERSION_KEY,
    ArtifactRecord,
    FingerprintMismatchError,
    write_record,
)
from marin.execution.step_spec import StepSpec, _is_relative_path


def _artifact_path(name: str, version: str, prefix: str) -> str:
    """The explicit, hash-free address of an artifact under a storage prefix."""
    return f"{prefix}/{name}/{version}"


@dataclass(frozen=True)
class RunContext:
    """What a recipe resolves its config against — and the dividing line for identity.

    A config has two kinds of inputs, and the context is how they are kept apart:

    - **Identity-bearing** values are written as literals in ``build_config`` (the
      model, hyperparameters, dep *versions*). They define the artifact, so they
      enter the fingerprint.
    - **Non-identity** values are *pulled from the context* — the output path, the
      storage prefix, the region, and recipe-declared *run-args* (compute, etc.).
      They are where/how the step runs, not what it computes, so they must never
      fork identity.

    The rule is uniform: *pulled from ``ctx`` ⇒ not in the fingerprint; written as a
    literal ⇒ in the fingerprint.* :meth:`for_run` binds the context to the live
    environment at lazy-evaluation time; :meth:`for_fingerprint` substitutes
    placeholders, which is what realizes the exclusion (a region move, a prefix
    move, or a different TPU never re-fingerprints).

    Pull points:

    - ``ctx.out`` — this artifact's output path
    - ``ctx.path(dep)`` — a dependency's resolved (region-local) output path
    - ``ctx.prefix`` — the live storage prefix (resolve raw input globs against it)
    - ``ctx.region`` — the live GCP region (unknown until run time)
    - ``ctx.run_arg(key)`` — a recipe-declared run-arg, e.g. the TPU a dispatched
      job runs on
    """

    out: str
    prefix: str
    region: str | None
    _dep_ref: Callable[["Artifact"], str]
    _run_args: Mapping[str, Any]

    def path(self, dep: "Artifact") -> str:
        return self._dep_ref(dep)

    def run_arg(self, key: str) -> Any:
        """A recipe-declared run-arg: its real value at run time, a ``<key>``
        placeholder at fingerprint time. Pull execution choices (e.g. the TPU a
        dispatched job uses) through here so they reach the step but never bear on
        its identity."""
        try:
            return self._run_args[key]
        except KeyError:
            raise KeyError(
                f"run-arg {key!r} is not declared in the recipe's run_args {sorted(self._run_args)}"
            ) from None

    @staticmethod
    def for_run(
        out: str, prefix: str, *, region: str | None = None, run_args: Mapping[str, Any] | None = None
    ) -> "RunContext":
        return RunContext(
            out=out, prefix=prefix, region=region, _dep_ref=lambda d: d.path(prefix), _run_args=run_args or {}
        )

    @staticmethod
    def for_fingerprint(run_arg_keys: Iterable[str] = ()) -> "RunContext":
        return RunContext(
            out="<out>",
            prefix="<prefix>",
            region="<region>",
            _dep_ref=lambda d: f"{d.name}@{d.version}",
            _run_args={key: f"<{key}>" for key in run_arg_keys},
        )


@dataclass(frozen=True)
class Recipe:
    """How to build an artifact: the step fn, a config builder, deps, run-args."""

    fn: Callable[[Any], Any]
    """``fn(config)``, or a :class:`~marin.execution.remote.RemoteCallable`
    (``remote(fn, resources=…)``) to run the step on Fray with those resources.

    The contract is that a step produces its artifact by writing serialized data to
    its output path (``config`` carries it as ``ctx.out``) — this holds uniformly
    whether the step runs inline or on a Fray worker. As a convenience, an inline
    step may instead ``return`` a value and the runner persists it; a remote step
    cannot (a Fray job returns nothing to the caller), so it must write ``ctx.out``.

    Compute rides with the function, never on the step node, so it does not enter the
    fingerprint."""
    build_config: Callable[[RunContext], Any]
    """``build_config(ctx) -> config``. Pulls paths and live attributes from the RunContext."""
    deps: tuple["Artifact", ...] = ()
    run_args: Mapping[str, Any] = field(default_factory=dict)
    """Execution choices the config pulls via ``ctx.run_arg(key)`` — e.g. the TPU a
    dispatched job uses. Excluded from the fingerprint, so changing one never forks
    identity. A recipe may declare several (it may dispatch several jobs)."""


# eq=False gives handles object identity (hashable despite the unhashable
# ResourceConfig in their recipe), so a handle can key a mixture dict. Identity is
# the right model anyway: two handles are "the same" iff they are the same object;
# value-equality lives in name@version and the fingerprint.
@dataclass(frozen=True, eq=False)
class Artifact:
    """A lazy, content-free handle to a versioned artifact."""

    name: str
    version: str
    recipe: Recipe
    override_path: str | None = None
    """Pin the artifact to an existing location instead of ``{name}/{version}``.

    Used to reference already-materialized data (e.g. a content-addressed tokenize
    cache) without recomputing it. A relative pin is resolved against the prefix; an
    absolute one is used as-is. The pin does not affect identity/fingerprint."""
    adopt_source: str | None = None
    """Adopt pre-existing data at this location as a managed ``name@version``.

    Unlike a pin, an adopted artifact is *registered*: consumers resolve to
    ``adopt_source`` (the data is neither moved nor recomputed), but lowering writes a
    provenance record at the canonical ``{prefix}/{name}/{version}`` so the build-once
    guard governs the alias. Set via :func:`adopt`. A relative source is resolved
    against the prefix; an absolute one is used as-is. The source bears identity:
    re-adopting ``name@version`` from a different source re-fingerprints and is a
    guarded conflict."""
    expected_fingerprint: str | None = None
    """Pin the recipe fingerprint. When set, :func:`lower` raises
    :class:`~marin.execution.registry.FingerprintMismatchError` if the config now
    fingerprints to something else — making the config↔identity contract explicit and
    review-visible, and catching an edit *before* the first build (the build-once guard
    only fires once a record exists). Leave ``None`` to opt out."""

    def __post_init__(self) -> None:
        if self.adopt_source is not None and self.override_path is not None:
            raise ValueError(f"{self.name}@{self.version}: an artifact cannot be both adopted and pinned")

    def fingerprint_payload(self) -> str:
        """The canonical config bytes this artifact's fingerprint hashes
        (:mod:`marin.execution.fingerprint`).

        For a computed artifact, the config built with dependency versions in place of
        paths and context placeholders for everything pulled from ``ctx``. For an
        adopted artifact, its source location (so re-adopting from a different source
        re-fingerprints and trips the guard)."""
        if self.adopt_source is not None:
            return json.dumps({"adopt_source": self.adopt_source}, sort_keys=True)
        config = self.recipe.build_config(RunContext.for_fingerprint(self.recipe.run_args.keys()))
        return canonical_json(config)

    def fingerprint(self) -> str:
        """Stable id of *how* this artifact is built. Changes iff a config value or a
        dep version changes; independent of the storage prefix/region."""
        return fingerprint_hash(self.fingerprint_payload())

    def path(self, prefix: str | None = None) -> str:
        resolved_prefix = prefix or marin_prefix()
        # An adopted artifact resolves to its source data; a pin to its pinned location;
        # otherwise to the canonical name@version address.
        location = self.adopt_source if self.adopt_source is not None else self.override_path
        if location is not None:
            if _is_relative_path(location):
                return f"{resolved_prefix}/{location}"
            return location
        return _artifact_path(self.name, self.version, resolved_prefix)


@dataclass(frozen=True, eq=False)
class Dataset(Artifact):
    """A tokenized dataset handle (routes to dataset consumers)."""


@dataclass(frozen=True, eq=False)
class Checkpoint(Artifact):
    """A Levanter checkpoint handle (routes to ``initialize_from_checkpoint_path``)."""


def _adopt_noop(config: Any) -> None:
    raise AssertionError("adopted artifacts are registered, not computed")


_ADOPT_RECIPE = Recipe(fn=_adopt_noop, build_config=lambda _ctx: None)


def adopt(name: str, version: str, source: str, *, kind: type[Artifact] = Dataset) -> Artifact:
    """Register pre-existing data at ``source`` as a managed ``name@version``.

    The data is neither moved nor recomputed: a consumer that depends on the returned
    handle resolves to ``source``. Lowering writes a provenance record at the canonical
    ``{prefix}/{name}/{version}`` (with ``source`` recorded) so the build-once guard
    governs the alias — re-adopting ``name@version`` from a different source raises
    :class:`~marin.execution.registry.ImmutableArtifactError`. A relative source is
    resolved against the prefix; an absolute one is used as-is.

    ``kind`` selects the handle type for consumer routing (:class:`Dataset` by default,
    or :class:`Checkpoint`).
    """
    return kind(name=name, version=version, recipe=_ADOPT_RECIPE, adopt_source=source)


def materialized_config(artifact: Artifact, prefix: str) -> Any:
    """The concrete config a run would receive, for inspection/golden tests.

    Resolves ``ctx.out`` / ``ctx.path(dep)`` against ``prefix`` without running
    anything.
    """
    return artifact.recipe.build_config(
        RunContext.for_run(out=artifact.path(prefix), prefix=prefix, run_args=artifact.recipe.run_args)
    )


def _output_path_spec(artifact: Artifact) -> str:
    """The relative output path StepSpec/ExecutorStep should use: the pin if set,
    else the explicit ``{name}/{version}``."""
    return artifact.override_path or f"{artifact.name}/{artifact.version}"


def lower(artifact: Artifact) -> StepSpec:
    """Lower a handle graph into a ``StepSpec`` graph the ``StepRunner`` can run.

    Each handle becomes a step addressed by its explicit ``{name}/{version}`` (or its
    pin); the recipe fingerprint and version travel in ``hash_attrs`` so the runner
    applies the build-once guard before serving a cached output. A computed or adopted
    step records an :class:`~marin.execution.registry.ArtifactRecord` on success; a
    pin references existing data and records nothing.

    The graph is compute-agnostic: resources ride on the fn (``remote(fn,
    resources=…)``), so the same graph runs inline or dispatches to Fray by the fn's
    type alone — placement is the fn's concern, not the graph's.
    """
    dep_specs = [lower(dep) for dep in artifact.recipe.deps]
    payload = artifact.fingerprint_payload()
    fingerprint = fingerprint_hash(payload)
    if artifact.expected_fingerprint is not None and fingerprint != artifact.expected_fingerprint:
        raise FingerprintMismatchError(
            f"{artifact.name}@{artifact.version}: config fingerprint is {fingerprint}, but "
            f"expected_fingerprint pins {artifact.expected_fingerprint}. The recipe changed — update "
            f"the pin, and bump the version if this is meant to be a different artifact."
        )
    dep_refs = [f"{d.name}@{d.version}" for d in artifact.recipe.deps]

    # Provenance is captured in the launching process (which holds the git checkout)
    # and closed over, so a step that runs remotely still records the launch commit.
    record_provenance = artifact.override_path is None
    git_commit = get_git_commit() if record_provenance else None
    user = get_user() if record_provenance else None

    def fn(output_path: str, _artifact: Artifact = artifact) -> Any:
        if _artifact.adopt_source is not None:
            # Adoption registers pre-existing data: no compute, the record points at it.
            source = _artifact.path(marin_prefix())
            fs = url_to_fs(source, use_listings_cache=False)[0]
            if not fs.exists(source):
                raise FileNotFoundError(f"cannot adopt {_artifact.name}@{_artifact.version}: no data at {source}")
            result = None
        else:
            ctx = RunContext.for_run(
                out=output_path, prefix=marin_prefix(), region=marin_region(), run_args=_artifact.recipe.run_args
            )
            result = _artifact.recipe.fn(_artifact.recipe.build_config(ctx))
            source = None
        if record_provenance:
            write_record(
                ArtifactRecord(
                    name=_artifact.name,
                    version=_artifact.version,
                    fingerprint=fingerprint,
                    output_path=output_path,
                    source=source,
                    git_commit=git_commit,
                    user=user,
                    created_at=created_now(),
                    deps=dep_refs,
                    fingerprint_payload=payload,
                )
            )
        return result

    return StepSpec(
        name=artifact.name,
        override_output_path=_output_path_spec(artifact),
        deps=dep_specs,
        hash_attrs={FINGERPRINT_KEY: fingerprint, VERSION_KEY: artifact.version, "deps": dep_refs},
        fingerprint_payload=payload,
        fn=fn,
    )
