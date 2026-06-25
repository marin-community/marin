# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy artifacts addressed by an explicit ``name@version``.

Prototype of the executor redesign (see
``.agents/projects/executor-lazy-artifact-extraction.md``).

The model:

- An :class:`Artifact` is a lazy handle: ``(name, version, recipe)``. Building one
  runs nothing.
- A :class:`Recipe` is ``(fn, build_config, deps, resources)``. ``build_config`` is
  a pure function of a :class:`RunContext`, the live run environment it pulls from:
  ``ctx.out`` (this artifact's output path), ``ctx.path(dep)`` (a dependency's
  output path), ``ctx.prefix``/``ctx.region``/``ctx.resources`` (the live storage
  prefix, GCP region, and resolved compute). This replaces ``THIS_OUTPUT_PATH`` and
  ``InputName``/``.cd()`` with plain typed calls.
- ``fn(config) -> result`` is the step function. The config already carries its
  output path (``ctx.out``), matching the existing ``ExecutorStep`` fn convention.
- Identity is the explicit ``{prefix}/{name}/{version}`` path — no hash.
- The *recipe fingerprint* is the config built with dependency **versions** in
  place of paths (so it captures hyperparameters + dep identities but not
  region-specific paths). It is recorded for the build-once-immutability guard,
  never in the path.

Two ways to materialize:

- :func:`lower` turns a pure handle graph into the existing :class:`StepSpec`
  graph; the existing ``StepRunner`` runs it (cache → lock → run). This is the
  target engine — no content-addressing.
- :func:`run` / :func:`to_executor_step` is the **bridge** for configs whose data
  still embeds legacy ``ExecutorStep``/``InputName`` placeholders (e.g. a
  not-yet-migrated dataset catalog like ``nemotron_mix``). The top artifact keeps
  its explicit ``name@version`` identity; its legacy dependency graph is resolved
  by the existing ``Executor``. This bridge disappears as catalogs migrate.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fray.types import ResourceConfig
from rigging.filesystem import marin_prefix, marin_region

from marin.execution.executor import Executor
from marin.execution.provenance import created_now, get_git_commit, get_user
from marin.execution.registry import FINGERPRINT_KEY, VERSION_KEY, ArtifactRecord, write_record
from marin.execution.step_spec import StepSpec, _is_relative_path
from marin.execution.types import ExecutorStep
from marin.utilities.json_encoder import CustomJsonEncoder


def _artifact_path(name: str, version: str, prefix: str) -> str:
    """The explicit, hash-free address of an artifact under a storage prefix."""
    return f"{prefix}/{name}/{version}"


@dataclass(frozen=True)
class RunContext:
    """The *runtime-only* environment a recipe resolves its config against.

    The context carries only what cannot be known until the step actually runs —
    chiefly the live storage location and region. Compute (TPU type, etc.) is
    *not* here: it is known at authoring time and is part of the execution
    *description* (``Recipe.resources``), not something to discover at run time
    (a single recipe may even dispatch several jobs with different resources).

    A recipe's ``build_config`` *pulls* what it needs:

    - ``ctx.out`` — this artifact's output path
    - ``ctx.path(dep)`` — a dependency's resolved (region-local) output path
    - ``ctx.prefix`` — the live storage prefix (resolve raw input globs against it)
    - ``ctx.region`` — the live GCP region (unknown until run time)

    :meth:`for_run` binds these to the live environment at lazy-evaluation time.
    :meth:`for_fingerprint` supplies placeholders, so an artifact's identity stays
    independent of where it runs (a region or prefix move never re-fingerprints).
    """

    out: str
    prefix: str
    region: str | None
    _dep_ref: Callable[["Artifact"], str]

    def path(self, dep: "Artifact") -> str:
        return self._dep_ref(dep)

    @staticmethod
    def for_run(out: str, prefix: str, *, region: str | None = None) -> "RunContext":
        return RunContext(out=out, prefix=prefix, region=region, _dep_ref=lambda d: d.path(prefix))

    @staticmethod
    def for_fingerprint() -> "RunContext":
        return RunContext(out="<out>", prefix="<prefix>", region="<region>", _dep_ref=lambda d: f"{d.name}@{d.version}")


@dataclass(frozen=True)
class Recipe:
    """How to build an artifact: the step fn, a config builder, deps, resources."""

    fn: Callable[[Any], Any]
    """``fn(config) -> result``. The config carries its output path; result is persisted."""
    build_config: Callable[[RunContext], Any]
    """``build_config(ctx) -> config``. Pulls paths and live attributes from the RunContext."""
    deps: tuple["Artifact", ...] = ()
    resources: ResourceConfig | None = None


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

    def fingerprint(self) -> str:
        """Stable id of *how* this artifact is built: the config with dependency
        versions in place of paths. Changes iff a config value or a dep version
        changes; independent of the storage prefix/region."""
        config = self.recipe.build_config(RunContext.for_fingerprint())
        payload = json.dumps(config, sort_keys=True, cls=CustomJsonEncoder)
        return hashlib.md5(payload.encode()).hexdigest()[:8]

    def path(self, prefix: str | None = None) -> str:
        resolved_prefix = prefix or marin_prefix()
        if self.override_path is not None:
            if _is_relative_path(self.override_path):
                return f"{resolved_prefix}/{self.override_path}"
            return self.override_path
        return _artifact_path(self.name, self.version, resolved_prefix)


@dataclass(frozen=True, eq=False)
class Dataset(Artifact):
    """A tokenized dataset handle (routes to dataset consumers)."""


@dataclass(frozen=True, eq=False)
class Checkpoint(Artifact):
    """A Levanter checkpoint handle (routes to ``initialize_from_checkpoint_path``)."""


def materialized_config(artifact: Artifact, prefix: str) -> Any:
    """The concrete config a run would receive, for inspection/golden tests.

    Resolves ``ctx.out`` / ``ctx.path(dep)`` against ``prefix`` without running
    anything. Note: legacy ``ExecutorStep``/``InputName`` placeholders embedded in
    the config (the bridge case) are left unresolved here — :func:`to_executor_step`
    resolves those through the ``Executor``.
    """
    return artifact.recipe.build_config(RunContext.for_run(out=artifact.path(prefix), prefix=prefix))


def _output_path_spec(artifact: Artifact) -> str:
    """The relative output path StepSpec/ExecutorStep should use: the pin if set,
    else the explicit ``{name}/{version}``."""
    return artifact.override_path or f"{artifact.name}/{artifact.version}"


def lower(artifact: Artifact) -> StepSpec:
    """Lower a pure handle graph into the existing ``StepSpec`` graph.

    For artifacts whose config references only ``ctx.out`` / ``ctx.path(dep)`` (no
    embedded legacy ``ExecutorStep`` placeholders). The concrete config is rebuilt
    inside ``fn`` using the runner's ``marin_prefix()``, so dependency paths
    resolve region-locally rather than capturing the build environment's prefix.
    Identity is the explicit ``{name}/{version}`` (or the pin); the fingerprint and
    version ride in ``hash_attrs`` so the runner can apply the immutability guard
    before serving a cached output. On success the step writes an
    :class:`~marin.execution.registry.ArtifactRecord` (recipe fingerprint + launch
    provenance); a pinned artifact references existing external data and writes none.
    """
    dep_specs = [lower(dep) for dep in artifact.recipe.deps]
    fingerprint = artifact.fingerprint()
    dep_refs = [f"{d.name}@{d.version}" for d in artifact.recipe.deps]

    # Provenance is captured in the launching process (which holds the git checkout)
    # and closed over, so a step that runs remotely still records the launch commit.
    record_provenance = artifact.override_path is None
    git_commit = get_git_commit() if record_provenance else None
    user = get_user() if record_provenance else None

    def fn(output_path: str, _artifact: Artifact = artifact) -> Any:
        ctx = RunContext.for_run(out=output_path, prefix=marin_prefix(), region=marin_region())
        result = _artifact.recipe.fn(_artifact.recipe.build_config(ctx))
        if record_provenance:
            write_record(
                ArtifactRecord(
                    name=_artifact.name,
                    version=_artifact.version,
                    fingerprint=fingerprint,
                    output_path=output_path,
                    git_commit=git_commit,
                    user=user,
                    created_at=created_now(),
                    deps=dep_refs,
                )
            )
        return result

    return StepSpec(
        name=artifact.name,
        override_output_path=_output_path_spec(artifact),
        deps=dep_specs,
        hash_attrs={FINGERPRINT_KEY: fingerprint, VERSION_KEY: artifact.version, "deps": dep_refs},
        fn=fn,
        resources=artifact.recipe.resources,
    )


def to_executor_step(artifact: Artifact, prefix: str | None = None) -> ExecutorStep:
    """Bridge: build an ``ExecutorStep`` whose config may embed legacy placeholders.

    The artifact keeps its explicit ``name@version`` identity via
    ``override_output_path``; any ``ExecutorStep``/``InputName`` placeholders inside
    the built config (e.g. a legacy dataset mixture's tokenize steps) are left for
    the ``Executor`` to resolve. Used until those catalogs migrate to handles.
    """
    if artifact.recipe.deps:
        raise ValueError(
            f"{artifact.name}: to_executor_step is the legacy-dep bridge and does not lower handle deps; "
            "use lower() for pure handle graphs."
        )
    resolved_prefix = prefix or marin_prefix()
    config = artifact.recipe.build_config(
        RunContext.for_run(out=artifact.path(resolved_prefix), prefix=resolved_prefix, region=marin_region())
    )
    return ExecutorStep(
        name=artifact.name,
        fn=artifact.recipe.fn,
        config=config,
        override_output_path=_output_path_spec(artifact),
        resources=artifact.recipe.resources,
    )


def run(artifact: Artifact, *, prefix: str | None = None, **run_kwargs: Any) -> dict[ExecutorStep, str]:
    """Materialize an artifact through the bridge: resolve its legacy dependency
    graph and run it via the existing ``Executor``/``StepRunner``."""
    resolved_prefix = prefix or marin_prefix()
    executor = Executor(prefix=resolved_prefix, executor_info_base_path=f"{resolved_prefix}/experiments")
    return executor.run([to_executor_step(artifact, resolved_prefix)], **run_kwargs)
