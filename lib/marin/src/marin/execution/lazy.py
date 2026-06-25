# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy artifacts addressed by an explicit ``name@version``.

Prototype of the executor redesign (see
``.agents/projects/executor-lazy-artifact-extraction.md``).

The model:

- An :class:`Artifact` is a lazy handle: ``(name, version, recipe)``. Building one
  runs nothing.
- A :class:`Recipe` is ``(fn, build_config, deps, resources)``. ``build_config`` is
  a pure function of a :class:`BuildContext`, which resolves the two path
  references a config can need: ``ctx.out`` (this artifact's output path) and
  ``ctx.path(dep)`` (a dependency's output path). This replaces ``THIS_OUTPUT_PATH``
  and ``InputName``/``.cd()`` with plain typed calls.
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
from rigging.filesystem import marin_prefix

from marin.execution.executor import Executor
from marin.execution.step_spec import StepSpec
from marin.execution.types import ExecutorStep
from marin.utilities.json_encoder import CustomJsonEncoder


def _artifact_path(name: str, version: str, prefix: str) -> str:
    """The explicit, hash-free address of an artifact under a storage prefix."""
    return f"{prefix}/{name}/{version}"


@dataclass(frozen=True)
class BuildContext:
    """Resolves the path references a recipe's config can need.

    ``for_run`` yields real output paths (used to build the concrete config).
    ``for_fingerprint`` yields dependency ``name@version`` strings instead of
    paths, so the fingerprint captures dependency *identity* without baking in a
    region-specific prefix.
    """

    out: str
    _dep_ref: Callable[["Artifact"], str]

    def path(self, dep: "Artifact") -> str:
        return self._dep_ref(dep)

    @staticmethod
    def for_run(out: str, prefix: str) -> "BuildContext":
        return BuildContext(out=out, _dep_ref=lambda d: _artifact_path(d.name, d.version, prefix))

    @staticmethod
    def for_fingerprint() -> "BuildContext":
        return BuildContext(out="<out>", _dep_ref=lambda d: f"{d.name}@{d.version}")


@dataclass(frozen=True)
class Recipe:
    """How to build an artifact: the step fn, a config builder, deps, resources."""

    fn: Callable[[Any], Any]
    """``fn(config) -> result``. The config carries its output path; result is persisted."""
    build_config: Callable[[BuildContext], Any]
    """``build_config(ctx) -> config``. Reads ``ctx.out`` / ``ctx.path(dep)`` only for paths."""
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

    def fingerprint(self) -> str:
        """Stable id of *how* this artifact is built: the config with dependency
        versions in place of paths. Changes iff a config value or a dep version
        changes; independent of the storage prefix/region."""
        config = self.recipe.build_config(BuildContext.for_fingerprint())
        payload = json.dumps(config, sort_keys=True, cls=CustomJsonEncoder)
        return hashlib.md5(payload.encode()).hexdigest()[:8]

    def path(self, prefix: str | None = None) -> str:
        return _artifact_path(self.name, self.version, prefix or marin_prefix())


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
    out = _artifact_path(artifact.name, artifact.version, prefix)
    return artifact.recipe.build_config(BuildContext.for_run(out=out, prefix=prefix))


def lower(artifact: Artifact) -> StepSpec:
    """Lower a pure handle graph into the existing ``StepSpec`` graph.

    For artifacts whose config references only ``ctx.out`` / ``ctx.path(dep)`` (no
    embedded legacy ``ExecutorStep`` placeholders). The concrete config is rebuilt
    inside ``fn`` using the runner's ``marin_prefix()``, so dependency paths
    resolve region-locally rather than capturing the build environment's prefix.
    Identity is the explicit ``{name}/{version}``; the fingerprint rides in
    ``hash_attrs`` for the (later) immutability guard, not the path.
    """
    dep_specs = [lower(dep) for dep in artifact.recipe.deps]

    def fn(output_path: str, _artifact: Artifact = artifact) -> Any:
        ctx = BuildContext.for_run(out=output_path, prefix=marin_prefix())
        return _artifact.recipe.fn(_artifact.recipe.build_config(ctx))

    return StepSpec(
        name=artifact.name,
        override_output_path=f"{artifact.name}/{artifact.version}",
        deps=dep_specs,
        hash_attrs={
            "fingerprint": artifact.fingerprint(),
            "deps": [f"{d.name}@{d.version}" for d in artifact.recipe.deps],
        },
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
    out = _artifact_path(artifact.name, artifact.version, resolved_prefix)
    config = artifact.recipe.build_config(BuildContext.for_run(out=out, prefix=resolved_prefix))
    return ExecutorStep(
        name=artifact.name,
        fn=artifact.recipe.fn,
        config=config,
        override_output_path=f"{artifact.name}/{artifact.version}",
        resources=artifact.recipe.resources,
    )


def run(artifact: Artifact, *, prefix: str | None = None, **run_kwargs: Any) -> dict[ExecutorStep, str]:
    """Materialize an artifact through the bridge: resolve its legacy dependency
    graph and run it via the existing ``Executor``/``StepRunner``."""
    resolved_prefix = prefix or marin_prefix()
    executor = Executor(prefix=resolved_prefix, executor_info_base_path=f"{resolved_prefix}/experiments")
    return executor.run([to_executor_step(artifact, resolved_prefix)], **run_kwargs)
