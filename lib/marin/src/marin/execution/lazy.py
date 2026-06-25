# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SPIKE: lazy artifacts addressed by an explicit ``name@version``.

This is an exploratory prototype of the executor redesign (see
``.agents/projects/executor-lazy-artifact-extraction.md``). It is intentionally
small: it proves the *hardest* path — configs that reference their own output
path and a dependency's output path — without the content-addressing layer.

The model:

- An :class:`Artifact` is a lazy handle: ``(name, version, recipe)``. Building one
  runs nothing.
- A :class:`Recipe` is ``(fn, build_config, deps, resources)``. ``build_config`` is
  a pure function of a :class:`BuildContext`, which resolves the two path
  references a config can need: ``ctx.out`` (this artifact's output path) and
  ``ctx.path(dep)`` (a dependency's output path). This replaces ``THIS_OUTPUT_PATH``
  and ``InputName``/``.cd()`` with plain typed calls.
- Identity is the explicit ``{prefix}/{name}/{version}`` path — no hash.
- The *recipe fingerprint* is the config built with dependency **versions** in
  place of paths (so it captures hyperparameters + dep identities but not
  region-specific paths). It is recorded for the build-once-immutability guard,
  never in the path.
- ``lower(artifact)`` turns the handle graph into the existing :class:`StepSpec`
  graph; the existing ``StepRunner`` then materializes it (cache → lock → run).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fray.types import ResourceConfig
from rigging.filesystem import marin_prefix

from marin.execution.step_spec import StepSpec
from marin.utilities.json_encoder import CustomJsonEncoder


def _artifact_path(name: str, version: str, prefix: str) -> str:
    """The explicit, hash-free address of an artifact under a storage prefix."""
    return f"{prefix}/{name}/{version}"


@dataclass(frozen=True)
class BuildContext:
    """Resolves the path references a recipe's config can need.

    ``for_run`` yields real output paths (used on the worker to build the concrete
    config). ``for_fingerprint`` yields dependency ``name@version`` strings instead
    of paths, so the fingerprint captures dependency *identity* without baking in a
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

    fn: Callable[[str, Any], Any]
    """``fn(output_path, config) -> result``. The result is persisted by the runner."""
    build_config: Callable[[BuildContext], Any]
    """``build_config(ctx) -> config``. Reads ``ctx.out`` / ``ctx.path(dep)`` only for paths."""
    deps: tuple["Artifact", ...] = ()
    resources: ResourceConfig | None = None


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Dataset(Artifact):
    """A tokenized dataset handle (routes to dataset consumers)."""


@dataclass(frozen=True)
class Checkpoint(Artifact):
    """A Levanter checkpoint handle (routes to ``initialize_from_checkpoint_path``)."""


def lower(artifact: Artifact) -> StepSpec:
    """Lower a handle graph into the existing ``StepSpec`` graph.

    The concrete config is rebuilt on the worker (inside ``fn``) using the worker's
    ``marin_prefix()``, so dependency paths resolve to the region-local replica
    rather than capturing the build environment's prefix. The path scheme is the
    explicit ``{name}/{version}``; the recipe fingerprint rides in ``hash_attrs``
    for the (later) immutability guard, not the path.
    """
    dep_specs = [lower(dep) for dep in artifact.recipe.deps]

    def fn(output_path: str, _artifact: Artifact = artifact) -> Any:
        ctx = BuildContext.for_run(out=output_path, prefix=marin_prefix())
        config = _artifact.recipe.build_config(ctx)
        return _artifact.recipe.fn(output_path, config)

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


def materialized_config(artifact: Artifact, prefix: str) -> Any:
    """The concrete config a run would receive, for inspection/golden tests.

    Mirrors what ``lower(...).fn`` builds on the worker, without running anything.
    """
    ctx = BuildContext.for_run(out=_artifact_path(artifact.name, artifact.version, prefix), prefix=prefix)
    return artifact.recipe.build_config(ctx)
