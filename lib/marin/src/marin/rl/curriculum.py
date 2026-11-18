# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned RL curricula: which task sources a run trains on, at what weights, in what order.

A curriculum is the RL analog of :func:`marin.experiment.data.mixture`: the experiment states the
policy — which sources, at what weights or in what order — and the trainer executes it. The object
records each source's pinned dataset identity and recipe, a sampling mode, and the seed. The mode is
either ``weighted`` (weight-random draws whose every training window composes the weights'
largest-remainder integer apportionment, the vendored trainer's ``weighted_quotas`` behavior) or
``ordered`` (a deterministic stage sequence the trainer consumes in order).

The fingerprint covers the source pins, the weights, the stage order and quotas, and the seed.
An RL run's launch document records the intended curriculum, and the run's artifact record names
it. The training pool remains a separate input. See ``docs/references/rl-curricula.md``.
"""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from marin.execution.artifact import validate_path_segment
from marin.execution.fingerprint import canonical_json, fingerprint_hash


class SamplingMode(StrEnum):
    """How the trainer draws its task mix from a curriculum's sources."""

    WEIGHTED = "weighted"
    """Weighted random: every training window composes the weights' largest-remainder
    integer apportionment, so a window's per-source quotas are whole groups."""
    ORDERED = "ordered"
    """Ordered: the trainer consumes the declared stage sequence in order, deterministically."""


@dataclass(frozen=True)
class SourcePin:
    """A source's pinned upstream dataset identity."""

    name: str
    revision: str | None
    url: str


@dataclass(frozen=True)
class CurriculumEntry:
    """One source population a curriculum samples, with its pinned provenance and weight."""

    label: str
    """The sampling key the trainer resolves rows by: the pool row's ``teacher_route``
    value. A curriculum samples only a pool whose rows carry its labels."""
    source: str
    """The catalog invocation name (a ``runnable_sources()`` key)."""
    version: str
    """The source recipe's version."""
    pin: SourcePin
    recipe: str
    """The curation pipeline identity as ``module.qualname``."""
    weight: float | None = None
    """The weighted-mode mixture weight; ``None`` in ordered mode."""


@dataclass(frozen=True)
class CurriculumStage:
    """One ordered-mode stage visit: a source label and the draws that bound it."""

    label: str
    quota: int | None = None
    """Prompts consumed before the next stage; ``None`` draws the pool's row count."""


@dataclass(frozen=True)
class RlCurriculum:
    """A versioned RL task mix over pinned sources, fingerprinted like other artifacts."""

    name: str
    version: str
    mode: SamplingMode
    entries: tuple[CurriculumEntry, ...]
    stages: tuple[CurriculumStage, ...]
    seed: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        validate_path_segment("name", self.name)
        validate_path_segment("version", self.version)
        if not self.entries:
            raise ValueError(f"curriculum {self.name}@{self.version} needs at least one source")
        labels = [entry.label for entry in self.entries]
        if any(not label for label in labels):
            raise ValueError(f"curriculum {self.name}@{self.version} has an empty source label")
        if len(set(labels)) != len(labels):
            raise ValueError(f"curriculum {self.name}@{self.version} has duplicate source labels: {labels}")
        for entry in self.entries:
            if not entry.source:
                raise ValueError(f"curriculum {self.name}@{self.version} entry {entry.label!r} has no source")
            if not entry.version:
                raise ValueError(f"source {entry.source}@? has no version")
            if not entry.pin.name or not entry.pin.url:
                raise ValueError(f"source {entry.source}@{entry.version} is not pinned: name and url are required")
            if not entry.recipe:
                raise ValueError(f"source {entry.source}@{entry.version} has no recipe identity")
            if self.mode is SamplingMode.WEIGHTED:
                if entry.weight is None or not math.isfinite(entry.weight) or entry.weight <= 0:
                    raise ValueError(
                        f"weighted curriculum {self.name}@{self.version} source {entry.label!r} "
                        "needs a positive finite weight"
                    )
            elif entry.weight is not None:
                raise ValueError(
                    f"ordered curriculum {self.name}@{self.version} source {entry.label!r} "
                    "cannot carry a weight; ordered sampling consumes stages, not weights"
                )
        if self.mode is SamplingMode.WEIGHTED:
            if self.stages:
                raise ValueError(
                    f"weighted curriculum {self.name}@{self.version} cannot declare stages; "
                    "weight-random sampling has no order"
                )
        else:
            if not self.stages:
                raise ValueError(f"ordered curriculum {self.name}@{self.version} needs a non-empty stage order")
            declared = set(labels)
            for stage in self.stages:
                if stage.label not in declared:
                    raise ValueError(
                        f"ordered curriculum {self.name}@{self.version} stage {stage.label!r} "
                        f"names no declared source (declared: {sorted(declared)})"
                    )
                if stage.quota is not None and stage.quota <= 0:
                    raise ValueError(
                        f"ordered curriculum {self.name}@{self.version} stage {stage.label!r} "
                        f"quota must be positive, got {stage.quota}"
                    )
            unused = sorted(declared - {stage.label for stage in self.stages})
            if unused:
                raise ValueError(
                    f"ordered curriculum {self.name}@{self.version} declares sources no stage samples: {unused}"
                )
        object.__setattr__(self, "fingerprint", fingerprint_hash(canonical_json(self._fingerprint_payload())))

    def _fingerprint_payload(self) -> dict:
        """The canonical identity the fingerprint hashes: source pins, weights, stages, seed.

        Entries are sorted by label so declaration order never moves the fingerprint; the stage
        order is the declared sequence and is part of the identity."""
        return {
            "entries": sorted(map(entry_payload, self.entries), key=lambda payload: payload["label"]),
            "mode": self.mode.value,
            "seed": self.seed,
            "stages": list(map(stage_payload, self.stages)),
        }

    @property
    def identity(self) -> str:
        """The ``name@version:fingerprint`` identity a run record names."""
        return f"{self.name}@{self.version}:{self.fingerprint}"

    def weights(self) -> dict[str, float]:
        """The weighted-mode mixture as ``{label: weight}``, the trainer's ``domain_weights``."""
        if self.mode is not SamplingMode.WEIGHTED:
            raise ValueError(f"{self.identity} is an ordered curriculum and carries no weights")
        # __post_init__ leaves no None weight in weighted mode.
        return cast(dict[str, float], {entry.label: entry.weight for entry in self.entries})

    def sampling_config(self) -> dict:
        """The ``data.sampling`` recipe subtree this curriculum renders to."""
        if self.mode is SamplingMode.WEIGHTED:
            return {"kind": "domain-weighted", "domain_weights": self.weights(), "seed": self.seed}
        return {
            "kind": "ordered",
            "stages": [stage_payload(stage) for stage in self.stages],
            "seed": self.seed,
        }


def entry_payload(entry: CurriculumEntry) -> dict:
    """The canonical record of one source entry: identity, pin, recipe identity, and weight."""
    return {
        "label": entry.label,
        "source": entry.source,
        "version": entry.version,
        "pin": {"name": entry.pin.name, "revision": entry.pin.revision, "url": entry.pin.url},
        "recipe": entry.recipe,
        "weight": entry.weight,
    }


def stage_payload(stage: CurriculumStage) -> dict:
    """The canonical record of one ordered stage: its source label and draw quota."""
    return {"source": stage.label, "quota": stage.quota}
