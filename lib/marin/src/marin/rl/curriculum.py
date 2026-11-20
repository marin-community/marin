# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned RL curricula: which task sources a run trains on, at what weights, in what order.

A curriculum is the RL analog of :func:`marin.experiment.data.mixture`: the experiment states the
policy — which pools, at what weights or in what order — and the trainer executes it. Each entry
names the pool artifact the run trains on by its artifact identity, so the record and the training
data share one provenance chain and cannot drift apart. The mode is either ``weighted``
(weight-random draws that compose every training window into the exact weighted mixture, the
vendored trainer's ``weighted_quotas`` behavior) or ``ordered`` (a deterministic stage sequence the
trainer consumes in order).

The fingerprint covers the pool identities, the weights, the stage order and quotas, and the seed.
An RL run's launch document records the resolved sampling projection and the full curriculum, and
the run's artifact record names it. See ``docs/references/rl-curricula.md``.
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
    """Weighted random: every training window composes the exact weighted mixture."""
    ORDERED = "ordered"
    """Ordered: the trainer consumes the declared stage sequence in order, deterministically."""


@dataclass(frozen=True)
class CurriculumEntry:
    """One pool population a curriculum samples, with its artifact identity and weight."""

    label: str
    """The sampling key the trainer resolves rows by (a ``teacher_route`` value)."""
    identity: str
    """The pool artifact's ``name@version:fingerprint`` identity — the same pinned record the run trains on."""
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
            if not entry.identity:
                raise ValueError(f"curriculum {self.name}@{self.version} entry {entry.label!r} has no pool identity")
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
        """The weighted-mode mixture as ``{label: weight}``, the trainer's ``domain_weights``.

        Canonical (sorted) order: the trainer breaks remainder ties and seeds shuffles from this
        mapping's order, so two declarations of the same mixture must sample identically.
        """
        if self.mode is not SamplingMode.WEIGHTED:
            raise ValueError(f"{self.identity} is an ordered curriculum and carries no weights")
        # __post_init__ leaves no None weight in weighted mode.
        return cast(
            dict[str, float],
            {entry.label: entry.weight for entry in sorted(self.entries, key=lambda entry: entry.label)},
        )

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
    """The canonical record of one pool entry: its sampling label, pool identity, and weight."""
    return {"label": entry.label, "identity": entry.identity, "weight": entry.weight}


def stage_payload(stage: CurriculumStage) -> dict:
    """The canonical record of one ordered stage: its source label and draw quota."""
    return {"source": stage.label, "quota": stage.quota}
