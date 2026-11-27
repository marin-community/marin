# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned RL curricula: which task sources a run trains on, at what weights, in what order.

A curriculum is the RL analog of :func:`marin.experiment.data.mixture`: the experiment states the
policy — which sources, at what weights or in what order — and the trainer executes it. The object
records each source's pinned dataset identity and recipe, a sampling mode, and the seed. The mode is
either ``weighted`` (weight-random draws whose every training window composes the weights'
largest-remainder integer apportionment, the vendored trainer's ``weighted_quotas`` behavior) or
``ordered`` (a deterministic stage sequence the trainer consumes in order).

An RL run's launch document records the intended curriculum, and the run's artifact record names
it. The training pool remains a separate input.

Fixing a curriculum also fixes its validation split: a random holdout drawn from the curriculum's
own training distribution (:func:`draw_validation_split`), recorded in the curriculum's metadata,
and removed from the pools the run trains on. The validation mix matches the mixture the trainer
samples, so the held-out signal is a random subset of the training set rather than a hand-composed
selection. The fingerprint covers the source pins, the weights, the stage order and quotas, the
seed, and the validation split. An RL run's launch document records the resolved sampling
projection, the validation split, and the full curriculum, and the run's artifact record names it.
See ``docs/references/rl-curricula.md``.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
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
    """The curation callable identity or immutable served catalog pipeline reference."""
    weight: float | None = None
    """The weighted-mode mixture weight; ``None`` in ordered mode."""


@dataclass(frozen=True)
class CurriculumStage:
    """One ordered-mode stage visit: a source label and the draws that bound it."""

    label: str
    quota: int | None = None
    """Prompts consumed before the next stage; ``None`` draws the pool's row count."""


@dataclass(frozen=True)
class PoolRow:
    """One training-pool row's identity, as the validation draw consumes it."""

    id: str
    """The row's task identity (a ``task_spec.id``)."""
    sha256: str
    """The row task document's canonical SHA-256, verifying content at extraction time."""


@dataclass(frozen=True)
class ValidationRow:
    """One held-out validation prompt a fixed curriculum records."""

    label: str
    """The pool the row was drawn from: the curriculum entry label (a ``teacher_route`` value)."""
    id: str
    """The row's task identity (a ``task_spec.id``)."""
    sha256: str
    """The row task document's canonical SHA-256, verifying content at extraction time."""


@dataclass(frozen=True)
class CurriculumValidation:
    """The recorded holdout; size zero explicitly keeps the whole population in training."""

    size: int
    """How many prompts the holdout holds; ``rows`` records exactly this many."""
    seed: int
    """The seed the split was drawn with; the curriculum's own seed."""
    rows: tuple[ValidationRow, ...]

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"validation size must be nonnegative, got {self.size}")
        if len(self.rows) != self.size:
            raise ValueError(f"a validation split records {len(self.rows)} rows for size {self.size}")
        identities: set[tuple[str, str]] = set()
        for row in self.rows:
            if not row.label or not row.id or not row.sha256:
                raise ValueError(f"validation row {row!r} is incomplete: label, id and sha256 are required")
            identity = (row.label, row.id)
            if identity in identities:
                raise ValueError(f"validation split records row {row.label!r}/{row.id!r} twice")
            identities.add(identity)


@dataclass(frozen=True)
class ValidationSplit:
    """The partition a fixed curriculum's holdout induces on its pools.

    ``validation`` is the recorded holdout; ``training`` maps each label to the rows that remain
    trainable. The two are disjoint by construction: a data layer that writes the training pools
    from ``training`` and extracts the validation data from ``validation`` makes the holdout true."""

    validation: CurriculumValidation
    training: Mapping[str, tuple[PoolRow, ...]]


def _apportion(total: int, shares: Mapping[str, float]) -> dict[str, int]:
    """Apportion ``total`` draws across labels proportionally to ``shares``.

    Largest-remainder apportionment — the same integer behavior the trainer's
    ``weighted_quotas`` gives a training window — with remainder ties broken by sorted label
    order, so the split depends on the mixture alone, never on declaration order. Shares are
    proportions: their scale never moves the allocation."""
    total_share = sum(shares.values())
    weighted = {label: share / total_share * total for label, share in shares.items()}
    counts = {label: math.floor(weighted[label]) for label in weighted}
    remaining = total - sum(counts.values())
    by_remainder = sorted(weighted, key=lambda label: (-(weighted[label] - counts[label]), label))
    for label in by_remainder[:remaining]:
        counts[label] += 1
    return counts


def draw_validation_split(
    mixture: Mapping[str, float],
    rows_by_label: Mapping[str, Sequence[PoolRow]],
    *,
    size: int,
    seed: int,
) -> ValidationSplit:
    """Draw a curriculum's validation holdout from its pools and partition every row.

    ``mixture`` maps each pool label to its share of the training distribution — the curriculum's
    weighted-mode weights, or an ordered curriculum's per-label consumed prompts. ``rows_by_label``
    carries each pool's row identities. The holdout holds exactly ``size`` prompts: labels receive
    largest-remainder allocations of ``size``, and each label's rows are selected uniformly without
    replacement by the ``sha256(f"{seed}:{label}:{id}")`` order, so the draw is deterministic for a
    seed, independent of enumeration order, and follows the mixture the trainer samples. Every row
    lands in exactly one side of the returned :class:`ValidationSplit`."""

    if size < 0:
        raise ValueError(f"validation size must be nonnegative, got {size}")
    if not mixture:
        raise ValueError("the mixture names no pools")
    declared = set(mixture)
    if any(not math.isfinite(share) or share <= 0 for share in mixture.values()):
        raise ValueError(f"mixture shares must be positive and finite, got {mixture}")
    stray = sorted(set(rows_by_label) - declared)
    if stray:
        raise ValueError(f"pool rows name labels the mixture does not declare: {stray}")
    empty = sorted(label for label in declared if not rows_by_label.get(label))
    if empty:
        raise ValueError(f"the mixture names pools with no rows: {empty}")
    allocations = _apportion(size, mixture)
    validation_rows: list[ValidationRow] = []
    training: dict[str, tuple[PoolRow, ...]] = {}
    for label in sorted(declared):
        rows = rows_by_label[label]
        identities = [row.id for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError(f"pool {label!r} repeats a row identity; the holdout draw needs unique ids")
        if allocations[label] > len(rows):
            raise ValueError(
                f"validation size {size} allocates {allocations[label]} rows to pool {label!r}, "
                f"which holds {len(rows)}"
            )
        keyed = sorted(
            ((hashlib.sha256(f"{seed}:{label}:{row.id}".encode()).hexdigest(), row) for row in rows),
            key=lambda pair: (pair[0], pair[1].id),
        )
        drawn = {row.id for _, row in keyed[: allocations[label]]}
        validation_rows.extend(
            ValidationRow(label=label, id=row.id, sha256=row.sha256) for _, row in keyed[: allocations[label]]
        )
        training[label] = tuple(row for row in rows if row.id not in drawn)
    validation = CurriculumValidation(size=size, seed=seed, rows=tuple(sorted(validation_rows, key=_row_order)))
    return ValidationSplit(validation=validation, training=training)


def _row_order(row: ValidationRow) -> tuple[str, str]:
    return (row.label, row.id)


def validation_row_payload(row: ValidationRow) -> dict:
    """The canonical record of one held-out row: its pool label, task identity, and content hash."""
    return {"label": row.label, "id": row.id, "sha256": row.sha256}


def validation_payload(validation: CurriculumValidation) -> dict:
    """The canonical record of a validation split: size, seed, and the held-out rows.

    Rows are sorted by label then id, so the record — and every fingerprint and launch document
    that embeds it — is independent of the draw's iteration order."""
    return {
        "size": validation.size,
        "seed": validation.seed,
        "rows": sorted(
            map(validation_row_payload, validation.rows), key=lambda payload: (payload["label"], payload["id"])
        ),
    }


@dataclass(frozen=True)
class RlCurriculum:
    """A versioned RL task mix over pinned sources, fingerprinted like other artifacts."""

    name: str
    version: str
    mode: SamplingMode
    entries: tuple[CurriculumEntry, ...]
    stages: tuple[CurriculumStage, ...]
    seed: int
    validation: CurriculumValidation
    """The random holdout drawn from this curriculum's training distribution when it was fixed."""
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
        if self.validation.seed != self.seed:
            raise ValueError(
                f"curriculum {self.name}@{self.version} records validation seed {self.validation.seed}, "
                f"but its sampler seed is {self.seed}; the holdout is drawn with the curriculum's seed"
            )
        unknown = sorted({row.label for row in self.validation.rows} - set(labels))
        if unknown:
            raise ValueError(f"curriculum {self.name}@{self.version} holds out rows from undeclared sources: {unknown}")
        object.__setattr__(self, "fingerprint", fingerprint_hash(canonical_json(self._fingerprint_payload())))

    def _fingerprint_payload(self) -> dict:
        """The canonical identity the fingerprint hashes: source pins, weights, stages, seed, holdout.

        Entries are sorted by label so declaration order never moves the fingerprint; the stage
        order is the declared sequence and is part of the identity; the validation split is the
        recorded holdout, so two curricula that differ only in their split are different objects."""
        return {
            "entries": sorted(map(entry_payload, self.entries), key=lambda payload: payload["label"]),
            "mode": self.mode.value,
            "seed": self.seed,
            "stages": list(map(stage_payload, self.stages)),
            "validation": validation_payload(self.validation),
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
    return {"source": stage.label, "quota": stage.quota}


def consumed_route_quotas(
    curriculum: RlCurriculum, source_routes: Mapping[str, str], batch_groups: int
) -> dict[str, int]:
    """Derive the rollout buffer's per-route window quotas from a curriculum's own mixture.

    The curriculum's entry weights aggregate over the run's ``source_routes`` —
    a label with no mapping is its own route — and ``batch_groups`` apportions
    over the route shares by the same largest remainder the validation split
    uses, remainder ties breaking by sorted route order. Every admitted route
    draws at least one group per window: a route the apportionment starves takes
    its seat from the route with the largest allocation, so the realized window
    never drops an admitted source and always sums to the batch. Ordered
    curricula carry no weights and derive no window quotas."""
    if curriculum.mode is not SamplingMode.WEIGHTED:
        raise ValueError(
            f"{curriculum.identity} is an ordered curriculum; its stages, not window quotas, bound its draws"
        )
    if batch_groups <= 0:
        raise ValueError(f"batch groups must be positive, got {batch_groups}")
    shares: dict[str, float] = {}
    for entry in curriculum.entries:
        route = source_routes.get(entry.label, entry.label)
        # __post_init__ leaves no None weight in weighted mode.
        shares[route] = shares.get(route, 0.0) + float(entry.weight or 0.0)
    if batch_groups < len(shares):
        raise ValueError(f"a batch of {batch_groups} groups cannot admit {len(shares)} routes at one group each")
    quotas = _apportion(batch_groups, shares)
    for route in sorted(quotas):
        if quotas[route] != 0:
            continue
        quotas[route] = 1
        donor = sorted(quotas, key=lambda name: (-quotas[name], name))[0]
        quotas[donor] -= 1
    return dict(sorted(quotas.items()))
