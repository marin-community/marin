# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every bound of one run, stated in the run's ``policy.json``; no field has a default.

``POLICY`` reads and writes a ``LoopPolicy`` as JSON. Every field of ``LoopPolicy``, of its
``ValidationPolicy`` and of that policy's ``band``, ``sampling`` and ``deadlines``, and of the
``band_rules`` and each of their ``BandRule``s, is required and an unknown key is an error, so a run
config states each bound. A ``RetryBackoff`` is written as its four fields (``initial``, ``maximum``,
``factor``, ``jitter``), all required; a ``BandChoice`` as its value (``accept`` or ``reject``).
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, get_type_hints

from pydantic import PlainSerializer, PlainValidator, TypeAdapter

from taskforge.content_hash import digest
from taskforge.llm.policy import LLMPolicy
from taskforge.review.rules import BandRule, BandRules
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, RetryBackoff


@dataclass(frozen=True)
class LoopPolicy:
    """Every bound of one run.

    Attributes:
        proposals_per_idea: ``n`` in ``ProposalSource.propose(idea, n)``.
        max_idea_reproposals: Re-proposals of an idea whose batch yielded zero proposals.
        max_build_revisions: Author revisions per item after a build failure or a no-op repair.
        max_repairs: Review ``Repair`` decisions per item; 0, because this loop cannot repair a task.
        max_validation_retries: Review ``Retry`` decisions per item per launch; then the item is abandoned.
        retry_backoff: Wait before a ``Retry`` re-enters validation; the k-th retry waits the k-th interval.
        band_rules: Per band kind (too easy, too hard), the review ``Repair`` decisions a solve rate
            outside the band may trigger, then whether a task still outside it is accepted (labelled
            with where it fell) or rejected. Each rule's ``repairs`` is 0, as ``max_repairs`` is.
        validation: The policy of every validation round.
    """

    proposals_per_idea: int
    max_idea_reproposals: int
    max_build_revisions: int
    max_repairs: int
    max_validation_retries: int
    retry_backoff: RetryBackoff
    band_rules: BandRules
    validation: ValidationPolicy

    def __post_init__(self) -> None:
        if self.proposals_per_idea < 1:
            raise ValueError("A loop policy needs proposals_per_idea >= 1")
        bounds = {
            "max_idea_reproposals": self.max_idea_reproposals,
            "max_build_revisions": self.max_build_revisions,
            "max_repairs": self.max_repairs,
            "max_validation_retries": self.max_validation_retries,
        }
        negative = sorted(name for name, value in bounds.items() if value < 0)
        if negative:
            raise ValueError(f"Loop policy bounds must be non-negative: {negative}")
        repairs = {
            "max_repairs": self.max_repairs,
            "band_rules.too_easy.repairs": self.band_rules.too_easy.repairs,
            "band_rules.too_hard.repairs": self.band_rules.too_hard.repairs,
        }
        asked = sorted(name for name, value in repairs.items() if value)
        if asked:
            raise ValueError(
                f"This loop cannot repair a task: authoring adopts the template unchanged, so a repair rebuilds "
                f"the same task. Set {asked} to 0"
            )

    @property
    def digest(self) -> str:
        """The canonical digest of every field, including the validation policy's own digest."""
        return digest({"policy": POLICY.dump_python(self, mode="json"), "validation": self.validation.digest})


def _strict_dataclass(cls: type, substitutes: Mapping[Any, Any]) -> Any:
    """``cls`` as a pydantic type whose JSON object must hold exactly its fields.

    Field types are taken from ``cls``'s annotations, with ``substitutes`` replacing the nested
    dataclasses that need this same strictness.
    """
    hints: dict[str, Any] = get_type_hints(cls)
    adapters: dict[str, TypeAdapter] = {
        field.name: TypeAdapter(substitutes.get(hints[field.name], hints[field.name]))
        for field in dataclasses.fields(cls)
    }

    def validate(value: object) -> object:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{cls.__name__} must be a JSON object, got {type(value).__name__}")
        missing = sorted(set(adapters) - set(value))
        unknown = sorted(set(value) - set(adapters))
        if missing or unknown:
            raise ValueError(f"{cls.__name__}: missing fields {missing}, unknown fields {unknown}")
        return cls(**{name: adapter.validate_python(value[name]) for name, adapter in adapters.items()})

    def serialize(value: object) -> dict[str, object]:
        return {name: adapter.dump_python(getattr(value, name), mode="json") for name, adapter in adapters.items()}

    serializer: Callable[[object], dict[str, object]] = serialize
    return Annotated[cls, PlainValidator(validate), PlainSerializer(serializer)]


_VALIDATION_JSON = _strict_dataclass(
    ValidationPolicy,
    {
        RetryBackoff: _strict_dataclass(RetryBackoff, {}),
        CalibrationBand: _strict_dataclass(CalibrationBand, {}),
        LLMPolicy: _strict_dataclass(LLMPolicy, {}),
        Deadlines: _strict_dataclass(Deadlines, {}),
    },
)
_LOOP_JSON = _strict_dataclass(
    LoopPolicy,
    {
        RetryBackoff: _strict_dataclass(RetryBackoff, {}),
        BandRules: _strict_dataclass(BandRules, {BandRule: _strict_dataclass(BandRule, {})}),
        ValidationPolicy: _VALIDATION_JSON,
    },
)

POLICY: TypeAdapter[LoopPolicy] = TypeAdapter(_LOOP_JSON)
"""Reads and writes the run's ``policy.json``: every field required, unknown keys rejected."""
