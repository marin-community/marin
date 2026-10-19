# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rubric an authoring model writes for a model-graded verifier: what a correct deliverable has.

A rubric is data the author states once. Which grader applies it, and how, belongs to that grader's
module (``spec.judge`` for an LLM judge).
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_RUBRIC_ITEMS = 48


class ItemRole(StrEnum):
    SCORED = "scored"
    GATE = "gate"
    CAP = "cap"
    DISQUALIFIER = "disqualifier"


class RubricItem(BaseModel):
    """One property a correct deliverable has, stated as its intent; "satisfied" is always the good outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, description="A short snake_case name, unique in the rubric.")
    intent: str = Field(min_length=1, description="What property of the answer this item establishes.")
    pass_criterion: str = Field(
        min_length=1,
        description="When an expert marks the item satisfied, in plain language: the meaning, never a word or "
        "pattern ('the answer recommends design B', not 'contains recommend'), with the reference values, "
        "tolerances and units.",
    )
    equivalences: list[str] = Field(
        description="Forms that also satisfy the item: notation, units, synonyms, order, equivalent expressions."
    )
    negation_handling: str = Field(
        description="How negated, quoted, hypothetical or retracted mentions count for this item ('' if none can arise)."
    )
    weight: float = Field(ge=0, description="The item's share of the reward; 0 for a pure gate or cap.")
    role: ItemRole = Field(
        description="scored; gate or disqualifier (failing it caps the whole reward); cap (failing it caps the reward)."
    )
    cap_value: float | None = Field(
        description="For gate, cap and disqualifier items, the most the reward can be when the item is not "
        "satisfied (0 zeroes it); null for a scored item."
    )

    @model_validator(mode="after")
    def capped(self) -> Self:
        if (self.role is ItemRole.SCORED) != (self.cap_value is None):
            raise ValueError(f"rubric item {self.id!r}: cap_value is set exactly when the role is not scored")
        if self.cap_value is not None and not 0 <= self.cap_value < 1:
            raise ValueError(f"rubric item {self.id!r}: cap_value lies in [0, 1)")
        return self


class Rubric(BaseModel):
    """What a correct answer delivers, item by item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deliverable: str = Field(
        min_length=1,
        description="The deliverable as the task states it to the solver, plus the given data a grader needs.",
    )
    items: list[RubricItem] = Field(min_length=1, max_length=MAX_RUBRIC_ITEMS)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"rubric item ids repeat: {sorted({i for i in ids if ids.count(i) > 1})}")
        if not any(item.role is ItemRole.SCORED and item.weight > 0 for item in self.items):
            raise ValueError("the rubric needs a scored item with a positive weight")
        return self
