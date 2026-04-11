# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data types for alignment pipeline."""

from dataclasses import dataclass, field
from enum import Enum


class StatementType(Enum):
    """Type of policy statement."""

    GUIDELINE = "GUIDELINE"
    REQUIREMENT = "REQUIREMENT"
    PROHIBITION = "PROHIBITION"


class AuthorityLevel(Enum):
    """Authority level for a statement."""

    PLATFORM = "PLATFORM"
    DEVELOPER = "DEVELOPER"
    USER = "USER"
    GUIDELINE = "GUIDELINE"


@dataclass
class Example:
    """A good/bad response example for a statement."""

    description: str
    user_query: str
    good_response: str
    bad_response: str


@dataclass
class Statement:
    """A policy statement from a behavioral specification.

    Attributes:
        id: Unique identifier for the statement.
        text: The full text of the policy statement.
        type: Type of statement (GUIDELINE, REQUIREMENT, PROHIBITION).
        authority_level: Authority level (PLATFORM, DEVELOPER, USER, GUIDELINE).
        section: High-level section in the spec.
        subsection: Subsection within the section.
        examples: List of good/bad response examples for calibration.
        related_statements: List of related statement IDs.
    """

    id: str
    text: str
    type: StatementType
    authority_level: AuthorityLevel
    section: str
    subsection: str
    examples: list[Example] = field(default_factory=list)
    related_statements: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Statement":
        """Create a Statement from a dictionary (parsed JSONL line)."""
        examples = []
        metadata = data.get("metadata", {})
        for ex in metadata.get("examples", []):
            examples.append(
                Example(
                    description=ex.get("description", ""),
                    user_query=ex.get("user_query", ""),
                    good_response=ex.get("good_response", ""),
                    bad_response=ex.get("bad_response", ""),
                )
            )

        return cls(
            id=data["id"],
            text=data["text"],
            type=StatementType(data["type"]),
            authority_level=AuthorityLevel(data["authority_level"]),
            section=data.get("section", ""),
            subsection=data.get("subsection", ""),
            examples=examples,
            related_statements=data.get("related_statements", []),
        )


@dataclass
class ComplianceResult:
    """Result of a compliance judgment.

    score and compliant are None when the judge response could not be parsed
    into a valid score. Downstream aggregation must skip None-scored results
    rather than coerce them to a default — conflating parse failures with
    low/high scores biases mean score and compliance rate.
    """

    score: int | None
    compliant: bool | None
    confidence: float
    explanation: str
    highlights: list[str] = field(default_factory=list)
    raw_response: str | None = None

    @classmethod
    def from_judge_output(cls, output: dict, raw_response: str | None = None) -> "ComplianceResult":
        raw_score = output.get("score")
        score: int | None
        compliant: bool | None
        if raw_score is None:
            score = None
            compliant = None
        else:
            score = int(raw_score)
            compliant = score >= 7
        return cls(
            score=score,
            compliant=compliant,
            confidence=output.get("confidence", 0.5),
            explanation=output.get("explanation", ""),
            highlights=output.get("highlights", []),
            raw_response=raw_response,
        )
