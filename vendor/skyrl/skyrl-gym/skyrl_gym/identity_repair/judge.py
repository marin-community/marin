"""Judge contract for verifying attribution claims.

`SemanticJudge` is the seam between this package's deterministic detection and a semantic verifier. Production
binds an OpenAI-compatible endpoint in configuration; this package stays I/O-free and ships only the protocol and
a deterministic test double.

`ClaimKind` and `AttributionClaim` live here rather than in `detector` because both the detector and the judge
reference them; keeping the claim types in this lower module keeps imports acyclic. `detector` re-exports them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from skyrl_gym.identity_repair.facts import FactManifest


class ClaimKind(StrEnum):
    """The identity fact a claim asserts about the assistant."""

    NAME = "name"
    CREATOR = "creator"
    ORGANIZATION = "organization"


@dataclass(frozen=True)
class AttributionClaim:
    """An affirmative first-person identity claim found in an assistant turn."""

    kind: ClaimKind
    text: str  # exact claimed span substring of the turn text
    char_start: int  # offset into the turn text
    char_end: int
    asserted_entity: str  # what the assistant claimed to be (normalized)


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge evaluation of a claim against the fact manifest."""

    incorrect: bool
    ambiguous: bool
    revised_char_span: tuple[int, int] | None  # refined claim span within the turn text
    version: str


class SemanticJudge(Protocol):
    """A semantic verifier of attribution claims against the fact manifest."""

    def verify_claim(self, text: str, claim: AttributionClaim, manifest: FactManifest) -> JudgeVerdict: ...


class StaticJudge:
    """Deterministic test double that returns verdicts from a supplied callable."""

    def __init__(self, verdict_for: Callable[[str, AttributionClaim, FactManifest], JudgeVerdict]):
        self.verdict_for = verdict_for

    def verify_claim(self, text: str, claim: AttributionClaim, manifest: FactManifest) -> JudgeVerdict:
        return self.verdict_for(text, claim, manifest)
