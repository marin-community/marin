# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure source-budget definitions for the 100B-token science curricula."""

import dataclasses
import re
from dataclasses import dataclass
from enum import StrEnum

CONTEXT = 262_144
BATCH = 256
START_STEP = 157_000
STEPS = 1_491
FINAL_STEP = START_STEP + STEPS
TOKENS = STEPS * BATCH * CONTEXT
MIXTURE_BLOCK_SIZE = STEPS * BATCH
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ScienceMix(StrEnum):
    """The three size-controlled curricula from the source audit."""

    BALANCED = "balanced"
    PROOF_FIRST = "proof-first"
    SCIENCE_FORWARD = "science-forward"


@dataclass(frozen=True)
class MixBudget:
    """Billions of tokens allocated to each source group."""

    proof: float
    swallow: float
    ultradata: float
    biocollection: float
    science: float
    curriculum: float
    replay: float

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


MIX_BUDGETS = {
    ScienceMix.BALANCED: MixBudget(14.5, 15.0, 10.0, 41.65, 14.5, 0.35, 4.0),
    ScienceMix.PROOF_FIRST: MixBudget(14.5, 22.0, 18.0, 24.65, 14.5, 0.35, 6.0),
    ScienceMix.SCIENCE_FORWARD: MixBudget(10.0, 10.0, 7.0, 51.0, 14.5, 0.35, 7.15),
}


def run_id(mix: ScienceMix, version: str) -> str:
    """Return a stable identity that cannot alias another curriculum run."""
    if not _VERSION_RE.fullmatch(version):
        raise ValueError("Version must contain only letters, digits, '.', '_', and '-'")
    return f"grug-67b-sft-science-{mix}-100b-{version}"
