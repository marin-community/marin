# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Recipe-owned reference matching and abstention policy."""

import json
import os
import re
import string
from dataclasses import asdict
from pathlib import Path

from verifyit.grade import Reward, infra_error, scored
from verifyit.modes.grade_judge import boxed_answer, normalize

ABSTENTION_LIMIT = 64 * 1024
ABSTENTIONS = frozenset({"idk", "i dont know", "unknown", "unanswerable", "cannot answer"})


def grade_reference_candidate(references: tuple[str, ...], candidate: str) -> Reward:
    """Score exact open-QA answers and leave semantic paraphrases ungraded."""
    if not candidate.strip():
        return scored(0.0)
    if normalize(boxed_answer(candidate)) in {normalize(reference) for reference in references}:
        return scored(1.0)
    return infra_error("The source semantic reference judge is not bound")


def abstention_normalized(text: str) -> str:
    """Use the source abstention gate's narrower normalization, without LaTeX or Unicode folding."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def grade_abstention_candidate(references: tuple[str, ...], abstention_token: str | None, candidate: str) -> Reward:
    """Keep source exact matches ahead of abstention rejection and an unbound semantic judge."""
    normalized = abstention_normalized(boxed_answer(candidate[:ABSTENTION_LIMIT]))
    if normalized in {abstention_normalized(reference) for reference in references}:
        return scored(1.0)
    if not normalized:
        return scored(0.0)
    if normalized in ABSTENTIONS and not abstention_token:
        return scored(0.0)
    return infra_error("The source semantic abstention judge is not bound")


def main() -> None:
    tests = Path(os.environ["VERIFYIT_TESTS_DIR"])
    workspace = Path(os.environ["VERIFYIT_WORKSPACE"])
    config = json.loads((tests / "config.json").read_text())
    answer = (workspace / "answer.txt").read_text()
    verdict = (
        grade_abstention_candidate(tuple(config["references"]), config["abstention_token"], answer)
        if "abstention_token" in config
        else grade_reference_candidate(tuple(config["references"]), answer)
    )
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "verdict.json").write_text(json.dumps(asdict(verdict)))


if __name__ == "__main__":
    main()
