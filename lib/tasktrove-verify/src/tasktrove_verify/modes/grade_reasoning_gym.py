# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode reasoning-gym: score the answer file with the reasoning-gym dataset's own scorer.

The task ships the generated entry (its gold answer and metadata) as JSON under tests/; the spec
names the dataset it came from. ``score_answer`` returns a float in [0, 1], which becomes the
reward directly -- several reasoning-gym datasets award partial credit. A dataset name the library
does not know, or an entry file that is missing or not an entry, is a task defect.
"""

import json
import math
from pathlib import Path

import reasoning_gym

from tasktrove_verify.grade import InvalidTask, Reward, read_output, scored
from tasktrove_verify.spec import ReasoningGymSpec

CANDIDATE_DETAIL_CHARS = 200


def load_entry(path: Path) -> dict:
    """The reasoning-gym entry at ``path``. Raises ``InvalidTask`` when it is absent or malformed."""
    if not path.is_file():
        raise InvalidTask(f"reasoning-gym entry not found: {path}")
    try:
        entry = json.loads(path.read_text())
    except ValueError as error:
        raise InvalidTask(f"reasoning-gym entry {path} is not JSON: {error}") from error
    if not isinstance(entry, dict) or "metadata" not in entry:
        raise InvalidTask(f"reasoning-gym entry {path} must be an object with a metadata field")
    return entry


def grade_reasoning_gym_candidate(spec: ReasoningGymSpec, entry: dict, candidate: str | None) -> Reward:
    """Score extracted text with the dataset's scorer, retaining partial credit.

    The caller owns isolation: upstream scorers can parse or execute candidate
    expressions. Scorer failures propagate; they are not incorrect answers.
    """
    try:
        score_answer = reasoning_gym.get_score_answer_fn(spec.dataset)
    except ValueError as error:
        raise InvalidTask(f"unknown reasoning-gym dataset {spec.dataset!r}") from error
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("source_dataset") != spec.dataset:
        raise InvalidTask("reasoning-gym entry dataset differs from its verifier")
    if not isinstance(entry.get("answer"), str) or not entry["answer"].strip():
        raise InvalidTask("reasoning-gym entry requires a nonempty answer")
    if candidate is None or not candidate.strip():
        return scored(0.0, reason="no_output")
    answer = candidate.strip()
    # pyrefly: ignore[bad-argument-count]  # The upstream bound scorer accepts answer and entry.
    score = score_answer(answer, entry)
    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise InvalidTask(f"reasoning-gym scorer for {spec.dataset} returned an invalid reward")
    return scored(float(score), dataset=spec.dataset, answer=answer[:CANDIDATE_DETAIL_CHARS])


def grade(spec: ReasoningGymSpec, tests_dir: Path, workspace: Path) -> Reward:
    entry = load_entry(tests_dir / spec.entry)
    return grade_reasoning_gym_candidate(spec, entry, read_output(spec, workspace))
