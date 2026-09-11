# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode reasoning-gym: score the answer file with the reasoning-gym dataset's own scorer.

The task ships the generated entry (its gold answer and metadata) as JSON under tests/; the spec
names the dataset it came from. ``score_answer`` returns a float in [0, 1], which becomes the
reward directly -- several reasoning-gym datasets award partial credit. A dataset name the library
does not know, or an entry file that is missing or not an entry, is a task defect.
"""

import json
from pathlib import Path

import reasoning_gym

from tasktrove_verify.grade import read_output
from tasktrove_verify.grade import InvalidTask, Reward, scored
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


def grade(spec: ReasoningGymSpec, tests_dir: Path, workspace: Path) -> Reward:
    entry = load_entry(tests_dir / spec.entry)
    try:
        score_answer = reasoning_gym.get_score_answer_fn(spec.dataset)
    except ValueError as error:
        raise InvalidTask(f"unknown reasoning-gym dataset {spec.dataset!r}") from error

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    answer = text.strip()
    try:
        # pyrefly: ignore[bad-argument-count]  # reasoning-gym annotates the factory's return as
        # Callable[[], float]; the bound score_answer takes (answer, entry).
        score = score_answer(answer, entry)
    except Exception as error:
        return scored(0.0, reason="scorer_error", error=f"{type(error).__name__}: {error}")
    if not isinstance(score, int | float):
        raise TypeError(f"reasoning-gym scorer for {spec.dataset} returned {type(score).__name__}")
    return scored(float(score), dataset=spec.dataset, answer=answer[:CANDIDATE_DETAIL_CHARS])
