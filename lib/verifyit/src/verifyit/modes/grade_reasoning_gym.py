# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode reasoning-gym: score the answer file with the reasoning-gym dataset's own scorer.

The task ships the generated entry (its reference answer or null and metadata) as JSON under tests/; the spec
names the dataset it came from. ``score_answer`` returns a float in [0, 1], which becomes the
reward directly -- several reasoning-gym datasets award partial credit. A dataset name the library
does not know, or an entry file that is missing or not an entry, is a task defect.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import reasoning_gym
from reasoning_gym.factory import DATASETS

from verifyit.file_ops.read import read_text
from verifyit.grade import InvalidTask, Reward, empty_output_policy, read_output, scored
from verifyit.json_objects import unique_object
from verifyit.spec import EmptyOutputPolicy, ReasoningGymSpec

CANDIDATE_DETAIL_CHARS = 200


def load_entry(path: Path) -> dict:
    """The reasoning-gym entry at ``path``. Raises ``InvalidTask`` when it is absent or malformed."""
    if not path.is_file():
        raise InvalidTask(f"reasoning-gym entry not found: {path}")
    try:
        text = read_text(path)
    except ValueError as error:
        raise InvalidTask(f"reasoning-gym entry {path} is not JSON: {error}") from error
    return parse_entry(text, str(path))


def parse_entry(text: str, source: str) -> dict:
    """The reasoning-gym entry in ``text``, read from ``source``. Raises ``InvalidTask`` when malformed."""
    try:
        entry = json.loads(text, object_pairs_hook=unique_object)
        json.dumps(entry, allow_nan=False)
    except ValueError as error:
        raise InvalidTask(f"reasoning-gym entry {source} is not JSON: {error}") from error
    if not isinstance(entry, dict) or "metadata" not in entry:
        raise InvalidTask(f"reasoning-gym entry {source} must be an object with a metadata field")
    return entry


def parse_params(text: str) -> dict:
    """The scorer configuration object in ``text``. Raises ``InvalidTask`` when malformed."""
    try:
        params = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        json.dumps(params, allow_nan=False)
    except ValueError as error:
        raise InvalidTask(f"invalid reasoning-gym params: {error}") from error
    return params


def _load_params(spec: ReasoningGymSpec, tests_dir: Path) -> dict | None:
    if spec.params is None:
        return None
    if not isinstance(spec.params, str) or not spec.params:
        raise InvalidTask("reasoning-gym params must name a JSON file")
    params_path = tests_dir / spec.params
    if not params_path.is_file():
        raise InvalidTask(f"reasoning-gym params not found: {params_path}")
    try:
        text = read_text(params_path)
    except UnicodeError as error:
        raise InvalidTask(f"invalid reasoning-gym params: {error}") from error
    return parse_params(text)


def _score_answer(dataset: str, params: dict | None) -> Callable[[str, dict], float]:
    if params is None:
        try:
            return cast(Callable[[str, dict], float], reasoning_gym.get_score_answer_fn(dataset))
        except ValueError as error:
            raise InvalidTask(f"unknown reasoning-gym dataset {dataset!r}") from error
    try:
        _, config_cls = DATASETS[dataset]
    except KeyError as error:
        raise InvalidTask(f"unknown reasoning-gym dataset {dataset!r}") from error
    # Upstream permits configs without validate (power_function and tsumego).
    try:
        config = config_cls(**params)
        validate = getattr(config, "validate", None)
        if validate is not None:
            validate()
    except (TypeError, ValueError, AssertionError) as error:
        raise InvalidTask(f"invalid reasoning-gym configuration: {error}") from error
    return reasoning_gym.create_dataset(dataset, **params).score_answer


def grade_reasoning_gym_candidate(
    spec: ReasoningGymSpec, entry: dict, candidate: str | None, *, params: dict | None = None
) -> Reward:
    """Score extracted text with the dataset's configured scorer, retaining partial credit.

    Supply decoded params for a spec that names a params file. The caller owns isolation:
    upstream scorers can parse or execute candidate expressions. Scorer failures propagate.
    """
    if spec.params is not None and params is None:
        raise InvalidTask("reasoning-gym candidate grading requires decoded params")
    score_answer = _score_answer(spec.dataset, params)
    _validate_entry(spec, entry)
    return _grade_candidate(spec, entry, candidate, score_answer)


def _validate_entry(spec: ReasoningGymSpec, entry: dict) -> None:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("source_dataset") != spec.dataset:
        raise InvalidTask("reasoning-gym entry dataset differs from its verifier")
    if "answer" not in entry or (entry["answer"] is not None and not isinstance(entry["answer"], str)):
        raise InvalidTask("reasoning-gym entry requires an answer field containing a string or null")


def _grade_candidate(
    spec: ReasoningGymSpec, entry: dict, candidate: str | None, score_answer: Callable[[str, dict], float]
) -> Reward:
    policy = empty_output_policy(spec)
    if candidate is None or (not candidate.strip() and policy is EmptyOutputPolicy.ZERO):
        return scored(0.0, reason="no_output")
    answer = candidate.strip()
    score = score_answer(answer, entry)
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError(f"reasoning-gym scorer for {spec.dataset} returned {type(score).__name__}")
    return scored(float(score), dataset=spec.dataset, answer=answer[:CANDIDATE_DETAIL_CHARS])


def grade(spec: ReasoningGymSpec, tests_dir: Path, workspace: Path) -> Reward:
    entry = load_entry(tests_dir / spec.entry)
    params = _load_params(spec, tests_dir)
    score_answer = _score_answer(spec.dataset, params)
    _validate_entry(spec, entry)
    return _grade_candidate(spec, entry, read_output(spec, workspace), score_answer)
