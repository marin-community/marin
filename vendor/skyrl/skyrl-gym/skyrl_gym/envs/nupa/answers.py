"""Adapt SkyRL JSON ground truth to the shared NUPA source representation."""

import json

from taskcompendium.importers.evalchemy.scorers.nupa_preparation import NupaAnswerFormat


def parse_ground_truth(ground_truth: object) -> tuple[str, str]:
    """Return the (answer, answer_format) pair encoded in a reward-spec ground truth."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as error:
            raise ValueError(f"NUPA ground truth must be a JSON object: {ground_truth!r}") from error
    if not isinstance(ground_truth, dict):
        raise ValueError(f"NUPA ground truth must be a JSON object: {ground_truth!r}")
    answer, answer_format = ground_truth.get("answer"), ground_truth.get("answer_format")
    if not isinstance(answer, str) or not answer:
        raise ValueError("NUPA ground truth requires a nonempty 'answer' string")
    if answer_format not in tuple(NupaAnswerFormat):
        raise ValueError(f"Unsupported NUPA answer format: {answer_format!r}")
    return answer, answer_format
