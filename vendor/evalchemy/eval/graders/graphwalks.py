"""GraphWalks final-line node-set grading."""

import re
from collections.abc import Sequence
from typing import Any

from eval.contracts.eval_spec import EvalSpec, Grade

FINAL_ANSWER_PREFIX = "Final Answer:"
_FINAL_ANSWER = re.compile(r"\[.*\]")


def extract_answer(response: str) -> tuple[list[str], bool]:
    """Extract the last-line answer and report whether parsing failed."""
    line = response.rstrip().split("\n")[-1]
    if FINAL_ANSWER_PREFIX not in line:
        return [], True
    match = _FINAL_ANSWER.search(line)
    if match is None:
        return [], True
    return [node.strip() for node in match.group(0).strip("[]").split(",") if node.strip()], False


def graphwalks(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Score precision, recall, F1 and exact match over distinct node names."""
    extracted, failed_to_parse = extract_answer(str(responses[-1]))
    predicted, truth = set(extracted), set(spec.target)
    if failed_to_parse:
        precision = recall = f1 = 0.0
    elif not truth and not predicted:
        precision = recall = f1 = 1.0
    else:
        overlap = len(predicted & truth)
        recall = overlap / len(truth) if truth else 0.0
        precision = overlap / len(predicted) if predicted else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return Grade(
        {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "exact_match": float(predicted == truth and not failed_to_parse),
        },
        audit={"model_answer": extracted, "answer_extraction_errors": ["failed_to_parse"] if failed_to_parse else []},
        filtered_resps=[extracted],
    )
