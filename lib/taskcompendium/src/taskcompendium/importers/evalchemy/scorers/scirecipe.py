# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured metrics adapted from Thoth eval/eval_batch.py (MIT; see Thoth-MIT.txt).

Reference: a155710079da47bb26a021646fa84d5544aa180e. Preserve its greedy alignment,
line-count step match, exact-list order_strict, and unnormalized content_score (up to 1.5).
"""

import json
import re
from dataclasses import dataclass

from verifyit.grade import InvalidTask, Reward, scored

ROBERTA_PATH = "/opt/scirecipe/roberta-large"
MINILM_PATH = "/opt/scirecipe/all-MiniLM-L6-v2"

TAG_PATTERN = re.compile(
    r"<key>(.*?)</key>\s*<orc>(.*?)</orc>\s*<note>(.*?)</note>",
    re.DOTALL | re.IGNORECASE,
)
REASONING_TAG = re.compile(r"<think>", re.IGNORECASE)
FULL_PATTERN = re.compile(r"<think>.*?</think>\s*" + TAG_PATTERN.pattern, re.DOTALL | re.IGNORECASE)
STEP_PATTERN = re.compile(r"^Step\s+\d+\s*:\s*(\{.*\})\s*$", re.IGNORECASE)
STRUCTURED_METRICS = (
    "step_match",
    "order_strict",
    "order_lcs",
    "order_tau",
    "content_score",
    "avg_obj",
    "avg_para",
    "avg_mx",
)
TEXT_METRICS = (
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "bleu_avg",
    "meteor",
    "rouge1",
    "rouge2",
    "rougeL",
    "kw_precision",
    "kw_recall",
    "kw_f1",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
)


@dataclass(frozen=True)
class Step:
    action: str
    objects: tuple[str, ...]
    parameters: tuple[str, ...]


def step_lines(block: str) -> list[str]:
    return [line.rstrip() for line in block.strip().splitlines() if line.strip()]


def parse_steps(block: str) -> list[Step]:
    steps = []
    for line in step_lines(block):
        match = STEP_PATTERN.match(line)
        if match is None:
            continue
        try:
            value = json.loads(match[1])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not value:
            continue
        steps.append(
            Step(
                value.get("action", "").strip().lower(),
                tuple(x.strip().lower() for x in value.get("objects", []) if isinstance(x, str) and x.strip()),
                tuple(x.strip().lower() for x in value.get("parameters", []) if isinstance(x, str) and x.strip()),
            )
        )
    return steps


def keyword_set(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9%µμ\-\._]+", text.lower()) if token}


def intersection_over_union(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def structured_metrics(prediction: str, reference: str) -> dict[str, float]:
    metrics = dict.fromkeys(STRUCTURED_METRICS, 0.0)
    try:
        expected = parse_steps(reference)
    except (AttributeError, TypeError) as error:
        raise InvalidTask("Malformed typed fields in the SciRecipe reference") from error
    try:
        predicted = parse_steps(prediction)
    except (AttributeError, TypeError):
        # The reference assigns zero to every structured metric on malformed typed fields.
        return metrics
    metrics["step_match"] = float(len(step_lines(prediction)) == len(step_lines(reference)))
    pred_actions, gold_actions = [s.action for s in predicted], [s.action for s in expected]
    metrics["order_strict"] = float(pred_actions == gold_actions)
    previous = [0] * (len(expected) + 1)
    for action in pred_actions:
        current = [0]
        for j, gold in enumerate(gold_actions):
            current.append(previous[j] + 1 if action == gold else max(previous[j + 1], current[-1]))
        previous = current
    length = len(predicted) + len(expected)
    metrics["order_lcs"] = 2 * previous[-1] / length if length else 1.0
    pairs = []
    j = 0
    for i, action in enumerate(pred_actions):
        while j < len(expected) and gold_actions[j] != action:
            j += 1
        if j < len(expected):
            pairs.append((i, j))
            j += 1
    # Reference pairs are monotonic, so Kendall tau is 1 for at least two pairs.
    metrics["order_tau"] = float(len(pairs) > 1)
    for i, j in pairs:
        pred, gold = predicted[i], expected[j]
        objects = intersection_over_union(set(pred.objects), set(gold.objects))
        if objects == 0:
            objects = intersection_over_union(keyword_set(" ".join(pred.objects)), keyword_set(" ".join(gold.objects)))
        parameters = 0.0
        if objects >= 0.5:
            if not pred.parameters and not gold.parameters:
                parameters = 1.0
            elif pred.parameters and gold.parameters:
                parameters = intersection_over_union(
                    keyword_set(" ".join(pred.parameters)), keyword_set(" ".join(gold.parameters))
                )
        displacement = max(0.0, 1.0 - (abs(i - j) / max(1, len(expected))) ** 1.5)
        for name, value in (
            ("content_score", displacement * (objects + 0.5 * parameters)),
            ("avg_obj", objects),
            ("avg_para", parameters),
            ("avg_mx", displacement),
        ):
            metrics[name] += value
    for name in ("content_score", "avg_obj", "avg_para", "avg_mx"):
        metrics[name] /= max(1, len(pairs))
    return metrics


def prediction_blocks(response: str) -> tuple[str, str]:
    """Extract a complete block sequence, else return the whole response and an empty orc."""
    # Serving APIs can separate <think> into a reasoning channel. Only the final artifact is graded.
    pattern = FULL_PATTERN if REASONING_TAG.search(response) else TAG_PATTERN
    match = pattern.search(response)
    return (match[1], match[2]) if match else (response, "")


def grade_scirecipe(reference_key: str, reference_orc: str, response: str) -> Reward:
    # Optional model dependencies are installed only in this benchmark's grader environment.
    from .scirecipe_text import text_metrics  # noqa: PLC0415

    key, orc = prediction_blocks(response)
    metrics = {**structured_metrics(key, reference_key), **text_metrics(reference_orc, orc)}
    return scored(metrics["content_score"] / 1.5, metrics=metrics)
