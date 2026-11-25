# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute evaluation scores from verified task answers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import pyarrow as pa
from finestore.rl.run_records import EvaluationAnswerRow, EvaluationTaskRow

FAMILY_SLICE = "family"
SLICE_COLUMNS = (FAMILY_SLICE, "aai_category", "source", "topic", "difficulty", "agentic")
AAII_WEIGHTS = {"scientific_reasoning": 0.2, "coding": 0.2, "general": 0.3, "agents": 0.3}

ANSWER_SCHEMA = pa.schema(
    [
        ("evaluation", pa.string()),
        ("step", pa.int32()),
        ("task_id", pa.string()),
        ("task_digest", pa.string()),
        ("answer", pa.int32()),
        *((column, pa.bool_() if column == SLICE_COLUMNS[-1] else pa.string()) for column in SLICE_COLUMNS),
        ("score", pa.float64()),
        ("scored", pa.bool_()),
        ("response_token_count", pa.int32()),
    ]
)


def _has_verified_score(answer: EvaluationAnswerRow) -> bool:
    return answer.verification_status == "verified" and answer.verifier_score is not None


def _scored_answer(answer: EvaluationAnswerRow) -> bool:
    return _has_verified_score(answer) and not answer.error_type and not answer.error_treatment


def answer_table(evaluation_name: str, tasks: Sequence[EvaluationTaskRow]) -> pa.Table:
    """Arrange recorded task answers in a table for scoring."""
    rows = []
    ordinals = Counter()
    for task, answer in sorted(
        ((row, value) for row in tasks for value in row.answers), key=lambda pair: (pair[0].step, pair[1].ordinal)
    ):
        verified = _has_verified_score(answer)
        rows.append(
            {
                "evaluation": evaluation_name,
                "step": task.step,
                "task_id": task.task_id,
                "task_digest": task.task_digest,
                "answer": ordinals[task.step, task.task_id],
                **task.strata.model_dump(),
                "score": answer.verifier_score if verified else 0.0,
                "scored": _scored_answer(answer),
                "response_token_count": answer.response_token_count,
            }
        )
        ordinals[task.step, task.task_id] += 1
    return pa.Table.from_pylist(rows, schema=ANSWER_SCHEMA)


def score_slices(scores: pa.Table) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in scores.to_pylist():
        if not row["scored"]:
            continue
        evaluation = row["evaluation"]
        keys = [f"{evaluation}/mean"]
        for column in SLICE_COLUMNS:
            value = row[column]
            if value is not None:
                value = str(value).lower() if column == SLICE_COLUMNS[-1] else value
                key = f"{evaluation}/{column}/{value}"
                keys.append(key + "/mean")
        for key in keys:
            groups[key].append(row)
    for evaluation in {key.split("/", 1)[0] for key in groups}:
        agentic = [key for key in groups if key.startswith(f"{evaluation}/{SLICE_COLUMNS[-1]}/")]
        if len(agentic) == 1:
            # A task set without both kinds of task has no agentic split; its one slice repeats the overall mean.
            del groups[agentic[0]]
    return dict(groups)


def score_membership(session: pa.Table, statistic: str) -> dict:
    """Describe the stored task assignments consumed by an evaluation mean."""
    columns = {
        "family_mean": ("task_id", "family"),
        "aai_weighted_mean": ("task_id", "aai_category"),
    }.get(statistic, ("task_id",))
    if statistic.startswith("family/"):
        columns = ("task_id", "family")
    members = {tuple(row[column] for column in columns) for row in session.select(columns).to_pylist()}
    return {
        "members": sorted(members, key=lambda values: (values[0], tuple(value or "" for value in values[1:]))),
        "weights": AAII_WEIGHTS if statistic == "aai_weighted_mean" else None,
    }


def score_answers(scores: pa.Table) -> dict[str, float]:
    sessions = defaultdict(list)
    for row in scores.to_pylist():
        sessions[row["evaluation"]].append(row)
    result = {}
    for key, rows in score_slices(scores).items():
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["task_id"]].append(row["score"])
        result[key] = (
            math.fsum(row["score"] for row in rows) / len(rows)
            if key.count("/") == 1
            else math.fsum(math.fsum(values) / len(values) for values in tasks.values()) / len(tasks)
        )
    for evaluation, rows in sessions.items():
        if len({row["step"] for row in rows}) != 1:
            raise ValueError(f"select one weight step before aggregating {evaluation}")
        scored = [row for row in rows if row["scored"]]
        result[f"{evaluation}/attempted"] = len(rows)
        result[f"{evaluation}/scored"] = len(scored)
        result[f"{evaluation}/scored_fraction"] = len(scored) / len(rows)
        represented = {
            category: weight
            for category, weight in AAII_WEIGHTS.items()
            if any(row["aai_category"] == category for row in rows)
        }
        weight_sum = math.fsum(represented.values())
        result[f"{evaluation}/aai_category_weight_coverage"] = weight_sum
        if weight_sum:
            result.update(
                {
                    f"{evaluation}/aai_category/{category}/weight": weight / weight_sum
                    for category, weight in represented.items()
                }
            )
        if not scored:
            continue
        tasks = defaultdict(list)
        for row in scored:
            tasks[row["task_id"]].append(row)
        counts = [(sum(row["score"] == 1 for row in group), len(group)) for group in tasks.values()]
        for name, accepts in (
            ("mixed", lambda right, n: 0 < right < n),
            ("all_right", lambda right, n: right == n),
            ("all_wrong", lambda right, n: right == 0),
        ):
            if name == "all_right" or any(n > 1 for n in Counter(row["task_id"] for row in rows).values()):
                result[f"{evaluation}/{name}_fraction"] = sum(accepts(right, n) for right, n in counts) / len(counts)
        families = [
            value
            for key, value in result.items()
            if key.startswith(f"{evaluation}/{FAMILY_SLICE}/") and key.count("/") == 3
        ]
        if families:
            result[f"{evaluation}/family_mean"] = math.fsum(families) / len(families)
        if weight_sum and all(f"{evaluation}/aai_category/{category}/mean" in result for category in represented):
            result[f"{evaluation}/aai_weighted_mean"] = (
                math.fsum(
                    weight * result[f"{evaluation}/aai_category/{category}/mean"]
                    for category, weight in represented.items()
                )
                / weight_sum
            )
    return result
