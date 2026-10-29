# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.skyrl import read_skyrl_launch_config
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import snowball_rl_presets as presets

EVAL_SCORE_SCHEMA = pa.schema(
    [
        ("evaluation", pa.string()),
        ("harness", pa.string()),
        ("step", pa.int32()),
        ("item", pa.string()),
        ("attempt", pa.int32()),
        ("family", pa.string()),
        ("aaii_category", pa.string()),
        ("score", pa.float64()),
        ("scored", pa.bool_()),
    ]
)
PASS_RATES_NAME = f"rl-data/snowball-pass-rates/{presets.MODEL_REVISION}"
PASS_RATE_EVALUATION = replace(presets.EVALUATIONS["panel-910"], name="pass-rates-910x8", answers=8, post_run=False)


class InvalidScores(ValueError):
    """Evaluation answers cannot be joined or normalized for comparison."""


class InvalidPassRates(ValueError):
    """Pass-rate rows do not match the pinned initializer and panel."""


def evaluation_answers(run_output: str) -> list[tuple[int, StoragePath, dict]]:
    answers = []
    for session in presets.paths(StoragePath(run_output) / "exports/dumped_evals/*"):
        name = PurePosixPath(str(session)).name
        step = -1 if name == "eval_only" else int(name.removeprefix("global_step_").removesuffix("_evals"))
        for path in presets.paths(session / "*.jsonl"):
            if PurePosixPath(str(path)).name != "aggregated_results.jsonl":
                answers.extend(
                    (step, path, json.loads(line)) for line in presets.read_text(path).split("\n") if line.strip()
                )
    return answers


def panel_scores(
    evaluation: presets.Evaluation, run_output: str, answers: list[tuple[int, StoragePath, dict]] | None = None
) -> pa.Table:
    launch = read_skyrl_launch_config(run_output)
    manifest = {}
    for source in launch["inputs"]["validation_data"]:
        manifest.update(presets.task_manifest(StoragePath(source["uri"])))
    attempts = Counter()
    rows = []
    for step, path, row in evaluation_answers(run_output) if answers is None else answers:
        item = row.get("task_id")
        if not item:
            raise InvalidScores(f"evaluation answer lacks a TaskSpec id: {path}")
        metadata = manifest.get(item, {})
        if evaluation.file != "train" and not metadata:
            raise InvalidScores(f"panel task {item} is absent from its task manifest")
        verdict = row.get("verification")
        score = 0.0
        if verdict and verdict["status"] == "verified":
            value, lower, upper = (float(verdict[key]) for key in ("score", "score_min", "score_max"))
            if not all(math.isfinite(number) for number in (value, lower, upper)) or upper <= lower:
                raise InvalidScores(f"invalid verifier score or bounds for {item}")
            score = min(1.0, max(0.0, (value - lower) / (upper - lower)))
        rows.append(
            {
                "evaluation": evaluation.name,
                "harness": evaluation.harness.value,
                "step": step,
                "item": item,
                "attempt": attempts[step, item],
                "family": metadata.get("family"),
                "aaii_category": metadata.get("aaii_category"),
                "score": score,
                "scored": not row.get("exception_type") and not row.get("error_treatment"),
            }
        )
        attempts[step, item] += 1
    return pa.Table.from_pylist(rows, schema=EVAL_SCORE_SCHEMA)


def evaluation_statistics(scores: pa.Table) -> dict[str, float]:
    sessions = defaultdict(list)
    for row in scores.to_pylist():
        sessions[row["evaluation"]].append(row)
    result = {}
    for evaluation, rows in sessions.items():
        if len({row["step"] for row in rows}) != 1:
            raise ValueError(f"select one weight step before aggregating {evaluation}")
        scored = [row for row in rows if row["scored"]]
        result[f"{evaluation}/scored_fraction"] = len(scored) / len(rows)
        if not scored:
            continue
        result[f"{evaluation}/mean"] = math.fsum(row["score"] for row in scored) / len(scored)
        tasks = defaultdict(list)
        for row in scored:
            tasks[row["item"]].append(row)
        counts = [(sum(row["score"] == 1 for row in group), len(group)) for group in tasks.values()]
        for name, accepts in (
            ("mixed", lambda right, n: 0 < right < n),
            ("all_right", lambda right, n: right == n),
            ("all_wrong", lambda right, n: right == 0),
        ):
            result[f"{evaluation}/{name}_fraction"] = sum(accepts(right, n) for right, n in counts) / len(counts)
        for column in ("family", "aaii_category"):
            groups = defaultdict(list)
            for group in tasks.values():
                category = group[0][column]
                if category is not None:
                    groups[category].append(math.fsum(row["score"] for row in group) / len(group))
            means = {category: math.fsum(values) / len(values) for category, values in groups.items()}
            result.update({f"{evaluation}/{column}/{category}/mean": mean for category, mean in means.items()})
            if column == "family" and means:
                result[f"{evaluation}/family_mean"] = math.fsum(means.values()) / len(means)
    return result


def read_pass_rates(path: str) -> pa.Table:
    fs, key = filesystem_for(str(StoragePath(path) / "pass_rates.parquet"))
    with fs.open(key, "rb") as stream:
        table = pq.read_table(stream)
    rows = table.to_pylist()
    ids = {row["task_id"] for row in rows}
    expected = set(presets.task_manifest(StoragePath(presets.default_selection().path())))
    if (
        ids != expected
        or len(rows) != len(ids)
        or any(
            row["model_revision"] != presets.MODEL_REVISION
            or not 1 <= row["samples"] <= PASS_RATE_EVALUATION.answers
            or not 0 <= row["passed"] <= row["samples"]
            for row in rows
        )
    ):
        raise InvalidPassRates("pass-rate artifact differs from the pinned initializer or development panel")
    return table
