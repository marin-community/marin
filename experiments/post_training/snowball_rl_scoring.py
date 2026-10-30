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
from experiments.post_training.task_curation.inventory import TASK_MANIFEST_FILENAME

EVAL_SCORE_SCHEMA = pa.schema(
    [
        ("evaluation", pa.string()),
        ("step", pa.int32()),
        ("item", pa.string()),
        ("attempt", pa.int32()),
        ("family", pa.string()),
        ("aaii_category", pa.string()),
        ("agentic", pa.bool_()),
        ("score", pa.float64()),
        ("scored", pa.bool_()),
    ]
)
PASS_RATES_NAME = f"rl-data/snowball-pass-rates/{presets.MODEL_REVISION}"
PASS_RATE_REVISION = "2026.10.30"
PASS_RATE_EVALUATION = replace(presets.EVALUATIONS["panel-910"], name="pass-rates-910x8", answers=8, post_run=False)


class InvalidComparison(ValueError):
    """The recorded runs cannot support this calibrated comparison."""


def task_manifest(root: StoragePath) -> dict[str, dict]:
    fs, key = filesystem_for(str(root / TASK_MANIFEST_FILENAME))
    if not fs.exists(key):
        return {}
    with fs.open(key, "rb") as stream:
        return {row["task_id"]: row for row in pq.read_table(stream).to_pylist()}


def paths(pattern: StoragePath) -> list[StoragePath]:
    fs, key = filesystem_for(str(pattern))
    return [StoragePath(fs.unstrip_protocol(path)) for path in sorted(fs.glob(key))]


def evaluation_answers(run_output: str) -> list[tuple[int, StoragePath, dict]]:
    answers = []
    for session in paths(StoragePath(run_output) / "exports/dumped_evals/*"):
        name = PurePosixPath(str(session)).name
        step = -1 if name == "eval_only" else int(name.removeprefix("global_step_").removesuffix("_evals"))
        for path in paths(session / "*.jsonl"):
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
        manifest.update(task_manifest(StoragePath(source["uri"])))
    attempts = Counter()
    rows = []
    for step, path, row in evaluation_answers(run_output) if answers is None else answers:
        item = row.get("task_id")
        if not item:
            raise InvalidComparison(f"evaluation answer lacks a TaskSpec id: {path}")
        metadata = manifest.get(item, {})
        if evaluation.file != "train" and not metadata:
            raise InvalidComparison(f"panel task {item} is absent from its task manifest")
        verdict = row.get("verification")
        score = 0.0
        if verdict and verdict["status"] == "verified":
            value, lower, upper = (float(verdict[key]) for key in ("score", "score_min", "score_max"))
            if not all(math.isfinite(number) for number in (value, lower, upper)) or upper <= lower:
                raise InvalidComparison(f"invalid verifier score or bounds for {item}")
            score = min(1.0, max(0.0, (value - lower) / (upper - lower)))
        rows.append(
            {
                "evaluation": evaluation.name,
                "step": step,
                "item": item,
                "attempt": attempts[step, item],
                "family": metadata.get("family"),
                "aaii_category": metadata.get("aaii_category"),
                "agentic": metadata.get("agentic"),
                "score": score,
                "scored": not row.get("exception_type") and not row.get("error_treatment"),
            }
        )
        attempts[step, item] += 1
    return pa.Table.from_pylist(rows, schema=EVAL_SCORE_SCHEMA)


def score_slices(scores: pa.Table, rates: pa.Table | None = None) -> dict[str, list[dict]]:
    pass_rates = (
        {row["task_id"]: row["passed"] / row["samples"] for row in rates.to_pylist()} if rates is not None else {}
    )
    groups = defaultdict(list)
    for row in scores.to_pylist():
        if not row["scored"]:
            continue
        evaluation = row["evaluation"]
        keys = [f"{evaluation}/mean"]
        rate = pass_rates.get(row["item"])
        band = "hard" if rate is not None and rate < 0.1 else "easy" if rate is not None and rate > 0.7 else "learnable"
        for column in ("family", "aaii_category", "agentic"):
            value = row[column]
            if value is not None:
                value = str(value).lower() if column == "agentic" else value
                key = f"{evaluation}/{column}/{value}"
                keys.append(key + "/mean")
                if rate is not None and column != "agentic":
                    keys.append(f"{key}/difficulty/{band}/mean")
        if rate is not None:
            keys.append(f"{evaluation}/difficulty/{band}/mean")
        for key in keys:
            groups[key].append(row)
    return dict(groups)


def evaluation_statistics(scores: pa.Table, rates: pa.Table | None = None) -> dict[str, float]:
    sessions = defaultdict(list)
    for row in scores.to_pylist():
        sessions[row["evaluation"]].append(row)
    result = {}
    for key, rows in score_slices(scores, rates).items():
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["item"]].append(row["score"])
        result[key] = (
            math.fsum(row["score"] for row in rows) / len(rows)
            if key.count("/") == 1
            else math.fsum(math.fsum(values) / len(values) for values in tasks.values()) / len(tasks)
        )
    for evaluation, rows in sessions.items():
        if len({row["step"] for row in rows}) != 1:
            raise ValueError(f"select one weight step before aggregating {evaluation}")
        scored = [row for row in rows if row["scored"]]
        result[f"{evaluation}/scored_fraction"] = len(scored) / len(rows)
        if not scored:
            continue
        tasks = defaultdict(list)
        for row in scored:
            tasks[row["item"]].append(row)
        counts = [(sum(row["score"] == 1 for row in group), len(group)) for group in tasks.values()]
        for name, accepts in (
            ("mixed", lambda right, n: 0 < right < n),
            ("all_right", lambda right, n: right == n),
            ("all_wrong", lambda right, n: right == 0),
        ):
            if name == "all_right" or any(n > 1 for n in Counter(row["item"] for row in rows).values()):
                result[f"{evaluation}/{name}_fraction"] = sum(accepts(right, n) for right, n in counts) / len(counts)
        families = [
            value for key, value in result.items() if key.startswith(f"{evaluation}/family/") and key.count("/") == 3
        ]
        if families:
            result[f"{evaluation}/family_mean"] = math.fsum(families) / len(families)
    return result


def read_pass_rates(path: str) -> pa.Table:
    fs, key = filesystem_for(str(StoragePath(path) / "pass_rates.parquet"))
    with fs.open(key, "rb") as stream:
        table = pq.read_table(stream)
    rows = table.to_pylist()
    ids = {row["task_id"] for row in rows}
    expected = set(task_manifest(StoragePath(presets.default_selection().path())))
    if (
        ids != expected
        or len(rows) != len(ids)
        or any(
            row["model_revision"] != presets.MODEL_REVISION
            or not 1 <= row["samples"] <= PASS_RATE_EVALUATION.answers
            or not 0 <= row["passed"] <= row["samples"]
            or not math.isfinite(row["mean_score"])
            or not 0 <= row["mean_score"] <= 1
            for row in rows
        )
    ):
        raise InvalidComparison("pass-rate artifact differs from the pinned initializer or development panel")
    return table
