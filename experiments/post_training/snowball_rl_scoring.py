# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import PurePosixPath
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.skyrl_records import read_skyrl_launch_config
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.task_curation.selection import TASK_MANIFEST_FILENAME, TASK_SPEC_COLUMN, TRAIN_FILENAME

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
TRAIN_STEM = PurePosixPath(TRAIN_FILENAME).stem


class InvalidComparison(ValueError):
    """The recorded runs cannot support this calibrated comparison."""


def task_manifest(root: StoragePath) -> dict[str, dict]:
    fs, key = filesystem_for(str(root / TASK_MANIFEST_FILENAME))
    if not fs.exists(key):
        return {}
    with fs.open(key, "rb") as stream:
        return {row["task_id"]: row for row in pq.read_table(stream).to_pylist()}


def paths(pattern: StoragePath) -> list[StoragePath]:
    return sorted(pattern.glob(), key=str)


class EvaluationAnswer(NamedTuple):
    step: int
    path: StoragePath
    answer: dict


def _evaluation_answers(dump_root: StoragePath) -> list[EvaluationAnswer]:
    answers = []
    for session in paths(dump_root / "*"):
        name = PurePosixPath(str(session)).name
        step = -1 if name == "eval_only" else int(name.removeprefix("global_step_").removesuffix("_evals"))
        for path in paths(session / "*.jsonl"):
            if PurePosixPath(str(path)).name != "aggregated_results.jsonl":
                answers.extend(
                    EvaluationAnswer(step, path, json.loads(line))
                    for line in path.read_text().split("\n")
                    if line.strip()
                )
    return answers


def evaluation_answers(run_output: str) -> list[EvaluationAnswer]:
    launch = read_skyrl_launch_config(run_output)
    return _evaluation_answers(StoragePath(launch["artifacts"]["export_root"]) / "dumped_evals")


def recorded_sources(launch: dict, name: str) -> list[dict]:
    return launch["inputs"].get("evaluation_data", {}).get(name, launch["inputs"]["validation_data"])


def native_prefix(launch: dict, name: str) -> str:
    return f"eval/{name}/" if name in launch["inputs"].get("evaluation_data", {}) else "eval/"


def recorded_identity(source: dict) -> tuple[str, str]:
    name, separator, version = source["identity"].partition(":")[0].rpartition("@")
    if not name or not separator or not version:
        raise ValueError("expected NAME@VERSION")
    return name, version


class RecordedPanel(NamedTuple):
    task_ids: tuple[str, ...]
    task_set_sha256: str


def read_recorded_panel(launch: dict, name: str) -> RecordedPanel:
    """Read panel task IDs and a hash of the sorted raw TaskSpec strings."""
    pairs = []
    for source in recorded_sources(launch, name):
        path = StoragePath(source["uri"]) / source["relative_path"]
        fs, key = filesystem_for(str(path))
        with fs.open(key, "rb") as stream:
            for raw in pq.read_table(stream, columns=[TASK_SPEC_COLUMN])[TASK_SPEC_COLUMN].to_pylist():
                pairs.append((json.loads(raw)["id"], raw))
    payload = json.dumps(sorted(pairs), separators=(",", ":"), ensure_ascii=False).encode()
    return RecordedPanel(tuple(task_id for task_id, _ in pairs), hashlib.sha256(payload).hexdigest())


def evaluation_protocol(launch: dict, name: str) -> dict:
    """The evaluation request recorded by native configuration composition."""
    config = launch["skyrl"]
    generator = config["generator"]
    sampling = deepcopy(generator["eval_sampling_params"])
    samples = generator["eval_n_samples_per_prompt"]
    if name in launch["inputs"].get("evaluation_data", {}):
        profile = next(
            (
                profile
                for callback in config["trainer"]["callbacks"]
                if callback["type"] == "evaluation"
                for key, profile in (callback.get("additional_evaluations") or {}).items()
                if key == name
            ),
            None,
        )
        if profile is None:
            raise InvalidComparison(f"expected one recorded evaluation profile for {name}")
        sampling = deepcopy(profile["sampling_params"])
        samples = profile["n_samples_per_prompt"]
    return {
        "sampling_params": sampling,
        "n_samples_per_prompt": samples,
        "context_budget": {
            "max_new_tokens_per_turn": sampling["max_generate_length"],
            "max_turns": generator["max_turns"],
        },
        "max_prompt_length": config["trainer"]["max_prompt_length"],
        "chat_template": generator.get("chat_template"),
        "task_env": {key: value for key, value in launch["runtime"]["task_env"].items() if not key.startswith("VLLM_")},
    }


def panel_scores(
    evaluation: presets.Evaluation, run_output: str, primary_answers: list[EvaluationAnswer] | None = None
) -> pa.Table:
    launch = read_skyrl_launch_config(run_output)
    sources = recorded_sources(launch, evaluation.name)
    answers = (
        _evaluation_answers(StoragePath(launch["artifacts"]["export_root"]) / evaluation.name / "dumped_evals")
        if native_prefix(launch, evaluation.name) != "eval/"
        else evaluation_answers(run_output) if primary_answers is None else primary_answers
    )
    manifest = {}
    for source in sources:
        manifest.update(task_manifest(StoragePath(source["uri"])))
    attempts = Counter()
    rows = []
    for step, path, row in answers:
        item = row.get("task_id")
        if not item:
            raise InvalidComparison(f"evaluation answer lacks a TaskSpec id: {path}")
        metadata = manifest.get(item, {})
        if evaluation.held_out and not metadata:
            raise InvalidComparison(f"panel task {item} is absent from its task manifest")
        verdict = row.get("verification")
        verified = bool(verdict and verdict["status"] == "verified")
        score = 0.0
        if verified:
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
                "scored": verified and not row.get("exception_type") and not row.get("error_treatment"),
            }
        )
        attempts[step, item] += 1
    return pa.Table.from_pylist(rows, schema=EVAL_SCORE_SCHEMA)


def score_slices(scores: pa.Table) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in scores.to_pylist():
        if not row["scored"]:
            continue
        evaluation = row["evaluation"]
        keys = [f"{evaluation}/mean"]
        for column in ("family", "aaii_category", "agentic"):
            value = row[column]
            if value is not None:
                value = str(value).lower() if column == "agentic" else value
                key = f"{evaluation}/{column}/{value}"
                keys.append(key + "/mean")
        for key in keys:
            groups[key].append(row)
    for evaluation in {key.split("/", 1)[0] for key in groups}:
        agentic = [key for key in groups if key.startswith(f"{evaluation}/agentic/")]
        if len(agentic) == 1:
            # A panel without both kinds of task has no agentic split; its one slice repeats the overall mean.
            del groups[agentic[0]]
    return dict(groups)


def evaluation_statistics(scores: pa.Table) -> dict[str, float]:
    sessions = defaultdict(list)
    for row in scores.to_pylist():
        sessions[row["evaluation"]].append(row)
    result = {}
    for key, rows in score_slices(scores).items():
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
