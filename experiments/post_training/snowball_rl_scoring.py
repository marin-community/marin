# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import NamedTuple

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.execution.artifact import read_record
from marin.rl.rl_progress import Invariant
from marin.rl.skyrl import read_skyrl_launch_receipt
from marin.rl.skyrl_records import attempt_tracking, read_skyrl_launch_config, read_skyrl_metrics
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_result import Outcome

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


GATE_INPUTS_FILENAME = "gate_inputs.json"
GRADER_ERROR_LIMIT = 0.01


class GateInputs(BaseModel, frozen=True):
    preset_name: str
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: str | None = None
    requirements: tuple[str, ...] = ()


def read_gate_inputs(run_output: str) -> GateInputs:
    """Read the typed Snowball inputs saved beside a native launch receipt."""
    return GateInputs.model_validate_json((StoragePath(run_output) / GATE_INPUTS_FILENAME).read_text())


@dataclass(frozen=True)
class Requirement:
    key: str
    higher_is_better: bool
    bound: float | None


def parse_requirement(text: str) -> Requirement:
    match = re.fullmatch(r"([^<>=\s]+)(<=|>=)(control|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if match is None:
        raise click.BadParameter(f"expected KEY<=NUMBER, KEY>=NUMBER, KEY<=control or KEY>=control: {text}")
    key, operator, value = match.groups()
    bound = None if value == "control" else float(value)
    if bound is not None and not math.isfinite(bound):
        raise click.BadParameter(f"requirement bound must be finite: {text}")
    return Requirement(key, operator == ">=", bound)


def _mismatch_counts(rows: Sequence[Mapping[str, float]], prefix: str) -> list[float] | None:
    counts = [row.get(prefix + "/selected_tokens", math.nan) for row in rows]
    if all(math.isfinite(count) and count >= 0 for count in counts) and math.fsum(counts) > 0:
        return counts
    return None


def _window_values(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    rows = [metrics.get(step, {}) for step in preset.measured]
    phases = sorted(
        key
        for key in set().union(*(row.keys() for row in rows))
        if key.startswith("timing/step_wall/") and key.rsplit("/", 1)[-1] not in ("evaluation", "checkpoint_work")
    )
    values = {}
    declared = {metric.key: metric.window for metric in (*preset.guards, *preset.objectives)}
    extra = (
        set().union(*(row.keys() for row in metrics.values()))
        - declared.keys()
        - {"step", "attempt_id", "metric_attempt_ids"}
    )
    for key in (*declared, *sorted(extra)):
        window = declared.get(key, presets.WindowStatistic.MEAN)
        if key.startswith("startup/"):
            first = next((row[key] for _, row in sorted(metrics.items()) if key in row), None)
            if first is not None:
                values[key] = float(first)
            continue
        if window == presets.WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN:
            prefix = "/".join(key.split("/")[:3])
            counts = _mismatch_counts(rows, prefix)
            if counts is None:
                continue
            total = math.fsum(counts)
            nonempty = [(row, count) for row, count in zip(rows, counts, strict=True) if count > 0]
            if all(
                math.isfinite(row.get(key, math.nan)) and row.get(prefix + "/finite_fraction") == 1
                for row, _ in nonempty
            ):
                values[key] = math.fsum(count / total * row[key] for row, count in nonempty)
            continue
        if key == presets.UPDATE_SECONDS_METRIC:
            series = [math.fsum(row[key] for key in phases) for row in rows if phases and set(phases) <= set(row)]
        else:
            series = [row[key] for row in rows if key in row]
        if series and len(series) == len(rows):
            reducer = statistics.median if window == presets.WindowStatistic.MEDIAN else statistics.mean
            values[key] = float(reducer(series))
    return values


def _grader_errors(
    rows: Sequence[Mapping[str, float]], prefix: str, name: str, *, answers: int | None = None
) -> Invariant:
    recorded = all(prefix + outcome in row for row in rows for outcome in (*Outcome, "rollout_failure"))

    def count(*outcomes: str) -> float:
        return math.fsum(row.get(prefix + outcome, 0) for row in rows for outcome in outcomes)

    total = count(*Outcome)
    bad = count(Outcome.INFRA_ERROR, Outcome.UNAVAILABLE, Outcome.INVALID_TASK)
    failures = count("rollout_failure")
    upper = min(total, bad + failures) / total if total else 1.0
    if answers is not None and (not recorded or total != answers or failures > total):
        upper = 1.0
    return Invariant(
        name=name,
        value=upper,
        limit=GRADER_ERROR_LIMIT,
        comparator="<=",
        holds=upper <= GRADER_ERROR_LIMIT and recorded,
    )


def _session_invariants(
    name: str,
    table: pa.Table,
    evaluation: presets.Evaluation,
    preset: presets.Preset,
    expected: int,
    task_ids: Sequence[str],
) -> list[Invariant]:
    rows = table.to_pylist()
    counts = Counter(row["item"] for row in rows)
    holds = len(rows) == expected and set(counts.values()) == {evaluation.answers} and set(counts) == set(task_ids)
    result = [Invariant(name=f"{name}/batch_finished", value=len(rows), limit=expected, comparator="==", holds=holds)]
    if any(metric.key.startswith(evaluation.name + "/") for metric in preset.guards):
        coverage = len({row["item"] for row in rows if row["scored"]}) / (expected // evaluation.answers)
        result.append(
            Invariant(
                name=f"{name}/panel_scored_fraction",
                value=coverage,
                limit=presets.MINIMUM_PANEL_COVERAGE,
                comparator=">=",
                holds=coverage >= presets.MINIMUM_PANEL_COVERAGE,
            )
        )
    return result


@dataclass(frozen=True)
class RunData:
    config: dict
    path: str
    version: str
    panel: str | None
    provenance: dict
    values: dict[str, float]
    panel_ids: dict[str, frozenset[str]]
    invariants: tuple[Invariant, ...]
    missing_tokens: bool
    slices: dict[str, int]
    evaluation_identity: dict[str, tuple[str | None, dict]]
    h100_hours: float | None
    engine: dict
    attempt_id: str


def _recorded_panel(launch: dict, evaluation: presets.Evaluation) -> str:
    try:
        identities = {"@".join(recorded_identity(source)) for source in recorded_sources(launch, evaluation.name)}
        if len(identities) != 1:
            raise ValueError("expected one panel identity")
        identity = identities.pop()
        return identity
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise InvalidComparison(f"recorded panel version is missing or malformed: {error}") from error


def _panel_identity(preset: presets.Preset, launch: dict, inputs: GateInputs) -> str | None:
    evaluation = presets.panel_evaluation(preset)
    if evaluation is None:
        return None
    default = presets.default_selection(preset)
    selection = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
    selected = evaluation.tasks(selection, default, preset.eval_lite)
    panel = f"{selected.name.removeprefix(presets.SELECTION_PREFIX + '/')}@{selected.version}"
    binding = selected
    if evaluation.sample is not None:
        binding = presets.sampled_tasks_step(
            selected,
            evaluation.file,
            evaluation.sample,
            launch["skyrl"]["trainer"]["seed"],
            revision=preset.revision,
        )
    expected = f"{binding.name}@{binding.version}"
    sources = recorded_sources(launch, evaluation.name)
    recorded = _recorded_panel(launch, evaluation)
    if recorded != expected:
        raise InvalidComparison(
            f"recorded panel differs from preset: expected {panel} ({expected}); recorded {recorded}"
        )
    if evaluation.held_out and any(source["relative_path"] != f"{evaluation.file}.parquet" for source in sources):
        raise InvalidComparison(f"recorded panel file differs from preset {panel}: expected {evaluation.file}.parquet")
    return panel


def training_invariants(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> tuple[Invariant, ...]:
    """Check planned update completion and finite training loss and gradient evidence."""
    return preset.recipe.layout.workload.invariants(preset.updates, metrics)


def load_run(
    path: str,
    preset: presets.Preset,
    inputs: GateInputs | None = None,
) -> RunData:
    inputs = inputs or GateInputs(preset_name=preset.name)
    terminal = read_skyrl_launch_receipt(path)
    if terminal is None:
        raise FileNotFoundError(f"no launch receipt at {path}")
    if terminal.native.result.state != "succeeded":
        raise InvalidComparison(f"native job {terminal.native.result.iris_job_id} is {terminal.native.result.state}")
    if terminal.native.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    metrics = read_skyrl_metrics(path)
    workload = preset.recipe.layout.workload
    planned = workload.planned_steps(preset.updates)
    training = [row for step, row in sorted(metrics.items()) if step in planned]
    invariants = list(training_invariants(preset, metrics))
    answers = evaluation_answers(path) if preset.evaluations else []
    launch = terminal.native.config
    tracking = attempt_tracking(terminal)
    try:
        recorded_panels = {
            evaluation.name: read_recorded_panel(launch, evaluation.name) for evaluation in preset.evaluations
        }
        identities = {
            evaluation.name: (
                (recorded_panels[evaluation.name].task_set_sha256 if evaluation.held_out else None),
                evaluation_protocol(launch, evaluation.name),
            )
            for evaluation in preset.evaluations
        }
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidComparison(f"recorded evaluation identity is malformed: {error}") from error
    panel = _panel_identity(preset, launch, inputs)
    scores, initial_scores, panel_ids = [EVAL_SCORE_SCHEMA.empty_table()], [], {}
    default = presets.default_selection(preset)
    selection = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
    for evaluation in preset.evaluations:
        quality = evaluation.held_out and any(metric.key.startswith(evaluation.name + "/") for metric in preset.guards)
        expected = presets.evaluation_answer_count(evaluation, evaluation.tasks(selection, default, preset.eval_lite))
        table = panel_scores(evaluation, path, answers)
        initial = table.filter(pc.equal(table["step"], 0))
        table = table.filter(pc.equal(table["step"], workload.scored_step(preset.updates)))
        scores.append(table)
        initial_scores.append(initial)
        ids = frozenset(table["item"].to_pylist())
        if evaluation.held_out:
            panel_ids[evaluation.name] = ids
        sessions = [(evaluation.name, table, workload.metric_step(preset.updates))]
        if evaluation.at_start:
            if initial.num_rows and frozenset(initial["item"].to_pylist()) != ids:
                raise InvalidComparison(f"initial and final panel task IDs differ: {path}")
            sessions.append((evaluation.name + "/initial", initial, 0))
        prefix = native_prefix(launch, evaluation.name) + "all/outcome/"
        for session_name, session, metric_step in sessions:
            invariants.extend(
                _session_invariants(
                    session_name, session, evaluation, preset, expected, recorded_panels[evaluation.name].task_ids
                )
            )
            if quality:
                invariants.append(
                    _grader_errors(
                        [metrics.get(metric_step, {})],
                        prefix,
                        session_name + "/grader_error_fraction",
                        answers=session.num_rows,
                    )
                )
    prefix = workload.outcome_prefix
    invariants.append(_grader_errors(training, prefix, "grader_error_fraction"))
    fixed = preset.recipe.fixed_reply_tokens
    if fixed is not None:
        holds = workload.fixed_reply_holds(fixed, training, answers)
        invariants.append(Invariant(name="fixed_reply_tokens", value=holds, limit=True, comparator="==", holds=holds))
    missing = any(
        metric.key.startswith("policy/mismatch/")
        and _mismatch_counts([metrics.get(step, {}) for step in preset.measured], metric.key.rsplit("/", 1)[0]) is None
        for metric in preset.guards
    )
    final = pa.concat_tables(scores)
    values = {
        **_window_values(preset, metrics),
        **evaluation_statistics(final),
    }
    for key, value in evaluation_statistics(pa.concat_tables([scores[0], *initial_scores])).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    result = terminal.native.result
    h100_hours = None
    if result.started_at and result.finished_at:
        seconds = (presets.utc_millis(result.finished_at) - presets.utc_millis(result.started_at)) / 1000
        if seconds > 0:
            allocation = launch["iris"]["allocation"]
            h100_hours = seconds * allocation["num_nodes"] * allocation["gpus_per_node"] / 3600
    slices = {key: len({row["item"] for row in rows}) for key, rows in score_slices(final).items()}
    record = read_record(path)
    if record is None:
        raise FileNotFoundError(f"no artifact record at {path}")
    return RunData(
        config=launch,
        path=path,
        version=record.version,
        panel=panel,
        provenance=terminal.native.result.provenance,
        values=values,
        panel_ids=panel_ids,
        invariants=tuple(invariants),
        missing_tokens=missing,
        slices=slices,
        evaluation_identity=identities,
        h100_hours=h100_hours,
        engine=tracking.get("engine", {}),
        attempt_id=terminal.attempt_id,
    )


def with_objectives(candidate: RunData, preset: presets.Preset) -> RunData:
    values = dict(candidate.values)
    for metric in preset.objectives:
        if metric.key.endswith("/gain_per_h100_hour"):
            gain = values.get(metric.key.removesuffix("_per_h100_hour"))
            if gain is not None and candidate.h100_hours:
                values[metric.key] = gain / candidate.h100_hours
    return replace(candidate, values=values)


def with_requirements(candidate: RunData, requirements: Sequence[str]) -> RunData:
    invariants = list(candidate.invariants)
    values = candidate.values
    for text in requirements:
        requirement = parse_requirement(text)
        value = values.get(requirement.key)
        if value is None or not math.isfinite(value):
            raise InvalidComparison(f"missing or non-finite requirement metric: {requirement.key}")
        if requirement.bound is not None:
            holds = value >= requirement.bound if requirement.higher_is_better else value <= requirement.bound
            invariants.append(
                Invariant(
                    name=f"require/{requirement.key}",
                    value=value,
                    limit=requirement.bound,
                    comparator=">=" if requirement.higher_is_better else "<=",
                    holds=holds,
                )
            )
    return replace(candidate, invariants=tuple(invariants))
