# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist coordinator wall spans without confusing them with GPU work time."""

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import SkyRLRun
from rigging.event_log import Event, EventLog
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.archive_stage import archive_step
from experiments.post_training.long_loop.eval_runtime.audit_results import audit, require_evaluation_coverage
from experiments.post_training.long_loop.eval_runtime.profile import SKIPPED_BENCHMARKS, SMOKE_PATH_COUNT
from experiments.post_training.long_loop.finalization import terminal_outcome
from experiments.post_training.long_loop.timing import report


def prepare_reporting_inputs(
    *,
    output: str,
    prepared_inputs: str,
    evaluated: str,
    source_sha256: str,
    data_relative_path: str,
    policy_sha256: str,
) -> None:
    """Bind report coverage and native result paths to the exact canonical evaluation."""
    canonical = json.loads((StoragePath(evaluated) / "canonical-evaluation.json").read_bytes())
    if canonical["canonical_spec_sha256"] != policy_sha256:
        raise ValueError("Reporting policy differs from the completed canonical evaluation")
    raw = (StoragePath(prepared_inputs) / data_relative_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    rows = pq.read_table(pa.BufferReader(raw), columns=["task_rollout_spec"]).to_pylist()
    identities = [
        {
            "dataset_row_uid": str(index),
            "data_source": json.loads(row["task_rollout_spec"])["task"]["source"]["dataset"],
        }
        for index, row in enumerate(rows)
    ]
    run = SkyRLRun.raw_load(evaluated)
    if not run.attempt_id:
        raise ValueError("Canonical evaluation must identify its completed attempt")
    selection = {
        "attempts": len(identities),
        "attempts_per_data_source": dict(Counter(row["data_source"] for row in identities)),
        "identities": identities,
        "results_uri": prefix_join(evaluated, f"exports/dumped_evals/{run.attempt_id}/eval_only"),
        "source_sha256": source_sha256,
        "policy_sha256": policy_sha256,
    }
    root = StoragePath(output)
    (root / "selection.json").write_text(json.dumps(selection, indent=2))
    (root / "environment.json").write_text(json.dumps({"prepared_inputs": prepared_inputs, "sha256": source_sha256}))
    (root / "rollouts.parquet").write_bytes(raw)


def timed(step: ArtifactStep) -> ArtifactStep:
    run = step.run

    def execute(config):
        attempt = uuid4().hex
        fs, key = filesystem_for(prefix_join(step.path(), f"timing/{attempt}.jsonl"))
        with tempfile.TemporaryDirectory() as directory:
            log = EventLog(Path(directory) / "events.jsonl", step.version, step.name, attempt)
            try:
                with log.span("stage_wall_including_queue"):
                    fs.pipe_file(key, log.path.read_bytes())
                    return run(config)
            finally:
                fs.pipe_file(key, log.path.read_bytes())

    return replace(step, run=execute)


def write_report(
    *,
    output: str,
    stages: list[str],
    expected_rollouts: int | None,
    sft_geometry: str,
    evaluation_selection: str | None = None,
    additional_timing: list[dict] | None = None,
    accepted_unscored_attempts: list[dict] | None = None,
    rl_terminal: str | None = None,
    sft_selection: str | None = None,
) -> None:
    events = []
    sources = []
    for stage in stages:
        for directory, pattern in (("timing", "**/*.jsonl"), ("telemetry", "**/phases.jsonl")):
            fs, prefix = filesystem_for(prefix_join(stage, directory))
            for key in sorted(fs.glob(prefix_join(prefix, pattern))):
                events.extend(Event(**json.loads(line)) for line in fs.cat_file(key).decode().splitlines())
                sources.append(fs.unstrip_protocol(key))
    for item in additional_timing or []:
        if item["source"] in sources:
            continue
        source_fs, source_key = filesystem_for(item["source"])
        content = source_fs.cat_file(source_key)
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Pinned native timing journal changed: {item['source']}")
        events.extend(Event(**json.loads(line)) for line in content.decode().splitlines() if line.strip())
        sources.append(item["source"])
    # Reused stage artifacts have different original run IDs; preserve those in the sources.
    original_runs = sorted({event.run for event in events})
    text = report([replace(event, run=output) for event in events], synthetic=False)
    text += "\nOriginal stage run IDs: " + ", ".join(original_runs) + ". Raw events retain these IDs.\n"
    text += "\n" + sft_geometry + "\n"
    if sft_selection is not None:
        selected = json.loads((StoragePath(sft_selection) / "selected.json").read_text())
        text += f"\nSFT completed {selected['completed_updates']} updates; selected native step {selected['step']}.\n"
        if "training_completion" in selected:
            completion = selected["training_completion"]
            text += f"SFT stop reason: {completion['stop_reason']}; maximum {completion['maximum_updates']} updates.\n"
    outcome = None
    if rl_terminal is not None:
        outcome = terminal_outcome(rl_terminal)
        text += "\nRL terminal outcome (preserved independently of finalization): " + json.dumps(outcome) + ".\n"
    text += (
        "\nStage spans include scheduling and startup; they are not active compute measurements.\n"
        "Queue wait, allocation release, native learner phases, and per-benchmark timing require the\n"
        "Iris/SkyRL/SFT telemetry for this run. Missing measurements are not counted as zero.\n"
    )
    fs, root = filesystem_for(output)
    if outcome is not None:
        fs.pipe_file(prefix_join(root, "rl-terminal.json"), json.dumps(outcome, indent=2).encode())
    fs.pipe_file(prefix_join(root, "timing.md"), text.encode())
    fs.pipe_file(prefix_join(root, "timing-sources.json"), json.dumps(sources, indent=2).encode())
    expected_by_data_source = None
    expected_identities = None
    results_uri = None
    if evaluation_selection is not None:
        selection_fs, selection_key = filesystem_for(prefix_join(evaluation_selection, "selection.json"))
        selection = json.loads(selection_fs.cat_file(selection_key))
        if expected_rollouts is None:
            expected_rollouts = selection["attempts"]
        if selection["attempts"] != expected_rollouts:
            raise ValueError("Prepared evaluation attempt count differs from the resolved run spec")
        expected_by_data_source = selection["attempts_per_data_source"]
        expected_identities = selection["identities"]
        results_uri = selection.get("results_uri")
    if expected_rollouts is None:
        raise ValueError("Report requires a pinned evaluation cohort or an explicit attempt count")
    evaluation = audit(
        stages[-1],
        expected_rollouts=expected_rollouts,
        expected_by_data_source=expected_by_data_source,
        expected_identities=expected_identities,
        results_uri=results_uri,
    )
    evaluation["excluded_benchmarks"] = SKIPPED_BENCHMARKS
    evaluation["accepted_unscored_attempts"] = accepted_unscored_attempts or []
    require_evaluation_coverage(evaluation)
    fs.pipe_file(prefix_join(root, "evaluation.json"), json.dumps(evaluation, indent=2).encode())
    if not evaluation["all_verified"]:
        text += (
            "\nEvaluation limitations: the following attempts are explicitly unscored and excluded from reward means: "
        )
        text += json.dumps(evaluation["unscored_attempts"], sort_keys=True) + ". Full scoring is not certified.\n"
        fs.pipe_file(prefix_join(root, "timing.md"), text.encode())


def report_step(
    version: str,
    stages: list[ArtifactStep],
    archive_plan: dict | None = None,
    *,
    expected_rollouts: int | None = SMOKE_PATH_COUNT,
    evaluation_selection: ArtifactStep | None = None,
    additional_timing: list[dict] | None = None,
    accepted_unscored_attempts: list[dict] | None = None,
    sft_geometry: str = "Plumbing SFT (32k/32 GPU), not target: 15 updates, 125,829,120 packed token positions.",
    rl_terminal: ArtifactStep | None = None,
    sft_selection: ArtifactStep | None = None,
) -> ArtifactStep:
    completed = apply(
        "long-loop/report",
        write_report,
        version=version,
        output=OUT,
        stages=stages,
        expected_rollouts=expected_rollouts,
        sft_geometry=sft_geometry,
        evaluation_selection=evaluation_selection,
        additional_timing=additional_timing,
        accepted_unscored_attempts=accepted_unscored_attempts,
        rl_terminal=rl_terminal,
        sft_selection=sft_selection,
    )
    return archive_step(version, archive_plan, completed) if archive_plan is not None else completed
