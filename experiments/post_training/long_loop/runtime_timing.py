# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist coordinator wall spans without confusing them with GPU work time."""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.long_loop.archive_stage import archive_step
from experiments.post_training.long_loop.eval_runtime.audit_results import audit, require_evaluation_coverage
from experiments.post_training.long_loop.eval_runtime.profile import SKIPPED_BENCHMARKS, SMOKE_PATH_COUNT
from experiments.post_training.long_loop.finalization import terminal_outcome
from experiments.post_training.long_loop.timing import Event, EventLog, report


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
    expected_rollouts: int,
    sft_geometry: str,
    evaluation_selection: str | None = None,
    additional_timing: list[dict] | None = None,
    accepted_unscored_attempts: list[dict] | None = None,
    rl_terminal: str | None = None,
) -> None:
    events = []
    sources = []
    for stage in stages:
        fs, prefix = filesystem_for(prefix_join(stage, "timing"))
        for key in sorted(fs.glob(prefix_join(prefix, "**/*.jsonl"))):
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
    if rl_terminal is not None:
        outcome = terminal_outcome(rl_terminal)
        text += "\nRL terminal outcome (preserved independently of finalization): " + json.dumps(outcome) + ".\n"
    text += (
        "\nStage spans include scheduling and startup; they are not active compute measurements.\n"
        "Queue wait, allocation release, native learner phases, and per-benchmark timing require the\n"
        "Iris/SkyRL/SFT telemetry for this run. Missing measurements are not counted as zero.\n"
    )
    fs, root = filesystem_for(output)
    fs.pipe_file(prefix_join(root, "timing.md"), text.encode())
    fs.pipe_file(prefix_join(root, "timing-sources.json"), json.dumps(sources, indent=2).encode())
    expected_by_data_source = None
    expected_identities = None
    if evaluation_selection is not None:
        selection_fs, selection_key = filesystem_for(prefix_join(evaluation_selection, "selection.json"))
        selection = json.loads(selection_fs.cat_file(selection_key))
        if selection["attempts"] != expected_rollouts:
            raise ValueError("Prepared evaluation attempt count differs from the resolved run spec")
        expected_by_data_source = selection["attempts_per_data_source"]
        expected_identities = selection["identities"]
    evaluation = audit(
        stages[-1],
        expected_rollouts=expected_rollouts,
        expected_by_data_source=expected_by_data_source,
        expected_identities=expected_identities,
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
    expected_rollouts: int = SMOKE_PATH_COUNT,
    evaluation_selection: ArtifactStep | None = None,
    additional_timing: list[dict] | None = None,
    accepted_unscored_attempts: list[dict] | None = None,
    sft_geometry: str = "Plumbing SFT (32k/32 GPU), not target: 15 updates, 125,829,120 packed token positions.",
    rl_terminal: ArtifactStep | None = None,
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
    )
    return archive_step(version, archive_plan, completed) if archive_plan is not None else completed
