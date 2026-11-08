# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist coordinator wall spans without confusing them with GPU work time."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.long_loop.eval_runtime.audit_results import audit
from experiments.post_training.long_loop.eval_runtime.profile import SKIPPED_BENCHMARKS, SMOKE_PATH_COUNT
from experiments.post_training.long_loop.timing import Event, EventLog, report


def timed(step: ArtifactStep) -> ArtifactStep:
    run = step.run

    def execute(config):
        attempt = uuid4().hex
        fs, key = filesystem_for(step.path() + f"/timing/{attempt}.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            log = EventLog(Path(directory) / "events.jsonl", step.version, step.name, attempt)
            try:
                with log.span("stage_wall_including_queue"):
                    fs.pipe_file(key, log.path.read_bytes())
                    return run(config)
            finally:
                fs.pipe_file(key, log.path.read_bytes())

    return replace(step, run=execute)


def write_report(*, output: str, stages: list[str]) -> None:
    events = []
    sources = []
    for stage in stages:
        fs, prefix = filesystem_for(stage + "/timing")
        for key in sorted(fs.glob(prefix + "/*.jsonl")):
            events.extend(Event(**json.loads(line)) for line in fs.cat_file(key).decode().splitlines())
            sources.append(fs.unstrip_protocol(key))
    text = report(events, synthetic=False)
    text += (
        "\nStage spans include scheduling and startup; they are not active compute measurements.\n"
        "Queue wait, allocation release, native learner phases, and per-benchmark timing require the\n"
        "Iris/SkyRL/SFT telemetry for this run. Missing measurements are not counted as zero.\n"
    )
    fs, root = filesystem_for(output)
    fs.pipe_file(root + "/timing.md", text.encode())
    fs.pipe_file(root + "/timing-sources.json", json.dumps(sources, indent=2).encode())
    evaluation = audit(stages[-1], expected_rollouts=SMOKE_PATH_COUNT)
    evaluation["excluded_benchmarks"] = SKIPPED_BENCHMARKS
    fs.pipe_file(root + "/evaluation.json", json.dumps(evaluation, indent=2).encode())
    if not evaluation["all_verified"]:
        raise ValueError(f"Micro210 reduced smoke did not verify all {SMOKE_PATH_COUNT} selected paths")


def report_step(version: str, stages: list[ArtifactStep]) -> ArtifactStep:
    return apply("long-loop/report", write_report, version=version, output=OUT, stages=stages)
