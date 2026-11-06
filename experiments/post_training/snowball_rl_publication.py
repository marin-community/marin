# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain and publish Snowball panel results for one recorded attempt."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import SkyRLRun, read_skyrl_launch_config
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.inventory.loop_publication import LoopPublication, summary_publication
from experiments.post_training.inventory.loop_result import RESULT_FILENAME
from experiments.post_training.inventory.publish_loop_result import RunSummary

if TYPE_CHECKING:
    from experiments.post_training.snowball_rl_gate import GateReport


def _snowball_record(path: str, native: tuple[str, str]) -> dict:
    source = StoragePath(path) / "exports/attempts" / native[1] / RESULT_FILENAME
    record = json.loads(presets.read_text(source))
    if (record["run_id"], record["attempt_id"]) != native:
        raise ValueError(f"Snowball attempt record identity differs from its native run: {source}")
    return record


def _phase_result(
    scores: pa.Table,
    evaluation: presets.Evaluation,
    step: int,
    metrics: dict[int, dict],
    evidence: list[str],
) -> dict | None:
    table = scores.filter(pc.equal(scores["step"], step))
    rows = table.to_pylist()
    counts = Counter(row["item"] for row in rows)
    if any(count != 1 for count in counts.values()):
        evidence.append(f"Evaluation step {step} contains duplicate task IDs.")
        return None
    attempted, scored = len(rows), sum(row["scored"] for row in rows)
    observation = metrics.get(step, {})
    if observation.get("eval/all/num_attempted") != attempted or observation.get("eval/all/num_scored") != scored:
        evidence.append(
            f"Evaluation step {step} has no matching current-attempt counters; "
            f"dump attempted={attempted}, scored={scored}."
        )
        return None
    reward = scoring.evaluation_statistics(table).get(evaluation.name + "/mean")
    if reward is None:
        evidence.append(f"Evaluation step {step} has no scored reward; attempted={attempted}, scored={scored}.")
        return None
    return {"reward": reward, "attempted": attempted, "scored": scored}


def _retain_snowball_summary(training_path: str, training_version: str, gate_path: str | None, output: str) -> Artifact:
    launch = read_skyrl_launch_config(training_path)
    native = tuple(launch["run"][key] for key in ("id", "attempt_id"))
    metadata = _snowball_record(training_path, native)
    gate_record = presets.artifact_record(gate_path) if gate_path is not None else None
    gate_identity = (gate_record.name, gate_record.version, gate_record.fingerprint) if gate_record is not None else None
    report = gate_record.result if gate_record is not None else None
    preset = presets.PRESETS[metadata["label"]]
    if not presets.published(preset):
        raise ValueError("Snowball publication requires a trained panel run")
    evaluation = next(item for item in preset.evaluations if item.harness == presets.EvalHarness.PANEL)
    default = presets.default_selection()
    expected_tasks = (
        presets.evaluation_answer_count(evaluation, evaluation.tasks(default, default)) // evaluation.answers
    )
    metrics = {}
    for path in scoring.paths(StoragePath(training_path) / "exports/metrics/step-*.json"):
        row = json.loads(presets.read_text(path))
        if row.get("attempt_id") == native[1]:
            metrics[row["step"]] = row
    evidence = [
        *metadata["evidence"],
        f"Gate name/version/fingerprint={canonical_json(gate_identity)}.",
        "Evaluation input identities are declarations; "
        "runtime task-content and sampling-protocol hashes are unknown/unranked.",
    ]
    answers = scoring.evaluation_answers(training_path)
    phases = []
    for step in (0, launch["skyrl"]["trainer"]["max_steps"]):
        try:
            scores = scoring.panel_scores(evaluation, training_path, [answer for answer in answers if answer[0] == step])
            phases.append(_phase_result(scores, evaluation, step, metrics, evidence))
        except scoring.InvalidComparison as error:
            phases.append(None)
            evidence.append(f"Evaluation step {step}: {error}")
    baseline, final = phases
    summary = {
        **metadata,
        "attempt_id": (
            native[1] + "-gate-" + fingerprint_hash(canonical_json(gate_identity)) if gate_identity else native[1]
        ),
        "verdict": report["verdict"] if report is not None else None,
        "min_scored_fraction": presets.MINIMUM_PANEL_COVERAGE if report is not None else 1.0,
        "evaluation": {
            "name": evaluation.name,
            "version": "; ".join(
                source["identity"].partition(":")[0] for source in scoring.recorded_sources(launch, evaluation.name)
            ),
            "task_count": expected_tasks,
            "task_set_sha256": None,
            "protocol_sha256": None,
        },
        "baseline": baseline,
        "final": final,
        "baseline_source": prefix_join(training_path, "exports/dumped_evals/global_step_0_evals"),
        "artifact_uri": gate_path or str(StoragePath(training_path) / "exports/attempts" / native[1] / RESULT_FILENAME),
        "campaign": training_version,
        "series": preset.name,
        "evidence": evidence,
    }
    RunSummary.model_validate(summary)
    destination = StoragePath(output) / RESULT_FILENAME
    destination.parent.mkdirs()
    destination.write_text(json.dumps(summary, allow_nan=False, indent=2))
    return Artifact(path=output)


def publication_step(
    training: ArtifactStep[SkyRLRun] | ArtifactStep[Artifact], *, gate: ArtifactStep[GateReport] | None = None
) -> ArtifactStep[LoopPublication]:
    """Publish the recorded attempt using its saved gate or train-long result."""
    name = gate.name if gate is not None else training.name
    source = apply(
        f"{name}/result/{training.name}",
        _retain_snowball_summary,
        version=gate.version if gate is not None else training.version,
        training_path=training.path(),
        training_version=training.version,
        gate_path=gate,
        output=OUT,
    )
    return summary_publication(source, RESULT_FILENAME)
