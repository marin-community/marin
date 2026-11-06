# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a retained loop result independently of expensive training."""

import json
import math
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.rl.skyrl import IRIS_HUB_CLUSTER_CONFIG, SkyRLRun, read_skyrl_launch_config, read_skyrl_launch_receipt
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.inventory.loop_result import RESULT_FILENAME, LoopRunConfig, run_loop
from experiments.post_training.inventory.publish_loop_result import RunSummary, publish_result


@dataclass(frozen=True)
class PublicationConfig:
    summary_uri: str
    receipt_uri: str


class LoopPublication(Artifact):
    run_id: str
    attempt_id: str


def publish_summary(config: PublicationConfig) -> LoopPublication:
    """Read the retained summary and confirm one immutable dashboard attempt."""
    with StoragePath(config.summary_uri).open("rt") as stream:
        summary = json.load(stream)
    receipt = publish_result(summary)
    destination = StoragePath(config.receipt_uri)
    destination.parent.mkdirs()
    destination.write_text(json.dumps(receipt, allow_nan=False))
    return LoopPublication(path=config.receipt_uri, run_id=summary["run_id"], attempt_id=summary["attempt_id"])


def run_recorded_loop(config: LoopRunConfig) -> SkyRLRun:
    """Publish failed attempts too, while preserving the training failure."""
    try:
        return run_loop(config)
    except Exception as training_error:
        try:
            publish_summary(
                PublicationConfig(
                    summary_uri=prefix_join(config.skyrl.output.export_root, "loop-result.json"),
                    receipt_uri=prefix_join(config.skyrl.output.export_root, "loop-publication.json"),
                )
            )
        except Exception as publication_error:
            raise ExceptionGroup("Training and result publication failed", [training_error, publication_error]) from None
        raise


def _native_identity(path: str) -> tuple[str, str]:
    receipt = read_skyrl_launch_receipt(path)
    if receipt is None:
        raise ValueError(f"Publication requires a recorded native attempt: {path}")
    identity = tuple(receipt.config["run"][key] for key in ("id", "attempt_id"))
    if not all(isinstance(value, str) and value for value in identity):
        raise ValueError(f"Publication requires native run and attempt IDs: {path}")
    return identity


def _native_interval(job_id: str) -> tuple[str, str, float]:
    with ExitStack() as stack:
        info = get_job_info()
        client = stack.enter_context(
            IrisClient.in_cluster(info.controller_address)
            if info is not None and info.controller_address
            else open_iris_client(config_file=Path(IRIS_HUB_CLUSTER_CONFIG), workspace=presets.REPO_ROOT)
        )
        job = client.job_status(JobName.from_wire(job_id))
    if job.started_at is None or job.finished_at is None:
        raise ValueError(f"Native timing is unavailable for {job_id}; retry publication")
    start, finish = job.started_at.epoch_seconds(), job.finished_at.epoch_seconds()
    if finish < start:
        raise ValueError(f"Native completion precedes its start: {job_id}")
    return (
        datetime.fromtimestamp(start, UTC).isoformat(),
        datetime.fromtimestamp(finish, UTC).isoformat(),
        finish - start,
    )


def _phase_result(
    scores: pa.Table,
    evaluation: presets.Evaluation,
    step: int,
    task_ids: frozenset[str],
    metrics: dict[int, dict],
    evidence: list[str],
) -> dict | None:
    table = scores.filter(pc.equal(scores["step"], step))
    rows = table.to_pylist()
    counts = Counter(row["item"] for row in rows)
    if not counts.keys() <= task_ids or any(count != 1 for count in counts.values()):
        evidence.append(f"Evaluation step {step} contains duplicate or foreign task IDs.")
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


def _retain_snowball_summary(
    training_path: str,
    gate_path: str,
    native: tuple[str, str],
    gate_identity: tuple[str, str, str],
    output: str,
) -> Artifact:
    if _native_identity(training_path) != native:
        raise ValueError("Native attempt changed before summary creation; rebuild publication")
    gate_record = presets.artifact_record(gate_path)
    if (gate_record.name, gate_record.version, gate_record.fingerprint) != gate_identity:
        raise ValueError("Gate interpretation changed before summary creation; rebuild publication")
    report = gate_record.result
    preset = presets.PRESETS[report["preset"]]
    if preset.recipe.layout.role_plan.generate_only or preset.focus == presets.Focus.SPEED:
        raise ValueError("Snowball publication requires a trained panel run")
    receipt = read_skyrl_launch_receipt(training_path)
    launch = receipt.config
    if launch != read_skyrl_launch_config(training_path):
        raise ValueError("Native receipt and resolved configuration disagree")
    task_ids = scoring.evaluation_ids(launch["inputs"]["validation_data"])
    evaluation = next(item for item in preset.evaluations if item.harness == presets.EvalHarness.PANEL)
    if launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] != 1:
        raise ValueError("Snowball publication requires one answer per evaluation task")
    metrics = {}
    for path in scoring.paths(StoragePath(training_path) / "exports/metrics/step-*.json"):
        row = json.loads(presets.read_text(path))
        if row.get("attempt_id") == native[1]:
            metrics[row["step"]] = row
    completed = [
        row
        for step, row in metrics.items()
        if step > 0 and row.get("trainer/global_step") == step and "timing/step_wall/unaccounted" in row
    ]
    if not completed:
        raise ValueError("Current-attempt applied optimizer evidence is unavailable; retry publication")
    applied = 0
    for row in completed:
        count = row.get("policy/policy_update_steps")
        finite = [row.get(key) for key in ("policy/policy_loss", "policy/raw_grad_norm")]
        if (
            type(count) not in (int, float)
            or not math.isfinite(count)
            or count < 0
            or int(count) != count
            or not all(type(value) in (int, float) and math.isfinite(value) for value in finite)
        ):
            raise ValueError("Current-attempt applied optimizer evidence is incomplete")
        applied += int(count)
    evidence = [
        f"Native run_id={native[0]}, attempt_id={native[1]}, Iris job={receipt.result.iris_job_id}.",
        f"Gate name/version/fingerprint={canonical_json(gate_identity)}.",
        "Evaluation input identities are declarations; "
        "runtime task-content and sampling-protocol hashes are unknown/unranked.",
        "Optimizer steps count finite applied windows recorded by this native attempt.",
    ]
    try:
        scores = scoring.panel_scores(evaluation, training_path)
        baseline = _phase_result(scores, evaluation, 0, task_ids, metrics, evidence)
        final = _phase_result(scores, evaluation, launch["skyrl"]["trainer"]["max_steps"], task_ids, metrics, evidence)
    except scoring.InvalidComparison as error:
        baseline, final = None, None
        evidence.append(str(error))
    started, completed_at, elapsed = _native_interval(receipt.result.iris_job_id)
    provenance = receipt.result.provenance or {}
    record = presets.artifact_record(training_path)
    tracking = StoragePath(training_path) / "exports/tracking.json"
    summary = {
        "run_id": native[0],
        "attempt_id": native[1] + "-gate-" + fingerprint_hash(canonical_json(gate_identity)),
        "label": preset.name,
        "run_date": started,
        "date_basis": "launch_started",
        "started_at": started,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed,
        "timing_scope": "native_job_start_to_finish",
        "status": "succeeded" if receipt.result.state == "succeeded" else "failed",
        "verdict": report["verdict"],
        "min_scored_fraction": presets.MINIMUM_PANEL_COVERAGE,
        "optimizer_steps": applied,
        "source_revision": f"base={provenance.get('base_commit')} tree={provenance.get('tree_hash')}",
        "data_identity": "; ".join(source["identity"] for source in launch["inputs"]["train_data"]),
        "initializer_identity": launch["inputs"]["model"]["identity"],
        "evaluation": {
            "name": evaluation.name,
            "version": "; ".join(source["identity"].partition(":")[0] for source in launch["inputs"]["validation_data"]),
            "task_count": len(task_ids),
            "task_set_sha256": None,
            "protocol_sha256": None,
        },
        "baseline": baseline,
        "final": final,
        "baseline_source": prefix_join(training_path, "exports/dumped_evals/global_step_0_evals"),
        "artifact_uri": gate_path,
        "wandb_url": json.loads(presets.read_text(tracking))["wandb_url"] if presets.exists(tracking) else None,
        "campaign": record.version,
        "series": preset.name,
        "config_fingerprint": record.fingerprint,
        "layout": preset.recipe.layout.name,
        "iris_job_ids": [receipt.result.iris_job_id],
        "evidence": evidence,
    }
    RunSummary.model_validate(summary)
    destination = StoragePath(output) / RESULT_FILENAME
    destination.parent.mkdirs()
    destination.write_text(json.dumps(summary, allow_nan=False, indent=2))
    return Artifact(path=output)


def publication_step(
    training: ArtifactStep[SkyRLRun], *, gate: ArtifactStep | None = None
) -> ArtifactStep[LoopPublication]:
    """Publish cached training, or a completed Snowball gate for its recorded native attempt."""
    if gate is not None:
        native = _native_identity(training.path())
        record = presets.artifact_record(gate.path())
        identity = record.name, record.version, record.fingerprint
        summary = apply(
            f"{gate.name}/result-{fingerprint_hash(canonical_json((native, identity)))}",
            _retain_snowball_summary,
            version=gate.version,
            training_path=training,
            gate_path=gate,
            native=native,
            gate_identity=identity,
            output=OUT,
        )

        def summary_config(ctx: StepContext) -> PublicationConfig:
            return PublicationConfig(
                summary_uri=prefix_join(ctx.artifact_path(summary), RESULT_FILENAME),
                receipt_uri=prefix_join(ctx.output_path, "publication.json"),
            )

        return ArtifactStep(
            name=f"{summary.name}-report",
            version=summary.version,
            artifact_type=LoopPublication,
            run=publish_summary,
            build_config=summary_config,
            deps=(summary,),
        )

    def build_config(ctx: StepContext) -> PublicationConfig:
        return PublicationConfig(
            summary_uri=prefix_join(ctx.artifact_path(training), "exports/loop-result.json"),
            receipt_uri=prefix_join(ctx.output_path, "publication.json"),
        )

    return ArtifactStep(
        name=f"{training.name}-report",
        version=training.version,
        artifact_type=LoopPublication,
        run=publish_summary,
        build_config=build_config,
        deps=(training,),
    )
