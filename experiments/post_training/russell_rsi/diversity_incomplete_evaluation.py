# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a qualified SFT checkpoint after an incomplete calibration."""

from dataclasses import replace

import click
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import StatusFile
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.russell_rsi.bootstrap_loop import (
    ATTEMPTS_PER_TASK,
    IncompleteCalibrationError,
    calibration_measurements,
)
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import post_sft_evaluation_stages, post_sft_stages
from experiments.post_training.russell_rsi.launch_teacher_diversity_post_sft import validated_durable_diversity_training
from experiments.post_training.russell_rsi.teacher_diversity_study import (
    ValidatedDiversityPostInputs,
    run_diversity_calibration,
    validated_diversity_post_inputs,
)

PROTOCOL = "teacher-diversity-incomplete-calibration-evaluation-v1"
MISSING_SLOTS = 3


def bound_terminal_record(pins: dict, expected: ArtifactStep, status: str) -> dict:
    producer = PinnedFile(**pins["producer"])
    record = producer.read_json()
    identity = f"{record['name']}@{record['version']}:{record['fingerprint']}"
    status_pin = PinnedFile(**pins["status"])
    status_pin.read_bytes()
    if (
        identity != artifact_identity(expected)
        or producer.uri != prefix_join(record["output_path"], ".artifact.json")
        or status_pin.uri != prefix_join(record["output_path"], ".executor_status")
        or StatusFile(record["output_path"], worker_id="incomplete-calibration").status != status
    ):
        raise ValueError("Incomplete calibration cites a different terminal producer")
    prefix = record["output_path"].removesuffix(f"/{expected.name}/{expected.version}")
    if expected.path(prefix) != record["output_path"]:
        raise ValueError("Terminal producer output differs from its artifact path")
    context = StepContext.for_run(record["output_path"], prefix, deps=expected.deps, runtime_args=expected.runtime_args)
    if canonical_json(record["config"]) != canonical_json(expected.build_config(context)):
        raise ValueError("Terminal calibration producer changed its bound scientific settings")
    return record


def require_incomplete_calibration(
    original: dict, original_pin: dict, amendment: dict, inputs: ValidatedDiversityPostInputs
) -> None:
    """Check the original partial grades and failed decision sealer."""
    if (
        amendment["protocol"] != PROTOCOL
        or amendment["original_config"] != original_pin
        or amendment["runtime_commit"] != MARIN_SKYRL.commit
        or amendment["model_identity"] != artifact_identity(inputs.model)
        or amendment["calibration_status"] != "incomplete_infrastructure"
        or amendment["signal_gate_passed"] is not None
        or amendment["rl_authorized"] is not False
        or amendment["repeated_issued_samples"] != 0
    ):
        raise ValueError("Incomplete calibration amendment changed its source or no-RL decision")
    stages = post_sft_stages(
        original,
        "calibrate",
        model=inputs.model,
        bank=inputs.bank,
        retention=inputs.retention,
        source_plan=inputs.source_plan,
        source=inputs.source,
        export_uri=inputs.export_uri,
        study=inputs.study,
        calibration_runner=run_diversity_calibration,
    )
    producer = bound_terminal_record(amendment["calibration"], stages["calibration"], "SUCCESS")
    expected_decision = stages["decision"]
    info_pin = PinnedFile(**amendment["failed_decision"]["executor_info"])
    decision = info_pin.read_json()
    decision_root = decision["output_path"]
    prefix = decision_root.removesuffix(f"/{expected_decision.name}/{expected_decision.version}")
    status_pin = PinnedFile(**amendment["failed_decision"]["status"])
    status_pin.read_bytes()
    if (
        info_pin.uri != prefix_join(decision_root, ".executor_info")
        or status_pin.uri != prefix_join(decision_root, ".executor_status")
        or StatusFile(decision_root, worker_id="incomplete-calibration").status != "FAILED"
        or decision["name"] != expected_decision.name
        or decision["config"]["fingerprint"] != expected_decision.fingerprint()
        or decision["config"]["version"] != expected_decision.version
        or decision_root != expected_decision.path(prefix)
        or decision["dependencies"]
        != [replace(dep.lower(), output_path_prefix=prefix).output_path for dep in expected_decision.deps]
        or decision["config"]["deps"] != [f"{dep.name}@{dep.version}" for dep in expected_decision.deps]
    ):
        raise ValueError("Failed decision is not the original terminal calibration sealer")
    foreground = PinnedFile(**amendment["foreground"]).read_json()
    if (
        foreground["exit_code"] != 1
        or foreground["error_type"] != "IncompleteCalibrationError"
        or foreground["original_config"] != original_pin
        or foreground["decision_identity"] != artifact_identity(stages["decision"])
        or foreground["decision_output_path"] != decision["output_path"]
    ):
        raise ValueError("Incomplete calibration lacks its actual failed foreground decision")
    summary_pin = PinnedFile(**amendment["calibration"]["summary"])
    if summary_pin.uri != prefix_join(producer["output_path"], "failure_summary.json"):
        raise ValueError("Partial summary is outside the completed calibration")
    summary = summary_pin.read_json()
    try:
        calibration_measurements(
            summary, inputs.source_plan.task_bank, artifact_identity(inputs.model), artifact_identity(inputs.bank)
        )
    except IncompleteCalibrationError as error:
        missing_task_ids = error.missing_task_ids
    else:
        raise ValueError("SFT-only amendment requires actual incomplete calibration")
    journal_pin = amendment["calibration"]["journal_binding"]
    if journal_pin["uri"] != prefix_join(producer["output_path"], "journal/binding.json"):
        raise ValueError("Calibration binding is outside its actual journal")
    journal = PinnedFile(**journal_pin).read_json()
    if canonical_json(journal["config"]) != canonical_json(producer["config"]):
        raise ValueError("Calibration journal differs from its actual producer")
    slots = amendment["missing_slots"]
    missing = {
        key: ATTEMPTS_PER_TASK - len(rewards)
        for key, rewards in summary["task_rewards"].items()
        if key in missing_task_ids
    }
    if len(slots) != MISSING_SLOTS:
        raise ValueError("Amendment must preserve the three original missing slots")
    tasks = {task.task_id: task for task in inputs.source_plan.task_bank}
    seen = set()
    counts = {}
    for slot in slots:
        key = f"{slot['task_id']}/{slot['sample']}"
        if key in seen:
            raise ValueError("Missing calibration slot is repeated")
        seen.add(key)
        pin = PinnedFile(**slot["result"])
        if pin.uri != prefix_join(producer["output_path"], f"journal/task/{key}/result.json"):
            raise ValueError("Missing grade is outside the original calibration journal")
        envelope = pin.read_json()
        binding = envelope["binding"]
        record = envelope["result"]["record"]
        task = tasks[slot["task_id"]]
        if (
            binding["key"] != key
            or binding["kind"] != "task"
            or binding["task_sha256"] != journal["attempts"]["task"].get(key)
            or canonical_json(binding["evaluation"]) != canonical_json(journal)
            or record["task_id"] != task.task_id
            or record["grade"]["status"] not in ("unavailable", "infra_error")
            or record["grade"]["reward"] is not None
        ):
            raise ValueError("Missing calibration slot differs from its frozen task or grade")
        counts[task.task_id] = counts.get(task.task_id, 0) + 1
    if counts != missing:
        raise ValueError("Amendment does not cover every missing calibration grade")


def incomplete_diversity_evaluation(config: dict) -> dict[str, ArtifactStep]:
    original_pin = config["original_config"]
    original = PinnedFile(**original_pin).read_json()
    trained = validated_durable_diversity_training(original)
    inputs = validated_diversity_post_inputs(original, trained=trained)
    amendment = PinnedFile(**config["amendment"]).read_json()
    require_incomplete_calibration(original, original_pin, amendment, inputs)
    if (
        amendment["evaluation"]
        != {
            "version": config["version"],
            "conditions": ["sft"],
            "coding_limit": 32,
            "retention_limit": 3,
        }
        or config["version"] == original["version"]
    ):
        raise ValueError("SFT-only evaluation must use its new frozen output version")
    return post_sft_evaluation_stages(
        {**original, "version": config["version"]},
        model=inputs.model,
        retention=inputs.retention,
        export_uri=inputs.export_uri,
        checkpoints=[("sft", inputs.model)],
        barriers=(),
        outputs={},
        study=inputs.study,
    )


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@foreground_build_options
def main(config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = PinnedFile(config_uri, config_sha256).read_json()
    if resolve_version(PROTOCOL, None) != config["version"]:
        raise click.UsageError("Incomplete-calibration evaluation version differs from its config")
    return [incomplete_diversity_evaluation(config)["terminal"]]


if __name__ == "__main__":
    main()
