# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the original diversity post-SFT gates to completed durable training."""

import json

import click
import numpy as np
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_sft import (
    COLLECTION_VERSION,
    SFT_VERSION,
    durable_sft_stages,
)
from experiments.post_training.russell_rsi.launch_teacher_sft import SFT_LEARNING_RATE
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.teacher_diversity_study import (
    CONTEXT_TOKENS,
    NAMESPACE,
    PASSES,
    PROTOCOL,
    ROWS,
    UPDATES,
    diversity_workflow,
    validated_diversity_post_workflow,
)
from experiments.post_training.russell_rsi.teacher_four_pass import pinned_record

AMENDMENT_PROTOCOL = "teacher-diversity-durable-sft-binding-v1"
LAUNCH_PROTOCOL = "russell-rsi-foreground-launch-proof-v1"
PERMITTED_CHANGES = ["training_version", "trainer_id", "tracker", "reload_binding"]
METRIC_KEYS = {
    "loss": "train/loss",
    "learning_rate": "optim/learning_rate",
    "gradient_norm": "grad/norm/total",
    "update_norm": "updates/norm/total",
}


def completed_durable_producer(config: dict, role: str, expected: ArtifactStep, amendment: dict) -> dict:
    """Check actual completed metadata against its frozen launch and source review."""
    launch = PinnedFile(**config[f"{role}_launch_proof"]).read_json()
    review = PinnedFile(**config["sft_source_review"]).read_json()
    config_pin = {"uri": config["sft_config_uri"], "sha256": config["sft_config_sha256"]}
    binding = amendment[role]
    if (
        launch["protocol"] != LAUNCH_PROTOCOL
        or launch["source_head"] != amendment["source"]["head"]
        or launch["runtime_commit"] != MARIN_SKYRL.commit
        or launch["source_files"] != amendment["source"]["files"]
        or launch["config"] != config_pin
        or launch["source_review"] != config["sft_source_review"]
        or review["status"] != "approved"
        or review["source_head"] != launch["source_head"]
        or review["runtime_commit"] != MARIN_SKYRL.commit
        or launch["producer_identity"] != artifact_identity(expected)
        or launch["producer_identity"] != binding["identity"]
        or launch["output_path"] != binding["output_path"]
        or launch["rl_authorized"] is not False
        or launch["signal_gate_passed"] is not None
    ):
        raise ValueError("Durable SFT producer differs from its reviewed launch or source")
    request = PinnedFile(**launch["request"]).read_json()
    if (
        request["stage"] != role
        or request["version"] != SFT_VERSION
        or request["source_head"] != launch["source_head"]
        or request["runtime_commit"] != MARIN_SKYRL.commit
        or request["config_uri"] != config_pin["uri"]
        or request["config_sha256"] != config_pin["sha256"]
    ):
        raise ValueError("Durable SFT launch request changed its stage, config or source")
    preflight = PinnedFile(**launch["preflight"]).read_json()
    if (
        preflight["exit_code"] != 0
        or preflight["identity"]["source_head"] != launch["source_head"]
        or preflight["identity"]["request_sha256"] != launch["request"]["sha256"]
    ):
        raise ValueError("Durable SFT producer lacks its successful exact preflight")
    producer_pin = PinnedFile(**config[f"{role}_producer"])
    if producer_pin.uri != prefix_join(launch["output_path"], ".artifact.json"):
        raise ValueError("Durable SFT record is outside its producer output")
    record = producer_pin.read_json()
    identity = f"{record['name']}@{record['version']}:{record['fingerprint']}"
    if (
        identity != launch["producer_identity"]
        or record["version"] != SFT_VERSION
        or record["output_path"] != launch["output_path"]
        or canonical_json(record["config"]) != canonical_json(launch["bound_config"])
        or StatusFile(record["output_path"], worker_id="diversity-post-sft").status != STATUS_SUCCESS
        or len(record["provenance"]["base_commit"]) < 9
        or not launch["source_head"].startswith(record["provenance"]["base_commit"])
        or record["provenance"]["dirty"] is not False
    ):
        raise ValueError("Durable SFT requires its exact successful producer")
    return record


def qualified_optimizer_telemetry(qualification: dict, amendment: dict, trained: dict) -> None:
    """Bind the four qualified optimizer steps to complete durable event bytes."""
    telemetry = qualification["optimizer_telemetry"]
    trainer = trained["config"]["train_config"]["trainer"]
    if (
        telemetry["skip_bad_steps"] is not False
        or trained["config"]["train_config"]["optimizer"]["skip_bad_steps"] is not False
        or telemetry["crash_on_nan"] is not True
        or telemetry["crash_on_inf"] is not True
        or trainer["crash_on_nan"] is not True
        or trainer["crash_on_inf"] is not True
        or telemetry["learning_rate_dtype"] not in ("float32", "float64")
        or telemetry["learning_rate_dtype"] != trainer["mp"]["param_dtype"]["__dtype__"]
        or not telemetry["files"]
    ):
        raise ValueError("Durable optimizer evidence lacks the executed no-skip or numeric settings")
    seen = set()
    merged = {}
    for entry in telemetry["files"]:
        pin = PinnedFile(**entry)
        if pin.uri in seen:
            raise ValueError("Durable optimizer event is listed more than once")
        seen.add(pin.uri)
        event = pin.read_json()
        step = event["step"]
        if (
            event["tracker"] != "json_logger"
            or event["event"] != "log"
            or event["run_id"] != amendment["sft"]["run_id"]
            or type(step) is not int
            or step not in range(UPDATES)
            or pin.uri != prefix_join(amendment["sft"]["metric_destination"], f"step-{step}-{pin.sha256}.json")
        ):
            raise ValueError("Durable optimizer event differs from its run, step or content path")
        metrics = merged.setdefault(step, {})
        for key, value in event["metrics"].items():
            if key in metrics and metrics[key] != value:
                raise ValueError("Durable optimizer events contain conflicting values for one step")
            metrics[key] = value
    steps = qualification["optimizer_steps"]
    if sorted(merged) != list(range(UPDATES)) or [step["step"] for step in steps] != list(range(UPDATES)):
        raise ValueError("Durable optimizer events do not cover all four qualified steps")
    for step in steps:
        for field, key in METRIC_KEYS.items():
            expected = step[field]
            if field == "learning_rate":
                expected = float(np.asarray(expected, dtype=telemetry["learning_rate_dtype"]))
            if key not in merged[step["step"]] or merged[step["step"]][key] != expected:
                raise ValueError("Durable optimizer metrics differ from the qualified update")


def durable_diversity_post_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    """Apply the original study gates to completed training with durable optimizer evidence."""
    study = pinned_record(config, "sft_config")
    if study["version"] != COLLECTION_VERSION or study["collection_version"] != COLLECTION_VERSION:
        raise ValueError("Durable post-SFT must retain its exact version 15 collection inputs")
    stages = durable_sft_stages(diversity_workflow(study))
    amendment = PinnedFile(**config["sft_telemetry_amendment"]).read_json()
    expected_science = {
        "rows": ROWS,
        "passes": PASSES,
        "batch_size": ROWS,
        "updates": UPDATES,
        "context_tokens": CONTEXT_TOKENS,
        "learning_rate": SFT_LEARNING_RATE,
    }
    if (
        amendment["protocol"] != AMENDMENT_PROTOCOL
        or amendment["config"] != {"uri": config["sft_config_uri"], "sha256": config["sft_config_sha256"]}
        or amendment["source"]["runtime_commit"] != MARIN_SKYRL.commit
        or amendment["changes"] != PERMITTED_CHANGES
        or amendment["source_only_changes"] != ["exclude_stale_parent_weight_manifest"]
        or amendment["science"] != expected_science
        or amendment["collection_identity"] != artifact_identity(stages["collect"])
        or amendment["sft"]["version"] != SFT_VERSION
        or amendment["reload"]["version"] != SFT_VERSION
        or amendment["sft"]["output_path"] != config["sft_uri"]
        or amendment["sft"]["run_id"] != f"russell-rsi-{NAMESPACE}-sft-{SFT_VERSION}"
        or amendment["sft"]["metric_destination"] != prefix_join(config["sft_uri"], "optimizer-telemetry")
    ):
        raise ValueError("Durable SFT binding changed the collection, science or training producer")
    trained = completed_durable_producer(config, "sft", stages["train"], amendment)
    reload = completed_durable_producer(config, "reload", stages["reload"], amendment)
    if (
        trained["config"]["train_config"]["trainer"]["id"] != amendment["sft"]["run_id"]
        or trained["config"]["train_config"]["trainer"]["tracker"][0]["metric_destination"]
        != amendment["sft"]["metric_destination"]
        or reload["config"]["model"]["identity"] != artifact_identity(stages["train"])
        or reload["config"]["model"]["location"] != prefix_join(config["sft_uri"], "hf/step-3")
        or reload["config"]["evals"] != "mmlu-smoke"
        or reload["config"]["limit"] != 1
    ):
        raise ValueError("Durable SFT executed different telemetry or reload settings")
    qualification = pinned_record(config, "qualification")
    qualified_optimizer_telemetry(qualification, amendment, trained)
    reload_evidence = qualification["serving_reload"]
    if reload_evidence["evidence_uri"] not in reload["result"]["results_paths"]:
        raise ValueError("Diversity qualification cites a different serving reload")
    pinned_bytes(reload_evidence["evidence_uri"], reload_evidence["evidence_sha256"])
    return validated_diversity_post_workflow(config, stage, trained=stages["train"])


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@click.option("--stage", type=click.Choice(["calibrate", "rl", "evaluate"]), required=True)
@foreground_build_options
def main(
    config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str, stage: str
) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if resolve_version(PROTOCOL, None) != config["version"]:
        raise click.UsageError("Durable diversity post-SFT version differs from its frozen config")
    return [durable_diversity_post_workflow(config, "train" if stage == "rl" else stage)["terminal"]]


if __name__ == "__main__":
    main()
