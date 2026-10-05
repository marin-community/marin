# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume bounded Russell pilots with separately reviewed construction inputs."""

import hashlib
import json
import os
from dataclasses import asdict

import click
from fray.types import ResourceConfig
from iris.client.context_state import has_current_context
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.bootstrap_loop import MAX_GLM_RESPONSES, LoopState, restored_round
from experiments.post_training.russell_rsi.calibration_recovery import (
    calibration_recovery_step,
    grade_only_recovery_step,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingPanel, PanelItem
from experiments.post_training.russell_rsi.feedback import SKILL_DESCRIPTIONS, CodingSkill
from experiments.post_training.russell_rsi.launch import (
    MODEL,
    MODEL_REVISION,
    LoopPredecessor,
    ReviewedConstructionInputs,
    run_bootstrap_loop,
)
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV, IRIS_TASK_ID_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256


def canonical_labels(capabilities_bytes: bytes) -> tuple[str, ...]:
    """Parse a canonical capability file and return its unique labels."""
    record = json.loads(capabilities_bytes)
    if set(record) != {"skills"} or any(set(skill) != {"label", "description"} for skill in record["skills"]):
        raise ValueError("Capability feedback does not match the canonical schema")
    labels = tuple(skill["label"] for skill in record["skills"])
    if len(labels) != len(set(labels)):
        raise ValueError("Capability feedback contains duplicate labels")
    for skill in record["skills"]:
        if SKILL_DESCRIPTIONS[CodingSkill(skill["label"])] != skill["description"]:
            raise ValueError("Capability feedback changes a canonical description")
    return labels


def execute_loop(config: dict) -> None:
    """Resolve reviewed bank inputs and seal progress at each stage boundary."""
    reviewed_feedback_by_pilot = config["reviewed_feedback"]

    def adopted(value: dict, kind: type = Artifact) -> ArtifactStep:
        return ArtifactStep.adopt(
            value["name"], value["version"], value["uri"], kind=kind, config=value["identity_config"]
        )

    seed = adopted(config["seed_bank"])
    parent = adopted(config["parent"], LevanterCheckpoint)
    retention = adopted(config["retention"])
    panel_value = json.loads(pinned_bytes(config["panel_uri"], config["panel_sha256"]))
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])
    recovery = config.get("initial_calibration_recovery")
    initial_calibration = (
        calibration_recovery_step(
            recovery, config["version"], seed, parent, RuntimeBundle(**config["runtime_bundle"]), MODEL, MODEL_REVISION
        )
        if recovery is not None
        else None
    )

    def next_construction_inputs(
        feedback_identity: str, raw_bytes: bytes, state: LoopState, response_cap: int
    ) -> ReviewedConstructionInputs | None:
        supplied = reviewed_feedback_by_pilot.get(str(state.completed_pilots))
        if supplied is None:
            return None
        prior = compact_json_sha256({"tasks": [asdict(task) for task in state.bank]})
        if supplied["raw_feedback_identity"] != feedback_identity:
            raise ValueError("Reviewed feedback does not identify this raw coding feedback")
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if supplied["raw_capabilities_sha256"] != raw_sha256:
            raise ValueError("Reviewed feedback does not identify the raw capability bytes")
        reviewed = supplied["reviewed_feedback"]
        review_record = json.loads(pinned_bytes(reviewed["review_record_uri"], reviewed["review_record_sha256"]))
        reviewed_bytes = pinned_bytes(prefix_join(reviewed["uri"], "capabilities.json"), reviewed["capabilities_sha256"])
        raw_labels = canonical_labels(raw_bytes)
        reviewed_labels = canonical_labels(reviewed_bytes)
        if any(label not in raw_labels for label in reviewed_labels):
            raise ValueError("Reviewed feedback adds a label absent from raw feedback")
        if (
            review_record["source_capabilities_sha256"] != raw_sha256
            or review_record["reviewed_capabilities_sha256"] != reviewed["capabilities_sha256"]
        ):
            raise ValueError("Review record does not identify the raw and reviewed capability bytes")
        reviewed_feedback = ArtifactStep.adopt(
            reviewed["name"],
            reviewed["version"],
            reviewed["uri"],
            config={
                **reviewed["identity_config"],
                "raw_feedback_identity": feedback_identity,
                "raw_capabilities_sha256": raw_sha256,
                "reviewed_capabilities_sha256": reviewed["capabilities_sha256"],
                "review_record_uri": reviewed["review_record_uri"],
                "review_record_sha256": reviewed["review_record_sha256"],
            },
        )
        bank_value = supplied.get("bank")
        if bank_value is None:
            return ReviewedConstructionInputs(reviewed_feedback, reviewed_bytes, None)
        if supplied["prior_bank_sha256"] != prior or supplied["response_cap"] != response_cap:
            raise ValueError("Reviewed construction does not identify this prior bank and response budget")
        bank_record = json.loads(pinned_bytes(prefix_join(bank_value["uri"], "bank.json"), bank_value["bank_sha256"]))
        if bank_record["feedback_identity"] != reviewed["capabilities_sha256"]:
            raise ValueError("Reviewed construction bank does not cite the reviewed feedback")
        bank = ArtifactStep.adopt(
            bank_value["name"],
            bank_value["version"],
            bank_value["uri"],
            config={
                **bank_value["identity_config"],
                "bank_sha256": bank_value["bank_sha256"],
                "prior_bank_sha256": prior,
                "feedback_identity": artifact_identity(reviewed_feedback),
                "capabilities_sha256": reviewed["capabilities_sha256"],
            },
        )
        return ReviewedConstructionInputs(reviewed_feedback, reviewed_bytes, bank)

    predecessor_value = config.get("predecessor")
    predecessor = None
    if predecessor_value is not None:
        predecessor = LoopPredecessor(
            restored_round(
                json.loads(pinned_bytes(predecessor_value["round_uri"], predecessor_value["round_file_sha256"]))
            ),
            pinned_bytes(predecessor_value["raw_capabilities_uri"], predecessor_value["raw_capabilities_sha256"]),
            predecessor_value["round_file_sha256"],
        )
    continuation_calibration = None
    recovery = config.get("continuation_calibration_recovery")
    if recovery is not None:
        if predecessor is None:
            raise ValueError("Grade-only continuation recovery requires a predecessor")
        construction = next_construction_inputs(
            predecessor.round.result.feedback_identity,
            predecessor.raw_capabilities,
            predecessor.round.state,
            MAX_GLM_RESPONSES,
        )
        if construction is None or construction.bank is None:
            raise ValueError("Grade-only continuation recovery requires a reviewed successor bank")
        continuation_calibration = grade_only_recovery_step(
            recovery, config["version"], construction.bank, parent, RuntimeBundle(**config["runtime_bundle"])
        )
    run_bootstrap_loop(
        seed,
        parent,
        retention,
        panel,
        config["heldout_manifest_uri"],
        config["heldout_manifest_sha256"],
        config["parent_coding_evidence_uri"],
        config["parent_coding_evidence_sha256"],
        config["parent_retention_evidence_uri"],
        config["parent_retention_evidence_sha256"],
        config["version"],
        RuntimeBundle(**config["runtime_bundle"]),
        config["machine_config"],
        config["relay_job"],
        StoragePath(config["manifest_prefix"]),
        next_construction_inputs,
        initial_calibration=initial_calibration,
        predecessor=predecessor,
        continuation_calibration=continuation_calibration,
    )


def run_loop(config: dict) -> None:
    remote(
        execute_loop,
        resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@build_options
def main(config_uri: str, config_sha256: str) -> ArtifactStep[Artifact]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if click.get_current_context().params.get("do_run") and not (
        has_current_context() or os.environ.get(IRIS_TASK_ID_ENV)
    ):
        raise click.UsageError("Run this CPU coordinator inside the CW02 Iris context")
    version = resolve_version("russell-rsi-bootstrap-loop", None)
    if config["version"] != version:
        raise click.UsageError("The pinned loop config and artifact version differ")
    if not config["manifest_prefix"].startswith(("gs://", "s3://")):
        raise click.UsageError("The loop requires an explicit regional object-storage manifest prefix")
    return ArtifactStep(
        # The cache uses name and version. Each reviewed config must execute the coordinator.
        name=f"documents/russell-rsi-bootstrap-loop-{config_sha256}",
        version=version,
        artifact_type=Artifact,
        build_config=lambda ctx: config,
        run=run_loop,
    )


if __name__ == "__main__":
    main()
