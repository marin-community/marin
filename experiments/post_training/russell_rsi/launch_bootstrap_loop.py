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
from marin.execution.lazy import ArtifactStep, artifact_identity, resolve
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.bootstrap_loop import LoopState
from experiments.post_training.russell_rsi.calibration_recovery import calibration_recovery_step
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingPanel, PanelItem
from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION, run_bootstrap_loop
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV, IRIS_TASK_ID_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256


def execute_loop(config: dict) -> None:
    """Resolve reviewed bank inputs and seal progress at each stage boundary."""

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

    def next_bank(
        feedback: ArtifactStep[Artifact], state: LoopState, response_cap: int
    ) -> ArtifactStep[Artifact] | None:
        supplied = config["reviewed_banks"].get(str(state.completed_pilots))
        if supplied is None:
            return None
        prior = compact_json_sha256({"tasks": [asdict(task) for task in state.bank]})
        if supplied["prior_bank_sha256"] != prior or supplied["feedback_identity"] != artifact_identity(feedback):
            raise ValueError("Reviewed construction does not identify this prior bank and coding feedback")
        feedback_result = resolve(feedback)
        actual_feedback_sha256 = hashlib.sha256(
            StoragePath(prefix_join(feedback_result.path, "capabilities.json")).read_bytes()
        ).hexdigest()
        if supplied["capabilities_sha256"] != actual_feedback_sha256:
            raise ValueError("Reviewed construction changed its canonical feedback bytes")
        record = json.loads(pinned_bytes(prefix_join(supplied["uri"], "bank.json"), supplied["bank_sha256"]))
        if record["feedback_identity"] != supplied["capabilities_sha256"] or supplied["response_cap"] != response_cap:
            raise ValueError("Reviewed construction changed its canonical feedback or response budget")
        return ArtifactStep.adopt(
            supplied["name"],
            supplied["version"],
            supplied["uri"],
            config={
                **supplied["identity_config"],
                "bank_sha256": supplied["bank_sha256"],
                "prior_bank_sha256": prior,
                "feedback_identity": artifact_identity(feedback),
                "capabilities_sha256": actual_feedback_sha256,
            },
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
        next_bank,
        initial_calibration=initial_calibration,
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
