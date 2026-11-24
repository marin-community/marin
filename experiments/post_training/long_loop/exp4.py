# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Exp4 from a qualified V7 artifact, the Exp4 curriculum and frozen Micro210."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import click
import draccus
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.cli import rl_build_options
from marin.rl.curriculum import RlCurriculum
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.evaluation.micro210_policy import Micro210Model, micro210_evaluation
from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.execution_policy import RL_TRAINING_BUDGET
from experiments.post_training.long_loop.exp3_production import (
    GEOMETRY,
    INITIALIZER_RECEIPT,
    INITIALIZER_SHA256,
    REFERENCE_INPUTS,
    DataRelease,
    StoragePin,
    baseline_files,
    production_plan,
)
from experiments.post_training.long_loop.exp3_recipe import (
    SftLrSchedule,
    verified_initializer_model,
    whole_run_lr_receipt,
)
from experiments.post_training.long_loop.exp3_rl_restart import restart_recipe
from experiments.post_training.long_loop.exp4_data import NEMOTRON_SOURCE, pool_steps
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.shared_sft import shared_sft_spec
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.exp4_curriculum import (
    SOURCE_ROUTES,
    exp4_rl_curriculum,
    exp4_validation_from_manifest,
)
from experiments.post_training.task_curation.images.build import environment_artifact

OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4"
REFERENCE_URI = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/restart53/frozen51-reference.json"
REFERENCE_SHA256 = "897f22035802380a63fb748739ae8a66ef208ea67f1fe1e5470bbec417b3e1bb"
# Exp3 completed 388 updates inside the same 20-hour budget. Leave 28 updates
# of margin; retain V7's absolute mixture phase coordinates rather than rescaling them.
SFT_UPDATES = 360

QUALIFIED_V7 = StoragePin(
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.18-v7-option-g/release-v7/release-complete-v7.json",
    "9ad020729305cd9d9ca0ae554850abc6a2575611145a5e7369ef0f656c1a617b",
)


V7_EVIDENCE_NAMES = ("inventory", "native_consumer_receipt", "native_measurement_job")


def verify_release_admission(
    *, qualification: StoragePin, output: str, publication: str, qualified: str, native_check: str
) -> None:
    document = json.loads(qualification.read())
    publisher = StoragePin(**document["publisher_manifest"])
    if (
        publisher.uri != publication
        or qualified != qualification.uri
        or native_check != document["artifact_graph"]["uri"]
    ):
        raise ValueError("SFT release graph dependencies differ from the sealed qualification")
    publisher.read()
    if document["training_ready"] is not True:
        raise ValueError("SFT release is not data-qualified")
    StoragePin(**{k: v for k, v in document["artifact_graph"].items() if k in {"uri", "sha256"}}).read()
    (StoragePath(output) / "admission.json").write_text(
        json.dumps(
            {
                "qualification": {"uri": qualification.uri, "sha256": qualification.sha256},
                "training_ready": True,
                "readiness_scope": document["readiness_scope"],
                "gpu_training_approved_by_data_owner": document["gpu_training_approved"],
                "publisher": document["publisher_manifest"],
                "native_check": document["artifact_graph"],
            },
            indent=2,
        )
    )


def release_artifact(
    uri: str, sha256: str, *, qualification: StoragePin, evidence_names: tuple[str, ...], require_ready: bool
) -> tuple[DataRelease, ArtifactStep]:
    """Reconstruct the published handle after its seal, record and members verify."""
    document = json.loads(qualification.read())
    if document["schema_version"] != 1 or document["geometry"] != asdict(GEOMETRY):
        raise ValueError("SFT release changes the controlled data geometry")
    if require_ready and document["training_ready"] is not True:
        raise ValueError("SFT release is not admitted for training")
    members = {name: StoragePin(**pin) for name, pin in document["artifacts"].items()}
    for pin in members.values():
        pin.read()
    evidence = {name: StoragePin(**document[name]) for name in ("publisher_manifest", *evidence_names)}
    for pin in evidence.values():
        pin.read()
    data = json.loads(members["data.json"].read())
    if data["cache_catalog"] != members["cache_catalog.json"].uri:
        raise ValueError("SFT release data names another cache artifact")
    if data["train_weights"] != json.loads(members["train_weights.json"].read()):
        raise ValueError("SFT release weights differ from the pinned member")
    if data["num_validation_sequences"] != document["num_validation_sequences"]:
        raise ValueError("SFT validation reservations differ from the qualified seal")
    release = DataRelease(qualification, document["training_ready"], members, evidence, data, document)
    raw = StoragePin(uri, sha256).read()
    record = json.loads(raw)
    configuration = record["config"]
    document = release.document
    if document["artifact_graph"] != {"uri": uri, "sha256": sha256}:
        raise ValueError("Qualified seal belongs to another artifact record")
    publisher = release.evidence["publisher_manifest"]
    if configuration["sha256"] != publisher.sha256:
        raise ValueError("Qualified seal belongs to another publication")
    if configuration["members"] != {name: asdict(pin) for name, pin in release.artifacts.items()}:
        raise ValueError("Published artifact members differ from the qualified seal")
    handle = ArtifactStep.adopt(record["name"], record["version"], record["source"], config=configuration)
    if handle.fingerprint() != record["fingerprint"]:
        raise ValueError("Reconstructed release handle differs from its published identity")
    name = record["name"].removesuffix("/release")
    publication = ArtifactStep.adopt(
        name + "/publisher-manifest", record["version"], publisher.uri, config={"sha256": publisher.sha256}
    )
    qualified = ArtifactStep.adopt(
        name + "/qualified-seal", record["version"], qualification.uri, config={"sha256": qualification.sha256}
    )
    native = ArtifactStep.adopt(name + "/native-check-record", record["version"], uri, config={"sha256": sha256})
    admission = apply(
        name + "/admission",
        verify_release_admission,
        version=record["version"],
        qualification=qualification,
        output=OUT,
        publication=publication,
        qualified=qualified,
        native_check=native,
    )
    return release, admission


def exp4_plan(release: DataRelease, source_revision: str, version: str, curriculum: RlCurriculum) -> dict:
    resolved = production_plan(release, source_revision, baseline_files(), initial_eval_inputs=REFERENCE_INPUTS)
    sft = resolved["sft"]
    sft["stop_after_steps"] = SFT_UPDATES
    sft["trainer"]["num_train_steps"] = SFT_UPDATES
    sft["target_positions"] = SFT_UPDATES * sft["batch_size"] * sft["seq_len"]
    sft["optimizer_schedule"] = {
        "horizon": SFT_UPDATES,
        "warmup": 0,
        "decay": SFT_UPDATES,
        "min_lr_ratio": 0.1,
        "lr_schedule": "linear",
    }
    raw = StoragePath(REFERENCE_URI).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Exp3 comparison reference changed")
    reference = json.loads(raw)
    rl = reference["plan"]["rl"]
    recipe = restart_recipe(reference)
    recipe["extra_env"]["NCCL_BUFFSIZE"] = "4194304"
    recipe["terminal_bench"]["harbor"].pop("eval_override_storage_mb", None)
    # The curriculum derives the rollout buffer's per-route window quotas at the
    # run's batch size; a declared value here would have to match it exactly.
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    recipe["trainer"]["rollout_buffer"]["source_routes"] = {
        **SOURCE_ROUTES,
        NEMOTRON_SOURCE: "nonagentic",
    }
    recipe["trainer"]["callbacks"] = [
        callback for callback in recipe["trainer"]["callbacks"] if callback["type"] != "wall_time_budget"
    ]
    resolved["rl"] = {
        **rl,
        "name": "mochi-exp4-rl",
        "config": recipe,
        "wall_clock_budget": dict(RL_TRAINING_BUDGET),
    }
    resolved["name"] = "long-loop-exp4/round4"
    resolved["storage"]["output_prefix"] = OUTPUT_PREFIX
    resolved["execution"]["coordinator"]["coordinator_name"] = "mochi-exp4"
    canonical = micro210_evaluation(
        Micro210Model(
            "s3://unresolved/terminal",
            "sha256:" + "0" * 64,
            resolved["tokenizer"]["repo"],
            resolved["tokenizer"]["revision"],
        ),
        run_id=f"mochi-exp4@{version}",
        output_uri="<resolved-at-build>",
    )
    resolved["evaluation"] = {
        **canonical.policy()["evaluation"],
        "name": "users/benfeuer/checkpoints/mochi-exp4-micro210",
        "expected_rollouts": 440,
    }
    resolved["datasets"]["nonagentic"]["source_names"] = [
        *resolved["datasets"]["nonagentic"]["source_names"],
        NEMOTRON_SOURCE,
    ]
    resolved["archive"]["hf_upload"]["repo_prefix"] = "open-athena/Snowball-LongLoop-Round4"
    resolved["archive"]["evaldash"]["run_prefix"] = "long-loop-round4"
    resolved["curriculum"] = {**asdict(curriculum), "identity": curriculum.identity}
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return resolved


@click.command(help=__doc__)
@click.option("--sft-artifact-record", required=True)
@click.option("--sft-artifact-record-sha256", required=True)
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@rl_build_options
def main(
    sft_artifact_record: str, sft_artifact_record_sha256: str, source_revision: str, validation_output: Path
) -> ArtifactStep:
    configure_coreweave_s3()
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    release, handle = release_artifact(
        sft_artifact_record,
        sft_artifact_record_sha256,
        qualification=QUALIFIED_V7,
        evidence_names=V7_EVIDENCE_NAMES,
        require_ready=click.get_current_context().params["do_run"],
    )
    version = resolve_version("mochi-exp4", None)
    os.environ.update(MARIN_PREFIX=OUTPUT_PREFIX, MARIN_TEMP_PREFIX="s3://marin-us-east-08a/tmp")
    pool_version = "2026.11.18.71"
    producer = pool_steps(pool_version, grader_environment=environment_artifact(GRADER_PACKAGES))["pools"]
    path = producer.path(OUTPUT_PREFIX)
    record_bytes = (StoragePath(path) / ".artifact.json").read_bytes()
    producer_record = json.loads(record_bytes)
    if producer_record["fingerprint"] != producer.fingerprint():
        raise ValueError("Regional Exp4 pool producer differs")
    manifest_bytes = (StoragePath(path) / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    curriculum = exp4_rl_curriculum(exp4_validation_from_manifest(manifest["validation"]))
    if manifest["curriculum"] != curriculum.identity:
        raise ValueError("Regional pool was prepared for another curriculum")
    resolved = exp4_plan(release, source_revision, version, curriculum)
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    pins = [
        {"uri": prefix_join(path, name), "sha256": sha256}
        for name, sha256 in {**manifest["files"], "manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}.items()
    ]
    pools = {
        "pools": ArtifactStep.adopt(
            producer.name,
            producer.version,
            path,
            config={
                "producer_fingerprint": producer.fingerprint(),
                "producer_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "files": pins,
            },
        )
    }
    resolved["datasets"] = {
        **{key: value for key, value in resolved["datasets"].items() if key not in ("agentic", "nonagentic", "heldout")},
        "pools": {
            "uri": path,
            "rows": manifest["rows"],
            "files": pins,
            "composition": manifest["composition"],
            "routes": manifest["routes"],
        },
    }
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    terminal = build_spec_graph(
        resolved, version, through="archive", data_artifacts=pools, curriculum=curriculum, sft_release=handle
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
    (validation_output / "optimizer-groups.json").write_text(
        json.dumps(optimizer_group_receipt(resolved["sft"]), indent=2)
    )
    (validation_output / "sft-lr-schedule.json").write_text(
        json.dumps(
            whole_run_lr_receipt(
                shared_sft_spec("long-loop-exp4/round4/sft", version, resolved["sft"]),
                schedule=draccus.decode(SftLrSchedule, resolved["sft"]["optimizer_schedule"]),
            ),
            indent=2,
        )
    )
    model = verified_initializer_model(INITIALIZER_RECEIPT, INITIALIZER_SHA256)
    validate_graph(
        terminal, validation_output, OUTPUT_PREFIX, model_config=(StoragePath(model) / "config.json").read_bytes()
    )
    return terminal


if __name__ == "__main__":
    main()
