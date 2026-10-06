# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal the four admitted v3 DEV rows and bind one later promoted comparison."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from marin.evaluation.records import record_path
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.parquet import read_task_records, write_task_records

from experiments.post_training.russell_rsi.bootstrap_loop import checkpoint_score, promotes, write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION
from experiments.post_training.russell_rsi.launch_post_teacher_sft import qualified_four_update_sft
from experiments.post_training.russell_rsi.launch_rsi_continuation import qualified_champion
from experiments.post_training.russell_rsi.launch_supplementary import SelectedCheckpoint
from experiments.post_training.russell_rsi.teacher_diversity_study import PROTOCOL as STUDY_PROTOCOL

PROTOCOL = "unused-supplementary-qualification-v3"
TASK_ORDER = (
    "iris-terminal-history-transaction-chunks",
    "iris-inmemory-stream-log-presence",
    "fray-tpu-topology-resource-defaults",
    "marin-structural-status-path",
)


@dataclass(frozen=True)
class PanelAssemblyConfig:
    handoff: PinnedFile
    protocol: PinnedFile
    parquets: tuple[PinnedFile, PinnedFile]
    task_jsons: tuple[PinnedFile, ...]
    output_path: Path


def assemble_unused_panel(config: PanelAssemblyConfig) -> dict:
    """Copy admitted raw task records without adding current schema defaults."""
    handoff = config.handoff.read_json()
    protocol_bytes = config.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    tasks = handoff["tasks"]
    if (
        handoff["protocol"] != PROTOCOL
        or protocol["protocol"] != PROTOCOL
        or handoff["comparison_protocol_sha256"] != config.protocol.sha256
        or tuple(handoff["task_order"]) != TASK_ORDER
        or tuple(row["contract_id"] for row in tasks) != TASK_ORDER
        or handoff["task_count"] != 4
        or handoff["source_split"] != "dev"
        or handoff["repository_holdout"] is not False
        or [row["source_commit"] for row in tasks] != sorted(row["source_commit"] for row in tasks)
        or any(row["qualified"] is not True for row in tasks)
        or len(config.task_jsons) != 4
    ):
        raise ValueError("Panel assembly changed the frozen four-task v3 handoff")
    admission_files = {}
    if (
        len(handoff["admission_artifacts"]) != 2
        or len({row["artifact_uri"] for row in handoff["admission_artifacts"]}) != 2
    ):
        raise ValueError("Panel assembly requires two distinct admission artifacts")
    for admission in handoff["admission_artifacts"]:
        completion_pin = admission["completion"]
        completion = PinnedFile(completion_pin["path"], completion_pin["sha256"]).read_json()
        evidence_pin = admission["evidence_manifest"]
        evidence = PinnedFile(evidence_pin["path"], evidence_pin["sha256"]).read_json()
        if (
            completion["status"] != "passed"
            or completion["task_count"] != 2
            or completion["evidence_manifest_sha256"] != evidence_pin["sha256"]
            or completion["provider_requests"] != 0
            or evidence["provider_requests"] != 0
            or evidence["source_split"] != "dev"
            or evidence["repository_holdout"] is not False
        ):
            raise ValueError("Panel assembly requires the completed model-free admissions")
        admission_files[admission["artifact_uri"]] = evidence["files"]
    expected_parquets = {row["admitted_cohort_parquet_sha256"] for row in tasks}
    if len(expected_parquets) != 2 or {pin.sha256 for pin in config.parquets} != expected_parquets:
        raise ValueError("Panel inputs differ from the two admitted Parquets")
    records = {}
    for pin in config.parquets:
        with tempfile.TemporaryDirectory() as temporary:
            verified = Path(temporary) / "admitted.parquet"
            verified.write_bytes(pin.read_bytes())
            cohort = list(read_task_records(str(verified)))
        if len(cohort) != 2:
            raise ValueError("Each admitted cohort requires its exact two records")
        for raw in cohort:
            value = json.loads(raw)
            if value["id"] in records:
                raise ValueError("Admitted cohorts contain duplicate task identities")
            records[value["id"]] = raw
    ordered = []
    for row, task_pin in zip(tasks, config.task_jsons, strict=True):
        files = admission_files[row["admission_artifact_uri"]]
        if (
            task_pin.sha256 != row["sealed_task_json_file_sha256"]
            or files[f"contracts/{row['contract_id']}/task.json"] != task_pin.sha256
            or files["supplementary-evaluation.parquet"] != row["admitted_cohort_parquet_sha256"]
            or files[f"contracts/{row['contract_id']}/result.json"] != row["admission_result_sha256"]
        ):
            raise ValueError("Panel task inputs differ from their admission evidence")
        value = task_pin.read_json()
        raw = records.pop(value["id"])
        if json.loads(raw) != value or digest(value) != row["task_sha256"]:
            raise ValueError("Panel task differs from the admitted raw TaskSpec")
        ordered.append(raw)
    if records:
        raise ValueError("Panel assembly left an unexpected admitted task")
    destination = config.output_path
    destination.mkdir(parents=True, exist_ok=False)
    parquet = destination / "supplementary.parquet"
    write_task_records(str(parquet), ordered)
    if list(read_task_records(str(parquet))) != ordered:
        raise ValueError("Panel serialization changed an admitted raw record")
    manifest = {
        "protocol": PROTOCOL,
        "status": "sealed-before-model-calls",
        "handoff": {"uri": config.handoff.uri, "sha256": config.handoff.sha256},
        "comparison_protocol_sha256": config.protocol.sha256,
        "task_count": 4,
        "task_order": list(TASK_ORDER),
        "tasks": [{**row, "task_id": json.loads(raw)["id"]} for row, raw in zip(tasks, ordered, strict=True)],
        "parquet_filename": parquet.name,
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "admission_artifacts": handoff["admission_artifacts"],
        "runtime_bundle": handoff["runtime_bundle"],
        "source_split": "dev",
        "repository_holdout": False,
        "model_calls": 0,
    }
    write_once(StoragePath(str(destination / "panel-manifest.json")), manifest)
    (destination / "comparison-protocol.json").write_bytes(protocol_bytes)
    return manifest


def completed_record(pin: PinnedFile, status: PinnedFile) -> dict:
    record = pin.read_json()
    if (
        pin.uri != prefix_join(record["output_path"], ".artifact.json")
        or status.uri != prefix_join(record["output_path"], ".executor_status")
        or status.read_bytes().decode().strip() != "SUCCESS"
    ):
        raise ValueError("Comparison requires the exact completed producer")
    return record


def promoted_study_checkpoint(config: dict) -> SelectedCheckpoint:
    """Return the qualified promoted export, with a separate historical comparator."""
    if config["protocol"] != PROTOCOL or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise ValueError("Unused comparison source and runtime are not explicitly bound")
    panel = PinnedFile(
        prefix_join(config["panel"]["uri"], "panel-manifest.json"), config["panel"]["identity_config"]["manifest_sha256"]
    ).read_json()
    if (
        panel["protocol"] != PROTOCOL
        or tuple(panel["task_order"]) != TASK_ORDER
        or [row["task_id"] for row in panel["tasks"]] != config["task_ids"]
        or panel["parquet_sha256"] != config["panel"]["identity_config"]["panel_sha256"]
        or panel["comparison_protocol_sha256"] != config["panel"]["identity_config"]["comparison_protocol_sha256"]
        or panel["runtime_bundle"] != config["runtime_bundle"]
    ):
        raise ValueError("Comparison changed the unused v3 panel or its fixed task order")
    selection_producer = completed_record(
        PinnedFile(**config["selection_producer"]), PinnedFile(**config["selection_status"])
    )
    decision_pin = PinnedFile(**config["selection"])
    if decision_pin.uri != prefix_join(selection_producer["output_path"], "post-sft-selection.json"):
        raise ValueError("Promotion record is outside its completed selection producer")
    decision = decision_pin.read_json()
    if (
        decision["protocol"] != STUDY_PROTOCOL
        or selection_producer["name"] != f"documents/russell-rsi-{STUDY_PROTOCOL}-selection"
    ):
        raise ValueError("Promotion is not the completed diversity study selection")
    incumbent = checkpoint_score(decision["incumbent"])
    selected = checkpoint_score(decision["selected"])
    incumbent_source = PinnedFile(**config["incumbent_producer"]).read_json()
    incumbent_qualification = PinnedFile(**config["incumbent_qualification"]).read_json()
    qualified_champion(incumbent_qualification, incumbent_source)
    incumbent_identity = f"{incumbent_source['name']}@{incumbent_source['version']}:{incumbent_source['fingerprint']}"
    if (
        incumbent.checkpoint_identity != incumbent_identity
        or checkpoint_score(decision["promoted"]) != selected
        or not promotes(selected, incumbent)
        or selected not in [checkpoint_score(decision[key]) for key in ("sft", "sft_rl") if decision[key] is not None]
    ):
        raise ValueError("Unused comparison requires actual promotion against the fixed update8 incumbent")
    historical = PinnedFile(**config["historical_comparison"]).read_json()
    historical_launch = PinnedFile(**config["historical_launch"]).read_json()
    if (
        historical_launch["config_sha256"] != config["historical_comparison"]["sha256"]
        or historical_launch["config_readback_verified"] is not True
        or config["incumbent_producer"]
        != {"uri": historical["checkpoint_record_uri"], "sha256": historical["checkpoint_record_sha256"]}
        or incumbent_identity != historical["selected_candidate"]
        or incumbent != checkpoint_score(historical_launch["selected_checkpoint"])
        or config["parent"] != historical["parent"]
        or config["parent"]["artifact_identity"] != decision["original_parent"]["checkpoint_identity"]
        or config["parent"]["artifact_identity"] == incumbent_identity
    ):
        raise ValueError("Comparator differs from the independent historical SFT parent")
    producer_pin = PinnedFile(**config["candidate_producer"])
    producer = completed_record(producer_pin, PinnedFile(**config["candidate_status"]))
    identity = f"{producer['name']}@{producer['version']}:{producer['fingerprint']}"
    if identity != selected.checkpoint_identity:
        raise ValueError("Candidate producer does not identify the promoted checkpoint")
    return SelectedCheckpoint(
        qualified_candidate_export(config, producer), identity, producer_pin.sha256, decision_pin.sha256
    )


def qualified_candidate_export(config: dict, producer: dict) -> str:
    """Return the selected producer export after its training and reload checks."""
    identity = f"{producer['name']}@{producer['version']}:{producer['fingerprint']}"
    expected_reload_deps: list[str]
    reload_model_identity = identity
    if producer["result_type"] == "marin.training.training.LevanterCheckpoint":
        qualification_pin = PinnedFile(**config["candidate_qualification"])
        qualification = qualification_pin.read_json()
        training = completed_record(
            PinnedFile(**config["candidate_training_producer"]), PinnedFile(**config["candidate_training_status"])
        )
        reload_model_identity = f"{training['name']}@{training['version']}:{training['fingerprint']}"
        expected_reload_deps = [f"{training['name']}@{training['version']}"]
        export = qualified_four_update_sft(qualification, identity=reload_model_identity, root=training["output_path"])
        if (
            producer["config"] != {"sft": reload_model_identity, "qualification_sha256": qualification_pin.sha256}
            or producer["source"] != export
            or training["result_type"] != "marin.training.training.LevanterCheckpoint"
        ):
            raise ValueError("Selected qualified HF model differs from its original SFT producer and qualification")
        reload_evidence = qualification["serving_reload"]
    elif producer["result_type"] == "marin.rl.skyrl.SkyRLRun":
        export = producer["result"]["hf_model_uri"]
        optimizer = completed_record(
            PinnedFile(**config["candidate_optimizer_producer"]), PinnedFile(**config["candidate_optimizer_status"])
        )
        if (
            producer["result"]["global_step"] != 4
            or producer["result"]["tokenizer_uri"] != MODEL
            or producer["result"]["tokenizer_revision"] != MODEL_REVISION
            or not export
            or export != prefix_join(producer["output_path"], "exports/global_step_4/policy")
            or optimizer["name"] != f"documents/russell-rsi-{STUDY_PROTOCOL}-optimizer-gate"
            or optimizer["deps"] != [f"{producer['name']}@{producer['version']}"]
            or optimizer["config"] != {"expected_updates": 4, "actual_updates": 4, "export_uri": export}
        ):
            raise ValueError("Promoted RL requires its completed four-update optimizer gate and fixed tokenizer")
        expected_reload_deps = [
            f"{producer['name']}@{producer['version']}",
            f"{optimizer['name']}@{optimizer['version']}",
        ]
        reload_evidence = config["candidate_reload_evidence"]
    else:
        raise ValueError("Unsupported promoted checkpoint producer")
    reload = completed_record(
        PinnedFile(**config["candidate_reload_producer"]), PinnedFile(**config["candidate_reload_status"])
    )
    if reload["deps"] != expected_reload_deps:
        raise ValueError("Serving reload differs from the selected producer dependencies")
    evidence_pin = PinnedFile(reload_evidence["evidence_uri"], reload_evidence["evidence_sha256"])
    record = evidence_pin.read_json()
    if (
        record["run_id"] not in reload["result"]["run_ids"]
        or evidence_pin.uri != record_path(reload["result"]["records_prefix"], record["run_id"])
        or reload["config"]["model"]["identity"] != reload_model_identity
        or reload["config"]["model"]["location"] != export
        or reload["config"]["evals"] != "mmlu-smoke"
        or reload["config"]["limit"] != 1
        or record["status"] != "succeeded"
        or record["error"] is not None
        or not record["metrics"]
        or record["model"]["config"]["identity"] != reload_model_identity
        or record["model"]["location"] != export
        or record["eval"]["name"] != "mmlu-smoke"
        or record["eval"]["evalchemy"]["max_eval_instances"] != 1
    ):
        raise ValueError("Promoted export lacks its own exact successful serving reload")
    return export


def main() -> None:
    """Assemble a new local panel from the reviewed input pins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    arguments = parser.parse_args()
    record = json.loads(arguments.config_path.read_text())
    if len(record["parquets"]) != 2:
        raise ValueError("Local panel assembly requires exactly two pinned Parquets")
    config = PanelAssemblyConfig(
        PinnedFile(**record["handoff"]),
        PinnedFile(**record["protocol"]),
        (PinnedFile(**record["parquets"][0]), PinnedFile(**record["parquets"][1])),
        tuple(PinnedFile(**pin) for pin in record["task_jsons"]),
        Path(record["output_path"]),
    )
    manifest = assemble_unused_panel(config)
    print(
        json.dumps(
            {"task_count": manifest["task_count"], "parquet_sha256": manifest["parquet_sha256"], "model_calls": 0}
        )
    )


if __name__ == "__main__":
    main()
