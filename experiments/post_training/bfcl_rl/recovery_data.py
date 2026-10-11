# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind successful BFCL collection receipts to exact-token preference caches."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from levanter.store.cache import write_levanter_cache
from marin.execution.artifact import Artifact
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import DATA_URI, MODELS, ModelSource
from experiments.post_training.bfcl_rl.data import (
    DATASET_COMMIT,
    FULL_TASK_COUNT,
    PARITY_TASK_COUNT,
    PARTITION_MANIFEST_SHA256,
    BFCLPartition,
    TaskIdentity,
)
from experiments.post_training.bfcl_rl.preferences import select_training_pairs
from experiments.post_training.bfcl_rl.retained_preferences import (
    CollectionIdentity,
    PretokenizedPreference,
    RetainedRollout,
    pretokenized_preference,
    read_retained_archives,
)


@dataclass(frozen=True)
class CollectionReceipt:
    identity: CollectionIdentity
    task_names: frozenset[str]
    conditions_digest: str
    trajectory_root: str


@dataclass(frozen=True)
class RecoveryCacheConfig:
    teacher_terminal_uri: str
    student_terminal_uri: str
    data_root: str
    max_length: int
    output_path: str


class RecoveryPreferenceCache(Artifact):
    num_preferences: int
    tokenizer_uri: str
    tokenizer_revision: str
    max_length: int
    selection_manifest_uri: str


def load_audited_partition(data_root: str) -> BFCLPartition:
    """Require the pinned partition manifest and the verified-copy completion marker."""
    root = StoragePath(data_root)
    payload = (root / "bfcl_complement-manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != PARTITION_MANIFEST_SHA256:
        raise ValueError("BFCL partition manifest differs from the campaign audit")
    completion = json.loads((root / ".verified-copy-manifest.json").read_text())
    copied_manifest = next(file for file in completion["files"] if file["path"] == "bfcl_complement-manifest.json")
    if copied_manifest["sha256"] != PARTITION_MANIFEST_SHA256:
        raise ValueError("verified release contains a different BFCL partition manifest")
    manifest = json.loads(payload)
    partition = BFCLPartition(
        manifest["dataset_commit"],
        tuple(TaskIdentity(**task) for task in manifest["complement"]),
        tuple(TaskIdentity(**task) for task in manifest["parity"]),
    )
    if partition.dataset_commit != DATASET_COMMIT:
        raise ValueError("BFCL partition belongs to a different dataset revision")
    if len(partition.complement) != FULL_TASK_COUNT - PARITY_TASK_COUNT or len(partition.parity) != PARITY_TASK_COUNT:
        raise ValueError("BFCL partition task counts differ from the audited complement and holdout")
    return partition


def generation_collection_receipt(
    terminal: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    model: ModelSource,
    harness: str,
    partition: BFCLPartition,
) -> CollectionReceipt:
    """Bind native terminal and worker receipts to an explicit model and audited complement."""
    config = terminal["config"]
    result = terminal["result"]
    if result["state"] != "succeeded" or result["run_id"] != config["run"]["id"]:
        raise ValueError("recovery requires a successful collection with matching terminal identity")
    if result["attempt_id"] != config["run"]["attempt_id"] or not result["iris_job_id"]:
        raise ValueError("collection attempt or Iris job identity is missing or inconsistent")
    return generation_collection_config_receipt(config, resolved, model=model, harness=harness, partition=partition)


def generation_collection_config_receipt(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    model: ModelSource,
    harness: str,
    partition: BFCLPartition,
) -> CollectionReceipt:
    """Validate actual generation inputs independently of whether the producer has finished."""
    if config["runtime"]["entrypoint"] != "skyrl_train.entrypoints.terminal_bench_generate":
        raise ValueError("recovery collection must be generation-only")
    expected = model
    locator = config["inputs"]["model"]
    if (locator["uri"], locator["tokenizer_uri"], locator["tokenizer_revision"]) != (
        expected.uri,
        expected.model,
        expected.revision,
    ):
        raise ValueError("collection model or tokenizer differs from the mandated revision")
    sources = config["inputs"]["train_data"]
    if len(sources) != 1 or sources[0]["uri"] != DATA_URI or config["inputs"]["validation_data"]:
        raise ValueError("recovery collection must use only the audited BFCL complement")
    source = sources[0]
    relative_path = source["relative_path"]
    task_names = frozenset(task.name for task in partition.complement)
    if relative_path != "bfcl_complement":
        parts = relative_path.split("/")
        if len(parts) != 2 or parts[0] != "bfcl_complement" or parts[1] not in task_names:
            raise ValueError("collection selection is outside the BFCL complement")
        task_names = frozenset({parts[1]})
    actual_sources = resolved["train_data_sources"]
    if len(actual_sources) != 1 or any(
        actual_sources[0][key] != source[key] for key in ("uri", "identity", "relative_path")
    ):
        raise ValueError("worker data source differs from the collection launch")
    if resolved["val_data_sources"]:
        raise ValueError("recovery collection cannot consume parity validation tasks")
    skyrl = resolved["config"]["skyrl"]
    policy = skyrl["trainer"]["policy"]["model"]
    if policy["source_identity"] != locator["identity"] or policy["source_uri"] != expected.uri:
        raise ValueError("worker model source differs from the collection launch")
    retention = skyrl["generator"]["trajectory_retention"]
    if (
        not retention["enabled"]
        or not retention["required"]
        or retention["sample_fraction"] != 1.0
        or retention["phases"] != ["eval"]
    ):
        raise ValueError("recovery requires complete generation trajectory retention")
    if skyrl["generator"]["n_samples_per_prompt"] != 1:
        raise ValueError("initial recovery requires one paired rollout per model and task")
    harbor = skyrl["terminal_bench_config"]["harbor"]
    if harbor.get("environment_type") == "daytona":
        if harbor.get("import_path"):
            raise ValueError("Daytona collection must use the native Harbor environment")
    elif (
        harbor.get("container_profile") != "gvisor"
        or harbor.get("import_path") != "marinskyrl.iris_harbor_environment:IrisEnvironment"
    ):
        raise ValueError("collection must use native Daytona or historical Iris gVisor task sandboxes")
    if not config["ingress"]["record_literal"] or not skyrl["trainer"]["algorithm"]["tito_full"]:
        raise ValueError("recovery requires literal model-token capture and full token evidence")
    model_info = skyrl["terminal_bench_config"]["model_info"]
    if (model_info["max_input_tokens"], model_info["max_output_tokens"]) != (32768, 8192):
        raise ValueError("recovery collection context differs from the fixed Pi policy")
    conditions_harbor = dict(harbor)
    if harness == "native":
        for key in ("name", "version", "agent_profiles"):
            conditions_harbor.pop(key, None)
    conditions = {
        "harbor": conditions_harbor,
        "model_info": model_info,
        "sampling_params": skyrl["generator"]["sampling_params"],
        "max_input_length": skyrl["generator"]["max_input_length"],
    }
    conditions_digest = hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()
    task_root = Path(actual_sources[0]["local_path"]) / "bfcl_complement"
    identity = CollectionIdentity(
        retention["run_id"],
        locator["identity"],
        expected.revision,
        harness,
        partition.dataset_commit,
        str(task_root),
    )
    return CollectionReceipt(identity, task_names, conditions_digest, retention["output_path"])


def collection_receipt(
    terminal: Mapping[str, Any], resolved: Mapping[str, Any], *, model: str, partition: BFCLPartition
) -> CollectionReceipt:
    """Bind recovery generation receipts to the mandated model and fixed Pi conditions."""
    harbor = resolved["config"]["skyrl"]["terminal_bench_config"]["harbor"]
    if (harbor["name"], harbor["version"], harbor["thinking_format"]) != ("pi", "0.87.0", "chat-template"):
        raise ValueError("recovery collection must use the fixed policy Pi harness")
    return generation_collection_receipt(
        terminal, resolved, model=MODELS[model], harness="pi@0.87.0", partition=partition
    )


def recovery_preference_rows(
    teacher: Sequence[RetainedRollout],
    student: Sequence[RetainedRollout],
    *,
    teacher_receipt: CollectionReceipt,
    student_receipt: CollectionReceipt,
    partition: BFCLPartition,
    max_length: int,
) -> tuple[list[PretokenizedPreference], dict[str, Any]]:
    """Require complete paired collections and export only sole-correct preferences."""
    if teacher_receipt.task_names != student_receipt.task_names:
        raise ValueError("teacher and student collections selected different tasks")
    if teacher_receipt.conditions_digest != student_receipt.conditions_digest:
        raise ValueError("teacher and student collections used different harness or sampling conditions")
    selections = select_training_pairs(
        [record.rollout for record in teacher],
        [record.rollout for record in student],
        complement_source_ids=frozenset(task.source_id for task in partition.complement),
        parity_source_ids=frozenset(task.source_id for task in partition.parity),
    )
    expected = teacher_receipt.task_names
    task_names = {task.source_id: task.name for task in partition.complement}
    for records in (teacher, student):
        if {task_names[record.rollout.task_source_id] for record in records} != expected:
            raise ValueError("retained records do not cover the complete collection selection")
        if any(record.rollout.repetition != 0 for record in records):
            raise ValueError("initial recovery collection contains unexpected repetitions")
    retained = {record.rollout.trajectory_uri: record for record in (*teacher, *student)}
    rows = []
    pairs = []
    excluded_pairs = []
    for selection in selections:
        pair = selection.pair
        if pair is None:
            continue
        chosen = retained[pair.chosen.trajectory_uri]
        rejected = retained[pair.rejected.trajectory_uri]
        if chosen.steps[0].prompt_token_ids != rejected.steps[0].prompt_token_ids:
            excluded_pairs.append({"reason": "initial_prompt_mismatch", "pair": asdict(pair)})
            continue
        rows.append(pretokenized_preference(chosen, rejected, max_length=max_length))
        pairs.append(asdict(pair))
    report = {
        "dataset_commit": partition.dataset_commit,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "teacher": asdict(teacher_receipt.identity),
        "student": asdict(student_receipt.identity),
        "conditions_digest": teacher_receipt.conditions_digest,
        "dispositions": dict(Counter(selection.disposition.value for selection in selections)),
        "preferences": pairs,
        "excluded_preferences": excluded_pairs,
        "max_length": max_length,
    }
    return rows, report


def write_recovery_cache(rows: Sequence[PretokenizedPreference], report: Mapping[str, Any], output_path: str) -> None:
    """Write exact-token columns using Levanter's cache writer and retain preference provenance."""
    root = StoragePath(output_path)
    root.mkdirs()
    report_text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    (root / "selection.json").write_text(report_text)
    if not rows:
        raise ValueError("no verifier-discriminated pairs exist; recovery must perform no update")
    records = ({key: np.asarray(value, dtype=np.int32) for key, value in row.items()} for row in rows)
    metadata = {
        "preference_provenance_uri": str(root / "selection.json"),
        "preference_provenance_sha256": hashlib.sha256(report_text.encode()).hexdigest(),
    }
    write_levanter_cache(records, str(root / "train"), metadata=metadata)
    stats = {
        "total_elements": len(rows),
        "total_tokens": sum(len(row["chosen_input_ids"]) + len(row["rejected_input_ids"]) for row in rows),
    }
    (root / "train" / ".stats.json").write_text(json.dumps(stats, sort_keys=True) + "\n")


def recovery_cache_value(output_path: str) -> RecoveryPreferenceCache:
    """Read the completed cache's selection metadata for artifact persistence."""
    manifest_uri = str(StoragePath(output_path) / "selection.json")
    report = json.loads(StoragePath(manifest_uri).read_text())
    return RecoveryPreferenceCache(
        path=output_path,
        num_preferences=len(report["preferences"]),
        tokenizer_uri=MODELS["student"].model,
        tokenizer_revision=report["student"]["model_revision"],
        max_length=report["max_length"],
        selection_manifest_uri=manifest_uri,
    )


def build_recovery_cache(config: RecoveryCacheConfig) -> RecoveryPreferenceCache:
    """Bind successful collection receipts and archive evidence into a recovery artifact."""
    partition = load_audited_partition(config.data_root)
    receipts = []
    collections = []
    for model, uri in (("teacher", config.teacher_terminal_uri), ("student", config.student_terminal_uri)):
        terminal = json.loads(StoragePath(uri).read_text())
        resolved_uri = terminal["config"]["artifacts"]["resolved_config_uri"]
        resolved = json.loads(StoragePath(resolved_uri).read_text())
        receipt = collection_receipt(terminal, resolved, model=model, partition=partition)
        archives = sorted(
            str(path)
            for path in (StoragePath(receipt.trajectory_root) / "schema_v6" / "archives" / "**" / "*.zip").glob()
        )
        records = read_retained_archives(archives, identity=receipt.identity, partition=partition)
        receipts.append(receipt)
        collections.append(records)
    rows, report = recovery_preference_rows(
        collections[0],
        collections[1],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=partition,
        max_length=config.max_length,
    )
    report["terminal_manifests"] = [config.teacher_terminal_uri, config.student_terminal_uri]
    write_recovery_cache(rows, report, config.output_path)
    return recovery_cache_value(config.output_path)
