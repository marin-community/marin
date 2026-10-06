# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select from completed coding v9 and retention v10 without evaluator dependencies."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import PurePosixPath

from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.runner import EndpointRoute
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.parquet import read_tasks

from experiments.evaluation.pipeline import EvalStepConfig
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingEvidenceConfig,
    CodingPanel,
    PanelItem,
    coding_evidence_payload,
    collect_coding_eval_evidence,
)
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.interrupted_calibration import (
    OUTPUT_PROTOCOL,
    InterruptedSelectionConfig,
    foreground_coding_batch,
    seal_interrupted_selection,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    StudyBaseline,
    StudySelectionConfig,
    post_sft_selection_stages,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_PROBES, preflight_task

PROTOCOL = "russell-rsi-completed-sft-selection-v1"
VERSION = "2026.10.06.11"
LAUNCH_PROTOCOL = "russell-rsi-foreground-launch-proof-v1"
CODING_SOURCE_FILE = "experiments/post_training/russell_rsi/coding_transport_replacement.py"
RETENTION_SOURCE_FILE = "experiments/post_training/russell_rsi/retention_continuation.py"
JOURNAL_SOURCE_FILE = "experiments/post_training/russell_rsi/interrupted_calibration.py"
PRODUCER_SOURCE_HEADS = {
    "2026.10.06.9": "5744330c06fe8d8f5c64c99e991fc3d350e2317f",
    "2026.10.06.10": "6d589e113cbef9209118f7e667e8c60a65ff1727",
}


def pinned_json(value: dict) -> dict:
    return PinnedFile(**value).read_json()


def pinned_at(value: dict, path: str) -> dict:
    if value["uri"] != path:
        raise ValueError("Completed evidence is outside its producer output")
    return pinned_json(value)


@dataclass(frozen=True)
class CompletedProducer:
    pins: dict
    launch: dict
    record: dict
    config: dict

    @property
    def output_path(self) -> str:
        return self.record["output_path"]


def completed_producer(pins: dict, version: str) -> CompletedProducer:
    """Validate the completed record against its frozen launch, config, and source review."""
    launch = pinned_json(pins["launch_proof"])
    config = pinned_json(pins["config"])
    if (
        launch["protocol"] != LAUNCH_PROTOCOL
        or launch["source_head"] != PRODUCER_SOURCE_HEADS[version]
        or launch["runtime_commit"] != MARIN_SKYRL.commit
        or launch["config"] != pins["config"]
        or launch["calibration_status"] != "incomplete_infrastructure"
        or launch["signal_gate_passed"] is not None
        or launch["rl_authorized"] is not False
        or config["version"] != version
    ):
        raise ValueError("Completed producer changed its frozen launch or runtime")
    review = pinned_json(launch["source_review"])
    if review["status"] != "approved" or review["source_head"] != launch["source_head"]:
        raise ValueError("Completed producer has no matching approved source")
    PinnedFile(**launch["request"]).read_bytes()
    preflight = pinned_json(launch["preflight"])
    if (
        preflight["exit_code"] != 0
        or preflight["identity"]["source_head"] != launch["source_head"]
        or preflight["identity"]["request_sha256"] != launch["request"]["sha256"]
    ):
        raise ValueError("Completed producer has no matching successful artifact-main preflight")
    record = pinned_at(pins["producer_record"], str(StoragePath(launch["output_path"]) / ".artifact.json"))
    identity = f"{record['name']}@{record['version']}:{record['fingerprint']}"
    if (
        identity != launch["producer_identity"]
        or record["version"] != version
        or record["output_path"] != launch["output_path"]
        or canonical_json(record["config"]) != canonical_json(launch["bound_config"])
        or StatusFile(record["output_path"], worker_id="completed-selection").status != STATUS_SUCCESS
        or len(record["provenance"]["base_commit"]) < 9
        or not launch["source_head"].startswith(record["provenance"]["base_commit"])
        or record["provenance"]["dirty"] is not False
    ):
        raise ValueError("Selection requires the exact completed producer")
    return CompletedProducer(pins, launch, record, config)


def coding_result(producer: CompletedProducer, source_pin: dict, original_config: EvalStepConfig) -> dict:
    """Validate the saved batch and reservation using the frozen coding source hashes."""
    config = producer.config
    if {
        "uri": config["source_config_uri"],
        "sha256": config["source_config_sha256"],
    } != source_pin or "path" in producer.record["result"]:
        raise ValueError("Coding substituted its original science configuration")
    evaluation = producer.record["config"]["evaluation"]
    expected = json.loads(json.dumps(asdict(replace(original_config, version="2026.10.06.9"))))
    if canonical_json(evaluation) != canonical_json(expected):
        raise ValueError("Coding changed the original model, panels, or sampling")
    reservation = pinned_at(
        producer.pins["reservation"], str(StoragePath(producer.output_path) / "journal/coding/reservation.json")
    )
    saved = pinned_at(producer.pins["result"], str(StoragePath(producer.output_path) / "journal/coding/result.json"))
    result = {"path": producer.output_path, **producer.record["result"]}
    amendment_pin = {"uri": config["transport_amendment_uri"], "sha256": config["transport_amendment_sha256"]}
    amendment = pinned_json(amendment_pin)
    transport = reservation["transport_replacement"]
    if (
        saved != {"binding": reservation, "result": result}
        or reservation["protocol"] != OUTPUT_PROTOCOL
        or reservation["config"] != evaluation
        or reservation["job_policy"] != {"failure_retries": 0, "preemption_retries": 0}
        or transport["coding_config"] != producer.pins["config"]
        or transport["amendment"] != amendment_pin
        or transport["source_sha256"] != producer.launch["source_files"][CODING_SOURCE_FILE]
        or transport["settings"] != amendment["replacement"]
        or len(transport["evaluations"]) != 2
        or len(result["run_ids"]) != 2
        or len(result["results_paths"]) != 2
    ):
        raise ValueError("Completed coding journal differs from its frozen producer")
    expected_evaluations = []
    batch = foreground_coding_batch(replace(original_config, version="2026.10.06.9"), EndpointRoute.CAPABILITY)
    for item in batch.evaluations:
        if not isinstance(item.executor, EvalchemyExecutor):
            raise ValueError("Completed coding requires the original Evalchemy panels")
        expected_evaluations.append(
            {"route": item.endpoint_route.value, "executor_config": asdict(item.executor.config)}
        )
    if canonical_json(transport["evaluations"]) != canonical_json(expected_evaluations):
        raise ValueError("Coding changed the original executor or sampling configuration")
    return result


def normalized_worker_provenance(value: dict) -> dict:
    modules = {}
    root = PurePosixPath(value["branch_root"])
    for name, entry in value["modules"].items():
        path = PurePosixPath(entry["path"])
        if name == "skyrl_train.inference_engines.chat_continuation":
            normalized_path = name
        elif path.is_relative_to(root):
            normalized_path = str(path.relative_to(root))
        else:
            raise ValueError("Completed worker loaded a branch module outside its checkout")
        modules[name] = {"path": normalized_path, "sha256": entry["sha256"]}
    return {"modules": modules, "skyrl": value["skyrl"]}


@dataclass(frozen=True)
class CompletedRetention:
    evaluation: dict
    binding: dict
    records: dict[tuple[str, str], dict]


def completed_retention_records(
    producer: CompletedProducer, original_config: dict, task_ids: tuple[str, ...]
) -> CompletedRetention:
    """Validate the frozen producer, all five journal records, and the token preflight."""
    config = producer.config
    evaluation = producer.record["config"]["evaluation"]
    binding = pinned_at(
        producer.pins["journal_binding"], str(StoragePath(producer.output_path) / "journal/binding.json")
    )
    continuation = binding["continuation"]
    tasks = list(read_tasks(evaluation["tasks_path"]))
    expected_tasks = {f"{task.id}/0": digest(task.model_dump(mode="json")) for task in tasks}
    expected_probes = [
        preflight_task(index, instruction, value).model_dump(mode="json")
        for index, (instruction, value) in enumerate(PREFLIGHT_PROBES, 1)
    ]
    amendment = {"uri": config["repair_amendment_uri"], "sha256": config["repair_amendment_sha256"]}
    if (
        canonical_json(evaluation) != canonical_json(original_config)
        or binding["protocol"] != OUTPUT_PROTOCOL
        or binding["job_policy"] != {"failure_retries": 0, "preemption_retries": 0, "timeout_hours": 6}
        or binding["config"] != evaluation
        or binding["worker_source_sha256"] != producer.launch["source_files"][JOURNAL_SOURCE_FILE]
        or continuation["worker_source_sha256"] != producer.launch["source_files"][RETENTION_SOURCE_FILE]
        or continuation["retention_config"] != producer.pins["config"]
        or continuation["launch_failure"] != amendment
        or continuation["protocol"] != config["protocol"]
        or normalized_worker_provenance(binding["worker_provenance"])
        != normalized_worker_provenance(producer.launch["worker_provenance"])
        or binding["parquet_sha256"] != hashlib.sha256(StoragePath(evaluation["tasks_path"]).read_bytes()).hexdigest()
        or set(binding["attempts"]) != {"preflight", "task"}
        or set(binding["attempts"]["preflight"]) != {"1", "2"}
        or set(binding["attempts"]["task"]) != {f"{task_id}/0" for task_id in task_ids}
        or binding["attempts"]["task"] != expected_tasks
    ):
        raise ValueError("Retention changed its frozen model, tasks, worker, or journal")
    PinnedFile(**amendment).read_bytes()
    expected_slots = {(kind, key) for kind, keys in binding["attempts"].items() for key in keys}
    slots = {(item["kind"], item["key"]) for item in producer.pins["attempts"]}
    if slots != expected_slots or len(producer.pins["attempts"]) != 5:
        raise ValueError("Selection requires exactly five completed retention attempts")
    completed = {}
    for item in producer.pins["attempts"]:
        kind, key = item["kind"], item["key"]
        directory = StoragePath(producer.output_path) / "journal" / kind / key
        reservation = pinned_at(item["reservation"], str(directory / "reservation.json"))
        result = pinned_at(item["result"], str(directory / "result.json"))
        expected = {"evaluation": binding, "kind": kind, "key": key, "task_sha256": binding["attempts"][kind][key]}
        if reservation != expected or result["binding"] != expected:
            raise ValueError("Completed retention attempt has a different binding")
        completed[(kind, key)] = result["result"]
    preflight = pinned_at(
        producer.pins["preflight_summary"], str(StoragePath(producer.output_path) / "token-preflight.json")
    )
    probes = [completed[("preflight", key)] for key in ("1", "2")]
    if (
        preflight["attempts"] != probes
        or preflight["status"] != "passed"
        or not any(probe["status"] == "passed" for probe in probes)
        or any(probe.get("contract_failure") for probe in probes)
        or preflight["fixtures"] != expected_probes
        or any(
            compact_json_sha256(fixture) != binding["attempts"]["preflight"][str(index)]
            for index, fixture in enumerate(preflight["fixtures"], 1)
        )
    ):
        raise ValueError("Retention does not preserve the original token preflight gate")
    return CompletedRetention(evaluation, binding, completed)


def retention_result_summary(completed: CompletedRetention, task_ids: tuple[str, ...]) -> dict:
    """Summarize the three valid canonical grades."""
    evaluation = completed.evaluation
    rewards = {}
    categories: Counter = Counter()
    starts: Counter = Counter()
    failed = []
    for task_id in task_ids:
        result = completed.records[("task", f"{task_id}/0")]
        record = result["record"]
        grade = record["grade"]
        if record["task_id"] != task_id:
            raise ValueError("Retention result changed its task identity")
        starts.update(result["startup_counts"])
        reward = grade["reward"]
        if grade["status"] != "graded" or reward not in (0, 1):
            raise ValueError("Selection requires a valid canonical grade for each retention task")
        rewards[task_id] = [reward]
        categories["passed" if reward == 1 else "incorrect"] += 1
        if reward == 0:
            failed.append(task_id)
    return {
        "model_identity": evaluation["model_identity"],
        "tasks_path": evaluation["tasks_path"],
        "tasks_identity": evaluation["tasks_identity"],
        "count": 3,
        "samples_per_task": 1,
        "startup_attempts": 1,
        "startup_counts": dict(starts),
        "informative_groups": 0,
        "task_rewards": rewards,
        "categories": dict(categories),
        "failed_task_ids": sorted(failed),
    }


def retention_summary(producer: CompletedProducer, original_config: dict, task_ids: tuple[str, ...]) -> dict:
    """Require all three canonical grades and reproduce the original completed summary."""
    completed = completed_retention_records(producer, original_config, task_ids)
    summary = retention_result_summary(completed, task_ids)
    if pinned_at(producer.pins["summary"], str(StoragePath(producer.output_path) / "failure_summary.json")) != summary:
        raise ValueError("Retention summary differs from its completed canonical results")
    return summary


@dataclass(frozen=True)
class CompletedSelectionConfig:
    selection: InterruptedSelectionConfig
    input_pin: PinnedFile
    coding_identity: str
    retention_identity: str


def seal_completed_selection(config: CompletedSelectionConfig) -> None:
    write_once(
        StoragePath(config.selection.selection.record.output_path) / "completed-producers.json",
        {
            "protocol": PROTOCOL,
            "input": asdict(config.input_pin),
            "coding_source": "replacement_v9",
            "retention_source": "repaired_v10",
            "coding_producer": config.coding_identity,
            "retention_producer": config.retention_identity,
            "calibration_status": "incomplete_infrastructure",
            "signal_gate_passed": None,
            "rl_authorized": False,
        },
    )
    seal_interrupted_selection(config.selection)


def completed_sft_selection_stages(
    config: dict, input_pin: PinnedFile, original: dict[str, ArtifactStep]
) -> dict[str, ArtifactStep]:
    """Validate frozen completed producers and build only CPU extraction and selection."""
    if (
        config != input_pin.read_json()
        or set(config) != {"protocol", "version", "source_config", "coding", "retention"}
        or config["protocol"] != PROTOCOL
        or config["version"] != VERSION
        or set(config["coding"])
        not in (
            {"config", "launch_proof", "producer_record", "reservation", "result"},
            {"config", "launch_proof", "producer_record", "reservation", "result", "evidence"},
        )
        or set(config["retention"])
        != {
            "config",
            "launch_proof",
            "producer_record",
            "journal_binding",
            "summary",
            "preflight_summary",
            "attempts",
        }
    ):
        raise ValueError("Completed selection changed its frozen protocol or schema")
    source = pinned_json(config["source_config"])
    completed = completed_coding_evidence(config, original)
    coding_evidence, coding, model, baseline = (
        completed.evidence,
        completed.producer,
        completed.model,
        completed.baseline,
    )
    retained = completed_producer(config["retention"], "2026.10.06.10")
    old_retention = original["retention-sft"]
    retained_config = old_retention.build_config(
        StepContext.for_run(retained.output_path, source["recovery_artifact_prefix"], deps=old_retention.deps)
    )
    retention_summary(retained, json.loads(json.dumps(asdict(retained_config))), baseline.record.retention_task_ids)
    completed_retention = ArtifactStep.adopt(
        f"evals/{PROTOCOL}-completed-retention",
        VERSION,
        retained.output_path,
        config={"producer_identity": retained.launch["producer_identity"], **retained.pins["producer_record"]},
    )
    outputs = post_sft_selection_stages(
        version=VERSION,
        checkpoints=[("sft", model)],
        outputs={"coding-sft": coding_evidence, "retention-sft": completed_retention},
        panel_sha256=baseline.record.panel_sha256,
        retention=old_retention.deps[0],
        retention_task_ids=baseline.record.retention_task_ids,
        parent_score=baseline.record.parent,
        study=StudyBaseline(
            PROTOCOL, baseline.record.parent, baseline.original_parent, baseline.record.retention_task_ids
        ),
    )
    selection = outputs["selection"]

    def selection_config(ctx: StepContext) -> CompletedSelectionConfig:
        return CompletedSelectionConfig(
            InterruptedSelectionConfig(
                selection.build_config(ctx),
                source["calibration_interruption_uri"],
                source["calibration_interruption_sha256"],
            ),
            input_pin,
            coding.launch["producer_identity"],
            retained.launch["producer_identity"],
        )

    final = replace(selection, build_config=selection_config, run=seal_completed_selection)
    return {**outputs, "selection": final, "terminal": final}


EXTRACTION_PROTOCOL = "russell-rsi-completed-coding-extraction-v1"


@dataclass(frozen=True)
class CompletedCodingEvidence:
    evidence: ArtifactStep
    producer: CompletedProducer
    model: ArtifactStep
    baseline: StudySelectionConfig


def completed_coding_evidence(config: dict, original: dict[str, ArtifactStep]) -> CompletedCodingEvidence:
    """Validate saved coding and prepare only the existing CPU evidence extractor."""
    source = pinned_json(config["source_config"])
    coding = completed_producer(config["coding"], "2026.10.06.9")
    old_coding, model = original["coding-sft"].deps
    terminal = original["terminal"]
    baseline = terminal.build_config(
        StepContext.for_run(
            terminal.path(source["recovery_artifact_prefix"]), source["recovery_artifact_prefix"], deps=terminal.deps
        )
    ).selection
    old_config = old_coding.build_config(
        StepContext.for_run(
            coding.output_path,
            source["recovery_artifact_prefix"],
            deps=old_coding.deps,
            runtime_args=old_coding.runtime_args,
        )
    )
    result = coding_result(coding, config["source_config"], old_config)
    panel_value = PinnedFile(source["panel_uri"], source["panel_sha256"]).read_json()
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])
    evidence_config = CodingEvidenceConfig(
        result["records_prefix"],
        tuple(result["run_ids"]),
        tuple(result["results_paths"]),
        artifact_identity(model),
        panel,
        "",
    )
    if "evidence" in coding.pins:
        evidence = pinned_json(coding.pins["evidence"])
        expected_evidence = coding_evidence_payload(evidence_config)
        if evidence != expected_evidence:
            raise ValueError("Completed coding evidence differs from its canonical archives")
        coding_evidence = ArtifactStep.adopt(
            f"documents/{PROTOCOL}-completed-coding",
            VERSION,
            str(StoragePath(coding.pins["evidence"]["uri"]).parent),
            config={"producer_identity": coding.launch["producer_identity"], **coding.pins["evidence"]},
        )
    else:
        coding_evidence = ArtifactStep(
            name=f"documents/{PROTOCOL}-coding-evidence",
            version=VERSION,
            artifact_type=Artifact,
            deps=(),
            build_config=lambda ctx: replace(evidence_config, output_path=ctx.output_path),
            run=collect_coding_eval_evidence,
        )
    return CompletedCodingEvidence(coding_evidence, coding, model, baseline)


def completed_coding_extraction_stages(
    config: dict, input_pin: PinnedFile, original: dict[str, ArtifactStep]
) -> dict[str, ArtifactStep]:
    """Extract coding evidence without requiring a completed retention producer."""
    if (
        config != input_pin.read_json()
        or set(config) != {"protocol", "version", "source_config", "coding"}
        or config["protocol"] != EXTRACTION_PROTOCOL
        or config["version"] != VERSION
    ):
        raise ValueError("Completed coding extraction changed its frozen protocol or schema")
    evidence = completed_coding_evidence(config, original).evidence
    return {"coding-sft": evidence, "terminal": evidence}
