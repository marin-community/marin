# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate immutable evidence before continuing a frozen difficulty campaign."""

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import (
    PROTOCOL_ID,
    PROTOCOLS,
    DifficultyAttempt,
    DifficultyMeasurement,
    SamplingSettings,
    Solver,
    SolverIdentity,
)
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, network_policy_qualified, read_source_runtime

from experiments.post_training.task_curation.difficulty_prepared import PreparedSource


@dataclass(frozen=True)
class RecoveredSource:
    attempts: list[DifficultyAttempt]
    attempt_receipts: list[ContentEvidenceReference]
    measurement: DifficultyMeasurement | None


@dataclass(frozen=True)
class PreviousCampaign:
    root: StoragePath
    sources: dict[str, PreparedSource]
    serving: dict


def previous_campaign(root: str, serving_sha256: str, source_queue_sha256: str) -> PreviousCampaign:
    """Bind a continuation to an explicitly selected prior serving and source queue."""
    path = StoragePath(root)
    if not (path / "trials" / "model-requests-final.json").exists():
        raise ValueError("Previous worker has not drained and sealed its final model traffic")
    serving_path = path / "serving.json"
    if file_sha256(serving_path) != serving_sha256:
        raise ValueError("Previous serving receipt differs from its requested digest")
    prepared = json.loads((path / "prepared.json").read_text())
    if prepared["source_queue_sha256"] != source_queue_sha256:
        raise ValueError("Continuation cannot change the frozen source queue")
    sources = [PreparedSource.model_validate(row) for row in prepared["sources"]]
    return PreviousCampaign(
        path, {source.sample.source_id: source for source in sources}, json.loads(serving_path.read_text())
    )


def recovered_source(
    previous: PreviousCampaign,
    source: PreparedSource,
    solver: Solver,
    identity: SolverIdentity,
    settings: SamplingSettings,
    parameters: dict,
    judge_identity: SolverIdentity,
) -> RecoveredSource:
    """Recover every completed attempt, including tasks whose source aggregate was interrupted."""
    serving = previous.serving
    revision = (
        f"unexposed:{serving['tools_thinking_probe']['completion']['system_fingerprint']}"
        if solver == Solver.HOSTED
        else serving["requested_revision"]
    )
    if (
        serving["protocol"] != PROTOCOL_ID
        or serving["solver"] != solver.value
        or serving["model"] != identity.model
        or revision != identity.revision
        or serving["tokenizer_revision"] != identity.tokenizer_revision
        or serving["chat_template_sha256"] != identity.chat_template_sha256
        or serving["sampling"] != parameters
    ):
        raise ValueError("Continuation cannot mix solver, revision, tokenizer, template or sampling policies")
    old = previous.sources[source.sample.source_id]
    if old.sample != source.sample or old.sample_receipt != source.sample_receipt:
        raise ValueError("Continuation cannot change a source's frozen sample")
    old_tasks = {task.task_id: task for task in old.tasks}
    tasks = {task.task_id: task for task in source.tasks}
    if set(old_tasks) != set(tasks) or set(tasks) != {task.task_id for task in source.sample.tasks}:
        raise ValueError("Prepared task identities differ from the frozen sample")
    for task in source.tasks:
        if task.model_dump(exclude={"path", "reason"}) != old_tasks[task.task_id].model_dump(exclude={"path", "reason"}):
            raise ValueError("Continuation cannot change a task's runtime, backend or grader")
    key = hashlib.sha256(source.sample.source_id.encode()).hexdigest()
    base = previous.root / "trials" / key
    aggregate = base / "attempts.json"
    measurement_path = base / "measurement.json"
    measurement = (
        DifficultyMeasurement.model_validate_json(measurement_path.read_text()) if measurement_path.exists() else None
    )
    if measurement is not None and (
        measurement.sample != source.sample
        or measurement.sample_receipt != source.sample_receipt
        or measurement.solver != solver
        or measurement.settings != settings
        or measurement.model is None
        or measurement.model.model_dump(exclude={"serving_receipt"}) != identity.model_dump(exclude={"serving_receipt"})
    ):
        raise ValueError("Previous measurement differs from the continuation's source or model contract")
    refs = []
    if aggregate.exists():
        refs = [
            ContentEvidenceReference.model_validate(row)
            for row in json.loads(aggregate.read_text()).get("attempt_receipts", [])
        ]
    # Initial campaigns already persist individual attempts before aggregating a source.
    for path in sorted((base / "attempts" / "*" / "*" / "attempt.json").glob(), key=str):
        ref = ContentEvidenceReference(uri=str(path), sha256=file_sha256(path))
        if ref not in refs:
            refs.append(ref)
    attempts = []
    selected = {task.task_id for task in source.sample.tasks}
    scored = set()
    for ref in refs:
        path = StoragePath(ref.uri)
        if file_sha256(path) != ref.sha256:
            raise ValueError("Previous attempt evidence changed after publication")
        payload = json.loads(path.read_text())
        attempt = DifficultyAttempt.model_validate({name: payload[name] for name in DifficultyAttempt.model_fields})
        if attempt.task_id not in selected:
            raise ValueError("Previous attempt is outside the selected sample")
        if attempt.task_id in scored:
            raise ValueError("Previous campaign retried a task after its first scored result")
        if attempt.status == "scored":
            task = tasks[attempt.task_id]
            if task.judge_model is not None:
                prior_judge = serving.get("judge")
                expected = judge_identity.model_dump(exclude={"serving_receipt"})
                if prior_judge is None or (
                    any(prior_judge.get(name) != value for name, value in expected.items())
                    or prior_judge.get("effective_temperature") != PROTOCOLS[PROTOCOL_ID].temperature
                ):
                    raise ValueError("Continuation cannot mix judge identities or unqualified judge temperature")
            archive = StoragePath(payload["archive"])
            data = archive.read_bytes()
            if hashlib.sha256(data).hexdigest() != payload["archive_sha256"]:
                raise ValueError("Previous scored trial archive changed after publication")
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as stream:
                result = stream.extractfile("trial/result.json")
                if result is None:
                    raise ValueError("Previous scored trial has no archived result")
                archived_result = json.load(result)
                if archived_result.get("verifier_result", {}).get("rewards", {}).get("reward") != attempt.reward:
                    raise ValueError("Previous scored reward differs from its source's archived grade")
                if task.workload == "task_spec":
                    if task.path is None:
                        raise ValueError("Current task packet is unavailable for network qualification")
                    policy_file = stream.extractfile("trial/network-policy.json")
                    if policy_file is None:
                        raise ValueError("Previous scored trial has no network policy receipt")
                    policy = TaskNetworkPolicyReceipt.model_validate_json(policy_file.read())
                    binding = read_source_runtime(Path(task.path))
                    if (
                        binding is None
                        or binding.runtime_sha256 != task.runtime_sha256
                        or not network_policy_qualified(binding, policy)
                    ):
                        raise ValueError("Previous scored trial does not qualify against the original source runtime")
            scored.add(attempt.task_id)
        attempts.append(attempt)
    return RecoveredSource(attempts, refs, measurement)
