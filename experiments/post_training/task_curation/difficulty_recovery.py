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
from pydantic import BaseModel, ConfigDict
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
    parent: "PreviousCampaign | None"


class CampaignParent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root: str
    serving_receipt: ContentEvidenceReference
    source_queue_sha256: str


def reconstruct_sealed_traffic(root: StoragePath, prepared: dict) -> None:
    """Rebuild a killed worker's traffic seal from its durable per-source receipts.

    The worker died before sealing, but every completed source's traffic archive
    is durably written and digest-bound. The manifest assembles exactly those
    receipts — no traffic is fabricated; the in-flight source's ungraded
    requests stay absent.
    """
    sources = []
    input_tokens = output_tokens = 0
    for row in prepared["sources"]:
        source_key = hashlib.sha256(row["sample"]["source_id"].encode()).hexdigest()
        aggregate = root / "trials" / source_key / "attempts.json"
        if not aggregate.exists():
            continue
        ledger = json.loads(aggregate.read_text())
        model = ContentEvidenceReference.model_validate(
            {"uri": ledger["model_requests"], "sha256": ledger["model_requests_sha256"]}
        )
        if file_sha256(StoragePath(model.uri)) != model.sha256:
            raise ValueError("Previous per-source traffic archive changed after publication")
        judge = ledger.get("judge_model_requests")
        if judge is not None:
            judge_ref = ContentEvidenceReference.model_validate(judge)
            if file_sha256(StoragePath(judge_ref.uri)) != judge_ref.sha256:
                raise ValueError("Previous per-source judge traffic archive changed after publication")
        input_tokens += ledger["input_tokens"]
        output_tokens += ledger["output_tokens"]
        sources.append(
            {
                "source_id": row["sample"]["source_id"],
                "model_requests": {"uri": ledger["model_requests"], "sha256": ledger["model_requests_sha256"]},
                "judge_model_requests": judge,
                "input_tokens": ledger["input_tokens"],
                "output_tokens": ledger["output_tokens"],
            }
        )
    if not sources:
        raise ValueError("Previous worker has not drained and sealed its final model traffic")
    manifest = {
        "reconstructed": True,
        "basis": (
            "assembled from the durable per-source traffic receipts of a worker that died before sealing; "
            "the in-flight source's ungraded traffic is absent"
        ),
        "sources": sources,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    (root / "trials" / "model-requests-final.json").write_bytes(data)


def previous_campaign(root: str, serving_sha256: str, source_queue_sha256: str) -> PreviousCampaign:
    """Bind a continuation to an explicitly selected prior serving and source queue."""
    chain = []
    seen = set()
    while True:
        path = StoragePath(root)
        if str(path) in seen:
            raise ValueError("Campaign ancestry contains a cycle")
        seen.add(str(path))
        if not (path / "trials" / "model-requests-final.json").exists():
            prepared_path = path / "prepared.json"
            if not prepared_path.exists():
                raise ValueError("Previous worker has not drained and sealed its final model traffic")
            reconstruct_sealed_traffic(path, json.loads(prepared_path.read_text()))
            if not (path / "trials" / "model-requests-final.json").exists():
                raise ValueError("Previous worker has not drained and sealed its final model traffic")
        serving_path = path / "serving.json"
        if file_sha256(serving_path) != serving_sha256:
            raise ValueError("Previous serving receipt differs from its requested digest")
        prepared = json.loads((path / "prepared.json").read_text())
        if prepared["source_queue_sha256"] != source_queue_sha256:
            raise ValueError("Continuation cannot change the frozen source queue")
        sources = [PreparedSource.model_validate(row) for row in prepared["sources"]]
        chain.append(
            (path, {source.sample.source_id: source for source in sources}, json.loads(serving_path.read_text()))
        )
        parent_path = path / "trials" / "parent-campaign.json"
        if not parent_path.exists():
            break
        parent = CampaignParent.model_validate_json(parent_path.read_text())
        if parent.source_queue_sha256 != source_queue_sha256 or parent.serving_receipt.uri != str(
            StoragePath(parent.root) / "serving.json"
        ):
            raise ValueError("Campaign ancestry differs from its source queue or serving binding")
        root, serving_sha256 = parent.root, parent.serving_receipt.sha256
    previous = None
    for path, sources, serving in reversed(chain):
        previous = PreviousCampaign(path, sources, serving, previous)
    assert previous is not None
    return previous


def recovered_source(
    previous: PreviousCampaign,
    source: PreparedSource,
    solver: Solver,
    identity: SolverIdentity,
    settings: SamplingSettings,
    parameters: dict,
    judge_identity: SolverIdentity | None,
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
    inherited = (
        recovered_source(previous.parent, source, solver, identity, settings, parameters, judge_identity)
        if previous.parent is not None
        else None
    )
    if measurement is None and inherited is not None:
        measurement = inherited.measurement
    refs = [] if inherited is None else list(inherited.attempt_receipts)
    if aggregate.exists():
        for ref in [
            ContentEvidenceReference.model_validate(row)
            for row in json.loads(aggregate.read_text()).get("attempt_receipts", [])
        ]:
            if ref not in refs:
                refs.append(ref)
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
                if judge_identity is None:
                    raise ValueError("Scored judged history requires a qualified judge identity")
                prior_judge = serving.get("judge")
                expected = judge_identity.model_dump(exclude={"serving_receipt"})
                if prior_judge is None or (
                    any(prior_judge.get(name) != value for name, value in expected.items())
                    or prior_judge.get("effective_temperature") != PROTOCOLS[PROTOCOL_ID].temperature
                ):
                    raise ValueError("Continuation cannot mix judge identities or unqualified judge temperature")
            if task.workload == "native" and serving.get("native_client_context_safety_buffer_tokens") != 0:
                raise ValueError("Native continuation requires the corrected full-context worker receipt")
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
