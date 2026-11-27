# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only supersession evidence for a qualified solver request-policy change."""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from finestore.rl.difficulty import PROTOCOL_ID, SOLVER_MODELS, Solver
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.ledger import digest
from experiments.post_training.task_curation.difficulty.state import LedgerRow, Outcome
from experiments.post_training.task_curation.difficulty_output_budget import (
    LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID,
    require_solver_protocol,
)
from experiments.post_training.task_curation.difficulty_protocol import PINS
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


class ProtocolActivation(BaseModel):
    """Immutable qualification and terminal-cleanup bindings for the Large cutover."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    solver: Literal["large"] = "large"
    previous_protocol_id: str
    solver_protocol_id: str
    activated_at: str
    qualification: ArtifactRef
    native_cleanup: ArtifactRef


def activation_path(ledger: str) -> StoragePath:
    return StoragePath(ledger) / "v1/protocol-activations/large" / (LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID + ".json")


def verified_activation(record: ProtocolActivation) -> None:
    """Reject incomplete canaries, wrong frozen identities and live engine trees."""
    if (record.previous_protocol_id, record.solver_protocol_id) != (PROTOCOL_ID, LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID):
        raise ValueError("Activation does not bind the authorized Large protocol transition")
    activated = datetime.fromisoformat(record.activated_at)
    if activated.tzinfo is None:
        raise ValueError("Protocol activation requires a timezone")
    proof = read_ref(record.qualification)
    expected = {
        "protocol_id": record.solver_protocol_id,
        "model": SOLVER_MODELS[Solver.LARGE],
        "revision": PINS[Solver.LARGE].revision,
        "context_tokens": 65536,
        "safety_margin_tokens": 64,
        "minimum_output_floor_tokens": 16384,
        "tensor_parallel": 16,
        "data_parallel": 2,
        "request_count": 16,
    }
    if any(proof.get(key) != value for key, value in expected.items()) or any(
        proof.get(key) is not True
        for key in ("passed", "native_server_closed", "compilation_enabled", "early_stops_are_model_failures")
    ):
        raise ValueError("Large protocol activation requires the completed strict dynamic-output canary")
    completions = proof.get("completions", [])
    if (
        len(completions) != 16
        or {item["index"] for item in completions} != set(range(16))
        or any(item.get("qualified") is not True for item in completions)
    ):
        raise ValueError("Large qualification must include all sixteen qualified responses")
    if datetime.fromisoformat(proof["completed_at"]) > activated:
        raise ValueError("Protocol activation predates its completed qualification")
    native = read_ref(record.native_cleanup)
    jobs = {job["job_id"]: job for job in native["jobs"]}
    root = proof["attempt"].rsplit("/", 1)[0]
    if not proof.get("engine_jobs") or root not in jobs or jobs[root]["state"] != "JOB_STATE_SUCCEEDED":
        raise ValueError("Canary parent must be natively SUCCEEDED before activation")
    for job_id in [root, *proof["engine_jobs"]]:
        job = jobs[job_id]
        if (
            job["state"] not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_KILLED"}
            or job["failures"] != 0
            or not job["task_states"]
            or any(state not in {"TASK_STATE_SUCCEEDED", "TASK_STATE_KILLED"} for state in job["task_states"])
        ):
            raise ValueError("Canary engine tree is not terminal and failure-free")


def write_protocol_activation(ledger: str, record: ProtocolActivation) -> None:
    """Stage once before supersessions and the final parked-record installation."""
    verified_activation(record)
    target = conditional_object(str(activation_path(ledger)))
    payload = record.model_dump_json().encode()
    previous = target.read()
    if previous is not None:
        if previous.data != payload:
            raise ValueError("Protocol activation is immutable; preserve its original timestamp and evidence")
        return
    target.write(payload, expected_version=None)


def require_protocol_activation(ledger: str, protocol_id: str) -> ProtocolActivation | None:
    """Require explicit v2 selection and verified evidence before the first reconcile."""
    require_solver_protocol(Solver.LARGE, protocol_id)
    target = activation_path(ledger)
    if protocol_id == PROTOCOL_ID:
        if target.exists():
            raise ValueError("Large v2 activation exists; controller must explicitly select that protocol")
        return None
    if not target.exists():
        raise ValueError("Large v2 has no immutable protocol activation record")
    record = ProtocolActivation.model_validate_json(target.read_text())
    verified_activation(record)
    return record


def late_protocol_supersessions(rows: Sequence[LedgerRow], ledger: str, protocol_id: str) -> list[LedgerRow]:
    """Require pre-cutover links, then link late legacy finishes with the same activation."""
    if protocol_id == PROTOCOL_ID:
        return []
    activation = require_protocol_activation(ledger, protocol_id)
    assert activation is not None
    additions = protocol_supersessions(
        rows,
        solver=Solver.LARGE,
        previous_protocol_id=activation.previous_protocol_id,
        replacement_protocol_id=activation.solver_protocol_id,
        evidence_uri=str(activation_path(ledger)),
        activated_at=activation.activated_at,
    )
    originals = {row.attempt_id: row for row in rows}
    activated_at = datetime.fromisoformat(activation.activated_at)
    for row in additions:
        assert row.supersedes_attempt_id is not None
        if datetime.fromisoformat(originals[row.supersedes_attempt_id].recorded_at) <= activated_at:
            raise ValueError("Stage initial Large supersessions before the final FACET install and first reconcile")
    return additions


def protocol_supersessions(
    rows: Sequence[LedgerRow],
    *,
    solver: Solver,
    previous_protocol_id: str,
    replacement_protocol_id: str,
    evidence_uri: str,
    activated_at: str,
) -> list[LedgerRow]:
    """Link historical attempts to the new policy without rewriting grades or cost.

    The caller supplies the immutable activation record's timestamp on every retry.
    This function produces evidence only; it neither activates a protocol nor dispatches work.
    """
    require_solver_protocol(solver, previous_protocol_id)
    require_solver_protocol(solver, replacement_protocol_id)
    if previous_protocol_id == replacement_protocol_id or not evidence_uri:
        raise ValueError("Protocol supersession requires a distinct policy and its qualification evidence")
    recorded = {
        row.supersedes_attempt_id
        for row in rows
        if row.outcome == Outcome.PROTOCOL_SUPERSEDED and row.solver_protocol_id == replacement_protocol_id
    }
    return [
        LedgerRow(
            row.registry_revision,
            row.source_id,
            row.task_sha256,
            row.solver,
            digest(["protocol-superseded", row.attempt_id, replacement_protocol_id, evidence_uri]),
            Outcome.PROTOCOL_SUPERSEDED,
            None,
            evidence_uri,
            0.0,
            activated_at,
            task_id=row.task_id,
            evidence_uri=evidence_uri,
            error_identity="protocol_superseded_budget_limited",
            solver_protocol_id=replacement_protocol_id,
            supersedes_attempt_id=row.attempt_id,
        )
        for row in sorted(rows, key=lambda row: row.attempt_id)
        if row.solver == solver
        and row.solver_protocol_id == previous_protocol_id
        and row.outcome in {Outcome.SCORED, Outcome.UNSCORED}
        and row.task_id is not None
        and row.task_sha256 is not None
        and row.attempt_id not in recorded
    ]
