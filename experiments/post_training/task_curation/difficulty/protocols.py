# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only supersession evidence for a qualified solver request-policy change."""

from collections.abc import Sequence

from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty.ledger import digest
from experiments.post_training.task_curation.difficulty.state import LedgerRow, Outcome
from experiments.post_training.task_curation.difficulty_output_budget import require_solver_protocol


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
