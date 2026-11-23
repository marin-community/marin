# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Committed stage visits, checked as rows arrive: at claim, on reconstruction and before a live commit."""

from collections.abc import Sequence

from taskforge.effects.events import Committed, Resolved, StageCompleted, digest, event_digest
from taskforge.effects.issues import LogFailure
from taskforge.effects.log import Record
from taskforge.model.stage import Stage
from taskforge.resources.rollouts import SolverRollout


class Commitments:
    """A journal's committed visits, each checked against the rows before it; claim and run start share this pass."""

    def __init__(self, records: Sequence[Record] = ()) -> None:
        self.visits: dict[str, Committed] = {}
        self.solver: dict[int, tuple[SolverRollout, str]] = {}  # physical resolutions by row: request, replay key
        self.stage: Stage | None = None
        self.last = 0
        self.extend(records)

    def extend(self, records: Sequence[Record]) -> None:
        """Add the rows past the last one added; ``records`` holds every row, numbered from 1."""
        for record in records[self.last :]:
            self.add(record.sequence, record.event)

    def add(self, sequence: int, event: object) -> None:
        """Place row ``sequence``; a commitment that is misplaced, conflicting or names a wrong row is ``LogFailure``."""
        self.last = sequence
        if isinstance(event, Resolved) and isinstance(request := event.request, SolverRollout) and not event.replayed:
            self.solver[sequence] = (request, digest((event_digest(request), event.fingerprints)))
        if not isinstance(event, StageCompleted):
            return
        previous, self.stage = self.stage, event.stage
        if (commit := event.committed) is None:
            return
        where = f"Row {sequence} commits visit {commit.visit[:12]}"
        if previous not in (Stage.VALIDATE, Stage.CONFIRM) or event.stage is not Stage.REVIEW or event.terminal:
            raise LogFailure(f"{where} outside a continuing review right after its visit")
        if self.visits.get(commit.visit, commit) != commit:
            raise LogFailure(f"{where} unlike its earlier commitment")
        trials = [trial.trial for trial in commit.result.solver]
        receipts = [trial.receipt.sequence for trial in commit.result.solver if trial.receipt is not None]
        if trials != sorted(set(trials)) or len(set(receipts)) != len(receipts):
            raise LogFailure(f"{where} with its trials out of order or a receipt named twice")
        for trial in commit.result.solver:
            if (receipt := trial.receipt) is None:
                continue
            request, key = self.solver.get(receipt.sequence, (None, ""))
            # A confirmation's carried baseline was bought by an earlier attempt; every other slot by its own.
            if (
                request is None
                or receipt.key != key
                or request.trial != trial.trial
                or not (
                    request.attempt == commit.attempt or (previous is Stage.CONFIRM and request.attempt < commit.attempt)
                )
            ):
                raise LogFailure(f"{where}: trial {trial.trial}'s receipt names no physical resolution of its slot")
        self.visits[commit.visit] = commit
