# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Committed stage visits, checked as rows arrive: at claim, on reconstruction and before a live commit."""

from collections.abc import Sequence
from typing import cast

from rolloutengine.models import GradedRollout

from taskforge.effects.events import Committed, Resolved, StageCompleted, digest, event_digest
from taskforge.effects.issues import LogFailure
from taskforge.effects.log import Record, physical
from taskforge.model.review import Cause
from taskforge.model.stage import Stage
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.classify import budget_exhausted, trial_cause


class Commitments:
    """A journal's committed visits, each checked against the rows before it; claim and run start share this pass."""

    def __init__(self, records: Sequence[Record] = ()) -> None:
        self.visits: dict[str, Committed] = {}
        self.rounds: dict[int, Committed] = {}  # the latest commitment of each evidence index
        for record in records:
            if isinstance(event := record.event, StageCompleted) and event.committed is not None:
                self.add(records, record.sequence, event)

    def add(self, records: Sequence[Record], sequence: int, event: StageCompleted) -> None:
        """Check and register the committed marker at row ``sequence`` against the rows before it.

        Raises:
            LogFailure: the marker is misplaced or conflicts with an earlier one, or a trial is not the resolution its
                own visit issued for that slot (a confirmation's carried trial: the shipped round's committed one).
        """
        commit = event.committed
        assert commit is not None
        markers = (i for i in range(sequence - 2, -1, -1) if isinstance(records[i].event, StageCompleted))
        end, start = next(markers, -1), next(markers, -1)
        visit = records[end].event if end >= 0 else None
        stage = visit.stage if isinstance(visit, StageCompleted) else None
        where = f"Row {sequence} commits visit {commit.visit[:12]}"
        if stage not in (Stage.VALIDATE, Stage.CONFIRM) or event.stage is not Stage.REVIEW or event.terminal:
            raise LogFailure(f"{where} outside a continuing review right after its visit")
        if self.visits.get(commit.visit, commit) != commit:
            raise LogFailure(f"{where} unlike its earlier commitment")
        trials = [trial.trial for trial in commit.result.solver]
        if trials != sorted(set(trials)):
            raise LogFailure(f"{where} with its trials out of order or repeated")
        issued = {  # each slot's resolution as the visit itself emitted it, replayed or bought
            (row.request.trial, row.request.attempt): row
            for record in records[start + 1 : end]
            if isinstance(row := record.event, Resolved) and isinstance(row.request, SolverRollout)
        }
        shipped = self.rounds.get(commit.result.confirms) if commit.result.confirms is not None else None
        carried = {trial.trial: trial for trial in shipped.result.solver} if shipped and stage is Stage.CONFIRM else {}
        for trial in commit.result.solver:
            if (own := issued.get((trial.trial, commit.attempt))) is None:
                if carried.get(trial.trial) == trial or trial == budget_exhausted(trial.trial):
                    continue
                raise LogFailure(f"{where}: trial {trial.trial} names a slot its visit never resolved")
            key = (event_digest(own.request), own.fingerprints)
            found = None if trial.receipt is None else physical(records, trial.receipt, key, sequence)
            if found is None or not isinstance(result := found.value, GradedRollout):
                raise LogFailure(f"{where}: trial {trial.trial}'s receipt names no resolution its visit issued")
            cause = trial_cause(result, cast(SolverRollout, own.request).spec.task)
            if (trial.cause, digest(trial.grade)) != (cause, digest(None if cause is Cause.HELD else result.grade)):
                raise LogFailure(f"{where}: trial {trial.trial}'s grade or cause differs from its physical resolution")
        self.visits[commit.visit] = self.rounds[commit.attempt] = commit
