# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from typing import Any

import pytest
from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.test_end_to_end import Envs, config, resources, script
from taskcompendium.models import TaskSpec

from taskforge.coordinator import run_workpiece
from taskforge.effects.events import Resolved, StageBoundary
from taskforge.effects.log import EventLog, Record
from taskforge.model.proposal import TaskProposal
from taskforge.model.workpiece import Terminal
from taskforge.stages.author import AuthorReference
from taskforge.stages.classify import replayable_grade
from taskforge.wiring import install


def reword(monkeypatch: pytest.MonkeyPatch, description: str) -> None:
    """Install a TaskCompendium whose ``JsonAnswer`` description reads ``description``."""
    schema = TaskSpec.model_json_schema

    def reworded() -> dict[str, Any]:
        installed = schema()
        installed["$defs"]["JsonAnswer"]["description"] = description
        return installed

    monkeypatch.setattr(TaskSpec, "model_json_schema", reworded)


def references(records: Sequence[Record]) -> list[tuple[str, bool]]:
    """Each recorded author reference, and whether it replayed."""
    return [
        (record.event.value, record.event.replayed)
        for record in records
        if isinstance(record.event, Resolved) and isinstance(record.event.request, AuthorReference)
    ]


def described(reference: str) -> str:
    return json.loads(reference)["TaskSpec"]["$defs"]["JsonAnswer"]["description"]


async def test_a_fresh_root_records_the_installed_wording_and_shows_it_to_the_author(
    proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch
) -> None:
    reword(monkeypatch, "Installed wording.")
    bundle = resources(Envs(), script())
    log = EventLog(replayable=replayable_grade)
    assert (await run_workpiece(proposal, config(), bundle, log, install=install)).terminal is Terminal.ACCEPTED
    ((reference, replayed),) = references(log.records)
    assert described(reference) == "Installed wording." and not replayed
    assert isinstance(bundle.inference, FakeInference)
    assert reference in bundle.inference.requests[0].messages[0].content


async def test_a_resumed_root_replays_its_recorded_reference_after_the_wording_changes(
    proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch
) -> None:
    reword(monkeypatch, "Recorded wording.")
    recorded = EventLog(replayable=replayable_grade)
    await run_workpiece(proposal, config(), resources(Envs(), script()), recorded, install=install)
    reword(monkeypatch, "Later wording.")
    resumed = resources(Envs(), ())  # an author call bought again exhausts this empty script
    # Cut before the author's boundary, so the resumed launch opens the author session again.
    cut = next(r.sequence for r in recorded.records if isinstance(r.event, StageBoundary) and r.event.workpiece.session)
    log = EventLog(replay_from=recorded.records[: cut - 1], replayable=replayable_grade)
    assert (await run_workpiece(proposal, config(), resumed, log, install=install)).terminal is Terminal.ACCEPTED
    assert isinstance(resumed.inference, FakeInference) and not resumed.inference.requests
    ((reference, replayed),) = references(log.records[cut - 1 :])
    assert described(reference) == "Recorded wording." and replayed
