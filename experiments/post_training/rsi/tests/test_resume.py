# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A restarted loop gives back the synthesized instances that are not in its feed, and knows its finished ideas."""

import pytest

# The loop runs in the marin-taskforge environment; other suites skip it.
pytest.importorskip("taskforge")

import asyncio
import io
import tarfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskforge.agentic import example_build
from taskforge.agentic.gate import Evidence
from taskforge.agentic.tools import TaskReview, ToolSpend
from taskforge.capability import CapabilityIdea

from experiments.post_training.rsi.feedback import SessionRecord, TaskRecord
from experiments.post_training.rsi.resume import known_ideas, untrained_instances

IDEA = CapabilityIdea(
    capability_id="failed-runs.ledgers",
    subject_id="failed-runs",
    subject_name="Failed runs",
    catalog_version="test",
    capability={"id": "failed-runs.ledgers"},
    prerequisite_edges=(),
    capability_hash="0",
)
# ShellSim has no Python, so the build program is a shell script that renames the example task.
BUILD = 'sed "s/ledger--00/$1/g" task.json > "$2"\n'


def workspace() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in (("task.json", example_build.task_spec("ledger--00").model_dump_json()), ("build.sh", BUILD)):
            member = tarfile.TarInfo(name)
            member.size = len(data.encode())
            archive.addfile(member, io.BytesIO(data.encode()))
    return buffer.getvalue()


def test_a_restarted_loop_gives_back_the_instances_that_are_not_in_the_feed(tmp_path: Path):
    output = tmp_path / "rsi"
    task_id = f"{IDEA.capability_id}.ledger--00"
    session = output / "ideas" / IDEA.capability_id
    (session / "tasks" / "ledger--00").mkdir(parents=True)
    (session / "tasks" / "ledger--00" / "workspace.tgz").write_bytes(workspace())
    review = TaskReview(failure_causes=(), option_lists=(), problems=(), summary="ok", accept=True)
    task = example_build.task_spec("ledger--00")
    saved = TaskRecord(task=task, evidence=Evidence(), review=review, program="build.sh")

    def write_record(held_out: bool) -> None:
        record = SessionRecord(
            idea=IDEA, held_out=held_out, tasks=(saved,), report=None, failure=None, agent_tokens=0, spend=ToolSpend()
        )
        (session / "session.json").write_text(record.model_dump_json())

    write_record(held_out=False)

    assert known_ideas(StoragePath(str(output))) == {IDEA.capability_id}
    in_feed = {task_id, f"{task_id}.train-0"}

    async def collect():
        stream = untrained_instances(StoragePath(str(output)), in_feed, 3, ShellSimMachineFactory())
        return [task.id async for task in stream]

    assert asyncio.run(collect()) == [f"{task_id}.train-1", f"{task_id}.train-2"]

    # The tasks of a held-out idea are only probes, also after a restart.
    write_record(held_out=True)
    assert asyncio.run(collect()) == []
