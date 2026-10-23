# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A preempted run continues from its last complete adapter and gives back the synthesized tasks it did not train."""

import pytest

# The loop runs in the marin-taskforge environment; other suites skip it.
pytest.importorskip("taskforge")

import asyncio
import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskforge.agentic import example_build
from taskforge.agentic.gate import Evidence
from taskforge.agentic.tools import TaskReview
from taskforge.capability import CapabilityIdea

from experiments.post_training.baby_rsi.resume import OPTIMIZER_FILE, known_ideas, read_progress, untrained_instances

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


def record(task_id: str, model: str, label: str) -> str:
    return json.dumps({"task_id": task_id, "model": model, "label": label}) + "\n"


def test_a_resumed_run_takes_the_last_complete_adapter_and_skips_what_it_trained(tmp_path: Path):
    output = tmp_path / "rsi"
    for step, complete in ((1, True), (2, True), (3, False)):
        adapter = output / "adapters" / f"step-{step:04d}"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        if complete:
            (adapter / OPTIMIZER_FILE).write_bytes(b"state")
    task_id = f"{IDEA.capability_id}.ledger--00"
    (output / "groups.jsonl").write_text(
        record("pool-1", "base", "calendar")
        + record(task_id, "step-0001", f"synthesized:{IDEA.capability_id}")
        + record(f"{task_id}.train-0", "step-0001", f"synthesized:{IDEA.capability_id}")
        + record("pool-2", "step-0002", "structural")  # Step 3 sampled it with adapter 2; step 3 did not finish.
    )
    directory = output / "ideas" / IDEA.capability_id / "tasks" / "ledger--00"
    directory.mkdir(parents=True)
    (output / "ideas" / IDEA.capability_id / "idea.json").write_text(IDEA.model_dump_json())
    (directory / "task.json").write_text(example_build.task_spec("ledger--00").model_dump_json())
    (directory / "evidence.json").write_text(json.dumps(asdict(Evidence())))
    review = TaskReview(failure_causes=(), option_lists=(), problems=(), summary="ok", accept=True)
    (directory / "review.json").write_text(review.model_dump_json())
    (directory / "program.txt").write_text("build.sh")
    (directory / "workspace.tgz").write_bytes(workspace())

    progress = read_progress(StoragePath(str(output)), tmp_path / "adapters")
    assert progress.step == 2
    assert [entry["task_id"] for entry in progress.records] == ["pool-1", task_id, f"{task_id}.train-0"]
    assert progress.pool_tasks == 1
    assert (tmp_path / "adapters" / "step-0002" / OPTIMIZER_FILE).read_bytes() == b"state"
    assert known_ideas(StoragePath(str(output))) == {IDEA.capability_id}

    async def collect():
        stream = untrained_instances(StoragePath(str(output)), progress.trained_ids, 3, ShellSimMachineFactory())
        return [task.id async for task in stream]

    assert asyncio.run(collect()) == [f"{task_id}.train-1", f"{task_id}.train-2"]
