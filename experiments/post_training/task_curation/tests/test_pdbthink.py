# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.chat import chat_reply_status
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.oracle_filter import initial_verifier_requirements
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.pdbthink.tasks import BASE, REVISION, convert, decode, sources
from experiments.post_training.task_curation.environment_fidelity import environment_unused
from experiments.post_training.task_curation.oracle_filter import native_verifier_profile
from experiments.post_training.task_curation.pdbthink_verifier_repair import (
    VerifierAllocationDeviation,
    repair_cohort,
    repair_task,
)
from experiments.post_training.task_curation.remediation_publish import TASK_SCHEMA

FIXTURE = Path(__file__).with_name("fixtures") / "pdbthink" / "g02-native.json"


def native_task():
    row = json.loads(FIXTURE.read_text())
    task = convert(
        RawRow(
            row["metadata"]["path"],
            Source(
                dataset="open-athena/pdbthink-coordinate-tasks",
                revision=REVISION,
                row=f"{row['source_file']}:{row['source_row']}",
                importer_revision="fixture",
            ),
            {**row["decoded"], "metadata": row["metadata"]},
        ),
        ConversionContext({}, None),
    )
    return task, row


def test_native_conversion_preserves_public_prompt_and_private_verifier_oracle_bytes():
    task, row = native_task()
    original_files = row["decoded"]["files"]
    prompt = json.loads(base64.b64decode(original_files["prompt.json"]))
    assert [(event.role, event.content) for event in task.context.events] == [
        ("system", prompt["system_prompt"]),
        ("user", prompt["user_prompt"]),
    ]
    public = json.dumps([event.model_dump(mode="json") for event in task.context.events])
    assert "C:T670:N" not in public
    mapped = {resource.path: resource_bytes(resource) for resource in (*task.resources.verifier, *task.resources.oracle)}
    for name, encoded in original_files.items():
        target = (
            name.removeprefix("tests/")
            if name.startswith("tests/")
            else (name if name.startswith("solution/") else "source_archive/" + name)
        )
        assert mapped[target] == base64.b64decode(encoded)
    assert (
        environment_unused(
            {"environment_requirements": task.grader.environment.model_dump(mode="json")},
            original_files,
            bases={BASE: BASE},
        )
        is None
    )
    assert task.completion_protocol == "tool_free_single" and not task.resources.worker and not task.resources.all


@pytest.mark.parametrize(
    "finish_reason,fields,expected",
    [
        ("stop", {}, 1.0),
        ("length", {}, 0.0),
        ("stop", {"refusal": "refused"}, 0.0),
        (
            "tool_calls",
            {"tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": "invalid JSON"}}]},
            0.0,
        ),
        ("stop", {"annotations": [{"type": "web_search_call"}]}, 0.0),
    ],
)
def test_original_verifier_grades_host_text_and_status_from_converted_resources(
    tmp_path, finish_reason, fields, expected
):
    task, _ = native_task()
    for resource in task.resources.verifier:
        path = tmp_path / "tests" / resource.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resource_bytes(resource))
    original_answer = next(
        resource_bytes(resource).decode() for resource in task.resources.oracle if resource.path == "solution/answer.txt"
    )
    text = "Reasoning before the final field.\n" + original_answer
    conversation = tmp_path / "conversation.json"
    conversation.write_text(json.dumps([{"role": "assistant", "content": text}]))
    answer = tmp_path / "answer.txt"
    adapter = runpy.run_path(str(tmp_path / "tests/adapter.py"))
    adapter["write_answer"](conversation, answer)
    assert answer.read_bytes() == text.encode()
    payload = {"choices": [{"finish_reason": finish_reason, "message": {"content": text, **fields}}]}
    status = chat_reply_status(payload)
    assert task.grader.reply_status_path is not None
    (tmp_path / Path(task.grader.reply_status_path).name).write_text(
        json.dumps(
            {
                "truncated": status.truncated,
                "provider_refusal": status.provider_refusal,
                "tool_events": status.tool_events,
            }
        )
    )
    reward = tmp_path / "reward"
    subprocess.run(
        (sys.executable, str(tmp_path / "tests/verify.py")),
        check=True,
        env={
            **os.environ,
            "PDBTHINK_TEST_ROOT": str(tmp_path / "tests"),
            "PDBTHINK_ANSWER": str(answer),
            "PDBTHINK_REWARD_DIR": str(reward),
        },
        capture_output=True,
    )
    assert float((reward / "reward.txt").read_text()) == expected


def test_training_decoder_rejects_heldout_rows_before_reading_archives():
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="held-out"):
            decode({"split": split}, ConversionContext({}, None))
    source = sources()[0].config.source
    names = ["data/train-00000.parquet", "data/validation-00000.parquet", "data/test-00000.parquet"]
    assert [name for name in names if any(Path(name).match(pattern) for pattern in source.files)] == [
        "data/train-00000.parquet"
    ]


def test_verifier_deviation_exports_one_gib_without_changing_actor_or_payload(tmp_path):
    task, _ = native_task()
    assert task.execution_requirements is not None
    repaired = repair_task(task)
    assert repaired.model_dump(exclude={"verifier_execution_requirements"}) == task.model_dump()
    directory = tmp_path / "native"
    write_harbor_task(repaired, directory, task.execution_requirements)
    resolved = initial_verifier_requirements(task.execution_requirements, native_verifier_profile(directory))
    assert task.execution_requirements.memory_mb == repaired.execution_requirements.memory_mb == 512
    assert resolved == task.execution_requirements.model_copy(update={"memory_mb": 1024})


def test_native_verifier_repair_writes_complete_hash_bound_population_and_keeps_runtime(tmp_path):
    task, _ = native_task()
    assert task.execution_requirements is not None
    tasks = [task, task.model_copy(update={"id": task.id + "-second"})]
    rows = []
    for task in tasks:
        encoded = task.model_dump_json()
        rows.append(
            {
                "id": task.id,
                "path": task.id,
                "task_spec": encoded,
                "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "runtime": "{ }",
            }
        )
    source = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), source)
    output = tmp_path / "repair"
    receipt = repair_cohort(source, output, expected_sha256=file_sha256(StoragePath(str(source))), expected_tasks=2)
    written = pq.read_table(output / "tasks.parquet").to_pylist()
    ledger = [json.loads(line) for line in (output / "verifier-repair.jsonl").read_text().splitlines()]
    assert receipt["tasks"] == len(ledger) == len(written) == 2
    for before, after, evidence in zip(rows, written, ledger, strict=True):
        original, repaired = json.loads(before["task_spec"]), json.loads(after["task_spec"])
        allocation = task.execution_requirements.model_copy(update={"memory_mb": 1024})
        assert repaired.pop("verifier_execution_requirements") == allocation.model_dump(mode="json")
        assert original == repaired
        assert before["runtime"] == after["runtime"] == "{ }"
        assert evidence["original_task_sha256"] == before["task_sha256"]
        assert evidence["repaired_task_sha256"] == after["task_sha256"] != before["task_sha256"]
        deviation = VerifierAllocationDeviation.model_validate(evidence["allocation_deviation"])
        assert deviation.original == task.execution_requirements and deviation.approved == allocation
        assert receipt["allocation_deviations"] == [deviation.model_dump(mode="json")]
