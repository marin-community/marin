# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from typing import cast

import pytest
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.harbor.snapshots import file_map_snapshot, task_snapshot
from taskcompendium.models import ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.tasktrove.harbor_export import (
    UnsupportedHarborTask,
    VerifierPayloadIdentity,
    archive_bytes,
    archive_file_mode,
    harbor_payload,
    harbor_record,
)


@pytest.mark.parametrize(
    "path,content,mode",
    [
        ("tests/test.sh", b"new wrapper", "755"),
        ("tests/Dockerfile", b"FROM different:tag", "644"),
        ("tests/helper.py", b"private checker", "755"),
        ("task.toml", b"new dispatch", "644"),
    ],
)
def test_verifier_identity_tracks_payload_recipe_and_modes(path, content, mode):
    files = {
        "tests/test.sh": b"original wrapper",
        "tests/Dockerfile": b"FROM source:tag",
        "tests/helper.py": b"private checker",
        "task.toml": b"dispatch",
        "instruction.md": b"public prompt",
    }
    original = VerifierPayloadIdentity()
    original.add(archive_bytes(files, {}))
    changed = VerifierPayloadIdentity()
    changed.add(archive_bytes({**files, path: content}, {path: mode}))
    assert changed.ref != original.ref
    equivalent = VerifierPayloadIdentity()
    equivalent.add(archive_bytes({**files, "instruction.md": b"different public prompt"}, {}))
    equivalent.add(archive_bytes(files, {}))
    assert equivalent.ref == original.ref


@pytest.mark.parametrize(
    "prompt,reference,leaked",
    [
        ("The answer is ALPHA\n  BETA GAMMA.", "alpha beta gamma", True),
        ("Name the first three letters.", "alpha beta gamma", False),
        ("Explain why red is a color.", "red", False),
    ],
)
def test_harbor_filters_disclosed_long_references_without_dropping_short_answers(prompt, reference, leaked):
    source = Source(dataset="fixture", revision="pinned", row="fixture/tasks.parquet:0", importer_revision="1")
    task = exact_answer_task(RawRow("fixture", source, {}), prompt=prompt, answers=(reference,), ignore_case=False)
    task = cast(TaskSpec, task)
    row = {"task_json": task.model_dump_json(), "source_row": source.row, "original_path": "fixture.tar.gz"}
    options = {
        "grader_image": "example.test/grader@sha256:" + "a" * 64,
        "family": "fixture",
        "fallback_actor_image": "python:3.12",
    }
    if leaked:
        with pytest.raises(UnsupportedHarborTask, match="gold_leak: expected value appears in instruction"):
            harbor_payload(row, **options)
    else:
        payload = harbor_payload(row, **options)
        assert payload.files["instruction.md"].startswith(prompt.encode())


def test_file_map_snapshot_matches_actual_export_archives():
    source = Source(dataset="fixture", revision="pinned", row="fixture/tasks.parquet:0", importer_revision="1")
    task = exact_answer_task(RawRow("fixture", source, {}), prompt="Name a color", answers=("red",), ignore_case=False)
    task = cast(TaskSpec, task)
    task = task.model_copy(
        update={
            "resources": ResourceGroups(
                worker=(inline_resource("app/input.txt", b"public input").model_copy(update={"mode": "600"}),),
                verifier=(inline_resource("helper.sh", b"private helper").model_copy(update={"mode": "700"}),),
                oracle=(inline_resource("solution/solve.sh", b"echo red").model_copy(update={"mode": "750"}),),
            )
        }
    )
    row = {"task_json": task.model_dump_json(), "source_row": source.row, "original_path": "fixture.tar.gz"}
    options = {
        "grader_image": "example.test/grader@sha256:" + "a" * 64,
        "family": "fixture",
        "fallback_actor_image": "python:3.12",
    }
    payload = harbor_payload(row, **options)
    record = harbor_record(row, **options)
    direct = {
        **file_map_snapshot(
            payload.files, {name: archive_file_mode(name, payload.modes) for name in payload.files}, "task"
        ),
        **file_map_snapshot(
            payload.solution,
            {name: archive_file_mode(name, payload.solution_modes) for name in payload.solution},
            "oracle",
        ),
    }
    archived = task_snapshot(
        record.source,
        record.path,
        "converted",
        task_binary=gzip.compress(gzip.decompress(record.task_binary), compresslevel=9, mtime=0),
        solution_binary=record.solution_binary,
    )
    assert direct == archived.files
    assert direct["task/environment/files/app/input.txt"].mode == 0o600
    assert direct["task/tests/helper.sh"].mode == 0o700
    assert direct["oracle/solution/solve.sh"].mode == 0o750
    assert direct["task/tests/test.sh"].mode == 0o755
