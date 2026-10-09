# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.harbor.export import UnsupportedHarborTask, VerifierPayloadIdentity, archive_bytes, harbor_payload
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow


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
    assert isinstance(task, TaskSpec)
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
