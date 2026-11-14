# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rebuild lowers native tasks onto the fixed bases with setup-at-start."""

import gzip
import io
import tarfile

import pytest

from experiments.post_training.task_curation.calibforge_rebuild import (
    DeferredRebuild,
    lowered_task_spec,
    parse_instructions,
    rebuild_task,
)


def native_task(dockerfile: str, files: dict[str, bytes] | None = None) -> bytes:
    """Build a gzip task tar in the native release layout."""
    members = {
        "environment/Dockerfile": dockerfile.encode(),
        "instruction.md": b"Solve the task.\n",
        "tests/test.sh": b"#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n",
        "tests/test_outputs.py": b"def test_placeholder():\n    assert True\n",
    }
    members["task.toml"] = b'[task]\nid = "fixture"\n\n[environment]\nworkdir = "/app"\n'
    for name, content in (files or {"data/input.csv": b"a,b\n1,2\n"}).items():
        members[f"environment/{name}"] = content
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(buffer.getvalue(), mtime=0)


def unpack(task_binary: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(task_binary))) as archive:
        return {
            name: archive.extractfile(member).read()
            for name, member in ((m.name, m) for m in archive.getmembers() if m.isfile())
        }


@pytest.mark.parametrize(
    "dockerfile,reason",
    [
        ("FROM registry.example/weird:latest\nWORKDIR /app\n", "unmapped_base"),
        ("FROM ubuntu:24.04\nRUN <<EOF\necho hi\nEOF\n", "heredoc"),
        ("FROM ubuntu:24.04\nRUN service nginx start\n", "unsupported_run"),
        ("FROM ubuntu:24.04\nENV TZ\nWORKDIR /app\n", "env_support"),
        ("FROM ubuntu:24.04\nRUN --mount=type=cache apt-get update\n", "run_flags_support"),
        ("FROM python:3.11-slim-bookworm AS build\nFROM python:3.11-slim-bookworm\n", "multistage"),
        ("FROM ubuntu:24.04\nCOPY files/absent.csv /app/\n", "missing_source"),
    ],
)
def test_unmapped_and_unsupported_recipes_defer_with_reasons(dockerfile, reason):
    with pytest.raises(DeferredRebuild, match=reason):
        rebuild_task(native_task(dockerfile))


def test_parse_instructions_handles_continuations_comments_and_json_run():
    instructions = parse_instructions(
        'FROM ubuntu:24.04\n# comment\nRUN echo one \\\n  two\nRUN ["apt-get", "install", "-y", "jq"]\n'
    )
    assert [item.command for item in instructions] == ["from", "run", "run"]
    assert instructions[1].args == ["echo", "one", "two"]
    assert instructions[2].json_form and instructions[2].args == ["apt-get", "install", "-y", "jq"]


def test_a_task_declaring_more_than_the_normalized_shape_defers():
    members = {
        "environment/Dockerfile": b"FROM ubuntu:24.04\nWORKDIR /app\n",
        "task.toml": b'[task]\nid = "fixture"\n\n[environment]\nworkdir = "/app"\nmemory = "8G"\n',
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    with pytest.raises(DeferredRebuild, match="larger_machine_shape"):
        rebuild_task(gzip.compress(buffer.getvalue(), mtime=0))


def test_lowered_spec_separates_public_fixtures_and_private_verifier_resources():
    rebuilt = rebuild_task(
        native_task(
            "FROM ubuntu:24.04\nWORKDIR /app\nCOPY files/ /app/files/\n",
            {"files/input.csv": b"a\n1\n"},
        )
    )
    spec = lowered_task_spec("fixture", rebuilt.task_binary)
    task = spec["task"]
    assert task["environment_requirements"]["docker_image"] == rebuilt.base_ref
    assert task["environment_requirements"]["command_semantics"] == "linux_process"
    staged = {resource["path"] for resource in task["resources"]["all"]}
    assert staged == {"setup_files/calibforge-fixtures.tar.gz", "setup_files/calibforge-initialize.sh"}
    assert {resource["path"] for resource in task["resources"]["verifier"]} == {"tests/test.sh", "tests/test_outputs.py"}
