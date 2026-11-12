# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rebuild lowers native tasks onto the fixed bases with setup-at-start."""

import gzip
import io
import tarfile
import tomllib

import pytest
from taskcompendium.models import CommandSemantics, EnvironmentRequirements

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


def test_pinned_verifier_base_passes_through_and_data_files_stay_out_of_the_image():
    dockerfile = (
        "FROM aweaiteam/calibforge:tb2-verifier-base-v1@sha256:0f61b2776c40\n"
        "WORKDIR /app\n"
        "COPY files/data.csv /app/data.csv\n"
        "COPY make_fixtures.py /app/make_fixtures.py\n"
        "RUN python3 /app/make_fixtures.py\n"
    )
    rebuilt = rebuild_task(native_task(dockerfile, {"files/data.csv": b"x\n", "make_fixtures.py": b"print(1)\n"}))
    output = unpack(rebuilt.task_binary)
    assert b"FROM aweaiteam/calibforge:tb2-verifier-base-v1@sha256:0f61b2776c40" in output["environment/Dockerfile"]
    assert "environment/files" not in " ".join(_names(rebuilt))
    script = output["setup_files/calibforge-initialize.sh"].decode()
    assert "cp -a" in script and "make_fixtures.py" in script


def _names(rebuilt) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(rebuilt.task_binary))) as archive:
        return archive.getnames()


def test_unmapped_and_unsupported_recipes_defer_with_reasons():
    with pytest.raises(DeferredRebuild, match="unmapped_base"):
        rebuild_task(native_task("FROM registry.example/weird:latest\nWORKDIR /app\n"))
    with pytest.raises(DeferredRebuild, match="heredoc"):
        rebuild_task(native_task("FROM ubuntu:24.04\nRUN <<EOF\necho hi\nEOF\n"))
    with pytest.raises(DeferredRebuild, match="unsupported_run"):
        rebuild_task(native_task("FROM ubuntu:24.04\nRUN service nginx start\n"))
    with pytest.raises(DeferredRebuild, match="env_support"):
        rebuild_task(native_task("FROM ubuntu:24.04\nENV TZ\nWORKDIR /app\n"))


def test_package_flags_env_and_inline_env_normalize_without_deferring():
    rebuilt = rebuild_task(
        native_task(
            "FROM ubuntu:24.04\n"
            "ENV PYTHONDONTWRITEBYTECODE=1\n"
            "WORKDIR /app\n"
            "COPY files/ /app/files/\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends jq\n"
            "RUN rm -f /app/files/stale.csv\n"
            "RUN DEBIAN_FRONTEND=noninteractive apt-get install -y curl\n",
            {"files/input.csv": b"a\n1\n", "files/stale.csv": b"old\n"},
        )
    )
    script = unpack(rebuilt.task_binary)["setup_files/calibforge-initialize.sh"].decode()
    assert "export PYTHONDONTWRITEBYTECODE=1" in script
    assert "--no-install-recommends" in script
    assert "rm -f /app/files/stale.csv" in script
    assert (
        script.count("DEBIAN_FRONTEND=noninteractive") == 3
    )  # each apt segment (update + two installs; the inline one keeps its own)


def test_instruction_level_run_flags_still_defer():
    with pytest.raises(DeferredRebuild, match="run_flags_support"):
        rebuild_task(native_task("FROM ubuntu:24.04\nRUN --mount=type=cache apt-get update\n"))


def test_multistage_and_missing_sources_defer():
    with pytest.raises(DeferredRebuild, match="multistage"):
        rebuild_task(native_task("FROM python:3.11-slim-bookworm AS build\nFROM python:3.11-slim-bookworm\n"))
    with pytest.raises(DeferredRebuild, match="missing_source"):
        rebuild_task(native_task("FROM ubuntu:24.04\nCOPY files/absent.csv /app/\n"))


def test_parse_instructions_handles_continuations_comments_and_json_run():
    instructions = parse_instructions(
        'FROM ubuntu:24.04\n# comment\nRUN echo one \\\n  two\nRUN ["apt-get", "install", "-y", "jq"]\n'
    )
    assert [item.command for item in instructions] == ["from", "run", "run"]
    assert instructions[1].args == ["echo", "one", "two"]
    assert instructions[2].json_form and instructions[2].args == ["apt-get", "install", "-y", "jq"]


def test_the_machine_shape_is_normalized_so_snapshots_collapse():
    rebuilt = rebuild_task(
        native_task(
            "FROM ubuntu:24.04\nWORKDIR /app\nCOPY files/ /app/files/\n",
            {"files/input.csv": b"a\n1\n"},
        )
    )
    config = tomllib.loads(unpack(rebuilt.task_binary)["task.toml"].decode())
    environment = config["environment"]
    assert environment["cpus"] == 2
    assert environment["memory"] == "4G"
    assert environment["storage"] == "10G"


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


def test_a_lowered_spec_stages_fixtures_through_setup_commands():
    """The rollout runner installs resources then runs setup_commands; both are present."""
    rebuilt = rebuild_task(
        native_task(
            "FROM ubuntu:24.04\nWORKDIR /app\nCOPY files/ /app/files/\n",
            {"files/input.csv": b"a\n1\n"},
        )
    )
    spec = lowered_task_spec("fixture", rebuilt.task_binary)
    task = spec["task"]
    assert task["environment_requirements"]["setup_commands"] == ["sh /setup_files/calibforge-initialize.sh"]
    assert task["environment_requirements"]["docker_image"] == rebuilt.base_ref
    assert task["environment_requirements"]["command_semantics"] == "linux_process"
    staged = {resource["path"] for resource in task["resources"]["all"]}
    assert staged == {"setup_files/calibforge-fixtures.tar.gz", "setup_files/calibforge-initialize.sh"}
    assert {resource["path"] for resource in task["resources"]["verifier"]} == {"tests/test.sh", "tests/test_outputs.py"}
    assert spec["runtime"]["task_machine"] == {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}


def test_an_unpinned_base_cannot_reach_a_lowered_spec():
    """A floating tag is a schema error, so a missed pin fails loudly."""
    with pytest.raises(ValueError, match="sha256"):
        EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="ubuntu:24.04")
