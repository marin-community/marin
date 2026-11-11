# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rebuild lowers native tasks onto the fixed bases with setup-at-start."""

import gzip
import io
import tarfile

import pytest

from experiments.post_training.task_curation.calibforge_rebuild import (
    FIXED_BASES,
    DeferredRebuild,
    parse_instructions,
    rebuild_task,
)


def native_task(dockerfile: str, files: dict[str, bytes] | None = None) -> bytes:
    """Build a gzip task tar in the native release layout."""
    members = {"environment/Dockerfile": dockerfile.encode()}
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


def test_dependency_installs_move_into_the_setup_script():
    dockerfile = (
        "FROM ubuntu:24.04\n"
        "WORKDIR /app\n"
        "COPY files/ /app/files/\n"
        "RUN apt-get update && apt-get install -y jq\n"
    )
    rebuilt = rebuild_task(native_task(dockerfile, {"files/input.csv": b"a\n1\n"}))
    assert rebuilt.base_ref == "ubuntu:24.04"
    script = unpack(rebuilt.task_binary)["setup_files/calibforge-initialize.sh"].decode()
    assert "DEBIAN_FRONTEND=noninteractive" in script
    assert "apt-get" in script and "jq" in script
    assert "tar -xzf /setup_files/calibforge-fixtures.tar.gz" in script
    output = unpack(rebuilt.task_binary)
    assert output["environment/Dockerfile"] == b"FROM ubuntu:24.04\nWORKDIR /app\n"
    toml = output["task.toml"].decode()
    assert 'docker_image = "ubuntu:24.04"' in toml
    assert "setup" in toml


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


def test_the_ten_from_census_maps_onto_nine_bases():
    census = [
        "ubuntu:24.04",
        "aweaiteam/calibforge:tb2-verifier-base-v1",
        "python:3.13-slim-bookworm",
        "python:3.12-slim-bookworm",
        "python:3.11-slim-bookworm",
        "python:3.11-slim",
        "python:3.10-slim-bookworm",
        "debian:12-slim",
        "debian:13.0-slim",
        "aweaiteam/calibforge:fasttext-base-v1",
    ]
    refs = {FIXED_BASES[from_ref.split("@")[0]] for from_ref in census}
    assert len(refs) == 9  # python:3.11-slim folds into python:3.11-slim-bookworm
