# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The identity counter reproduces shellbox's snapshot keys exactly."""

import base64
import gzip
import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

from daytona import Resources
from shellbox.image import DockerfileSource, image_source_key

from experiments.post_training.task_curation.daytona_identity import (
    AGENTIC_DEFAULT_IMAGE_KEY,
    ContextFile,
    DEFAULT_RESOURCES,
    count_identities,
    identities_from_lowered_specs,
    identities_from_task_tars,
    image_source_key_from_context,
    registry_image_key,
    resources_json,
    snapshot_identity,
)



def _stage(tmp_path: Path, files: dict[str, bytes | str]) -> Path:
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content) if isinstance(content, str) else target.write_bytes(content)
    return tmp_path


def test_context_key_matches_shellbox_on_a_real_staged_context(tmp_path):
    context = _stage(
        tmp_path / "context",
        {
            "Dockerfile": "FROM python:3.11-bookworm\nCOPY files/ /app/files\n",
            "files/a.csv": b"one\ntwo\n",
            "files/b.json": b'{"k": 1}',
        },
    )
    shellbox_key = image_source_key(DockerfileSource(context=context, dockerfile=context / "Dockerfile"))
    files = [
        ContextFile(name="Dockerfile", mode=(context / "Dockerfile").stat().st_mode, content=b"FROM python:3.11-bookworm\nCOPY files/ /app/files\n"),
        ContextFile(name="files", mode=(context / "files").stat().st_mode),
        ContextFile(name="files/a.csv", mode=(context / "files/a.csv").stat().st_mode, content=b"one\ntwo\n"),
        ContextFile(name="files/b.json", mode=(context / "files/b.json").stat().st_mode, content=b'{"k": 1}'),
    ]
    assert image_source_key_from_context("Dockerfile", files) == shellbox_key


def test_two_tasks_sharing_a_context_yield_one_snapshot(tmp_path):
    dockerfile = "FROM python:3.11-bookworm\n"
    first = _stage(tmp_path / "first", {"Dockerfile": dockerfile})
    second = _stage(tmp_path / "second", {"Dockerfile": dockerfile})
    key = image_source_key(DockerfileSource(context=first, dockerfile=first / "Dockerfile"))
    assert image_source_key(DockerfileSource(context=second, dockerfile=second / "Dockerfile")) == key
    entry = ContextFile(name="Dockerfile", mode=(first / "Dockerfile").stat().st_mode, content=dockerfile.encode())
    assert snapshot_identity(key) == snapshot_identity(image_source_key_from_context("Dockerfile", [entry]))


def test_task_tars_differ_when_data_is_baked_into_the_context(tmp_path):
    import gzip
    import io
    import tarfile

    def tar_with(dockerfile: str, data: bytes) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, content in (("environment/Dockerfile", dockerfile.encode()), ("environment/files/data.csv", data)):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return gzip.compress(buffer.getvalue())

    identical = tar_with("FROM ubuntu:24.04\n", b"a,b\n1,2\n")
    distinct = tar_with("FROM ubuntu:24.04\n", b"a,b\n3,4\n")
    count = count_identities(identities_from_task_tars([identical, identical, distinct]))
    assert count.distinct == 2
    assert sorted(count.sizes.values(), reverse=True) == [2, 1]


def test_lowered_specs_without_shapes_collapse_to_shared_identities():
    def spec(dockerfile: str) -> dict:
        return {
            "task": {
                "environment_requirements": {"docker_image": None},
                "resources": {
                    "verifier": [
                        {
                            "path": "environment/Dockerfile",
                            "source": {"kind": "inline_file", "content_base64": base64.b64encode(dockerfile.encode()).decode()},
                            "mode": "644",
                        }
                    ]
                },
            },
            "runtime": {"task_machine": None, "verifier_machine": None},
        }

    shared = spec("FROM aweaiteam/calibforge:tb2-verifier-base-v1@sha256:0f61\nWORKDIR /app\n")
    other = spec("FROM ghcr.io/penfever/smartie-grader@sha256:bcd0\nWORKDIR /app\n")
    count = count_identities(identities_from_lowered_specs([shared, shared, shared, other]))
    assert count.distinct == 3  # two verifier contexts + one shared default task-machine identity
    assert count.sizes[snapshot_identity(AGENTIC_DEFAULT_IMAGE_KEY, DEFAULT_RESOURCES)] == 4
    assert sorted(count.sizes.values(), reverse=True) == [4, 3, 1]


def test_resource_shapes_and_registry_references_distinguish_identities():
    plain = snapshot_identity(registry_image_key("registry.example/img@sha256:aa"))
    shaped = snapshot_identity(registry_image_key("registry.example/img@sha256:aa"), resources_json(cpu=4, memory_mb=32768))
    assert plain != shaped
    assert json.dumps(asdict(Resources()), sort_keys=True) == DEFAULT_RESOURCES
