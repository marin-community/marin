# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import tarfile
import tomllib

import pytest

from experiments.post_training.task_curation.calibforge_fixtures import fixture_bundle
from experiments.post_training.task_curation.calibforge_release import transformed_task
from experiments.post_training.task_curation.tasktrove.harbor_export import archive_bytes


def test_packaging_preserves_private_verifier_and_checks_source_identity(tmp_path):
    task_id = "fixture/task"
    context = tmp_path / "source" / task_id / "environment"
    context.mkdir(parents=True)
    dockerfile = b"FROM ubuntu:24.04\nWORKDIR /app\nCOPY public.txt /app/input.txt\n"
    (context / "Dockerfile").write_bytes(dockerfile)
    (context / "public.txt").write_bytes(b"public input\n")
    instructions = [
        {"command": command, "args": args, "flags": [], "heredocs": []}
        for command, args in (
            ("from", ["ubuntu:24.04"]),
            ("workdir", ["/app"]),
            ("copy", ["public.txt", "/app/input.txt"]),
        )
    ]
    bundle = fixture_bundle(context, instructions)
    bundles = tmp_path / "bundles"
    destination = bundles / "tasks" / task_id
    destination.mkdir(parents=True)
    (destination / "calibforge-fixtures.tar.gz").write_bytes(bundle.payload)
    (destination / "calibforge-initialize.sh").write_text(bundle.setup_script)
    files = {
        "environment/Dockerfile": dockerfile,
        "environment/public.txt": b"public input\n",
        "instruction.md": b"Task instructions",
        "task.toml": b'[environment]\ndocker_image="old-task-image"\n',
        "tests/test.sh": b"private verifier sentinel",
    }
    row = {"path": task_id, "task_binary": archive_bytes(files, {name: "644" for name in files})}
    record = {
        "base_image": bundle.base_image,
        "workdir": bundle.workdir,
        "source_dockerfile_sha256": hashlib.sha256(dockerfile).hexdigest(),
        "payload_sha256": hashlib.sha256(bundle.payload).hexdigest(),
        "setup_sha256": hashlib.sha256(bundle.setup_script.encode()).hexdigest(),
    }
    result = transformed_task(row, record, bundles, tmp_path / "source", 120)
    with tarfile.open(fileobj=io.BytesIO(result)) as archive:
        assert archive.extractfile("tests/test.sh").read() == files["tests/test.sh"]
        assert archive.extractfile("instruction.md").read() == files["instruction.md"]
        config = tomllib.loads(archive.extractfile("task.toml").read().decode())
        assert config["environment"]["setup"] == {
            "command": "sh /setup_files/calibforge-initialize.sh",
            "timeout_sec": 120,
        }
        assert config["environment"]["workdir"] == "/app"
        assert "environment/public.txt" not in archive.getnames()
        payload = archive.extractfile("setup_files/calibforge-fixtures.tar.gz").read()
        with tarfile.open(fileobj=io.BytesIO(payload)) as public:
            assert public.getnames() == ["public.txt"]
            assert public.extractfile("public.txt").read() == b"public input\n"
    with pytest.raises(ValueError):
        transformed_task(row, {**record, "workdir": "/workspace"}, bundles, tmp_path / "source", 120)
    (context / "public.txt").write_bytes(b"changed mutable source")
    with pytest.raises(ValueError):
        transformed_task(row, record, bundles, tmp_path / "source", 120)
