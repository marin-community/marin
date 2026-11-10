# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from experiments.post_training.task_curation.calibforge_captured_fixtures import captured_payload
from experiments.post_training.task_curation.calibforge_equivalence import filesystem_manifest
from experiments.post_training.task_curation.calibforge_fixtures import fixture_bundle


@pytest.fixture(scope="module")
def parser_image():
    context = Path(__file__).parents[1] / "calibforge_parser"
    return subprocess.check_output(["docker", "build", "-q", str(context)], text=True).strip()


def _parse(dockerfile, parser_image):
    result = subprocess.check_output(
        ["docker", "run", "--rm", "-i", "--network", "none", parser_image],
        input=json.dumps({"id": "fixture", "dockerfile": dockerfile}).encode() + b"\n",
    )
    return json.loads(result)["instructions"]


def _snapshot(container):
    data = subprocess.check_output(["docker", "cp", container + ":/app", "-"])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for entry in archive:
            if entry.isfile():
                result[entry.name] = (entry.mode, entry.uid, entry.gid, archive.extractfile(entry).read())
            elif entry.isdir():
                result[entry.name] = (entry.mode, entry.uid, entry.gid, None)
            else:
                raise AssertionError(f"Unexpected fixture type: {entry.name}")
    return result


@pytest.mark.docker
@pytest.mark.parametrize(
    "pre_copy,post_copy,base",
    [
        ("", "", "ubuntu:24.04"),
        ("RUN mkdir -p /app/empty\n", "RUN chmod 600 /app/tool\n", "ubuntu:24.04"),
        ("", 'RUN ["touch", "/app/a file"]\n', "ubuntu:24.04"),
        ("", "RUN python3 /app/generate.py\nRUN rm -f /app/generate.py\n", "python:3.13-slim-bookworm"),
    ],
)
def test_fixture_initialization_matches_native_buildkit_copy(tmp_path, pre_copy, post_copy, base, parser_image):
    context = tmp_path / "context"
    context.mkdir()
    inputs = context / "files"
    inputs.mkdir()
    (inputs / "nested").mkdir()
    (inputs / "nested" / "input.txt").write_text("public fixture\n")
    (inputs / "generate.py").write_text(
        "from pathlib import Path\nPath('/app/generated.txt').write_text('generated fixture')\n"
    )
    executable = inputs / "tool"
    executable.write_text("#!/bin/sh\necho task-tool\n")
    executable.chmod(0o751)
    dockerfile = (
        f"FROM {base}\nWORKDIR /app\n"
        + pre_copy
        + 'COPY files/ /app/data/\nCOPY ["files/tool", "/app/tool"]\nCOPY files/generate.py /app/generate.py\n'
        + post_copy
    )
    (context / "Dockerfile").write_text(dockerfile)
    bundle = fixture_bundle(context, _parse(dockerfile, parser_image))
    image = subprocess.check_output(["docker", "build", "-q", str(context)], text=True).strip()
    reference = subprocess.check_output(["docker", "create", image], text=True).strip()
    candidate = subprocess.check_output(["docker", "create", base, "sleep", "infinity"], text=True).strip()
    try:
        subprocess.run(["docker", "start", candidate], check=True, capture_output=True)
        subprocess.run(["docker", "exec", candidate, "mkdir", "-p", "/setup_files"], check=True)
        payload = tmp_path / "calibforge-fixtures.tar.gz"
        payload.write_bytes(bundle.payload)
        script = tmp_path / "calibforge-initialize.sh"
        script.write_text(bundle.setup_script)
        for path in (payload, script):
            subprocess.run(["docker", "cp", str(path), candidate + ":/setup_files/"], check=True)
        subprocess.run(["docker", "exec", candidate, "sh", "/setup_files/calibforge-initialize.sh"], check=True)
        assert _snapshot(candidate) == _snapshot(reference)
        subprocess.run(["docker", "exec", candidate, "test", "!", "-e", "/setup_files"], check=True)
    finally:
        subprocess.run(["docker", "rm", "-f", reference, candidate], check=True, capture_output=True)


@pytest.mark.docker
def test_native_capture_preserves_generated_bytes_and_file_timestamps(tmp_path):
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.13-slim-bookworm\nWORKDIR /app\n"
        "RUN python3 -c \"import gzip; gzip.open('input.gz', 'wb').write(b'public input')\"\n"
        "RUN mkdir /data && echo public-data > /data/input.txt\n"
    )
    image = subprocess.check_output(
        ["docker", "build", "--platform", "linux/amd64", "-q", str(tmp_path)], text=True
    ).strip()
    payload, _ = captured_payload(image, ("app", "data"))
    archive = tmp_path / "captured.tar.gz"
    archive.write_bytes(payload)
    reference = subprocess.check_output(
        ["docker", "create", "--platform", "linux/amd64", image, "sleep", "infinity"], text=True
    ).strip()
    candidate = subprocess.check_output(
        ["docker", "create", "--platform", "linux/amd64", "python:3.13-slim-bookworm", "sleep", "infinity"], text=True
    ).strip()
    try:
        subprocess.run(["docker", "start", reference, candidate], check=True, capture_output=True)
        subprocess.run(["docker", "cp", str(archive), candidate + ":/captured.tar.gz"], check=True)
        subprocess.run(["docker", "exec", candidate, "tar", "-xzf", "/captured.tar.gz", "-C", "/"], check=True)
        expected = {k: v for k, v in filesystem_manifest(reference).items() if k.split("/")[0] in {"app", "data"}}
        actual = {k: v for k, v in filesystem_manifest(candidate).items() if k.split("/")[0] in {"app", "data"}}
        assert actual == expected
    finally:
        subprocess.run(["docker", "rm", "-f", reference, candidate], check=True, capture_output=True)
