# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.evaluation.images.tau3.build import build_image


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Docker CLI boundary that persists pushed manifests in a local registry file."""
    state = tmp_path / "registry.json"
    state.write_text("{}")
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "state = pathlib.Path(os.environ['TEST_REGISTRY'])\n"
        "images = json.loads(state.read_text())\n"
        "if sys.argv[1:4] == ['buildx', 'imagetools', 'inspect']:\n"
        "    if os.environ.get('TEST_REGISTRY_DENIED'):\n"
        "        sys.stderr.write('unauthorized')\n"
        "        sys.exit(1)\n"
        "    tag = sys.argv[4]\n"
        "    if tag not in images:\n"
        "        sys.stderr.write('manifest unknown')\n"
        "        sys.exit(1)\n"
        "    print(json.dumps({'digest': images[tag]}))\n"
        "elif sys.argv[1:3] == ['buildx', 'build']:\n"
        "    tag = next(arg.removeprefix('--tag=') for arg in sys.argv if arg.startswith('--tag='))\n"
        "    images[tag] = 'sha256:' + 'a' * 64\n"
        "    state.write_text(json.dumps(images))\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TEST_REGISTRY", str(state))
    return state


def test_build_image_reuses_pushed_dependency_identity(registry: Path) -> None:
    first = build_image("example.test/rollout-images")
    assert first["main"] == "example.test/rollout-images@sha256:" + "a" * 64

    # A separately published digest must be returned without overwriting the tag.
    registry.write_text(json.dumps({first["tag"]: "sha256:" + "b" * 64}))
    second = build_image("example.test/rollout-images")
    assert second["main"] == "example.test/rollout-images@sha256:" + "b" * 64


def test_build_image_parent_change_publishes_separate_tag(registry: Path) -> None:
    first = build_image("example.test/rollout-images")
    second = build_image("example.test/rollout-images", "example.test/parent@sha256:" + "c" * 64)
    assert first["tag"] != second["tag"]
    assert set(json.loads(registry.read_text())) == {first["tag"], second["tag"]}


def test_build_image_registry_auth_failure_does_not_publish(registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_REGISTRY_DENIED", "1")
    with pytest.raises(subprocess.CalledProcessError):
        build_image("example.test/rollout-images")
    assert json.loads(registry.read_text()) == {}
