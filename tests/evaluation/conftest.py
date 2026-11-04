# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest
from marin.evaluation.harbor import runtime


@pytest.fixture
def harbor_runtime_sources(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("harbor-runtime")
    for directory in runtime._RUNTIME_DIRECTORIES:
        source = root / directory / "source.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("value = 1\n")
    for filename in runtime._RUNTIME_FILES:
        source = root / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("runtime input\n")
    (root / ".gitignore").write_text("ignored.py\n")
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    monkeypatch.setattr(runtime, "find_project_root", lambda _start: root)
    return root
