# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from cloud.iris import runtime_bundle


@pytest.fixture
def bundled_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "vendor/skyrl"
    files = {
        "cloud/iris/launch.py": "def main(): pass\n",
        "cloud/iris/runtime_bundle_files.txt": "cloud/iris/launch.py\n",
        "skyrl-train/skyrl_train/config/ppo_base_config.yaml": "trainer: {max_steps: 2}\n",
        "skyrl-gym/skyrl_gym/__init__.py": "",
        "chat_templates/model.jinja2": "{{ messages }}",
        "pyproject.toml": '[project]\nname = "marinskyrl"\n',
        "uv.lock": "version = 1\n",
    }
    for name, content in files.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    provenance = project / runtime_bundle.PROVENANCE_FILE
    provenance.parent.mkdir(parents=True)
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    provenance.write_text(json.dumps({"files": hashes}))
    monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", project)
    source = runtime_bundle.resolve_launcher_source()
    workspace = runtime_bundle.build_runtime_bundle(source.digest)
    yield source, workspace
    shutil.rmtree(workspace)


def test_vendored_runtime_bundles_frozen_dependencies_and_assets_without_git(bundled_source):
    source, workspace = bundled_source
    assert (workspace / "uv.lock").read_bytes() == (source.root / "uv.lock").read_bytes()
    assert (workspace / "chat_templates/model.jinja2").read_text() == "{{ messages }}"
    assert runtime_bundle.validate_bundled_runtime(workspace) == source.digest


def test_changed_bundled_training_config_is_rejected(bundled_source):
    _, workspace = bundled_source
    (workspace / "skyrl-train/skyrl_train/config/ppo_base_config.yaml").write_text("trainer: {max_steps: 999}\n")
    with pytest.raises(RuntimeError, match="does not match its recorded identity"):
        runtime_bundle.validate_bundled_runtime(workspace)
