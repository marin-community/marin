# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import pytest
from iris.cluster.client.bundle import collect_workspace_files

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
    provenance.write_text(json.dumps({"source_sha": "original-upstream-revision"}))
    monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", project)
    source = runtime_bundle.resolve_launcher_source()
    with runtime_bundle.build_runtime_bundle(source.digest) as workspace:
        yield source, workspace


def test_vendored_runtime_bundles_frozen_dependencies_and_assets_without_git(bundled_source):
    source, workspace = bundled_source
    assert (workspace / "uv.lock").read_bytes() == (source.root / "uv.lock").read_bytes()
    assert (workspace / "chat_templates/model.jinja2").read_text() == "{{ messages }}"
    assert runtime_bundle.validate_bundled_runtime(workspace) == source.digest


def test_sibling_harbor_source_is_verified_with_the_skyrl_runtime(bundled_source, tmp_path: Path):
    source, _ = bundled_source
    harbor = source.root.parent / "harbor"
    (harbor / "src/harbor").mkdir(parents=True)
    (harbor / "src/harbor/agents.py").write_text("pi_capture = True\n")
    (harbor / "pyproject.toml").write_text('[project]\nname = "harbor"\n')
    (harbor / "README.md").write_text("Harbor\n")
    (harbor / "LICENSE").write_text("Apache-2.0\n")
    documentation = harbor / "src/harbor/environments/apptainer/docs/workflow.md"
    documentation.parent.mkdir(parents=True)
    documentation.write_text("Not required at runtime\n")
    (source.root / "harbor").symlink_to("../harbor", target_is_directory=True)

    updated = runtime_bundle.resolve_launcher_source()
    assert updated.digest != source.digest
    with runtime_bundle.build_runtime_bundle(updated.digest) as workspace:
        bundled = workspace / "harbor/src/harbor/agents.py"
        assert bundled.read_text() == "pi_capture = True\n"
        assert runtime_bundle.validate_bundled_runtime(workspace) == updated.digest
        assert not (workspace / "harbor/src/harbor/environments/apptainer/docs/workflow.md").exists()
        uploaded = tmp_path / "uploaded"
        for file in collect_workspace_files(workspace):
            destination = uploaded / file.relative_to(workspace)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        assert runtime_bundle.validate_bundled_runtime(uploaded) == updated.digest
        bundled.write_text("pi_capture = False\n")
        with pytest.raises(RuntimeError, match="does not match its recorded identity"):
            runtime_bundle.validate_bundled_runtime(workspace)


def test_changed_bundled_training_config_is_rejected(bundled_source):
    _, workspace = bundled_source
    (workspace / "skyrl-train/skyrl_train/config/ppo_base_config.yaml").write_text("trainer: {max_steps: 999}\n")
    with pytest.raises(RuntimeError, match="does not match its recorded identity"):
        runtime_bundle.validate_bundled_runtime(workspace)


@pytest.mark.parametrize("path", ["cloud/iris/launch.py", "cloud/iris/new_runtime.py"])
def test_source_edits_get_a_new_identity_and_bundle_without_restaging(bundled_source, path):
    source, _ = bundled_source
    (source.root / path).write_text("updated = True\n")
    changed = runtime_bundle.resolve_launcher_source()
    assert changed.digest != source.digest
    with runtime_bundle.build_runtime_bundle(changed.digest) as workspace:
        assert (workspace / path).read_text() == "updated = True\n"
        assert runtime_bundle.validate_bundled_runtime(workspace) == changed.digest


@pytest.mark.parametrize("submission_succeeds", [True, False])
def test_runtime_workspace_is_removed_after_submission(bundled_source, submission_succeeds):
    source, _ = bundled_source
    outcome = nullcontext() if submission_succeeds else pytest.raises(RuntimeError, match="submission failed")
    with outcome, runtime_bundle.build_runtime_bundle(source.digest) as workspace:
        assert runtime_bundle.validate_bundled_runtime(workspace) == source.digest
        if not submission_succeeds:
            raise RuntimeError("submission failed")
    assert not workspace.exists()
    assert (source.root / "uv.lock").is_file()


def test_unrecorded_file_added_to_a_built_bundle_is_rejected(bundled_source):
    _, workspace = bundled_source
    (workspace / "cloud/iris/injected.py").write_text("extra = True\n")
    with pytest.raises(ValueError, match="do not match the selected runtime identity"):
        runtime_bundle.validate_bundled_runtime(workspace)
