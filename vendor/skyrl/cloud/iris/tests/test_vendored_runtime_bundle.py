# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from contextlib import nullcontext
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


def test_bundle_supplies_the_configured_task_factory_before_an_older_launcher(tmp_path):
    source = runtime_bundle.resolve_launcher_source()
    with runtime_bundle.build_runtime_bundle(source.digest) as workspace:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(workspace), str(source.root), environment.get("PYTHONPATH", "")]
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from pathlib import Path; from omegaconf import OmegaConf; "
                "import marinskyrl.task_machine_factories as module; "
                "factories = module.task_machine_factories(OmegaConf.create({})); "
                "print(json.dumps({'module': str(Path(module.__file__).resolve()), 'backends': sorted(factories)}))",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        observed = json.loads(completed.stdout)
        assert observed["module"] == str((workspace / "marinskyrl/task_machine_factories.py").resolve())
        assert "shellsim" in observed["backends"]
        assert "local" not in observed["backends"]
