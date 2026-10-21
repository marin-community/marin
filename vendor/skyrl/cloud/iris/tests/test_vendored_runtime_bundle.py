# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
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
        "pyproject.toml": '[project]\nname = "marinskyrl"\n[tool.uv.sources]\nmarin-rolloutengine = { path = "../../lib/rolloutengine", editable = true }\n',
        "uv.lock": "version = 1\n",
    }
    for name, content in files.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    module = tmp_path / "lib/rolloutengine"
    (module / "src/rolloutengine").mkdir(parents=True)
    (module / "src/rolloutengine/__init__.py").write_text("value = 1\n")
    (module / "pyproject.toml").write_text(
        '[project]\nname = "marin-rolloutengine"\n[tool.hatch.build.targets.wheel]\npackages = ["src/rolloutengine"]\n'
    )
    provenance = project / runtime_bundle.PROVENANCE_FILE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"source_sha": "original-upstream-revision"}))
    monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", project)
    source = runtime_bundle.resolve_launcher_source()
    with runtime_bundle.build_runtime_bundle(source.digest) as workspace:
        yield source, workspace


def test_vendored_runtime_bundles_frozen_dependencies_and_assets_without_git(bundled_source):
    source, workspace = bundled_source
    assert (workspace / "vendor/skyrl/uv.lock").read_bytes() == (source.root / "uv.lock").read_bytes()
    assert (workspace / "vendor/skyrl/chat_templates/model.jinja2").read_text() == "{{ messages }}"
    assert runtime_bundle.validate_bundled_runtime(workspace) == source.digest


@pytest.mark.parametrize(
    "path",
    [
        "vendor/skyrl/skyrl-train/skyrl_train/config/ppo_base_config.yaml",
        "lib/rolloutengine/src/rolloutengine/__init__.py",
    ],
)
def test_changed_bundled_source_is_rejected(bundled_source, path):
    _, workspace = bundled_source
    (workspace / path).write_text("changed = True\n")
    with pytest.raises(RuntimeError, match="does not match its recorded identity"):
        runtime_bundle.validate_bundled_runtime(workspace)


@pytest.mark.parametrize(
    "path",
    ["cloud/iris/launch.py", "cloud/iris/new_runtime.py", "../../lib/rolloutengine/src/rolloutengine/__init__.py"],
)
def test_source_edits_get_a_new_identity_and_bundle_without_restaging(bundled_source, path):
    source, _ = bundled_source
    (source.root / path).write_text("updated = True\n")
    changed = runtime_bundle.resolve_launcher_source()
    assert changed.digest != source.digest
    with runtime_bundle.build_runtime_bundle(changed.digest) as workspace:
        assert (workspace / "vendor/skyrl" / path).read_text() == "updated = True\n"
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
    (workspace / "vendor/skyrl/cloud/iris/injected.py").write_text("extra = True\n")
    with pytest.raises(ValueError, match="do not match the selected runtime identity"):
        runtime_bundle.validate_bundled_runtime(workspace)


def test_worker_bundle_installs_local_modules_without_the_checkout(tmp_path):
    source = runtime_bundle.resolve_launcher_source()
    local_source = runtime_bundle.local_module_source(
        "marin-rolloutengine", source.root.parents[1] / "lib/rolloutengine", source.digest
    )
    workspace = tmp_path / "worker"
    with runtime_bundle.build_runtime_bundle(source.digest) as built:
        shutil.copytree(built, workspace)
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    completed = subprocess.run(
        [
            os.environ["UV"],
            "run",
            "--project",
            str(workspace / "vendor/skyrl"),
            "--isolated",
            "--frozen",
            "--no-group",
            "dev",
            "--extra",
            "cpu",
            "python",
            "-I",
            "-c",
            "import json; from pathlib import Path; "
            "import rolloutengine, taskcompendium, verifyit; "
            "import marinskyrl.task_machine_factories as factory; "
            "from cloud.iris.runtime_bundle import local_module_source; "
            f"source = local_module_source('marin-rolloutengine', Path(rolloutengine.__file__).parents[2], {source.digest!r}); "
            "records = {module.__name__: str(Path(module.__file__).resolve()) "
            "for module in (rolloutengine, taskcompendium, verifyit, factory)}; "
            "records['module_digest'] = source.digest; print(json.dumps(records))",
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    observed = json.loads(completed.stdout)
    assert observed["module_digest"] == local_source.digest
    for name in ("rolloutengine", "taskcompendium", "verifyit"):
        assert Path(observed[name]).is_relative_to(workspace / "lib" / name / "src" / name)
    assert observed["marinskyrl.task_machine_factories"] == str(
        workspace / "vendor/skyrl/marinskyrl/task_machine_factories.py"
    )
    assert runtime_bundle.validate_bundled_runtime(workspace) == source.digest
