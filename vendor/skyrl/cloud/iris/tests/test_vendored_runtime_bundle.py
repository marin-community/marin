# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path

import pytest

from cloud.iris import runtime_bundle


@pytest.fixture
def runtime_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", project)
    return project


def test_runtime_bundle_copies_current_source_dependencies_and_assets(runtime_source):
    (runtime_source / "cloud/iris/launch.py").write_text("updated = True\n")
    (runtime_source / "cloud/iris/new_runtime.py").write_text("new = True\n")
    with runtime_bundle.build_runtime_bundle() as workspace:
        for path in runtime_source.rglob("*"):
            if path.is_file():
                assert (workspace / path.relative_to(runtime_source)).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("submission_succeeds", [True, False])
def test_runtime_workspace_is_removed_after_submission(runtime_source, submission_succeeds):
    outcome = nullcontext() if submission_succeeds else pytest.raises(RuntimeError, match="submission failed")
    with outcome, runtime_bundle.build_runtime_bundle() as workspace:
        assert (workspace / "uv.lock").read_bytes() == (runtime_source / "uv.lock").read_bytes()
        if not submission_succeeds:
            raise RuntimeError("submission failed")
    assert not workspace.exists()
    assert (runtime_source / "uv.lock").is_file()
