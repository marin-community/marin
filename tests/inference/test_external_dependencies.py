# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged external pins must match the fork descriptors they are generated from."""

import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT, VLLM_GPU_RELEASE
from rigging.config_discovery import find_project_root


def _workspace_root() -> Path:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to compare against")
    return root


def _descriptor_requirement(name: str) -> str:
    config = tomllib.loads((_workspace_root() / "config" / "external" / "vllm" / "tpu.toml").read_text())
    entry = config[name]
    return f"{name} @ git+{entry['repository']}@{entry['commit']}"


def _update_external():
    """Load config/update-external.py, which is a standalone script rather than a package module."""
    path = _workspace_root() / "config" / "update-external.py"
    spec = importlib.util.spec_from_file_location("update_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _promoted_manifest() -> dict:
    return {
        "release": {
            "status": "released",
            "tag": "marin-vllm-gpu-20260101-abcdef012345",
            "repository": "marin-community/vllm",
        },
        "validation": {"status": "passed"},
        "source": {"fork_commit": "a" * 40},
        "distribution": {"version": "0.0.0.dev20260101+marin.abcdef012345"},
        "abi": {"cuda_variant": "cu130", "torch_version": "2.11.0+cu130"},
        "platforms": [
            {
                "architecture": "x86_64",
                "sm_targets": ["9.0"],
                "wheel": {
                    "filename": "vllm-0.0.0.dev20260101+marin.abcdef012345-cp38-abi3-manylinux_2_28_x86_64.whl",
                    "sha256": "b" * 64,
                },
            },
        ],
    }


def test_skyrl_dependency_check_rejects_importable_but_incompatible_constructor(tmp_path):
    updater = _update_external()
    package = tmp_path / "rolloutengine"
    package.mkdir()
    (package / "__init__.py").write_text("")
    engine = package / "engine.py"
    engine.write_text("class RolloutEngine:\n    def __init__(self, model, factories): pass\n")
    consumer = tmp_path / "skyrl-train/skyrl_train/task_worker.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        "from rolloutengine.engine import RolloutEngine\n"
        "def run():\n    return RolloutEngine(None, {}, judge_connection=None)\n"
    )
    incompatible = subprocess.run(
        [sys.executable, str(updater.SKYRL_MARIN_CHECK), "--source", str(tmp_path)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert incompatible.returncode != 0
    assert "judge_connection" in incompatible.stderr
    engine.write_text("class RolloutEngine:\n    def __init__(self, model, factories, *, judge_connection=None): pass\n")
    compatible = subprocess.run(
        [sys.executable, str(updater.SKYRL_MARIN_CHECK), "--source", str(tmp_path)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert compatible.returncode == 0, compatible.stderr


@pytest.mark.parametrize("compatible", [False, True])
def test_skyrl_marin_update_preserves_working_files_when_validation_fails(tmp_path, monkeypatch, compatible):
    updater = _update_external()
    old_revision, new_revision = "a" * 40, "b" * 40
    manifest, lock = tmp_path / "pyproject.toml", tmp_path / "uv.lock"
    manifest.write_text(
        "\n".join(
            f'"marin-{name} @ git+https://github.com/marin-community/marin.git@{old_revision}#subdirectory=lib/{name}"'
            for name in ("verifyit", "rolloutengine", "shellbox", "taskcompendium")
        )
    )
    lock.write_text("original lock\n")
    original_manifest = manifest.read_bytes()
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: new_revision + "\n")

    def uv(command, **kwargs):
        if command[1] == "lock":
            lock.write_text("updated lock\n")
        elif not compatible:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", uv)
    if compatible:
        updater.update_skyrl_marin(tmp_path, new_revision)
        assert old_revision not in manifest.read_text()
        assert manifest.read_text().count(new_revision) == 4
        assert lock.read_text() == "updated lock\n"
    else:
        with pytest.raises(subprocess.CalledProcessError):
            updater.update_skyrl_marin(tmp_path, new_revision)
        assert manifest.read_bytes() == original_manifest
        assert lock.read_text() == "original lock\n"


def test_tpu_vllm_requirements_match_fork_descriptor():
    assert VLLM_FORK_REQUIREMENT == _descriptor_requirement("vllm")
    assert TPU_INFERENCE_FORK_REQUIREMENT == _descriptor_requirement("tpu-inference")


def test_gpu_release_pin_matches_its_descriptor():
    update_external = _update_external()
    descriptor = update_external.load_vllm_gpu_release(update_external.VLLM_GPU_RELEASE_CONFIG)

    assert VLLM_GPU_RELEASE.release_tag == descriptor.release_tag
    assert VLLM_GPU_RELEASE.source_commit == descriptor.source_commit
    assert VLLM_GPU_RELEASE.version == descriptor.version
    assert VLLM_GPU_RELEASE.torch_backend == descriptor.torch_backend
    assert VLLM_GPU_RELEASE.torch_version == descriptor.torch_version
    generated = {(w.architecture, w.sm_targets, w.url, w.sha256) for w in VLLM_GPU_RELEASE.wheels}
    pinned = {(w.architecture, w.sm_targets, w.url, w.sha256) for w in descriptor.wheels}
    assert generated == pinned


def test_render_gpu_release_toml_reencodes_the_wheel_url_and_round_trips(tmp_path):
    update_external = _update_external()
    rendered = update_external.render_gpu_release_toml(_promoted_manifest())

    # The manifest carries the raw '+' filename; the pin must percent-encode it so the
    # loader's quote(version, safe='') URL check passes.
    assert "%2Bmarin.abcdef012345-" in rendered
    assert "+marin.abcdef012345-" not in rendered

    path = tmp_path / "gpu.toml"
    path.write_text(rendered)
    release = update_external.load_vllm_gpu_release(path)
    assert release.release_tag == "marin-vllm-gpu-20260101-abcdef012345"
    assert release.source_commit == "a" * 40
    assert release.torch_backend == "cu130"
    assert release.torch_version == "2.11.0+cu130"
    assert [wheel.architecture for wheel in release.wheels] == ["x86_64"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m["release"].__setitem__("status", "candidate"),
        lambda m: m["validation"].__setitem__("status", "pending"),
        lambda m: m["release"].__setitem__("repository", "someone-else/vllm"),
    ],
    ids=["unpromoted", "unvalidated", "foreign-repository"],
)
def test_render_gpu_release_toml_refuses_an_unpromoted_manifest(mutation):
    update_external = _update_external()
    manifest = _promoted_manifest()
    mutation(manifest)
    with pytest.raises(ValueError):
        update_external.render_gpu_release_toml(manifest)


def test_promote_gpu_release_keeps_the_pin_when_the_rendered_wheel_fails_validation(tmp_path, monkeypatch):
    # A manifest can clear the render-time status/repository gate yet still carry a wheel
    # invariant (here a malformed SHA-256) that only the loader rejects. The existing pin
    # must survive that failure rather than be overwritten with an invalid descriptor.
    update_external = _update_external()
    pin = tmp_path / "gpu.toml"
    original = 'release_tag = "keep-me"\n'
    pin.write_text(original)
    monkeypatch.setattr(update_external, "VLLM_GPU_RELEASE_CONFIG", pin)

    manifest = _promoted_manifest()
    manifest["platforms"][0]["wheel"]["sha256"] = "not-a-sha"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        update_external.promote_gpu_release(manifest_path)
    assert pin.read_text() == original
    assert not list(tmp_path.glob("gpu.*.toml.tmp"))


@pytest.fixture
def gpu_pin_workspace(tmp_path, monkeypatch):
    update_external = _update_external()
    pin = tmp_path / "gpu.toml"
    generated = tmp_path / "external_dependencies.py"
    monkeypatch.setattr(update_external, "VLLM_GPU_RELEASE_CONFIG", pin)
    monkeypatch.setattr(update_external, "GENERATED_PINS", generated)
    generated.write_text("# previous generated pins\n")
    return update_external, pin, generated


def test_promote_gpu_release_updates_pin_and_generated_dependencies(tmp_path, gpu_pin_workspace):
    update_external, pin, generated = gpu_pin_workspace
    previous = _promoted_manifest()
    previous["source"]["fork_commit"] = "e" * 40
    previous["release"]["tag"] = "marin-vllm-gpu-20251201-eeeeeeeeeeee"
    pin.write_text(update_external.render_gpu_release_toml(previous))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_promoted_manifest()))

    update_external.promote_gpu_release(manifest_path)

    release = update_external.load_vllm_gpu_release(pin)
    packaged = runpy.run_path(str(generated))["VLLM_GPU_RELEASE"]
    assert release.release_tag == packaged.release_tag == _promoted_manifest()["release"]["tag"]
    assert release.source_commit == packaged.source_commit == "a" * 40


def test_gpu_manifest_generation_failure_preserves_the_existing_pin(tmp_path, monkeypatch, gpu_pin_workspace):
    update_external, pin, generated = gpu_pin_workspace
    original = update_external.render_gpu_release_toml(_promoted_manifest())
    pin.write_text(original)
    monkeypatch.setattr(update_external, "TPU_FORKS_CONFIG", tmp_path / "missing-tpu.toml")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_promoted_manifest()))
    with pytest.raises(FileNotFoundError):
        update_external.promote_gpu_release(manifest_path)
    assert pin.read_text() == original
    assert generated.read_text() == "# previous generated pins\n"
