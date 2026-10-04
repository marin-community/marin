# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged external pins must match the fork descriptors they are generated from."""

import importlib.util
import json
import subprocess
import tomllib
import zipfile
from email import message_from_bytes
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


@pytest.mark.timeout(180)
def test_marin_wheel_authors_recipes_with_its_minimum_pydantic_dependency(tmp_path: Path) -> None:
    root = _workspace_root()
    built = subprocess.run(
        ["uv", "build", str(root / "lib/marin"), "--wheel", "--out-dir", str(tmp_path / "dist")],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert built.returncode == 0, built.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = message_from_bytes(
            archive.read(next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA")))
        )
    requirement = next(row for row in metadata.get_all("Requires-Dist") if row.startswith("pydantic"))
    minimum = requirement.split(">=")[1].split(",")[0].split(";")[0].strip()
    environment = tmp_path / "environment"
    subprocess.run(
        ["uv", "venv", "--python", "3.12", str(environment)], check=True, capture_output=True, text=True, timeout=30
    )
    python = environment / "bin/python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), f"pydantic=={minimum}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    program = """
import importlib.util, json, pickle
from pathlib import Path
import marin.skyrl_recipe as schema
from marin.skyrl_recipe import Algorithm, ContextBudget, Generator, RecipePatch, SkyRLRecipe, Trainer
assert 'site-packages' in Path(schema.__file__).parts, schema.__file__
assert all(importlib.util.find_spec(name) is None for name in ('yaml','ray','torch','hydra','omegaconf'))
recipe = SkyRLRecipe.combine(
    budget=RecipePatch(context_budget=ContextBudget(
        request_window_tokens=512,max_new_tokens_per_turn=128,max_turns=1)),
    policy=RecipePatch(trainer=Trainer(algorithm=Algorithm(use_kl_loss=False)),
                       generator=Generator(backend='vllm')),
).with_settings(['context_budget.max_turns=4'])
assert pickle.loads(pickle.dumps(recipe)) == recipe
print(json.dumps(recipe.to_skyrl()))
"""
    result = subprocess.run([str(python), "-I", "-c", program], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "context_budget": {"request_window_tokens": 512, "max_new_tokens_per_turn": 128, "max_turns": 4},
        "trainer": {"algorithm": {"use_kl_loss": False}},
        "generator": {"backend": "vllm"},
    }
