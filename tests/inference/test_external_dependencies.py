# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged external pins must match the fork descriptors they are generated from."""

import hashlib
import importlib.util
import json
import runpy
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
            "candidate_tag": "marin-vllm-gpu-staged-candidate-aaaaaaaaaaaa",
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
            {
                "architecture": "aarch64",
                "sm_targets": ["10.0"],
                "wheel": {
                    "filename": "vllm-0.0.0.dev20260101+marin.abcdef012345-cp38-abi3-manylinux_2_28_aarch64.whl",
                    "sha256": "d" * 64,
                },
            },
        ],
    }


def _staged_candidate_manifest() -> dict:
    manifest = _promoted_manifest()
    manifest["release"] |= {
        "status": "candidate",
        "tag": "marin-vllm-gpu-staged-candidate-aaaaaaaaaaaa",
    }
    manifest["validation"] = {"status": "pending", "targets": []}
    return manifest


def _published_candidate(manifest: dict) -> dict:
    return {
        "tag_name": manifest["release"]["tag"],
        "target_commitish": manifest["source"]["fork_commit"],
        "draft": False,
        "prerelease": True,
        "assets": (
            [
                {
                    "name": "marin-vllm-gpu-manifest.json",
                    "state": "uploaded",
                    "digest": f"sha256:{hashlib.sha256(json.dumps(manifest).encode()).hexdigest()}",
                }
            ]
            + [
                {"name": p["wheel"]["filename"], "state": "uploaded", "digest": f"sha256:{p['wheel']['sha256']}"}
                for p in manifest["platforms"]
            ]
        ),
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
    assert [wheel.architecture for wheel in release.wheels] == ["x86_64", "aarch64"]


def test_render_gpu_release_toml_explicitly_pins_a_staged_candidate(tmp_path):
    update_external = _update_external()
    rendered = update_external.render_gpu_release_toml(_staged_candidate_manifest(), staged_candidate=True)

    path = tmp_path / "gpu.toml"
    path.write_text(rendered)
    release = update_external.load_vllm_gpu_release(path)

    assert release.release_tag == "marin-vllm-gpu-staged-candidate-aaaaaaaaaaaa"
    assert release.source_commit == "a" * 40
    assert release.wheels[0].sha256 == "b" * 64


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


def test_stage_gpu_candidate_rejects_a_promoted_release(tmp_path, monkeypatch):
    update_external = _update_external()
    pin = tmp_path / "gpu.toml"
    pin.write_text('release_tag = "keep-me"\n')
    monkeypatch.setattr(update_external, "VLLM_GPU_RELEASE_CONFIG", pin)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_promoted_manifest()))

    with pytest.raises(ValueError, match="expected a staged 'candidate' manifest"):
        update_external.stage_gpu_candidate(manifest_path)
    assert pin.read_text() == 'release_tag = "keep-me"\n'


def test_stage_gpu_candidate_moved_tip_preserves_the_current_pin(tmp_path, monkeypatch, gpu_pin_workspace):
    update_external, pin, generated = gpu_pin_workspace
    original = update_external.render_gpu_release_toml(_promoted_manifest())
    pin.write_text(original)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_staged_candidate_manifest()))
    monkeypatch.setattr(
        update_external.subprocess,
        "run",
        lambda args, **kwargs: update_external.subprocess.CompletedProcess(args, 0, stdout=f"{'c' * 40}\n", stderr=""),
    )

    with pytest.raises(ValueError, match="is not the current main-next tip"):
        update_external.stage_gpu_candidate(manifest_path)
    assert pin.read_text() == original
    assert generated.read_text() == "# previous generated pins\n"


@pytest.fixture
def gpu_pin_workspace(tmp_path, monkeypatch):
    update_external = _update_external()
    pin = tmp_path / "gpu.toml"
    generated = tmp_path / "external_dependencies.py"
    monkeypatch.setattr(update_external, "VLLM_GPU_RELEASE_CONFIG", pin)
    monkeypatch.setattr(update_external, "GENERATED_PINS", generated)
    generated.write_text("# previous generated pins\n")
    return update_external, pin, generated


def test_stage_and_promote_gpu_release_keep_generated_pins_consistent(tmp_path, monkeypatch, gpu_pin_workspace):
    update_external, pin, generated = gpu_pin_workspace
    pin.write_text(update_external.render_gpu_release_toml(_promoted_manifest()))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_staged_candidate_manifest()))
    monkeypatch.setattr(
        update_external.subprocess,
        "run",
        lambda args, **kwargs: update_external.subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                f"{'a' * 40}\n" if "--jq" in args else json.dumps(_published_candidate(_staged_candidate_manifest()))
            ),
            stderr="",
        ),
    )
    update_external.stage_gpu_candidate(manifest_path)
    staged_pin = update_external.load_vllm_gpu_release(pin)
    staged_generated = runpy.run_path(str(generated))["VLLM_GPU_RELEASE"]
    assert staged_generated.release_tag == staged_pin.release_tag == "marin-vllm-gpu-staged-candidate-aaaaaaaaaaaa"
    assert [(w.architecture, w.sha256) for w in staged_generated.wheels] == [
        (w.architecture, w.sha256) for w in staged_pin.wheels
    ]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_promoted_manifest()))
    update_external.promote_gpu_release(manifest_path)
    final_pin = update_external.load_vllm_gpu_release(pin)
    final_generated = runpy.run_path(str(generated))["VLLM_GPU_RELEASE"]
    assert final_generated.release_tag == final_pin.release_tag == "marin-vllm-gpu-20260101-abcdef012345"
    assert final_generated.source_commit == final_pin.source_commit == staged_pin.source_commit
    assert [(w.architecture, w.sha256) for w in final_generated.wheels] == [
        (w.architecture, w.sha256) for w in staged_pin.wheels
    ]
    assert update_external.regenerate_generated_pins(
        tuple(update_external.locked_dependency(project) for project in update_external.EXTERNAL_PROJECTS), check=True
    )


def test_promote_gpu_release_advances_an_ordinary_final_pin(tmp_path, gpu_pin_workspace):
    update_external, pin, generated = gpu_pin_workspace
    previous = _promoted_manifest()
    previous["source"]["fork_commit"] = "e" * 40
    previous["release"]["tag"] = "marin-vllm-gpu-20251201-eeeeeeeeeeee"
    pin.write_text(update_external.render_gpu_release_toml(previous))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_promoted_manifest()))

    update_external.promote_gpu_release(manifest_path)

    final = update_external.load_vllm_gpu_release(pin)
    packaged = runpy.run_path(str(generated))["VLLM_GPU_RELEASE"]
    assert final.release_tag == packaged.release_tag == _promoted_manifest()["release"]["tag"]
    assert final.source_commit == packaged.source_commit == "a" * 40


@pytest.mark.parametrize("change", ["source", "x86_64", "aarch64", "candidate", "platform-set", "abi", "unvalidated"])
def test_promote_gpu_release_rejection_preserves_pin_and_generated_dependencies(tmp_path, gpu_pin_workspace, change):
    update_external, pin, generated = gpu_pin_workspace
    original = update_external.render_gpu_release_toml(_staged_candidate_manifest(), staged_candidate=True)
    pin.write_text(original)
    manifest = _promoted_manifest()
    if change == "source":
        manifest["source"]["fork_commit"] = "c" * 40
    elif change in {"x86_64", "aarch64"}:
        next(p for p in manifest["platforms"] if p["architecture"] == change)["wheel"]["sha256"] = "c" * 64
    elif change == "candidate":
        manifest["release"]["candidate_tag"] = "marin-vllm-gpu-staged-candidate-cccccccccccc"
    elif change == "platform-set":
        manifest["platforms"].pop()
    elif change == "abi":
        manifest["abi"] = {"cuda_variant": "cu132", "torch_version": "2.13.0+cu132"}
    else:
        manifest["validation"]["status"] = "pending"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        update_external.promote_gpu_release(manifest_path)
    assert pin.read_text() == original
    assert generated.read_text() == "# previous generated pins\n"
    assert not list(tmp_path.glob("gpu.*.toml.tmp"))


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


def test_stage_gpu_candidate_stale_metadata_preserves_pin_and_generated_dependencies(
    tmp_path, monkeypatch, gpu_pin_workspace
):
    update_external, pin, generated = gpu_pin_workspace
    original = update_external.render_gpu_release_toml(_promoted_manifest())
    pin.write_text(original)
    manifest = _staged_candidate_manifest()
    published = _published_candidate(manifest)
    published["assets"][1]["digest"] = "sha256:" + "f" * 64
    monkeypatch.setattr(
        update_external.subprocess,
        "run",
        lambda args, **kwargs: update_external.subprocess.CompletedProcess(
            args,
            0,
            stdout=(f"{'a' * 40}\n" if "--jq" in args else json.dumps(published)),
            stderr="",
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata no longer matches"):
        update_external.stage_gpu_candidate(manifest_path)
    assert pin.read_text() == original
    assert generated.read_text() == "# previous generated pins\n"
