# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for Marin-owned package releases."""

import hashlib
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts.ci.package_release import (
    PublishedArtifact,
    artifact_manifest,
    cargo_compatible_version,
    latest_supported_version,
    merge_targeted_lock_change,
    next_development_version,
    packages_for_changes,
    python_compatible_version,
    reconcile_published_artifacts,
    release_plan,
    requirement_paths_for_packages,
    update_native_requirement,
    validate_targeted_lock_change,
)

RELEASE_WORKFLOW = Path(".github/workflows/marin-release-libs-wheels.yaml")


def _artifact_names(package: str, version: str) -> list[str]:
    if package == "python-libs":
        return [
            filename
            for distribution in (
                "marin_core",
                "marin_iris",
                "marin_fray",
                "marin_rigging",
                "marin_zephyr",
                "marin_levanter",
                "marin_haliax",
            )
            for filename in (
                f"{distribution}-{version}-py3-none-any.whl",
                f"{distribution}-{version}.tar.gz",
            )
        ]

    platform_wheels = [f"cp312-cp312-manylinux_2_28_{arch}.whl" for arch in ("x86_64", "aarch64")] + [
        f"cp312-cp312-macosx_11_0_{arch}.whl" for arch in ("x86_64", "arm64")
    ]
    if package == "iris":
        return [
            f"marin_iris_native-{version}-{platform_wheels[0]}",
            f"marin_iris_native-{version}-{platform_wheels[1]}",
            f"marin_iris_native-{version}-{platform_wheels[2]}",
            f"marin_iris_native-{version}-{platform_wheels[3]}",
            f"marin_iris_native-{version}.tar.gz",
        ]

    pure_distribution, native_distribution = {
        "dupekit": ("marin_dupekit", "marin_dupekit_native"),
        "finelog": ("marin_finelog", "marin_finelog_server"),
    }[package]
    return [
        f"{pure_distribution}-{version}-py3-none-any.whl",
        f"{pure_distribution}-{version}.tar.gz",
        *(f"{native_distribution}-{version}-{platform}" for platform in platform_wheels),
        f"{native_distribution}-{version}.tar.gz",
    ]


def _write_artifacts(root: Path, package: str, version: str) -> list[Path]:
    paths = []
    for index, name in enumerate(_artifact_names(package, version)):
        path = root / name
        path.write_bytes(f"artifact-{index}".encode())
        paths.append(path)
    return paths


def _workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    return data


def _triggers(workflow: dict) -> dict:
    # PyYAML 1.1 parses the unquoted workflow key `on` as True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def test_next_development_version_is_retry_stable_and_above_stable() -> None:
    first = next_development_version("0.1.3", "0.1.4", 30194118926)
    retry = next_development_version("0.1.3", first, 30194118926)
    later = next_development_version("0.1.3", "0.1.4", 30194120716)

    assert first == retry == "0.1.5.dev30194118926"
    assert later == "0.1.5.dev30194120716"


def test_development_version_is_compatible_with_cargo_build_manifests() -> None:
    assert cargo_compatible_version("0.1.4.dev30194118926") == "0.1.4-dev.30194118926"
    assert cargo_compatible_version("0.1.4") == "0.1.4"
    assert cargo_compatible_version("0.1.4+abc12345") == "0.1.4+abc12345"
    assert python_compatible_version("0.1.4+abc12345") == "0.1.4+abc12345"


def test_latest_supported_version_includes_development_and_ignores_unsupported_releases() -> None:
    versions = ["0.1.3", "0.1.4.dev30194118926", "0.1.2", "0.2.0rc1"]

    assert latest_supported_version(versions) == "0.1.4.dev30194118926"


def test_next_development_version_advances_past_legacy_timestamp_serial() -> None:
    version = next_development_version("0.2.11", "0.2.12.dev202607250802", 30220111744)

    assert version == "0.2.13.dev30220111744"


def test_change_detection_maps_shared_and_owned_sources() -> None:
    assert packages_for_changes(["lib/iris/rust/src/lib.rs"]) == ["iris"]
    assert packages_for_changes(["lib/dupekit/src/dupekit/__init__.py"]) == ["dupekit"]
    assert packages_for_changes(["lib/finelog/config/marin.yaml"]) == ["finelog"]
    assert packages_for_changes(["lib/finelog/src/finelog/client/log_client.py"]) == ["finelog"]
    assert packages_for_changes(["rust/Cargo.lock"]) == ["dupekit", "finelog", "iris"]
    assert packages_for_changes(["scripts/python_libs_package.py"]) == ["python-libs"]
    assert packages_for_changes(["scripts/ci/package_release.py"]) == [
        "dupekit",
        "finelog",
        "iris",
        "python-libs",
    ]


def test_pull_request_plan_uses_one_declarative_build_matrix() -> None:
    plan = release_plan(
        event_name="pull_request",
        ref="refs/pull/1/merge",
        input_mode="manual",
        input_package="all",
        input_version="",
        revision="abcdef123456",
        serial=42,
        changed_paths=["scripts/ci/package_release.py"],
        repo_root=Path.cwd(),
    )

    assert plan.packages == ("dupekit", "finelog", "iris", "python-libs")
    expected_builds = {
        ("dupekit", "ubuntu-latest", "linux"),
        ("dupekit", "macos-14", "macos"),
        ("dupekit", "ubuntu-latest", "sdist"),
        ("finelog", "ubuntu-latest", "linux"),
        ("finelog", "macos-14", "macos"),
        ("finelog", "ubuntu-latest", "sdist"),
        ("iris", "ubuntu-latest", "linux"),
        ("iris", "macos-14", "macos"),
        ("iris", "ubuntu-latest", "sdist"),
        ("python-libs", "ubuntu-latest", "python"),
    }
    actual_builds = {(entry["package"], entry["os"], entry["operation"]) for entry in plan.builds}
    assert actual_builds == expected_builds
    assert all(plan.versions[package].version.endswith("+abcdef12") for package in plan.packages)
    assert plan.versions["iris"].mode == "manual"
    assert plan.bump


def test_schedule_selects_only_coupled_python_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.ci.package_release.latest_family_version", lambda package: "0.2.0")

    plan = release_plan(
        event_name="schedule",
        ref="refs/heads/main",
        input_mode="manual",
        input_package="all",
        input_version="",
        revision="abcdef123456",
        serial=30220111744,
        changed_paths=[],
        repo_root=Path.cwd(),
    )

    assert plan.packages == ("python-libs",)
    assert plan.versions["python-libs"].version == "0.2.1.dev30220111744"
    assert plan.builds == ({"package": "python-libs", "os": "ubuntu-latest", "operation": "python"},)
    assert not plan.bump


def test_general_library_tag_selects_one_stable_family() -> None:
    plan = release_plan(
        event_name="push",
        ref="refs/tags/marin-libs-v0.3.0",
        input_mode="manual",
        input_package="all",
        input_version="",
        revision="abcdef123456",
        serial=30220111744,
        changed_paths=[],
        repo_root=Path.cwd(),
    )

    assert plan.packages == ("python-libs",)
    assert plan.versions["python-libs"].version == "0.3.0"
    assert plan.builds == ({"package": "python-libs", "os": "ubuntu-latest", "operation": "python"},)
    assert not plan.bump


def test_dispatch_rejects_one_explicit_version_for_multiple_families() -> None:
    with pytest.raises(ValueError, match="one package family"):
        release_plan(
            event_name="workflow_dispatch",
            ref="refs/heads/main",
            input_mode="development",
            input_package="all",
            input_version="0.3.0",
            revision="abcdef123456",
            serial=30220111744,
            changed_paths=[],
            repo_root=Path.cwd(),
        )


def test_shared_requirement_path_is_emitted_once() -> None:
    assert requirement_paths_for_packages(["finelog", "iris"]) == (Path("lib/iris/pyproject.toml"),)


def test_update_native_requirement_advances_floor_without_touching_neighbors() -> None:
    before = """\
dependencies = [
    "marin-finelog >= 0.2.10",
    "marin-finelog-server >= 0.2.12.dev202607250802",
    "marin-iris-native >= 0.1.3",
]
"""

    after, changed = update_native_requirement(before, "marin-iris-native", "0.1.4.dev30194118926")

    assert changed
    assert after == before.replace(
        '"marin-iris-native >= 0.1.3"',
        '"marin-iris-native >= 0.1.4.dev30194118926"',
    )


def test_update_native_requirement_never_downgrades_floor() -> None:
    text = 'dependencies = ["marin-iris-native >= 0.1.5.dev40000000000"]\n'

    assert update_native_requirement(text, "marin-iris-native", "0.1.4")[0] == text


@pytest.mark.parametrize("package", ["iris", "dupekit", "finelog", "python-libs"])
def test_artifact_manifest_requires_complete_release_pair(tmp_path: Path, package: str) -> None:
    version = "0.3.0.dev30194118926"
    paths = _write_artifacts(tmp_path, package, version)

    manifest = artifact_manifest(package, version, paths)

    assert set(manifest) == {path.name for path in paths}
    with pytest.raises(ValueError, match="artifact manifest"):
        artifact_manifest(package, version, paths[:-1])


def test_reconcile_published_artifacts_accepts_matching_partial_upload(tmp_path: Path) -> None:
    version = "0.1.4.dev30194118926"
    paths = _write_artifacts(tmp_path, "iris", version)
    manifest = artifact_manifest("iris", version, paths)
    published_name = paths[0].name
    published = {
        "marin-iris-native": [
            PublishedArtifact(filename=published_name, sha256=manifest[published_name]),
        ]
    }

    missing = reconcile_published_artifacts("iris", version, manifest, published)

    assert missing == set(manifest) - {published_name}


@pytest.mark.parametrize(
    "published",
    [
        {"marin-iris-native": [PublishedArtifact(filename="unexpected.whl", sha256="0" * 64)]},
        {
            "marin-iris-native": [
                PublishedArtifact(
                    filename="marin_iris_native-0.1.4.dev30194118926-cp312-cp312-manylinux_2_28_x86_64.whl",
                    sha256=hashlib.sha256(b"different").hexdigest(),
                )
            ]
        },
    ],
)
def test_reconcile_published_artifacts_rejects_collision(
    tmp_path: Path, published: dict[str, list[PublishedArtifact]]
) -> None:
    version = "0.1.4.dev30194118926"
    manifest = artifact_manifest("iris", version, _write_artifacts(tmp_path, "iris", version))

    with pytest.raises(ValueError, match="PyPI"):
        reconcile_published_artifacts("iris", version, manifest, published)


def test_validate_targeted_lock_change_rejects_unrelated_package_update() -> None:
    before = """\
version = 1
revision = 3

[[package]]
name = "marin-iris-native"
version = "0.1.3"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "typing-extensions"
version = "4.16.0"
source = { registry = "https://pypi.org/simple" }
"""
    targeted = before.replace('version = "0.1.3"', 'version = "0.1.4.dev30194118926"')
    unrelated = targeted.replace('version = "4.16.0"', 'version = "4.17.0"')

    validate_targeted_lock_change(before, targeted, "marin-iris-native", "0.1.4.dev30194118926")
    with pytest.raises(ValueError, match="typing-extensions"):
        validate_targeted_lock_change(before, unrelated, "marin-iris-native", "0.1.4.dev30194118926")


def test_merge_targeted_lock_change_preserves_unrelated_resolution() -> None:
    before = """\
version = 1
revision = 3

[[package]]
name = "marin-iris"
version = "0.1.0"
source = { virtual = "lib/iris" }
dependencies = [
    { name = "marin-iris-native" },
]

[package.metadata]
requires-dist = [{ name = "marin-iris-native", specifier = ">=0.1.3" }]

[[package]]
name = "marin-iris-native"
version = "0.1.3"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "typing-extensions"
version = "4.16.0"
source = { registry = "https://pypi.org/simple" }
"""
    resolved = (
        before.replace(
            'specifier = ">=0.1.3"',
            'specifier = ">=0.1.4.dev30194118926"',
        )
        .replace(
            'version = "0.1.3"',
            'version = "0.1.4.dev30194118926"',
        )
        .replace(
            'version = "4.16.0"',
            'version = "4.17.0"',
        )
    )

    merged = merge_targeted_lock_change(before, resolved, ("marin-iris", "marin-iris-native"))

    packages = {package["name"]: package for package in tomllib.loads(merged)["package"]}
    assert packages["marin-iris"]["metadata"]["requires-dist"] == [
        {"name": "marin-iris-native", "specifier": ">=0.1.4.dev30194118926"}
    ]
    assert packages["marin-iris-native"]["version"] == "0.1.4.dev30194118926"
    assert packages["typing-extensions"]["version"] == "4.16.0"
    validate_targeted_lock_change(
        before,
        merged,
        "marin-iris-native",
        "0.1.4.dev30194118926",
        allowed_packages=("marin-iris",),
    )


def test_merge_targeted_lock_change_requires_unique_target() -> None:
    lock = """\
version = 1

[[package]]
name = "marin-iris"
version = "0.1.0"
source = { virtual = "lib/iris" }
"""

    with pytest.raises(ValueError):
        merge_targeted_lock_change(lock, lock, ("marin-iris-native",))


def test_release_workflow_publishes_only_trusted_package_releases() -> None:
    workflow = _workflow(RELEASE_WORKFLOW)
    triggers = _triggers(workflow)
    push = triggers["push"]

    assert push["branches"] == ["main"]
    assert "schedule" in triggers
    assert "uv.lock" not in push["paths"]
    assert not any(item.endswith("pyproject.toml") for item in push["paths"])

    publish = workflow["jobs"]["publish"]
    assert publish["permissions"] == {"contents": "read", "id-token": "write"}
    assert "github.repository == 'marin-community/marin'" in publish["if"]
    assert "needs.plan.outputs.packages" in str(publish["strategy"])
    assert "needs.plan.outputs.build_matrix" in str(workflow["jobs"]["build"]["strategy"])


def test_release_workflow_uses_app_token_for_version_pr() -> None:
    workflow = _workflow(RELEASE_WORKFLOW)
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["bump"]["steps"]
    token_step = next(step for step in steps if step.get("id") == "app-token")
    assert token_step["uses"] == "actions/create-github-app-token@v3"
    pr_step = next(step for step in steps if step.get("name") == "Commit and open or update pull request")
    assert pr_step["env"]["GH_TOKEN"] == "${{ steps.app-token.outputs.token }}"
