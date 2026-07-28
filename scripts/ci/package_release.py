#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve, validate, and consume Marin-owned package releases."""

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/{distribution}/json"
PYPI_VERSION_JSON_URL = "https://pypi.org/pypi/{distribution}/{version}/json"
LOCK_RETRY_DELAYS = (0, 5, 15, 30, 60, 120)
PYTHON_LIBS_FAMILY = "python-libs"
_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:[.-]dev\.?(?P<dev>\d+))?$")


@dataclass(frozen=True)
class ArtifactExpectation:
    wheels: int
    sdists: int
    pure_python: bool


@dataclass(frozen=True)
class PythonBundle:
    script_path: Path


@dataclass(frozen=True)
class NativeBuild:
    import_name: str
    native_path: Path
    version_paths: tuple[Path, ...]
    requirement_path: Path
    requirement_distribution: str
    requirement_owner: str
    pure_path: Path | None = None


@dataclass(frozen=True)
class PackageFamily:
    declared_version_paths: tuple[Path, ...]
    artifacts: Mapping[str, ArtifactExpectation]
    tag_prefix: str
    source_patterns: tuple[str, ...]
    build_legs: tuple[tuple[str, str], ...]
    build: PythonBundle | NativeBuild


@dataclass(frozen=True)
class PublishedArtifact:
    filename: str
    sha256: str


class ReleaseMode(StrEnum):
    DEVELOPMENT = "development"
    STABLE = "stable"
    MANUAL = "manual"


class BuildOperation(StrEnum):
    LINUX = "linux"
    MACOS = "macos"
    PYTHON = "python"
    SDIST = "sdist"


@dataclass(frozen=True)
class ReleaseVersion:
    mode: str
    version: str
    build_version: str


@dataclass(frozen=True)
class ReleasePlan:
    packages: tuple[str, ...]
    versions: Mapping[str, ReleaseVersion]
    builds: tuple[Mapping[str, str], ...]
    bump: bool


PACKAGES: Mapping[str, PackageFamily] = MappingProxyType(
    {
        PYTHON_LIBS_FAMILY: PackageFamily(
            declared_version_paths=(
                Path("lib/marin/pyproject.toml"),
                Path("lib/iris/pyproject.toml"),
                Path("lib/fray/pyproject.toml"),
                Path("lib/rigging/pyproject.toml"),
                Path("lib/zephyr/pyproject.toml"),
                Path("lib/levanter/pyproject.toml"),
                Path("lib/haliax/src/haliax/__about__.py"),
            ),
            artifacts=MappingProxyType(
                {
                    distribution: ArtifactExpectation(wheels=1, sdists=1, pure_python=True)
                    for distribution in (
                        "marin-core",
                        "marin-iris",
                        "marin-fray",
                        "marin-rigging",
                        "marin-zephyr",
                        "marin-levanter",
                        "marin-haliax",
                    )
                }
            ),
            tag_prefix="marin-libs-v",
            source_patterns=("scripts/python_libs_package.py",),
            build_legs=(("ubuntu-latest", BuildOperation.PYTHON),),
            build=PythonBundle(script_path=Path("scripts/python_libs_package.py")),
        ),
        "iris": PackageFamily(
            declared_version_paths=(Path("lib/iris/rust/pyproject.toml"),),
            artifacts=MappingProxyType(
                {
                    "marin-iris-native": ArtifactExpectation(wheels=4, sdists=1, pure_python=False),
                }
            ),
            tag_prefix="iris-native-v",
            source_patterns=("lib/iris/rust/**", "rust/iris-*/**"),
            build_legs=(
                ("ubuntu-latest", BuildOperation.LINUX),
                ("macos-14", BuildOperation.MACOS),
                ("ubuntu-latest", BuildOperation.SDIST),
            ),
            build=NativeBuild(
                import_name="iris_native",
                native_path=Path("lib/iris/rust"),
                version_paths=(
                    Path("lib/iris/rust/pyproject.toml"),
                    Path("lib/iris/rust/Cargo.toml"),
                    Path("lib/iris/rust/pyext/Cargo.toml"),
                ),
                requirement_path=Path("lib/iris/pyproject.toml"),
                requirement_distribution="marin-iris-native",
                requirement_owner="marin-iris",
            ),
        ),
        "dupekit": PackageFamily(
            declared_version_paths=(Path("lib/dupekit/pyproject.toml"),),
            artifacts=MappingProxyType(
                {
                    "marin-dupekit": ArtifactExpectation(wheels=1, sdists=1, pure_python=True),
                    "marin-dupekit-native": ArtifactExpectation(wheels=4, sdists=1, pure_python=False),
                }
            ),
            tag_prefix="dupekit-v",
            source_patterns=(
                "lib/dupekit/src/**",
                "lib/dupekit/rust/**",
                "rust/dupekit-pyext/**",
            ),
            build_legs=(
                ("ubuntu-latest", BuildOperation.LINUX),
                ("macos-14", BuildOperation.MACOS),
                ("ubuntu-latest", BuildOperation.SDIST),
            ),
            build=NativeBuild(
                import_name="dupekit_native",
                native_path=Path("lib/dupekit/rust"),
                version_paths=(Path("lib/dupekit/pyproject.toml"), Path("lib/dupekit/rust/Cargo.toml")),
                requirement_path=Path("lib/dupekit/pyproject.toml"),
                requirement_distribution="marin-dupekit-native",
                requirement_owner="marin-dupekit",
                pure_path=Path("lib/dupekit"),
            ),
        ),
        "finelog": PackageFamily(
            declared_version_paths=(Path("lib/finelog/pyproject.toml"),),
            artifacts=MappingProxyType(
                {
                    "marin-finelog": ArtifactExpectation(wheels=1, sdists=1, pure_python=True),
                    "marin-finelog-server": ArtifactExpectation(wheels=4, sdists=1, pure_python=False),
                }
            ),
            tag_prefix="finelog-v",
            source_patterns=(
                "lib/finelog/config/**",
                "lib/finelog/src/**",
                "lib/finelog/rust/**",
                "rust/finelog/**",
                "rust/finelog-pyext/**",
            ),
            build_legs=(
                ("ubuntu-latest", BuildOperation.LINUX),
                ("macos-14", BuildOperation.MACOS),
                ("ubuntu-latest", BuildOperation.SDIST),
            ),
            build=NativeBuild(
                import_name="finelog_server",
                native_path=Path("lib/finelog/rust"),
                version_paths=(Path("lib/finelog/pyproject.toml"), Path("lib/finelog/rust/pyproject.toml")),
                requirement_path=Path("lib/iris/pyproject.toml"),
                requirement_distribution="marin-finelog-server",
                requirement_owner="marin-iris",
                pure_path=Path("lib/finelog"),
            ),
        ),
    }
)
SHARED_SOURCE_PATTERNS = ("rust/Cargo.toml", "rust/Cargo.lock")
SHARED_PR_PATTERNS = (
    "scripts/ci/package_release.py",
    ".github/workflows/marin-release-libs-wheels.yaml",
)
ZIG_VERSION = "0.15.2"
ZIG_MIRRORS = (
    "https://pkg.earth/zig",
    "https://pkg.hexops.org/zig",
    "https://zig.linus.dev/zig",
)
ZIG_OFFICIAL_BASE = "https://ziglang.org/download"
ZIG_DOWNLOAD_ATTEMPTS_PER_SOURCE = 2
LINUX_TARGETS = (
    ("x86_64-unknown-linux-gnu", "2_28"),
    ("aarch64-unknown-linux-gnu", "2_28"),
)
MAC_TARGETS = (
    ("x86_64-apple-darwin", None),
    ("aarch64-apple-darwin", None),
)


def canonical_version(version: str) -> str:
    """Return the canonical stable or development version used by native releases."""
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"Expected X.Y.Z or X.Y.Z.devN, got {version!r}")
    release = ".".join(match.group(part) for part in ("major", "minor", "patch"))
    dev = match.group("dev")
    return release if dev is None else f"{release}.dev{dev}"


def cargo_compatible_version(version: str) -> str:
    """Return a release version accepted by Cargo and normalized by Python."""
    if "+" in version:
        release, local = version.split("+", maxsplit=1)
        canonical_version(release)
        if re.fullmatch(r"[0-9A-Za-z.-]+", local) is None:
            raise ValueError(f"Invalid local version {version!r}")
        return version
    canonical = canonical_version(version)
    if ".dev" not in canonical:
        return canonical
    release, serial = canonical.split(".dev", maxsplit=1)
    return f"{release}-dev.{serial}"


def python_compatible_version(version: str) -> str:
    """Return the canonical Python version, preserving validated local metadata."""
    if "+" in version:
        cargo_compatible_version(version)
        return version
    return canonical_version(version)


def _version_key(version: str) -> tuple[int, int, int, int, int]:
    match = _VERSION_RE.fullmatch(canonical_version(version))
    assert match is not None
    dev = match.group("dev")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        int(dev is None),
        int(dev or 0),
    )


def next_development_version(declared: str, published: str | None, serial: int) -> str:
    """Return the next-patch development version with a retry-stable run serial."""
    if serial <= 0:
        raise ValueError(f"Development serial must be positive, got {serial}")
    if published is not None:
        published_match = _VERSION_RE.fullmatch(canonical_version(published))
        assert published_match is not None
        if published_match.group("dev") == str(serial):
            return canonical_version(published)
    base = max((declared, published), key=_version_key) if published is not None else declared
    major, minor, patch, _, _ = _version_key(base)
    return f"{major}.{minor}.{patch + 1}.dev{serial}"


def latest_supported_version(versions: Iterable[str]) -> str | None:
    """Return the greatest stable or development version from a PyPI release list."""
    supported = []
    for version in versions:
        try:
            canonical = canonical_version(version)
        except ValueError:
            continue
        supported.append(canonical)
    return max(supported, key=_version_key) if supported else None


def _version_from_path(path: Path) -> str:
    text = path.read_text()
    if path.suffix == ".toml":
        data = tomllib.loads(text)
        project = data.get("project")
        if isinstance(project, dict) and isinstance(project.get("version"), str):
            return canonical_version(project["version"])
        cargo_package = data.get("package")
        if isinstance(cargo_package, dict) and isinstance(cargo_package.get("version"), str):
            return canonical_version(cargo_package["version"])
    else:
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if match is not None:
            return canonical_version(match.group(1))
    raise ValueError(f"No project/package version in {path}")


def _declared_version(repo_root: Path, package: PackageFamily) -> str:
    versions = (_version_from_path(repo_root / path) for path in package.declared_version_paths)
    return max(versions, key=_version_key)


def latest_pypi_version(distribution: str) -> str | None:
    """Return the latest supported PyPI release, or None before the first release."""
    url = PYPI_PROJECT_JSON_URL.format(distribution=distribution)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    releases = data.get("releases", {})
    return latest_supported_version(releases)


def latest_family_version(package: PackageFamily) -> str | None:
    """Return the greatest supported release across a coupled package family."""
    versions = [version for distribution in package.artifacts if (version := latest_pypi_version(distribution))]
    return max(versions, key=_version_key) if versions else None


def resolve_version(
    package_name: str,
    mode: str,
    explicit_version: str | None,
    revision: str | None,
    development_serial: int | None,
    repo_root: Path,
) -> str:
    """Resolve one version for every artifact in a release workflow run."""
    package = PACKAGES[package_name]
    if explicit_version:
        return canonical_version(explicit_version)
    if mode == ReleaseMode.STABLE:
        raise ValueError("Stable mode requires an explicit version")
    declared = _declared_version(repo_root, package)
    if mode == ReleaseMode.DEVELOPMENT:
        if development_serial is None:
            raise ValueError("Development mode requires a release serial")
        return next_development_version(
            declared,
            latest_family_version(package),
            development_serial,
        )
    if mode == ReleaseMode.MANUAL:
        if revision is None:
            raise ValueError("Manual mode requires --revision")
        return f"{declared}+{revision[:8]}"
    raise ValueError(f"Unknown release mode {mode!r}")


def packages_for_changes(paths: Iterable[str]) -> list[str]:
    """Return package families affected by native source paths."""
    changed = tuple(paths)
    if any(fnmatch.fnmatch(path, pattern) for path in changed for pattern in SHARED_PR_PATTERNS):
        return sorted(PACKAGES)
    selected = []
    for name, package in PACKAGES.items():
        patterns = package.source_patterns
        if isinstance(package.build, NativeBuild):
            patterns = (*SHARED_SOURCE_PATTERNS, *patterns)
        if any(fnmatch.fnmatch(path, pattern) for path in changed for pattern in patterns):
            selected.append(name)
    return sorted(selected)


def package_for_tag(ref: str) -> tuple[str, str]:
    """Return the package family and version encoded in a release tag ref."""
    tag = ref.removeprefix("refs/tags/")
    matches = [
        (name, tag.removeprefix(package.tag_prefix))
        for name, package in PACKAGES.items()
        if tag.startswith(package.tag_prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown native package release tag {tag!r}")
    name, version = matches[0]
    return name, canonical_version(version)


def update_native_requirement(text: str, distribution: str, version: str) -> tuple[str, bool]:
    """Raise one quoted TOML requirement floor without changing surrounding text."""
    version = canonical_version(version)
    pattern = re.compile(rf'(?P<prefix>"{re.escape(distribution)}\s*>=\s*)(?P<version>[^"]+)(?P<suffix>")')
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected one {distribution!r} requirement, found {len(matches)}")
    current = canonical_version(matches[0].group("version").strip())
    if _version_key(current) >= _version_key(version):
        return text, False
    return pattern.sub(rf"\g<prefix>{version}\g<suffix>", text, count=1), True


def _normalized_distribution(distribution: str) -> str:
    return re.sub(r"[-_.]+", "_", distribution).lower()


def _artifact_distribution(package: PackageFamily, version: str, filename: str) -> tuple[str, str]:
    for distribution in package.artifacts:
        prefix = f"{_normalized_distribution(distribution)}-{version}"
        if filename == f"{prefix}.tar.gz":
            return distribution, "sdist"
        if filename.startswith(f"{prefix}-") and filename.endswith(".whl"):
            return distribution, "wheel"
    raise ValueError(f"Unexpected artifact in release manifest: {filename}")


def artifact_manifest(
    package_name: str,
    version: str,
    paths: Iterable[Path],
) -> dict[str, str]:
    """Validate the complete local artifact set and return filename-to-SHA256."""
    package = PACKAGES[package_name]
    version = canonical_version(version)
    counts = {distribution: {"wheel": 0, "sdist": 0} for distribution in package.artifacts}
    manifest: dict[str, str] = {}
    for path in paths:
        distribution, kind = _artifact_distribution(package, version, path.name)
        if path.name in manifest:
            raise ValueError(f"Duplicate artifact in release manifest: {path.name}")
        expectation = package.artifacts[distribution]
        if kind == "wheel":
            pure_python_wheel = path.name.endswith("-py3-none-any.whl")
            if pure_python_wheel != expectation.pure_python:
                raise ValueError(f"Wrong wheel kind in release manifest: {path.name}")
        counts[distribution][kind] += 1
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    errors = []
    for distribution, expectation in package.artifacts.items():
        actual = counts[distribution]
        expected = {"wheel": expectation.wheels, "sdist": expectation.sdists}
        if actual != expected:
            errors.append(f"{distribution}: expected {expected}, found {actual}")
    if errors:
        raise ValueError(f"Incomplete artifact manifest for {package_name}: {'; '.join(errors)}")
    return manifest


def reconcile_published_artifacts(
    package_name: str,
    version: str,
    local_manifest: Mapping[str, str],
    published: Mapping[str, Iterable[PublishedArtifact]],
) -> set[str]:
    """Return missing local filenames when PyPI contains only a matching subset."""
    package = PACKAGES[package_name]
    version = canonical_version(version)
    expected_distributions = set(package.artifacts)
    unexpected_distributions = set(published) - expected_distributions
    if unexpected_distributions:
        raise ValueError(f"Unexpected PyPI projects: {sorted(unexpected_distributions)}")

    existing = set()
    for distribution in expected_distributions:
        for artifact in published.get(distribution, ()):
            try:
                local_hash = local_manifest[artifact.filename]
            except KeyError:
                raise ValueError(f"Unexpected PyPI artifact for {distribution} {version}: {artifact.filename}") from None
            if local_hash != artifact.sha256:
                raise ValueError(f"PyPI hash mismatch for {artifact.filename}")
            existing.add(artifact.filename)
    return set(local_manifest) - existing


def published_artifacts(distribution: str, version: str) -> list[PublishedArtifact]:
    """Return artifacts already present for one exact PyPI project version."""
    url = PYPI_VERSION_JSON_URL.format(distribution=distribution, version=canonical_version(version))
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise
    return [
        PublishedArtifact(filename=item["filename"], sha256=item["digests"]["sha256"]) for item in data.get("urls", [])
    ]


def _canonical_package_entries(lock_text: str) -> tuple[dict, dict[str, list[str]]]:
    lock = tomllib.loads(lock_text)
    metadata = {key: value for key, value in lock.items() if key != "package"}
    packages: dict[str, list[str]] = {}
    for package in lock.get("package", []):
        packages.setdefault(package["name"], []).append(json.dumps(package, sort_keys=True))
    for entries in packages.values():
        entries.sort()
    return metadata, packages


def validate_targeted_lock_change(
    before: str,
    after: str,
    distribution: str,
    version: str,
    *,
    allowed_packages: Iterable[str] = (),
) -> None:
    """Reject lock changes outside the released distribution and requirement owner."""
    before_metadata, before_packages = _canonical_package_entries(before)
    after_metadata, after_packages = _canonical_package_entries(after)
    if before_metadata != after_metadata:
        raise ValueError("Targeted native lock update changed lockfile metadata")

    allowed = {distribution, *allowed_packages}
    changed = {
        name
        for name in set(before_packages) | set(after_packages)
        if before_packages.get(name) != after_packages.get(name)
    }
    unrelated = changed - allowed
    if unrelated:
        raise ValueError(f"Targeted native lock update changed unrelated packages: {sorted(unrelated)}")

    target_entries = [json.loads(entry) for entry in after_packages.get(distribution, ())]
    expected = canonical_version(version)
    if len(target_entries) != 1 or target_entries[0].get("version") != expected:
        versions = sorted(entry.get("version") for entry in target_entries)
        raise ValueError(f"Expected {distribution}=={expected} in uv.lock, found {versions}")
    if "registry" not in target_entries[0].get("source", {}):
        raise ValueError(f"Expected {distribution}=={expected} to resolve from a registry")


def _emit_github_output(path: Path | None, **values: str) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def bump_native_requirement(
    package_name: str,
    version: str,
    repo_root: Path,
) -> None:
    """Advance the compatibility floor and exact universal lock entry."""
    package = PACKAGES[package_name]
    build = package.build
    if not isinstance(build, NativeBuild):
        raise ValueError(f"{package_name} has no native dependency floor")
    version = canonical_version(version)
    requirement_path = repo_root / build.requirement_path
    lock_path = repo_root / "uv.lock"
    original_requirement = requirement_path.read_text()
    _, changed = update_native_requirement(
        original_requirement,
        build.requirement_distribution,
        version,
    )
    if not changed:
        return

    original_lock = lock_path.read_text()
    command = [
        "uv",
        "add",
        "--package",
        build.requirement_owner,
        f"{build.requirement_distribution}>={version}",
        "--upgrade-package",
        f"{build.requirement_distribution}=={version}",
        "--no-sync",
    ]
    for attempt, delay in enumerate(LOCK_RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        requirement_path.write_text(original_requirement)
        lock_path.write_text(original_lock)
        result = subprocess.run(command, cwd=repo_root)
        if result.returncode != 0:
            if attempt < len(LOCK_RETRY_DELAYS):
                continue
            requirement_path.write_text(original_requirement)
            lock_path.write_text(original_lock)
            raise RuntimeError(f"`{' '.join(command)}` failed after {attempt} attempts")
        updated_lock = lock_path.read_text()
        try:
            validate_targeted_lock_change(
                original_lock,
                updated_lock,
                build.requirement_distribution,
                version,
                allowed_packages=(build.requirement_owner,),
            )
        except ValueError:
            requirement_path.write_text(original_requirement)
            lock_path.write_text(original_lock)
            raise
        check = subprocess.run(["uv", "lock", "--check"], cwd=repo_root)
        if check.returncode != 0:
            if attempt < len(LOCK_RETRY_DELAYS):
                continue
            requirement_path.write_text(original_requirement)
            lock_path.write_text(original_lock)
            raise RuntimeError(f"`uv lock --check` failed after {attempt} attempts")
        return
    raise AssertionError("Unreachable lock retry loop")


def _zig_platform_key() -> str:
    architecture = {
        "x86_64": "x86_64",
        "AMD64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(platform.machine())
    operating_system = {"Darwin": "macos", "Linux": "linux"}.get(platform.system())
    if architecture is None or operating_system is None:
        raise ValueError(f"Unsupported Zig host: {platform.system()} {platform.machine()}")
    return f"{architecture}-{operating_system}"


def _ensure_zig(repo_root: Path) -> str:
    existing = shutil.which("zig")
    if existing:
        return existing
    tools_dir = repo_root / ".tools"
    platform_key = _zig_platform_key()
    zig_dir = tools_dir / f"zig-{platform_key}-{ZIG_VERSION}"
    zig = zig_dir / "zig"
    if zig.exists():
        return str(zig)

    filename = f"zig-{platform_key}-{ZIG_VERSION}.tar.xz"
    tools_dir.mkdir(parents=True, exist_ok=True)
    archive = tools_dir / filename
    sources = [f"{base}/{filename}" for base in ZIG_MIRRORS]
    sources.append(f"{ZIG_OFFICIAL_BASE}/{ZIG_VERSION}/{filename}")
    last_error: Exception | None = None
    for url in sources:
        for _ in range(ZIG_DOWNLOAD_ATTEMPTS_PER_SOURCE):
            try:
                urllib.request.urlretrieve(url, archive)
                with tarfile.open(archive, "r:xz") as tar:
                    tar.extractall(tools_dir, filter="data")
                archive.unlink()
                return str(zig)
            except (urllib.error.URLError, OSError) as error:
                last_error = error
                archive.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download Zig {ZIG_VERSION}") from last_error


def _maturin(repo_root: Path, build: NativeBuild, *args: str, env: Mapping[str, str] | None = None) -> None:
    subprocess.run(
        ["uvx", "--from", "maturin>=1.5,<2.0", "maturin", *args],
        cwd=repo_root / build.native_path,
        env=env,
        check=True,
    )


def _stamp_versions(repo_root: Path, build: NativeBuild, version: str) -> None:
    pattern = re.compile(r'^(version\s*=\s*)"[^"]+"', re.MULTILINE)
    for relative_path in build.version_paths:
        path = repo_root / relative_path
        updated, count = pattern.subn(rf'\1"{version}"', path.read_text(), count=1)
        if count != 1:
            raise ValueError(f"Could not stamp release version in {relative_path}")
        path.write_text(updated)


def _uv_build(repo_root: Path, pure_path: Path, dist_dir: Path, kind: str) -> None:
    subprocess.run(
        ["uv", "build", f"--{kind}", "--out-dir", str(dist_dir)],
        cwd=repo_root / pure_path,
        check=True,
    )
    (dist_dir / ".gitignore").unlink(missing_ok=True)


def _build_python_bundle(build: PythonBundle, version: str, operation: str, repo_root: Path) -> None:
    if operation != BuildOperation.PYTHON:
        raise ValueError(f"Python bundle does not support build operation {operation!r}")
    subprocess.run(
        [
            sys.executable,
            build.script_path.as_posix(),
            "--mode",
            "stable",
            "--version",
            python_compatible_version(version),
        ],
        cwd=repo_root,
        check=True,
    )
    (repo_root / "dist" / ".gitignore").unlink(missing_ok=True)


def _build_native_package(build: NativeBuild, version: str, operation: str, repo_root: Path) -> None:
    build_version = cargo_compatible_version(version)
    _stamp_versions(repo_root, build, build_version)
    dist_dir = repo_root / "dist"
    if operation == BuildOperation.SDIST:
        dist_dir.mkdir(exist_ok=True)
        _maturin(repo_root, build, "sdist", "--out", str(dist_dir))
        if build.pure_path is not None:
            _uv_build(repo_root, build.pure_path, dist_dir, "sdist")
        return

    targets = {BuildOperation.LINUX: LINUX_TARGETS, BuildOperation.MACOS: MAC_TARGETS}.get(operation)
    if targets is None:
        raise ValueError(f"Unknown native build operation {operation!r}")
    if operation == BuildOperation.MACOS and platform.system() != "Darwin":
        raise ValueError("macOS wheels require a macOS host")
    shutil.rmtree(dist_dir, ignore_errors=True)
    dist_dir.mkdir()
    subprocess.run(["rustup", "target", "add", *(target for target, _ in targets)], check=True)
    environment: Mapping[str, str] | None = None
    if operation == BuildOperation.LINUX:
        zig_dir = str(Path(_ensure_zig(repo_root)).parent)
        environment = {**os.environ, "PATH": f"{zig_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    for target, manylinux in targets:
        arguments = ["build", "--release", "--out", str(dist_dir), "--target", target]
        if manylinux:
            arguments.extend(("--manylinux", manylinux))
        if operation == BuildOperation.LINUX:
            arguments.append("--zig")
        _maturin(repo_root, build, *arguments, env=environment)
    if operation == BuildOperation.LINUX and build.pure_path is not None:
        _uv_build(repo_root, build.pure_path, dist_dir, "wheel")


def build_package(package_name: str, version: str, operation: str, repo_root: Path) -> None:
    """Build one package-family matrix leg."""
    build = PACKAGES[package_name].build
    if isinstance(build, PythonBundle):
        _build_python_bundle(build, version, operation, repo_root)
        return
    _build_native_package(build, version, operation, repo_root)


def validate_native_wheel(package_name: str, version: str, dist_dir: Path) -> None:
    """Install and import the compatible native wheel from one build leg."""
    package = PACKAGES[package_name]
    build = package.build
    if not isinstance(build, NativeBuild):
        raise ValueError(f"{package_name} has no native wheel")
    with tempfile.TemporaryDirectory(prefix="native-wheel-") as temporary:
        environment = Path(temporary)
        subprocess.run(["uv", "venv", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-index",
                "--find-links",
                str(dist_dir),
                "--no-deps",
                f"{build.requirement_distribution}=={python_compatible_version(version)}",
            ],
            check=True,
        )
        subprocess.run(
            [str(python), "-c", f"import {build.import_name}"],
            check=True,
        )


def release_plan(
    *,
    event_name: str,
    ref: str,
    input_mode: str,
    input_package: str,
    input_version: str,
    revision: str,
    serial: int,
    changed_paths: Iterable[str],
    repo_root: Path,
) -> ReleasePlan:
    """Resolve affected packages, immutable versions, and build matrix legs."""
    if event_name == "push" and ref.startswith("refs/tags/"):
        package_name, tag_version = package_for_tag(ref)
        selected = [package_name]
        mode = ReleaseMode.STABLE
        explicit_versions = {package_name: tag_version}
    elif event_name == "schedule":
        selected = [PYTHON_LIBS_FAMILY]
        mode = ReleaseMode.DEVELOPMENT
        explicit_versions = {}
    elif event_name == "workflow_dispatch":
        selected = sorted(PACKAGES) if input_package == "all" else [input_package]
        unknown = set(selected) - set(PACKAGES)
        if unknown:
            raise ValueError(f"Unknown package family: {sorted(unknown)}")
        if input_version and len(selected) != 1:
            raise ValueError("An explicit version requires one package family")
        mode = ReleaseMode(input_mode)
        if mode == ReleaseMode.STABLE and len(selected) != 1:
            raise ValueError("Stable dispatch requires one package family")
        explicit_versions = {selected[0]: input_version} if input_version else {}
    elif event_name == "push" and ref == "refs/heads/main":
        selected = packages_for_changes(changed_paths)
        mode = ReleaseMode.DEVELOPMENT
        explicit_versions = {}
    elif event_name == "pull_request":
        selected = packages_for_changes(changed_paths)
        mode = ReleaseMode.MANUAL
        explicit_versions = {}
    else:
        raise ValueError(f"Unsupported release event {event_name!r} on {ref!r}")
    if not selected:
        raise ValueError("Package release workflow did not find an affected package")

    versions = {}
    builds = []
    for name in selected:
        version = resolve_version(
            name,
            mode,
            explicit_versions.get(name),
            revision,
            serial,
            repo_root,
        )
        versions[name] = ReleaseVersion(
            mode=mode,
            version=version,
            build_version=cargo_compatible_version(version),
        )
        builds.extend(
            {"package": name, "os": operating_system, "operation": operation}
            for operating_system, operation in PACKAGES[name].build_legs
        )
    bump = any(isinstance(PACKAGES[name].build, NativeBuild) for name in selected)
    return ReleasePlan(packages=tuple(selected), versions=MappingProxyType(versions), builds=tuple(builds), bump=bump)


def _verify_command(args: argparse.Namespace) -> None:
    paths = sorted(path for path in args.dist_dir.iterdir() if path.is_file())
    manifest = artifact_manifest(args.package, args.version, paths)
    package = PACKAGES[args.package]
    remote = {distribution: published_artifacts(distribution, args.version) for distribution in package.artifacts}
    missing = reconcile_published_artifacts(args.package, args.version, manifest, remote)
    print(json.dumps({"artifacts": manifest, "missing": sorted(missing)}, indent=2, sort_keys=True))


def requirement_paths_for_packages(package_names: Iterable[str]) -> tuple[Path, ...]:
    """Return each native compatibility-floor path once."""
    paths = set()
    for name in package_names:
        build = PACKAGES[name].build
        if isinstance(build, NativeBuild):
            paths.add(build.requirement_path)
    return tuple(sorted(paths))


def _bump_releases_command(args: argparse.Namespace) -> None:
    versions = json.loads(args.versions)
    selected = [name for name in sorted(versions) if isinstance(PACKAGES[name].build, NativeBuild)]
    requirement_paths = requirement_paths_for_packages(selected)
    before = {path: (args.repo_root / path).read_text() for path in requirement_paths}
    for name in selected:
        bump_native_requirement(name, versions[name]["version"], args.repo_root)
    changed_paths = [path for path in requirement_paths if (args.repo_root / path).read_text() != before[path]]
    _emit_github_output(
        args.github_output,
        changed=str(bool(changed_paths)).lower(),
        requirement_paths=" ".join(path.as_posix() for path in changed_paths),
    )


def _plan_command(args: argparse.Namespace) -> None:
    changed_paths = args.changed_files.read_text().splitlines() if args.changed_files else ()
    plan = release_plan(
        event_name=args.event_name,
        ref=args.ref,
        input_mode=args.mode,
        input_package=args.package,
        input_version=args.version,
        revision=args.revision,
        serial=args.serial,
        changed_paths=changed_paths,
        repo_root=args.repo_root,
    )
    values = {
        "packages": json.dumps(plan.packages, separators=(",", ":")),
        "versions": json.dumps(
            {name: asdict(version) for name, version in plan.versions.items()},
            separators=(",", ":"),
        ),
        "build_matrix": json.dumps({"include": plan.builds}, separators=(",", ":")),
        "bump": str(plan.bump).lower(),
        "publish": str(next(iter(plan.versions.values())).mode != ReleaseMode.MANUAL).lower(),
    }
    print(json.dumps(values, indent=2, sort_keys=True))
    _emit_github_output(args.github_output, **values)


def _build_command(args: argparse.Namespace) -> None:
    build_package(args.package, args.version, args.operation, args.repo_root)


def _validate_wheel_command(args: argparse.Namespace) -> None:
    validate_native_wheel(args.package, args.version, args.dist_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    verify = subparsers.add_parser("verify-release")
    verify.add_argument("--package", choices=PACKAGES, required=True)
    verify.add_argument("--version", required=True)
    verify.add_argument("--dist-dir", type=Path, default=Path("dist"))
    verify.set_defaults(func=_verify_command)

    bump_releases = subparsers.add_parser("bump-releases")
    bump_releases.add_argument("--versions", required=True)
    bump_releases.add_argument("--repo-root", type=Path, default=Path.cwd())
    bump_releases.add_argument("--github-output", type=Path)
    bump_releases.set_defaults(func=_bump_releases_command)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--event-name", required=True)
    plan.add_argument("--ref", required=True)
    plan.add_argument("--mode", default="manual")
    plan.add_argument("--package", default="all")
    plan.add_argument("--version", default="")
    plan.add_argument("--revision", required=True)
    plan.add_argument("--serial", type=int, required=True)
    plan.add_argument("--changed-files", type=Path)
    plan.add_argument("--repo-root", type=Path, default=Path.cwd())
    plan.add_argument("--github-output", type=Path)
    plan.set_defaults(func=_plan_command)

    build = subparsers.add_parser("build")
    build.add_argument("--package", choices=PACKAGES, required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--operation", required=True)
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.set_defaults(func=_build_command)

    validate_wheel = subparsers.add_parser("validate-wheel")
    validate_wheel.add_argument("--package", choices=PACKAGES, required=True)
    validate_wheel.add_argument("--version", required=True)
    validate_wheel.add_argument("--dist-dir", type=Path, default=Path("dist"))
    validate_wheel.set_defaults(func=_validate_wheel_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
