"""Resolve, build, and verify the launcher runtime synced into each Iris task."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlparse

from cloud.iris.paths import PROJECT_ROOT

BUNDLE_FILE_MANIFEST = Path("cloud/iris/runtime_bundle_files.txt")
BUNDLE_IDENTITY_FILE = ".marinskyrl-runtime.json"
DISTRIBUTION_NAME = "marinskyrl"


class LauncherSourceKind(StrEnum):
    CHECKOUT = "checkout"
    INSTALLED = "installed"


@dataclass(frozen=True)
class LauncherSource:
    """Immutable source selected for launcher runtime files."""

    root: Path
    commit: str
    kind: LauncherSourceKind = LauncherSourceKind.CHECKOUT


@dataclass(frozen=True)
class RuntimeBundleFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class RuntimeBundleIdentity:
    launcher_commit: str
    files: tuple[RuntimeBundleFile, ...]


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout_root(path: Path) -> Path | None:
    """Return the enclosing MarinSkyRL checkout, or None when the path is outside one."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    root = Path(result.stdout.strip()).resolve()
    project_file = root / "pyproject.toml"
    if not project_file.is_file() or not (root / BUNDLE_FILE_MANIFEST).is_file():
        return None
    project = tomllib.loads(project_file.read_text()).get("project", {})
    return root if project.get("name") == DISTRIBUTION_NAME else None


def _installed_source() -> LauncherSource:
    try:
        distribution = importlib.metadata.distribution(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Cannot locate an installed marinskyrl distribution outside a checkout") from error
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        raise RuntimeError("Installed marinskyrl distribution has no direct_url.json source identity")

    source_identity = json.loads(direct_url)
    parsed_url = urlparse(source_identity.get("url", ""))
    if parsed_url.scheme == "file":
        checkout = Path(unquote(parsed_url.path)).resolve()
        if not checkout.is_dir():
            raise RuntimeError(f"Installed marinskyrl checkout is missing: {checkout}")
        root = _checkout_root(checkout)
        if root is None:
            raise RuntimeError(f"Installed marinskyrl checkout is missing or invalid: {checkout}")
        return LauncherSource(root=root, commit=_git_output(root, "rev-parse", "HEAD"))

    vcs_info = source_identity.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        raise RuntimeError("Installed marinskyrl distribution does not identify a Git commit")
    commit = vcs_info.get("commit_id")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("Installed marinskyrl distribution does not identify a Git commit")
    root = Path(distribution.locate_file("")).resolve()
    if not (root / BUNDLE_FILE_MANIFEST).is_file():
        raise RuntimeError(f"Installed marinskyrl distribution is missing {BUNDLE_FILE_MANIFEST}")
    return LauncherSource(root=root, commit=commit, kind=LauncherSourceKind.INSTALLED)


def resolve_launcher_source() -> LauncherSource:
    """Resolve the immutable runtime bytes a launch will ship."""
    checkout = _checkout_root(Path.cwd())
    if checkout is not None:
        return LauncherSource(root=checkout, commit=_git_output(checkout, "rev-parse", "HEAD"))
    return _installed_source()


def _bundle_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Runtime bundle path must stay below its workspace: {value}")
    return path


def read_manifest_paths(root: Path) -> tuple[str, ...]:
    """Return the path strings listed in the bundle manifest (no validation)."""
    manifest = root / BUNDLE_FILE_MANIFEST
    return tuple(
        value for line in manifest.read_text().splitlines() if (value := line.strip()) and not value.startswith("#")
    )


def _bundle_paths(source: LauncherSource) -> tuple[Path, ...]:
    manifest = source.root / BUNDLE_FILE_MANIFEST
    paths: list[Path] = []
    for value in read_manifest_paths(source.root):
        path = _bundle_relative_path(value)
        if path in paths:
            raise ValueError(f"Runtime bundle manifest contains a duplicate path: {value}")
        if not (source.root / path).is_file():
            raise FileNotFoundError(f"Runtime bundle file does not exist: {value}")
        paths.append(path)
    if not paths:
        raise ValueError(f"Runtime bundle manifest is empty: {manifest}")
    return tuple(paths)


def _reject_uncommitted_runtime(source: LauncherSource, paths: tuple[Path, ...]) -> None:
    if source.kind is LauncherSourceKind.INSTALLED:
        return
    relative_paths = [BUNDLE_FILE_MANIFEST.as_posix(), *(path.as_posix() for path in paths)]
    status = _git_output(source.root, "status", "--porcelain", "--untracked-files=all", "--", *relative_paths)
    if status:
        raise RuntimeError(
            "Runtime bundle inputs contain uncommitted changes, so launcher_commit cannot identify their bytes:\n"
            f"{status}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundled_runtime(workspace: Path | None = None) -> str:
    """Verify the files synced into an Iris task and return their launcher commit."""
    root = workspace or PROJECT_ROOT
    identity_path = root / BUNDLE_IDENTITY_FILE
    if not identity_path.is_file():
        raise RuntimeError(f"Runtime bundle identity is missing: {identity_path}")
    value = json.loads(identity_path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Runtime bundle identity is invalid: {identity_path}")
    launcher_commit = value.get("launcher_commit")
    file_values = value.get("files")
    if not isinstance(launcher_commit, str) or not isinstance(file_values, list):
        raise RuntimeError(f"Runtime bundle identity is invalid: {identity_path}")
    files: list[RuntimeBundleFile] = []
    for entry in file_values:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise RuntimeError(f"Runtime bundle identity is invalid: {identity_path}")
        files.append(RuntimeBundleFile(path=entry["path"], sha256=entry["sha256"]))
    for entry in files:
        relative_path = _bundle_relative_path(entry.path)
        bundled_file = root / relative_path
        if not bundled_file.is_file() or _sha256(bundled_file) != entry.sha256:
            raise RuntimeError(f"Runtime bundle file does not match its recorded identity: {entry.path}")
    return launcher_commit


def runtime_bundle_inputs(expected_launcher_commit: str) -> tuple[LauncherSource, tuple[Path, ...]]:
    """Validate and return the runtime files identified by a launch request."""
    source = resolve_launcher_source()
    if source.commit != expected_launcher_commit:
        raise ValueError(
            f"Selected launcher checkout commit {source.commit} does not match requested {expected_launcher_commit}"
        )
    paths = _bundle_paths(source)
    _reject_uncommitted_runtime(source, paths)
    return source, paths


def build_runtime_bundle(expected_launcher_commit: str) -> Path:
    """Copy the selected runtime files into an Iris workspace."""
    source, paths = runtime_bundle_inputs(expected_launcher_commit)

    workspace = Path(tempfile.mkdtemp(prefix="marinskyrl-runtime-bundle-"))
    files: list[RuntimeBundleFile] = []
    for relative_path in paths:
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.root / relative_path, destination)
        files.append(RuntimeBundleFile(path=relative_path.as_posix(), sha256=_sha256(destination)))
    identity = RuntimeBundleIdentity(launcher_commit=source.commit, files=tuple(files))
    (workspace / BUNDLE_IDENTITY_FILE).write_text(json.dumps(asdict(identity), indent=2, sort_keys=True) + "\n")
    return workspace
