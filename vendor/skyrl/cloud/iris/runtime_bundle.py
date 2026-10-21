"""Resolve, build, and verify the launcher runtime synced into each Iris task."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from cloud.iris.paths import PROJECT_ROOT

BUNDLE_FILE_MANIFEST = Path("cloud/iris/runtime_bundle_files.txt")
BUNDLE_IDENTITY_FILE = ".marinskyrl-runtime.json"
PROVENANCE_FILE = Path("marinskyrl/_vendor_provenance.json")
RUNTIME_SOURCE_PREFIXES = (
    "cloud/",
    "marinskyrl/",
    "chat_templates/",
    "skyrl-train/skyrl_train/",
    "skyrl-gym/skyrl_gym/",
    "open_mopd_versions.py",
    "harbor/src/",
)


@dataclass(frozen=True)
class LauncherSource:
    """Immutable source selected for launcher runtime files."""

    root: Path
    digest: str


@dataclass(frozen=True)
class RuntimeBundleFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class RuntimeBundleIdentity:
    source_digest: str
    files: tuple[RuntimeBundleFile, ...]


def resolve_launcher_source() -> LauncherSource:
    """Identify the checked-out runtime source, assets, and independent lock."""
    files = tuple(
        RuntimeBundleFile(path=path.as_posix(), sha256=_sha256(PROJECT_ROOT / path))
        for path in _bundle_paths(PROJECT_ROOT)
    )
    return LauncherSource(root=PROJECT_ROOT, digest=_files_digest(files))


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


def _bundle_paths(root: Path) -> tuple[Path, ...]:
    names = set(read_manifest_paths(root))
    names.update(
        {
            BUNDLE_FILE_MANIFEST.as_posix(),
            PROVENANCE_FILE.as_posix(),
            "pyproject.toml",
            "uv.lock",
            "harbor/pyproject.toml",
            "harbor/README.md",
            "harbor/LICENSE",
        }
    )
    harbor_root = root / "harbor"
    if harbor_root.is_symlink() and harbor_root.readlink() != Path("../harbor"):
        raise ValueError("Vendored Harbor source must point to ../harbor")
    for prefix in RUNTIME_SOURCE_PREFIXES:
        source = root / prefix
        if source.is_dir():
            names.update(path.relative_to(root).as_posix() for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            names.add(prefix)
    paths = []
    for name in sorted(names):
        path = _bundle_relative_path(name)
        if {"tests", "__pycache__"}.intersection(path.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        original = root / path
        allowed_root = harbor_root.resolve() if path.parts[0] == "harbor" else root.resolve()
        original.resolve().relative_to(allowed_root)
        if not original.is_file():
            raise FileNotFoundError(f"Runtime source file is missing: {name}")
        paths.append(path)
    return tuple(paths)


def _files_digest(files: tuple[RuntimeBundleFile, ...]) -> str:
    encoded = json.dumps([(entry.path, entry.sha256) for entry in files], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundled_runtime(workspace: Path | None = None) -> str:
    """Verify the files synced into an Iris task and return their source digest."""
    root = workspace or PROJECT_ROOT
    identity_path = root / BUNDLE_IDENTITY_FILE
    if not identity_path.is_file():
        raise RuntimeError(f"Runtime bundle identity is missing: {identity_path}")
    value = json.loads(identity_path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Runtime bundle identity is invalid: {identity_path}")
    source_digest = value.get("source_digest")
    file_values = value.get("files")
    if not isinstance(source_digest, str) or not isinstance(file_values, list):
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
    actual = tuple(RuntimeBundleFile(path=path.as_posix(), sha256=_sha256(root / path)) for path in _bundle_paths(root))
    if _files_digest(actual) != source_digest or _files_digest(tuple(files)) != source_digest:
        raise ValueError("Bundled source files do not match the selected runtime identity")
    return source_digest


def runtime_bundle_inputs(expected_source_digest: str) -> tuple[LauncherSource, tuple[Path, ...]]:
    """Validate and return the runtime files identified by a launch request."""
    source = resolve_launcher_source()
    if source.digest != expected_source_digest:
        raise ValueError(
            f"Selected runtime source identity {source.digest} does not match requested {expected_source_digest}"
        )
    paths = _bundle_paths(source.root)
    return source, paths


@contextmanager
def build_runtime_bundle(expected_source_digest: str) -> Iterator[Path]:
    """Own the copied Iris workspace until submission and supervision finish."""
    source, paths = runtime_bundle_inputs(expected_source_digest)

    with tempfile.TemporaryDirectory(prefix="marinskyrl-runtime-bundle-") as directory:
        workspace = Path(directory)
        files: list[RuntimeBundleFile] = []
        for relative_path in paths:
            destination = workspace / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.root / relative_path, destination)
            files.append(RuntimeBundleFile(path=relative_path.as_posix(), sha256=_sha256(destination)))
        identity = RuntimeBundleIdentity(source_digest=source.digest, files=tuple(files))
        if _files_digest(identity.files) != expected_source_digest:
            raise ValueError("Runtime source changed while building its bundle")
        (workspace / BUNDLE_IDENTITY_FILE).write_text(json.dumps(asdict(identity), indent=2, sort_keys=True) + "\n")
        yield workspace
