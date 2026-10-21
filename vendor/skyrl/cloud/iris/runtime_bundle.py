"""Copy the checked-out SkyRL runtime into an Iris task workspace."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cloud.iris.paths import PROJECT_ROOT

BUNDLE_FILE_MANIFEST = Path("cloud/iris/runtime_bundle_files.txt")
RUNTIME_SOURCE_PREFIXES = (
    "cloud/",
    "marinskyrl/",
    "chat_templates/",
    "skyrl-train/skyrl_train/",
    "skyrl-gym/skyrl_gym/",
    "open_mopd_versions.py",
)


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
    names.update({BUNDLE_FILE_MANIFEST.as_posix(), "pyproject.toml", "uv.lock"})
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
        original.resolve().relative_to(root.resolve())
        if not original.is_file():
            raise FileNotFoundError(f"Runtime source file is missing: {name}")
        paths.append(path)
    return tuple(paths)


@contextmanager
def build_runtime_bundle() -> Iterator[Path]:
    """Own the copied Iris workspace until submission and supervision finish."""
    paths = _bundle_paths(PROJECT_ROOT)
    with tempfile.TemporaryDirectory(prefix="marinskyrl-runtime-bundle-") as directory:
        workspace = Path(directory)
        for relative_path in paths:
            destination = workspace / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative_path, destination)
        yield workspace
