# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path-agnostic config discovery for cluster YAML files.

Generic YAML config discovery helpers. Callers (e.g. iris) pass the
directories to search; this module knows nothing about any particular
marin sub-package.
"""

import functools
import logging
import tomllib
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_YAML_SUFFIXES = (".yaml", ".yml")


@functools.lru_cache(maxsize=128)
def find_project_root(start: Path | str | None = None) -> Path | None:
    """Find the marin workspace root.

    Walks up from ``start`` (or the current working directory) looking for a
    ``pyproject.toml`` that declares ``[tool.uv.workspace]``. This uniquely
    identifies the top-level marin root and avoids matching a workspace
    member's pyproject (e.g. ``lib/iris/pyproject.toml``).

    Returns:
        The marin root ``Path``, or ``None`` when running outside a marin
        checkout (e.g. from an installed pip package).
    """
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()

    for directory in (current, *current.parents):
        pp = directory / "pyproject.toml"
        if pp.is_file() and _declares_uv_workspace(pp):
            logger.debug("Found marin workspace root: %s", directory)
            return directory

    logger.debug("No marin workspace root found starting from %s", current)
    return None


def _declares_uv_workspace(pyproject_path: Path) -> bool:
    """Return True if ``pyproject_path`` declares ``[tool.uv.workspace]``."""
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return "workspace" in data.get("tool", {}).get("uv", {})


def _resolve_dirs(dirs: Sequence[Path | str]) -> list[Path]:
    """Expand ``~`` and resolve relative dirs against the marin project root.

    An empty string resolves to the project root itself. Absolute paths are
    returned unchanged. Relative paths are joined onto the marin project root
    when one is found, and fall back to the current working directory otherwise.
    """
    root = find_project_root()
    resolved: list[Path] = []
    for raw in dirs:
        p = Path(raw).expanduser()
        if p.is_absolute():
            resolved.append(p)
        elif root is not None:
            resolved.append(root / p)
        else:
            resolved.append(Path.cwd() / p)
    return resolved


def find_configs(
    dirs: Sequence[Path | str],
    name: str | None = None,
) -> dict[str, Path]:
    """Discover YAML config files across ``dirs``.

    Relative ``dirs`` are resolved against the marin project root (see
    :func:`find_project_root`); absolute paths are used as-is; ``~`` is
    expanded. An empty string resolves to the project root itself.

    Args:
        dirs: Directories to search, in priority order.
        name: When given, only return entries whose stem equals ``name``.

    Returns:
        A dict mapping config stem (filename without ``.yaml``/``.yml``) to
        its resolved ``Path``. When the same stem appears in multiple dirs,
        the first (highest-priority) match wins.
    """
    configs: dict[str, Path] = {}
    for directory in _resolve_dirs(dirs):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix not in _YAML_SUFFIXES:
                continue
            stem = path.stem
            if name is not None and stem != name:
                continue
            if stem not in configs:
                configs[stem] = path
    return configs


def resolve_cluster_config(name: str, dirs: Sequence[Path | str]) -> Path:
    """Resolve a cluster name (or path) to an existing YAML config file.

    If ``name`` is already an existing file path, it is returned directly.
    Otherwise ``dirs`` are searched for a file whose stem matches ``name``
    (with ``.yaml`` or ``.yml`` extensions stripped from ``name`` before
    comparison).

    Args:
        name: Cluster name (e.g. ``"marin-dev"``) or path to an existing file.
        dirs: Directories to search.

    Returns:
        The resolved ``Path`` to the config file.

    Raises:
        FileNotFoundError: When no matching config file is found, with a
            message listing all searched locations.
    """
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate

    # Allow callers to pass either "marin-dev" or "marin-dev.yaml".
    name_path = Path(name)
    search_stem = name_path.stem if name_path.suffix in _YAML_SUFFIXES else name

    matches = find_configs(dirs, name=search_stem)
    if search_stem in matches:
        logger.debug("Resolved cluster config %r -> %s", name, matches[search_stem])
        return matches[search_stem]

    searched_str = "\n  ".join(str(d) for d in _resolve_dirs(dirs))
    raise FileNotFoundError(f"No config file found for cluster {name!r}.\nSearched directories:\n  {searched_str}")


def list_cluster_configs(dirs: Sequence[Path | str]) -> dict[str, Path]:
    """List all YAML cluster configs across ``dirs``.

    Thin alias over :func:`find_configs` for callers that want the full
    name-to-path mapping.
    """
    return find_configs(dirs)
