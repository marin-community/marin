# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path-agnostic config discovery for cluster YAML files.

Allows users to reference cluster configs by short name (e.g. ``"marin-dev"``)
instead of hardcoded relative paths (e.g. ``"lib/iris/examples/marin-dev.yaml"``).
"""

import functools
import importlib.resources
import logging
from pathlib import Path
from collections.abc import Sequence

logger = logging.getLogger(__name__)

_YAML_SUFFIXES = (".yaml", ".yml")
_GIT_MARKER = ".git"
_PYPROJECT_MARKER = "pyproject.toml"
_USER_CONFIG_DIR = Path("~/.config/marin/clusters").expanduser()


@functools.lru_cache(maxsize=128)
def find_project_root(start: Path | str | None = None) -> Path | None:
    """Walk up from ``start`` looking for a project root directory.

    A directory is considered a project root if it contains ``.git`` or
    ``pyproject.toml``.

    Args:
        start: Directory to begin the search. Defaults to the current working
            directory when ``None``.

    Returns:
        The project root ``Path``, or ``None`` if no root marker was found
        (e.g. running from an installed pip package).
    """
    current = Path(start).resolve() if start is not None else Path.cwd()

    for directory in (current, *current.parents):
        if (directory / _GIT_MARKER).exists() or (directory / _PYPROJECT_MARKER).exists():
            logger.debug("Found project root: %s", directory)
            return directory

    logger.debug("No project root found starting from %s", current)
    return None


def _search_dirs(search_paths: Sequence[Path | str] | None) -> list[Path]:
    """Build the ordered list of directories to search for config files."""
    dirs: list[Path] = []

    if search_paths:
        dirs.extend(Path(p) for p in search_paths)

    root = find_project_root()
    if root is not None:
        dirs.append(root / "infra")
        dirs.append(root / "lib" / "iris" / "examples")

    try:
        iris_examples = importlib.resources.files("iris").joinpath("examples")
        # Convert to a Path only if it's a real filesystem path
        iris_path = Path(str(iris_examples))
        if iris_path.is_dir():
            dirs.append(iris_path)
    except (ImportError, TypeError):
        pass

    dirs.append(_USER_CONFIG_DIR)

    return dirs


def resolve_cluster_config(name: str, search_paths: Sequence[Path | str] | None = None) -> Path:
    """Resolve a cluster name or path to an existing YAML config file.

    If ``name`` is already an existing file path, it is returned directly.
    Otherwise the function searches a prioritized set of directories for a
    file named ``{name}``, ``{name}.yaml``, or ``{name}.yml``.

    Search order:
      1. Any explicit ``search_paths`` passed in
      2. Current working directory
      3. ``{project_root}/infra/`` (if project root found)
      4. ``{project_root}/lib/iris/examples/`` (if project root found)
      5. Installed ``iris`` package ``examples/`` directory (if importable)
      6. ``~/.config/marin/clusters/``

    Args:
        name: Cluster name (e.g. ``"marin-dev"``) or path to an existing file.
        search_paths: Additional directories to search before the defaults.

    Returns:
        The resolved ``Path`` to the config file.

    Raises:
        FileNotFoundError: When no matching config file is found, with a
            message listing all searched locations.
    """
    # If the name is already an existing file, return it directly.
    candidate = Path(name)
    if candidate.is_file():
        return candidate

    searched: list[str] = []
    for directory in _search_dirs(search_paths):
        for suffix in ("", *_YAML_SUFFIXES):
            path = directory / f"{name}{suffix}"
            searched.append(str(path))
            if path.is_file():
                logger.debug("Resolved cluster config %r -> %s", name, path)
                return path

    searched_str = "\n  ".join(searched)
    raise FileNotFoundError(f"No config file found for cluster {name!r}.\nSearched locations:\n  {searched_str}")


def list_cluster_configs(search_paths: Sequence[Path | str] | None = None) -> dict[str, Path]:
    """Discover all cluster config files across the standard search locations.

    Searches the same directories as :func:`resolve_cluster_config`. When the
    same config name appears in multiple locations, the first-found (highest
    priority) path wins.

    Args:
        search_paths: Additional directories to search before the defaults.

    Returns:
        A ``dict`` mapping config stem (filename without ``.yaml``/``.yml``
        extension) to the resolved ``Path``.
    """
    configs: dict[str, Path] = {}

    for directory in _search_dirs(search_paths):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix not in _YAML_SUFFIXES:
                continue
            stem = path.stem
            if stem not in configs:
                configs[stem] = path

    return configs
