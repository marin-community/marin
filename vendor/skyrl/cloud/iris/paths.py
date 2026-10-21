"""Path resolution helpers for the Iris RL launcher.

``PROJECT_ROOT`` is the MarinSkyRL repository root. The launcher syncs this tree
to ``/app`` inside the task container, so config paths are resolved relative to it
both on the launch host and in the pod.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from marinskyrl.resource_locator import is_hugging_face_repo_id

# cloud/iris/paths.py -> cloud/iris -> cloud -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Root directory of the MarinSkyRL project."""


def resolve_repo_path(path_like: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# Common config/template/data file extensions used to distinguish file paths from
# HuggingFace repo IDs in resolve_paths_in_dict.
_PATH_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".py",
    ".sh",
    ".jinja",
    ".jinja2",
    ".j2",
    ".parquet",
    ".csv",
    ".tsv",
    ".arrow",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}


def looks_like_file_path(value: str) -> bool:
    """Check if a string looks like a repository file path, not an HF repo ID or a URI such as ``s3://...``."""
    if not isinstance(value, str) or not value:
        return False

    if is_hugging_face_repo_id(value) or "://" in value:
        return False

    if value.startswith("/") or value.startswith("~"):
        return True
    if value.startswith("./") or value.startswith("../"):
        return True

    lower = value.lower()
    if any(lower.endswith(ext) for ext in _PATH_EXTENSIONS):
        return True

    # Multiple slashes indicate a nested path (more than org/repo).
    if value.count("/") > 1:
        return True

    return False


def resolve_paths_in_dict(
    config: Dict[str, Any],
    base_dir: Optional[Path] = None,
    skip_keys: Optional[Set[str]] = None,
    _prefix: str = "",
) -> Dict[str, Any]:
    """Recursively resolve file paths in a config dictionary.

    Walks the dict, identifies values that look like file paths (not HF repo IDs),
    and resolves them to absolute paths using PROJECT_ROOT. Returns a new dict; the
    original is not modified.
    """
    skip_keys = skip_keys or set()
    result: Dict[str, Any] = {}

    for key, value in config.items():
        full_key = f"{_prefix}.{key}" if _prefix else key

        if full_key in skip_keys:
            result[key] = value
            continue

        if isinstance(value, dict):
            result[key] = resolve_paths_in_dict(value, base_dir, skip_keys, full_key)
        elif isinstance(value, list):
            resolved_list: List[Any] = []
            for item in value:
                if isinstance(item, str) and looks_like_file_path(item):
                    if base_dir:
                        resolved = (base_dir / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
                        resolved_list.append(str(resolved))
                    else:
                        resolved_list.append(str(resolve_repo_path(item)))
                elif isinstance(item, dict):
                    resolved_list.append(resolve_paths_in_dict(item, base_dir, skip_keys, full_key))
                else:
                    resolved_list.append(item)
            result[key] = resolved_list
        elif isinstance(value, str) and looks_like_file_path(value):
            if base_dir:
                resolved = (base_dir / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
                result[key] = str(resolved)
            else:
                result[key] = str(resolve_repo_path(value))
        else:
            result[key] = value

    return result
