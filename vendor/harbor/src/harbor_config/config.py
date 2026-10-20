"""Public Harbor job-configuration loading and canonicalization API."""

import hashlib
import importlib.resources
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from harbor_config.models.job.config import JobConfig

_CONFIG_SOURCE_FILES = (
    "config.py",
    "constants.py",
    "env.py",
    "errors.py",
    "models/agent/name.py",
    "models/environment_type.py",
    "models/job/config.py",
    "models/metric/config.py",
    "models/metric/type.py",
    "models/task/config.py",
    "models/task/id.py",
    "models/trial/config.py",
)


def load_config_document(document: Mapping[str, Any]) -> JobConfig:
    """Validate a parsed Harbor job configuration document."""
    return JobConfig.model_validate(document)


def load_config_path(path: Path) -> JobConfig:
    """Load and validate a YAML or JSON Harbor job configuration."""
    text = path.read_text()
    if path.suffix == ".yaml":
        document = yaml.safe_load(text)
    elif path.suffix == ".json":
        document = json.loads(text)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    return JobConfig.model_validate(document)


def apply_overlay(config: JobConfig, overlay: Mapping[str, Any]) -> JobConfig:
    """Apply a structured overlay and revalidate the resolved configuration.

    Mapping values merge recursively. Lists replace the existing list, so a
    caller overriding an agent list must supply each complete agent entry it
    intends to retain.
    """
    resolved = _merge_mappings(config.model_dump(mode="json"), overlay)
    return JobConfig.model_validate(resolved)


def canonical_json(config: JobConfig) -> bytes:
    """Serialize a resolved configuration deterministically for comparison."""
    return json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def schema_fingerprint() -> str:
    """Return the SHA-256 fingerprint of the current JobConfig JSON schema."""
    schema = json.dumps(
        JobConfig.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(schema).hexdigest()


def resolver_fingerprint() -> str:
    """Return the deterministic fingerprint of config schema and behavior."""
    digest = hashlib.sha256(schema_fingerprint().encode())
    package_files = importlib.resources.files("harbor_config")
    for relative_path in _CONFIG_SOURCE_FILES:
        digest.update(relative_path.encode())
        digest.update(package_files.joinpath(relative_path).read_bytes())
    return digest.hexdigest()


def _merge_mappings(
    base: Mapping[str, Any], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged
