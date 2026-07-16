"""Harbor config loading + agent-kwargs utilities.

Extracted from OT-Agent ``hpc/harbor_utils.py``. These functions have no
``hpc.*`` dependencies.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_harbor_config(harbor_config_path: str) -> Dict[str, Any]:
    """Load and parse Harbor config YAML."""
    try:
        with open(harbor_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def get_harbor_env_from_config(
    harbor_config: Any,
    default: str = "daytona",
) -> str:
    """Extract Harbor environment type from config's ``environment.type`` field."""
    if harbor_config is None:
        return default

    if isinstance(harbor_config, str):
        config_dict = load_harbor_config(harbor_config)
    else:
        config_dict = harbor_config

    env_config = config_dict.get("environment") or {}
    env_type = env_config.get("type")

    if env_type and isinstance(env_type, str):
        return env_type.lower()

    return default


def extract_agent_kwargs_from_config(harbor_config: dict, agent_name: Optional[str]) -> dict:
    """Extract kwargs for the specified agent from harbor config."""
    agents = harbor_config.get("agents", [])
    for agent in agents:
        if agent.get("name") == agent_name:
            return copy.deepcopy(agent.get("kwargs", {}))
    if agents and isinstance(agents[0], dict):
        return copy.deepcopy(agents[0].get("kwargs", {}))
    return {}


def apply_nested_key(target: dict, dotted_key: str, value: Any) -> None:
    """Apply a value to a nested dict using dotted key notation."""
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_agent_kwarg_strings(entries: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Parse --agent-kwarg CLI entries into overrides and passthrough."""
    overrides: Dict[str, Any] = {}
    passthrough: List[str] = []
    for entry in entries:
        if "=" not in entry:
            passthrough.append(entry)
            continue
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            passthrough.append(entry)
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides, passthrough


def serialize_agent_kwargs(kwargs: dict) -> List[str]:
    """Serialize agent kwargs dict to CLI argument strings."""
    serialized: List[str] = []
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            serialized.append(f"{key}={json.dumps(value)}")
        else:
            serialized.append(f"{key}={value}")
    return serialized


def merge_agent_kwargs(
    harbor_config_data: dict,
    agent_name: Optional[str],
    endpoint_meta: Optional[Dict[str, Any]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge agent kwargs from multiple sources with proper precedence.

    Precedence (lowest to highest):
    1. Base kwargs from Harbor YAML (agents[].kwargs)
    2. Endpoint-specific values (api_base, metrics_endpoint) for local vLLM
    3. Extra kwargs from datagen config (extra_agent_kwargs)
    4. CLI --agent-kwarg overrides (highest precedence, supports dotted keys)
    """
    agent_kwargs = extract_agent_kwargs_from_config(harbor_config_data, agent_name)

    if endpoint_meta:
        if endpoint_meta.get("metrics_endpoint"):
            agent_kwargs["metrics_endpoint"] = endpoint_meta["metrics_endpoint"]
        if endpoint_meta.get("api_base"):
            agent_kwargs["api_base"] = endpoint_meta["api_base"]
        if endpoint_meta.get("api_key"):
            agent_kwargs["api_key"] = endpoint_meta["api_key"]

    if extra_kwargs:
        for key, value in extra_kwargs.items():
            apply_nested_key(agent_kwargs, key, value)

    passthrough: List[str] = []
    if cli_overrides:
        override_kwargs, passthrough = parse_agent_kwarg_strings(cli_overrides)
        for dotted_key, override_value in override_kwargs.items():
            apply_nested_key(agent_kwargs, dotted_key, override_value)

    return agent_kwargs, passthrough


def resolve_jobs_dir_path(
    jobs_dir_value: Optional[str],
    repo_root: Optional[Path] = None,
) -> Path:
    """Resolve jobs_dir from Harbor config to an absolute path."""
    if repo_root is None:
        repo_root = Path.cwd()

    raw_value = jobs_dir_value or "jobs"
    path = Path(raw_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path
