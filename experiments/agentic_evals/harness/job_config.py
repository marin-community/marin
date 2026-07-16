"""Harbor ``JobConfig`` loading + metric filtering.

Extracted from OT-Agent ``scripts/harbor/job_config_utils.py`` (only the
functions the eval package needs: ``load_job_config``,
``_filter_supported_metrics``, ``normalize_trajectory_kwargs``,
``_normalize_job_config_agent_kwargs``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from harbor.models.job.config import JobConfig
from harbor.models.metric.type import MetricType

_UNSUPPORTED_METRIC_TYPES = frozenset({"mean-drop-ei", "accuracy-drop-ei"})


def _filter_supported_metrics(raw: Any) -> Any:
    """Drop metrics with types that the pinned harbor schema can't validate."""
    if not isinstance(raw, dict):
        return raw
    metrics = raw.get("metrics")
    if not isinstance(metrics, list):
        return raw
    allowed = {mt.value for mt in MetricType}
    filtered = []
    dropped = []
    for entry in metrics:
        if isinstance(entry, dict):
            mtype = entry.get("type")
            if mtype in _UNSUPPORTED_METRIC_TYPES or (mtype is not None and mtype not in allowed):
                dropped.append(mtype)
                continue
        filtered.append(entry)
    if dropped:
        print(
            f"[load_job_config] Dropped unsupported harbor metrics: {dropped}.",
            flush=True,
        )
    raw = dict(raw)
    raw["metrics"] = filtered
    return raw


def normalize_trajectory_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Convert legacy ``trajectory_configs`` into the singular ``trajectory_config``."""
    if kwargs is None:
        return {}

    normalized = dict(kwargs)
    legacy = normalized.pop("trajectory_configs", None)
    current = normalized.get("trajectory_config")

    def _ensure_mapping(value: Any, label: str) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping (received {type(value).__name__}).")
        return value

    merged: dict[str, Any] = {}
    legacy_map = _ensure_mapping(legacy, "trajectory_configs")
    if legacy_map:
        merged.update(legacy_map)
    current_map = _ensure_mapping(current, "trajectory_config")
    if current_map:
        merged.update(current_map)

    if merged:
        normalized["trajectory_config"] = merged
    elif current is None:
        normalized.pop("trajectory_config", None)

    return normalized


def _normalize_job_config_agent_kwargs(config: JobConfig) -> JobConfig:
    updated = config.model_copy(deep=True)
    normalized_agents = []
    for agent in updated.agents:
        agent_copy = agent.model_copy(deep=True)
        agent_copy.kwargs = normalize_trajectory_kwargs(agent_copy.kwargs)
        normalized_agents.append(agent_copy)
    updated.agents = normalized_agents
    return updated


def load_job_config(config_path: Path | str) -> JobConfig:
    """Load a Harbor ``JobConfig`` from YAML or JSON."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Harbor job config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = _filter_supported_metrics(yaml.safe_load(path.read_text()))
        config = JobConfig.model_validate(raw)
    elif suffix == ".json":
        import json as _json
        raw = _filter_supported_metrics(_json.loads(path.read_text()))
        config = JobConfig.model_validate(raw)
    else:
        raise ValueError(
            f"Unsupported Harbor job config format '{path.suffix}'. "
            "Expected one of: .yaml, .yml, .json."
        )

    return _normalize_job_config_agent_kwargs(config)


__all__ = [
    "load_job_config",
    "_filter_supported_metrics",
    "normalize_trajectory_kwargs",
    "_normalize_job_config_agent_kwargs",
]
