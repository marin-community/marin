# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Harbor evaluation presets through Harbor's pinned configuration artifact.

Marin owns evaluation intent and runtime wiring; Harbor owns the schema and semantics of a
``JobConfig``.  Normal launches choose a compact named preset.  Advanced operators can provide a
complete Harbor document and/or a structured overlay, but Marin always supplies the served endpoint,
dataset identity, durable job root, and generated job name.  That boundary keeps a launch flexible
without allowing a stale hand-written Marin schema to diverge from Harbor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from harbor_config import apply_overlay, canonical_json, load_config_document

from marin.evaluation.config_artifacts import HARBOR_CONFIG

DEFAULT_HARBOR_PRESET = "standard"


@dataclass(frozen=True)
class HarborRuntime:
    """Values that must be supplied by the serving/orchestration layer, not a user preset."""

    job_name: str
    jobs_dir: str
    dataset: str
    version: str
    task_limit: int | None
    served_model_name: str
    api_base: str


@dataclass(frozen=True)
class ResolvedHarborConfig:
    """A validated Harbor job document plus immutable provenance for its artifact."""

    document: dict[str, Any]
    canonical: bytes
    sha256: str

    def persisted_metadata(self) -> dict[str, Any]:
        """Return safe provenance for the durable eval output (without capability credentials)."""
        return {
            "harbor_revision": HARBOR_CONFIG.revision,
            "release_tag": HARBOR_CONFIG.release_tag,
            "schema_fingerprint": HARBOR_CONFIG.schema_fingerprint,
            "resolver_fingerprint": HARBOR_CONFIG.resolver_fingerprint,
            "config_sha256": self.sha256,
            "job_config": _redact_runtime_secrets(self.document),
        }


def resolve_harbor_config(
    *,
    preset: str,
    n_concurrent: int,
    agent: str,
    environment: str,
    default_max_output_tokens: int,
    agent_kwargs: Mapping[str, Any],
    runtime: HarborRuntime,
    document: Mapping[str, Any] | None = None,
    patch: Mapping[str, Any] | None = None,
) -> ResolvedHarborConfig:
    """Validate a named or supplied Harbor document and apply Marin's protected runtime overlay.

    ``document`` and ``patch`` deliberately retain the whole Harbor schema: retry policy, sandbox
    resources, timeouts, verifier policy, agent choice, and generation budget are Harbor concerns.
    The caller cannot redirect a run to a different served model/dataset or leak a stale jobs
    directory, because those runtime-owned fields are reapplied after the operator's document is
    validated.
    """
    base = load_config_document(
        dict(document) if document is not None else _preset_document(preset, n_concurrent, agent, environment)
    )
    if patch is not None:
        base = apply_overlay(base, dict(patch))
    resolved = load_config_document(
        _apply_runtime_overlay(base.model_dump(mode="json"), agent_kwargs, default_max_output_tokens, runtime)
    )
    canonical = canonical_json(resolved)
    return ResolvedHarborConfig(
        document=resolved.model_dump(mode="json"),
        canonical=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _preset_document(preset: str, n_concurrent: int, agent: str, environment: str) -> dict[str, Any]:
    """Return a compact named preset; additions here are Marin product policy, not YAML copies."""
    if preset != DEFAULT_HARBOR_PRESET:
        raise ValueError(f"unknown Harbor preset {preset!r}; available presets: {DEFAULT_HARBOR_PRESET!r}")
    return {
        "n_concurrent_trials": n_concurrent,
        "environment": {"type": environment},
        "agents": [{"name": agent}],
        "datasets": [{"name": "aime", "version": "1.0"}],
    }


def _apply_runtime_overlay(
    base: dict[str, Any],
    agent_kwargs: Mapping[str, Any],
    default_max_output_tokens: int,
    runtime: HarborRuntime,
) -> dict[str, Any]:
    """Preserve policy from a supplied document while replacing launcher-owned execution wiring."""
    agents = base.get("agents")
    datasets = base.get("datasets")
    if not isinstance(agents, list) or len(agents) != 1:
        raise ValueError("a Marin Harbor launch requires exactly one agent in its resolved document")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("a Marin Harbor launch requires exactly one dataset in its resolved document")

    agent = dict(agents[0])
    existing_kwargs = agent.get("kwargs") or {}
    if not isinstance(existing_kwargs, Mapping):
        raise ValueError("Harbor agent kwargs must be a mapping")
    if not isinstance(existing_kwargs.get("model_info") or {}, Mapping):
        raise ValueError("Harbor agent kwargs.model_info must be a mapping")
    runtime_kwargs = _deep_merge(
        {
            "model_info": {
                "max_input_tokens": 32768,
                "max_output_tokens": default_max_output_tokens,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        },
        agent_kwargs,
    )
    runtime_kwargs = _deep_merge(runtime_kwargs, dict(existing_kwargs))
    runtime_kwargs["api_base"] = runtime.api_base
    agent.update(
        {
            "model_name": f"hosted_vllm/{runtime.served_model_name}",
            "kwargs": runtime_kwargs,
        }
    )

    dataset = dict(datasets[0])
    # A Marin suite is a registry dataset. Remove mutually exclusive local/package identities before
    # restoring the selected registry identity, while preserving registry URL/cache/filter controls.
    dataset.pop("path", None)
    dataset.pop("ref", None)
    dataset.update({"name": runtime.dataset, "version": runtime.version})
    if runtime.task_limit is not None:
        dataset["n_tasks"] = runtime.task_limit

    resolved = dict(base)
    resolved.update(
        {"job_name": runtime.job_name, "jobs_dir": runtime.jobs_dir, "agents": [agent], "datasets": [dataset]}
    )
    return resolved


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _redact_runtime_secrets(value: Any, key: str = "") -> Any:
    """Keep a useful durable config record without retaining credential-bearing capability URLs."""
    normalized_key = key.lower()
    if normalized_key == "api_base" or any(
        marker in normalized_key for marker in ("api_key", "token", "password", "secret")
    ):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {name: _redact_runtime_secrets(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_runtime_secrets(item) for item in value]
    return value
