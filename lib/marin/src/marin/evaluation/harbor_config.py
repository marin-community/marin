# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact-backed Harbor policy resolution and the deliberately narrow runtime binding.

Harbor's configuration artifact owns every Harbor policy field. Marin resolves that document once
at launch time, then the runner binds only facts that do not exist until it has a live served
endpoint and a durable job directory. This keeps the runner from becoming a second ``JobConfig``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import Any

from harbor_config import apply_overlay, canonical_json, load_config_document, load_config_path

from marin.evaluation.config_artifacts import HARBOR_CONFIG

DEFAULT_HARBOR_PRESET = "standard"
GRUG_OPENCODE_ID_PRESET = "grug-opencode-id"
_HARBOR_PRESET_FILES = {GRUG_OPENCODE_ID_PRESET: "harbor_profiles/grug-opencode-id.yaml"}


@dataclass(frozen=True)
class HarborPolicyDefaults:
    """The compact defaults behind a named Marin Harbor suite, consumed only during resolution."""

    dataset: str
    version: str
    agent: str
    environment: str
    n_concurrent_trials: int
    max_output_tokens: int
    agent_kwargs: Mapping[str, Any]
    task_limit: int | None = None
    preset: str = DEFAULT_HARBOR_PRESET


@dataclass(frozen=True)
class HarborRuntimeBinding:
    """Values unavailable before the runner has created a job and served the model."""

    job_name: str
    jobs_dir: str
    served_model_name: str
    api_base: str


@dataclass(frozen=True)
class ResolvedHarborPolicy:
    """A complete, artifact-validated Harbor policy before endpoint-specific binding."""

    document: dict[str, Any]
    canonical: bytes
    sha256: str

    @property
    def dataset(self) -> str:
        return str(self.document["datasets"][0]["name"])

    @property
    def version(self) -> str | None:
        value = self.document["datasets"][0].get("version")
        return str(value) if value is not None else None

    @property
    def agent(self) -> str:
        value = self.document["agents"][0].get("name")
        if value is None:
            raise ValueError("a Marin Harbor policy requires a named agent")
        return str(value)

    @property
    def environment(self) -> str:
        value = self.document.get("environment", {}).get("type")
        if value is None:
            raise ValueError("a Marin Harbor policy requires an environment type")
        return str(value)

    def bind(self, runtime: HarborRuntimeBinding) -> BoundHarborConfig:
        """Apply only endpoint/job facts, then revalidate through Harbor's artifact schema."""
        agents = self.document.get("agents")
        if not isinstance(agents, list) or len(agents) != 1:
            raise ValueError("a Marin Harbor launch requires exactly one agent in its resolved policy")
        agent = dict(agents[0])
        kwargs = agent.get("kwargs") or {}
        if not isinstance(kwargs, Mapping):
            raise ValueError("Harbor agent kwargs must be a mapping")
        agent["model_name"] = f"hosted_vllm/{runtime.served_model_name}"
        agent["kwargs"] = _deep_merge(dict(kwargs), {"api_base": runtime.api_base})
        bound = dict(self.document)
        bound.update({"job_name": runtime.job_name, "jobs_dir": runtime.jobs_dir, "agents": [agent]})
        config = load_config_document(bound)
        canonical = canonical_json(config)
        return BoundHarborConfig(
            document=config.model_dump(mode="json"),
            policy_sha256=self.sha256,
            sha256=hashlib.sha256(canonical).hexdigest(),
        )


@dataclass(frozen=True)
class BoundHarborConfig:
    """The exact driver input plus safe provenance for durable eval artifacts."""

    document: dict[str, Any]
    policy_sha256: str
    sha256: str

    def persisted_metadata(self) -> dict[str, Any]:
        return {
            "harbor_revision": HARBOR_CONFIG.revision,
            "release_tag": HARBOR_CONFIG.release_tag,
            "schema_fingerprint": HARBOR_CONFIG.schema_fingerprint,
            "resolver_fingerprint": HARBOR_CONFIG.resolver_fingerprint,
            "policy_sha256": self.policy_sha256,
            "bound_config_sha256": self.sha256,
            "job_config": _redact_runtime_secrets(self.document),
        }


def resolve_harbor_policy(
    defaults: HarborPolicyDefaults,
    *,
    document: Mapping[str, Any] | None = None,
    patch: Mapping[str, Any] | None = None,
    task_limit_override: int | None = None,
) -> ResolvedHarborPolicy:
    """Resolve one complete Harbor policy without re-declaring Harbor's schema in Marin.

    A supplied document replaces the named preset; a Harbor-native patch then overrides any policy
    field. The selected suite's dataset identity is deliberately applied after that policy layer,
    because ``--evals`` is the launch's explicit benchmark selection rather than a hidden YAML
    default. ``--limit`` is similarly an explicit selection-level task cap.
    """
    base = load_config_document(dict(document) if document is not None else _preset_document(defaults))
    if patch is not None:
        base = apply_overlay(base, dict(patch))
    selected_limit = task_limit_override if task_limit_override is not None else defaults.task_limit
    config = load_config_document(_apply_selection(base.model_dump(mode="json"), defaults, selected_limit))
    _require_single_agent_and_dataset(config.model_dump(mode="json"))
    canonical = canonical_json(config)
    return ResolvedHarborPolicy(
        document=config.model_dump(mode="json"),
        canonical=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _preset_document(defaults: HarborPolicyDefaults) -> dict[str, Any]:
    if defaults.preset in _HARBOR_PRESET_FILES:
        resource = files("marin.evaluation").joinpath(_HARBOR_PRESET_FILES[defaults.preset])
        with as_file(resource) as path:
            return load_config_path(path).model_dump(mode="json")
    if defaults.preset != DEFAULT_HARBOR_PRESET:
        available = (DEFAULT_HARBOR_PRESET, *_HARBOR_PRESET_FILES)
        raise ValueError(f"unknown Harbor preset {defaults.preset!r}; available presets: {available!r}")
    agent_kwargs = _deep_merge(
        {
            "model_info": {
                "max_input_tokens": 32768,
                "max_output_tokens": defaults.max_output_tokens,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        },
        defaults.agent_kwargs,
    )
    return {
        "n_concurrent_trials": defaults.n_concurrent_trials,
        "environment": {"type": defaults.environment},
        "agents": [
            {
                "name": defaults.agent,
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [
            {
                "name": defaults.dataset,
                "version": defaults.version,
                "n_tasks": defaults.task_limit,
            }
        ],
    }


def _apply_selection(
    document: dict[str, Any], defaults: HarborPolicyDefaults, task_limit_override: int | None
) -> dict[str, Any]:
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("a Marin Harbor launch requires exactly one dataset in its resolved policy")
    dataset = dict(datasets[0])
    dataset.pop("path", None)
    dataset.pop("ref", None)
    dataset.update({"name": defaults.dataset, "version": defaults.version})
    if task_limit_override is not None:
        dataset["n_tasks"] = task_limit_override
    selected = dict(document)
    selected["datasets"] = [dataset]
    return selected


def _require_single_agent_and_dataset(document: Mapping[str, Any]) -> None:
    agents = document.get("agents")
    datasets = document.get("datasets")
    if not isinstance(agents, list) or len(agents) != 1:
        raise ValueError("a Marin Harbor launch requires exactly one agent in its resolved policy")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("a Marin Harbor launch requires exactly one dataset in its resolved policy")


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _redact_runtime_secrets(value: Any, key: str = "") -> Any:
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
