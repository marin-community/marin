# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical training-stream identities for StarCoder WSD80 comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from marin.execution.step_spec import StepSpec

STARCODER_COMPONENT = "dolma/starcoder"
FLOAT_DIGITS = 13


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Training identity contains a non-finite float")
        if value.is_integer():
            return int(value)
        return float(f"{value:.{FLOAT_DIGITS}g}")
    raise TypeError(f"Unsupported training-identity value: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    """Return a stable digest after normalizing semantically equal JSON numbers."""
    encoded = json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _training_components(train_config: Mapping[str, Any]) -> tuple[str, ...]:
    train_weights = train_config["data"]["train_weights"]
    components = {str(component) for _, phase_weights in train_weights for component in phase_weights}
    if STARCODER_COMPONENT not in components:
        raise ValueError(f"Missing {STARCODER_COMPONENT!r} from the training schedule")
    return tuple(sorted(components))


def _normalized_cache_path(path: str) -> str:
    marker = "/tokenized/"
    if marker in path:
        return "tokenized/" + path.split(marker, maxsplit=1)[1]
    if path.startswith("tokenized/"):
        return path
    raise ValueError(f"Training cache is not a region-local tokenized path: {path!r}")


def wandb_training_cache_paths(train_config: Mapping[str, Any]) -> dict[str, str]:
    """Extract physical training-cache identities from a completed W&B config."""
    components = train_config["data"]["components"]
    return {
        name: _normalized_cache_path(str(components[name]["cache_dir"])) for name in _training_components(train_config)
    }


def step_training_cache_paths(step_spec: StepSpec, train_config: Mapping[str, Any]) -> dict[str, str]:
    """Extract physical training-cache identities from a lowered Marin step."""
    dependencies = {dependency.name: dependency for dependency in step_spec.deps}
    result = {}
    for name in _training_components(train_config):
        dependency = dependencies.get(name)
        if dependency is None or dependency.override_output_path is None:
            raise ValueError(f"Lowered step has no physical cache dependency for {name!r}")
        result[name] = _normalized_cache_path(str(dependency.override_output_path))
    return result


def _policy_free_schedule(train_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    schedule = []
    for boundary_step, raw_weights in train_config["data"]["train_weights"]:
        weights = {str(key): float(value) for key, value in raw_weights.items()}
        starcoder_weight = weights.pop(STARCODER_COMPONENT)
        background_mass = sum(weights.values())
        if background_mass <= 0.0:
            raise ValueError("WSD80 identity requires positive background mass in every phase")
        schedule.append(
            {
                "boundary_step": int(boundary_step),
                "background_ratios": {key: value / background_mass for key, value in sorted(weights.items())},
                "policy_mass_check": starcoder_weight + background_mass,
            }
        )
    return schedule


def _dtype_name(value: Any) -> str:
    if isinstance(value, Mapping) and set(value) == {"__dtype__"}:
        return str(value["__dtype__"])
    name = str(value)
    return name.removeprefix("jax.numpy.")


def policy_coordinates(train_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the policy coordinate excluded from the stream-identity comparison."""
    return [
        {
            "boundary_step": int(boundary_step),
            "starcoder_weight": float(weights[STARCODER_COMPONENT]),
        }
        for boundary_step, weights in train_config["data"]["train_weights"]
    ]


def training_stream_identity(
    train_config: Mapping[str, Any],
    training_cache_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Build the policy-free identity that must match within a paired comparison."""
    data = train_config["data"]
    trainer = train_config["trainer"]
    identity = {
        "randomness": {
            "data_seed": train_config["data_seed"],
            "simulated_epoch_subset_seed": data["simulated_epoch_subset_seed"],
            "trainer_seed": trainer["seed"],
        },
        "data": {
            "experiment_budget": data["experiment_budget"],
            "target_budget": data["target_budget"],
            "mixture_block_size": data["mixture_block_size"],
            "permutation_type": data["permutation_type"],
            "shuffle": data["shuffle"],
            "shuffle_before_trainval_split": data["shuffle_before_trainval_split"],
            "stop_strategy": data["stop_strategy"],
            "block_cross_document_attention": data["block_cross_document_attention"],
            "cache_options": data["cache_options"],
            "training_cache_paths": dict(sorted(training_cache_paths.items())),
            "policy_free_schedule": _policy_free_schedule(train_config),
        },
        "model": train_config["model"],
        "optimizer": train_config["optimizer"],
        "train_seq_len": train_config["train_seq_len"],
        "trainer": {
            "allow_nondivisible_batch_size": trainer["allow_nondivisible_batch_size"],
            "jax_config": trainer["jax_config"],
            "mesh": {
                "axes": trainer["mesh"]["axes"],
                "batch_axis_name": trainer["mesh"]["batch_axis_name"],
                "compute_mapping": trainer["mesh"]["compute_mapping"],
                "dcn_axes": trainer["mesh"]["dcn_axes"],
                "param_mapping": trainer["mesh"]["param_mapping"],
            },
            "mp": {key: _dtype_name(value) for key, value in trainer["mp"].items()},
            "num_train_steps": trainer["num_train_steps"],
            "train_batch_size": trainer["train_batch_size"],
        },
    }
    return _canonicalize(identity)


def lowered_step_training_config(step_spec: StepSpec) -> dict[str, Any]:
    """Read a lowered training config from a Marin step fingerprint."""
    payload = json.loads(step_spec.fingerprint_payload)
    train_config = payload.get("train_config")
    if not isinstance(train_config, dict):
        raise ValueError(f"{step_spec.name}: fingerprint has no training config")
    return train_config


def lowered_step_stream_identity(step_spec: StepSpec) -> dict[str, Any]:
    """Build a stream identity from a lowered Marin training step."""
    train_config = lowered_step_training_config(step_spec)
    cache_paths = step_training_cache_paths(step_spec, train_config)
    return training_stream_identity(train_config, cache_paths)


def wandb_stream_identity(train_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stream identity from a completed W&B training config."""
    return training_stream_identity(train_config, wandb_training_cache_paths(train_config))


def identity_differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Return precise paths at which two canonical identities differ."""
    left = _canonicalize(left)
    right = _canonicalize(right)
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(identity_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            differences.extend(identity_differences(left_item, right_item, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]
