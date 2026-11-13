# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable comparison identity for a normalized evaluation model configuration."""

import hashlib
import json

from marin.evaluation.records import ModelConfigRef, ModelRef


def model_config_digest(config: ModelConfigRef) -> str:
    """Hash the complete normalized model YAML, independent of key order or whitespace."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def comparison_model_name(model: ModelRef) -> str:
    """Separate differently configured runs that share a display/serving model name."""
    config = model.source_config or model.config
    if config is None:
        return model.name
    return f"{config.name}@{model_config_digest(config)[:12]}"
