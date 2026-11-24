# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable comparison identity for a normalized evaluation model configuration."""

import hashlib
import json

from evalrecords.records import ModelConfigRef


def model_config_digest(config: ModelConfigRef) -> str:
    """Hash the complete normalized model YAML, independent of key order or whitespace."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
