# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin inference policy stored with exported models."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

MARIN_GENERATION_CONFIG_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
    }
)
