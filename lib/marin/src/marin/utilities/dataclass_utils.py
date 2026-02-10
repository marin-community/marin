# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from draccus.utils import DataclassInstance


def asdict_without_nones(obj: DataclassInstance) -> dict:
    """Convert dataclass to dictionary, omitting None values."""
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return dataclasses.asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
