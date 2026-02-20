# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build result tracking."""

from dataclasses import dataclass


@dataclass
class BuildResult:
    image_tag: str
    build_time_ms: int
    from_cache: bool
