# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from fray.v2.types import ResourceConfig


@dataclass(frozen=True)
class StepSpec:
    """Step identity, dependencies, and execution configuration."""

    # Identity
    name: str
    """Name of the step, used for readability and in the output path."""
    output_path_prefix: str | None = None
    """Output path prefix for the step. If not provided, it will be taken from the MARIN_PREFIX environment variable."""
    deps: "list[str | StepSpec]" = dataclasses.field(default_factory=list)
    """
    List of output paths that this step depends on. Used for tracking dependencies and cache invalidation.

    NOTE: If StepSpec instances are provided, their output paths will be automatically coerced.
    """
    hash_attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Attributes to include in the hash calculation for the step. Used for cache invalidation.

    Must be JSON-serializable.
    """
    override_output_path: str | None = None
    """Override the default output path for the step."""

    # Execution
    fn: Callable[[str], Any] | None = None
    """
    Callable that accepts the output path as the only argument, and produces the step output at that path
    when called. Usually this function would then call the specific function e.g. tokenize with the appropriate
    arguments. Usually you would specify this via a `lambda output_path: foo(output_path=output_path, bar=42)`.
    """

    def __post_init__(self):
        # Coerce StepSpec instances in deps to their output_path strings
        # NOTE: https://stackoverflow.com/a/54119384 - use setattr to handle the frozen class
        object.__setattr__(self, "deps", [dep.output_path if isinstance(dep, StepSpec) else dep for dep in self.deps])

    @cached_property
    def hash_id(self) -> str:
        """Hash ID of the step, used for cache invalidation and output path generation."""
        content = json.dumps(
            {"name": self.name, "attrs": self.hash_attrs, "deps": sorted(self.deps)},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

    @cached_property
    def name_with_hash(self) -> str:
        """Name of the step with hash, used for readability, reporting, and in the output path."""
        return f"{self.name}_{self.hash_id}"

    @cached_property
    def output_path(self) -> str:
        """Output path of the step"""
        if self.override_output_path is not None:
            return self.override_output_path

        prefix = self.output_path_prefix or os.environ["MARIN_PREFIX"]
        return f"{prefix}/{self.name_with_hash}"

    # Resources
    resources: ResourceConfig = dataclasses.field(default_factory=ResourceConfig.with_cpu)
    """CPU/GPU/TPU (defaults resolved)."""

    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)
    """Environment variables (defaults resolved)."""

    pip_dependency_groups: list[str] = dataclasses.field(default_factory=list)
    """Pip deps (defaults resolved)."""
