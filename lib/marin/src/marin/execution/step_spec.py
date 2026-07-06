# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from urllib.parse import urlparse

from fray.types import ResourceConfig
from rigging.filesystem import marin_prefix, prefix_join


def _is_relative_path(url_or_path: str) -> bool:
    """Return True if the path is relative (not a URL and doesn't start with /)."""
    if urlparse(url_or_path).scheme:
        return False
    return not url_or_path.startswith("/")


@dataclass(frozen=True)
class StepSpec:
    """Step identity, dependencies, and execution configuration.

    StepSpec is a pure data object: it describes *what* to run, not *how*.
    Caching, locking, heartbeats, and status writes are handled explicitly
    by the step runner.

    A StepSpec freezes nothing at construction: ``output_path`` resolves
    ``marin_prefix()`` lazily and ``fn`` pulls the live environment (prefix,
    region) from a ``RunContext`` at run time. So it is safe to build anywhere —
    it carries no separate build phase.
    """

    # Identity
    name: str
    """Name of the step, used for readability and in the output path."""
    output_path_prefix: str | None = None
    """Output path prefix for the step. If not provided, it will be taken from the MARIN_PREFIX environment variable."""
    deps: "list[StepSpec]" = dataclasses.field(default_factory=list)
    """Steps that this step depends on. Their output paths are used for dependency tracking and cache invalidation."""
    hash_attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Attributes to include in the hash calculation for the step. Used for cache invalidation.

    Must be JSON-serializable.
    """
    override_output_path: str | None = None
    """Override the default output path for the step."""

    fingerprint_payload: str | None = None
    """For a lazy artifact, the canonical config JSON behind its fingerprint
    (:mod:`marin.execution.fingerprint`). Recorded as provenance and used to render a
    field-level diff when the drift check sees a changed recipe. Diagnostics only: it
    never enters :attr:`hash_id`."""

    writes_record: bool = False
    """Whether the step's ``fn`` already writes its own :class:`~marin.execution.artifact.ArtifactRecord`.

    Lazy-artifact steps set this so the runner does not overwrite the full record with a
    minimal one. A plain ``StepSpec`` leaves it ``False`` and the runner saves the fn's
    return via :func:`~marin.execution.artifact.write_artifact`."""

    # Execution
    fn: Callable[[str], Any] | None = None
    """
    Callable that accepts the output path as the only argument, and produces the step output at that path
    when called. Usually this function would then call the specific function e.g. tokenize with the appropriate
    arguments. Usually you would specify this via a `lambda output_path: foo(output_path=output_path, bar=42)`.

    May be a :class:`~marin.execution.remote.RemoteCallable` for Fray dispatch.
    """

    resources: ResourceConfig | None = None
    """If set, the step runner submits ``fn`` as its own Fray job with these
    resources. If ``None``, dispatch falls back to ``fn``'s type: a
    :class:`~marin.execution.remote.RemoteCallable` is submitted via Fray; a
    plain callable runs inline in the runner thread.
    """

    @cached_property
    def dep_paths(self) -> list[str]:
        """Physical and resolved output paths of all dependencies."""
        return [dep.output_path for dep in self.deps]

    @cached_property
    def dep_names(self) -> list[str]:
        """Logical names with hashes of all dependencies, used for hashing and reporting."""
        return [dep.name_with_hash for dep in self.deps]

    @cached_property
    def hash_id(self) -> str:
        """Hash ID of the step, used for cache invalidation and output path generation."""
        content = json.dumps(
            {"name": self.name, "attrs": self.hash_attrs, "deps": sorted(self.dep_names)},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

    @cached_property
    def name_with_hash(self) -> str:
        """Name of the step with hash, used for readability, reporting, and in the output path."""
        return f"{self.name}_{self.hash_id}"

    @cached_property
    def output_path(self) -> str:
        """Output path of the step.

        If ``override_output_path`` is set and relative (no URL scheme, doesn't
        start with ``/``), it is automatically prefixed with ``output_path_prefix``
        or ``marin_prefix()``.
        """
        prefix = self.output_path_prefix or marin_prefix()
        if self.override_output_path is not None:
            if _is_relative_path(self.override_output_path):
                return prefix_join(prefix, self.override_output_path)
            return self.override_output_path
        return prefix_join(prefix, self.name_with_hash)
