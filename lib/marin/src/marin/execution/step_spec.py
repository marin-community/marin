# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from iris.marin_fs import marin_prefix


@dataclass(frozen=True)
class StepSpec:
    """Step identity, dependencies, and execution configuration."""

    # Identity
    name: str
    """Name of the step, used for readability and in the output path."""
    output_path_prefix: str | None = None
    """Output path prefix for the step. If not provided, it will be taken from the MARIN_PREFIX environment variable."""
    deps: list[StepSpec] = dataclasses.field(default_factory=list)
    """Steps that this step depends on. Their output paths are used for dependency tracking and cache invalidation."""
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

    @cached_property
    def dep_paths(self) -> list[str]:
        """Output paths of all dependencies."""
        return [dep.output_path for dep in self.deps]

    @cached_property
    def hash_id(self) -> str:
        """Hash ID of the step, used for cache invalidation and output path generation."""
        content = json.dumps(
            {"name": self.name, "attrs": self.hash_attrs, "deps": sorted(self.dep_paths)},
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

        prefix = self.output_path_prefix or marin_prefix()
        return f"{prefix}/{self.name_with_hash}"

    @cached_property
    def executable_fn(self) -> Callable[[str], Any]:
        """Fully-wrapped fn: disk_cache(distributed_lock(remote(raw_fn))).

        Lock acquisition, heartbeats, caching, and status writes all run on
        the executor node.  Only the raw step logic runs inside the remote
        job (if the step was decorated with ``@remote``).
        """
        from marin.execution.artifact import Artifact
        from marin.execution.disk_cache import disk_cache
        from marin.execution.executor_step_status import distributed_lock
        from marin.execution.remote import RemoteCallable

        raw_fn = self.fn.fn if isinstance(self.fn, RemoteCallable) else self.fn

        # If the step is remote, wrap the raw function in a RemoteCallable so
        # that the step logic runs on the remote worker, but keep lock/cache
        # handling on the executor node.
        #
        # Fray jobs can't return values, so the remote wrapper saves the
        # artifact inside the job.  The executor-side disk_cache skips the
        # save (no-op) and only uses load_fn on cache hits.
        inner_fn = raw_fn
        save_fn: Callable = Artifact.save
        if isinstance(self.fn, RemoteCallable):
            captured_raw = raw_fn
            assert captured_raw is not None

            def _remote_with_artifact_save(output_path: str):
                result = captured_raw(output_path)  # pyrefly: ignore[not-callable]
                Artifact.save(result, output_path)

            job_name = f"{self.name_with_hash}-{uuid.uuid4().hex[:8]}"
            remote_callable = self.fn.named(job_name)
            inner_fn = dataclasses.replace(remote_callable, fn=_remote_with_artifact_save)

            def _noop_save(result: Any, path: str) -> None:
                pass

            save_fn = _noop_save

        wrapped = disk_cache(
            distributed_lock(inner_fn),
            output_path=self.output_path,
            save_fn=save_fn,
            load_fn=Artifact.load,
        )

        return wrapped
