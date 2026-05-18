# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, TypeVar, overload

from pydantic import BaseModel
from rigging.filesystem import marin_prefix, open_url

from marin.execution.executor_step_status import STATUS_SUCCESS, get_status_path
from marin.execution.step_spec import StepSpec, _is_relative_path

T = TypeVar("T")


class Artifact:
    __artifact_file_name = "artifact.json"
    # Legacy filename written before the rename; read-only fallback so historical
    # GCS outputs remain loadable. Safe to remove once those prefixes are gone.
    __legacy_artifact_file_name = ".artifact"

    @overload
    @classmethod
    def from_path(cls, base_path: str | StepSpec, artifact_type: type[T]) -> T: ...

    @overload
    @classmethod
    def from_path(cls, base_path: str | StepSpec) -> "PathMetadata | dict[str, Any]": ...

    @classmethod
    def from_path(
        cls, base_path: str | StepSpec, artifact_type: type[T] | None = None
    ) -> "T | PathMetadata | dict[str, Any]":
        """Load an Artifact instance from the specified output base path.

        If ``base_path`` is a relative path (no URL scheme, doesn't start with ``/``),
        it is resolved against ``marin_prefix()``.

        If ``base_path`` has no ``artifact.json`` (or legacy ``.artifact``) file but
        its ``.executor_status`` file contains ``SUCCESS``, returns a
        :class:`PathMetadata` pointing at ``base_path`` — provided the caller asked
        for no specific type or for ``PathMetadata``.
        """

        if isinstance(base_path, StepSpec):
            base_path = base_path.output_path
        elif _is_relative_path(base_path):
            base_path = f"{marin_prefix()}/{base_path}"

        for file_name in (cls.__artifact_file_name, cls.__legacy_artifact_file_name):
            try:
                with open_url(f"{base_path}/{file_name}", "rb") as fd:
                    if artifact_type is None:
                        return json.load(fd)
                    if not issubclass(artifact_type, BaseModel):
                        raise TypeError(f"artifact_type must be a pydantic BaseModel subclass, got {artifact_type!r}")
                    return artifact_type.model_validate_json(fd.read())
            except FileNotFoundError:
                continue
        return cls._from_executor_status(base_path, artifact_type)

    @classmethod
    def _from_executor_status(cls, base_path: str, artifact_type: type[T] | None) -> "T | PathMetadata":
        """Fallback when no artifact file is present: synthesize a :class:`PathMetadata`
        if the step published ``.executor_status = SUCCESS``.

        Only valid when the caller wants no type or ``PathMetadata`` — other types
        cannot be reconstructed from a bare path.
        """
        if artifact_type is not None and artifact_type is not PathMetadata:
            raise FileNotFoundError(
                f"No {cls.__artifact_file_name} at {base_path}; cannot synthesize "
                f"{artifact_type!r} from {get_status_path(base_path)!r}"
            )
        with open_url(get_status_path(base_path), "r") as fd:
            status = fd.read().strip()
        if status != STATUS_SUCCESS:
            raise FileNotFoundError(
                f"No {cls.__artifact_file_name} at {base_path} and "
                f"{get_status_path(base_path)!r} is {status!r} (not {STATUS_SUCCESS!r})"
            )
        return PathMetadata(path=base_path)

    @classmethod
    def save(cls, artifact: T, base_path: str) -> None:
        """Saves an Artifact instance to the specified output base path"""
        with open_url(f"{base_path}/{cls.__artifact_file_name}", "wb") as fd:
            if isinstance(artifact, BaseModel):
                fd.write(artifact.model_dump_json().encode("utf-8"))
            elif is_dataclass(artifact):
                # `asdict` recursively converts nested dataclasses (for example ResourceConfig),
                # avoiding non-serializable objects in `__dict__`.
                fd.write(json.dumps(asdict(artifact)).encode("utf-8"))
            else:
                # TODO: should the error to serialize be ignored/logged instead of raising an exception?
                fd.write(json.dumps(artifact).encode("utf-8"))


class PathMetadata(BaseModel):
    """Represents a single output path.

    Also used as the synthetic return type of :meth:`Artifact.from_path` when the
    step published a ``.executor_status = SUCCESS`` marker but no ``.artifact``.
    """

    path: str


@dataclass
class PathsMetadata:
    """Represents a list of paths to the output files

    Useful for Zephyr steps to capture all the output shards.
    """

    parent_path: str
    paths: list[str]
