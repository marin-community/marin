# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import TypeVar, overload, Any
from dataclasses import dataclass, is_dataclass
from marin.execution.step_model import StepSpec

import fsspec
from pydantic import BaseModel

T = TypeVar("T")


class Artifact:
    __artifact_file_name = ".artifact"

    @overload
    @classmethod
    def load(cls, base_path: str | StepSpec, artifact_type: type[T]) -> T: ...

    @overload
    @classmethod
    def load(cls, base_path: str | StepSpec) -> dict[str, Any]: ...

    @classmethod
    def load(cls, base_path: str | StepSpec, artifact_type: type[T] | None = None) -> T | dict[str, Any]:
        """Loads an Artifact instance from the specified output base path"""

        if isinstance(base_path, StepSpec):
            base_path = base_path.output_path

        with fsspec.open(f"{base_path}/{cls.__artifact_file_name}", "rb") as fd:
            if artifact_type is None:
                return json.load(fd)
            if issubclass(artifact_type, BaseModel):
                return artifact_type.model_validate_json(fd.read())
            if is_dataclass(artifact_type):
                return artifact_type(**json.load(fd))  # type: ignore[not-callable]
            raise ValueError(f"Unsupported artifact type: {artifact_type!r}")

    @classmethod
    def save(cls, artifact: T, base_path: str) -> None:
        """Saves an Artifact instance to the specified output base path"""
        with fsspec.open(f"{base_path}/{cls.__artifact_file_name}", "wb") as fd:
            if isinstance(artifact, BaseModel):
                fd.write(artifact.model_dump_json().encode("utf-8"))
            elif is_dataclass(artifact):
                fd.write(json.dumps(artifact.__dict__).encode("utf-8"))
            else:
                # TODO: should the error to serialize be ignored/logged instead of raising an exception?
                fd.write(json.dumps(artifact).encode("utf-8"))


@dataclass
class PathMetadata:
    """Represents a single output path"""

    path: str


@dataclass
class PathsMetadata:
    """Represents a list of paths to the output files

    Useful for Zephyr steps to capture all the output shards.
    """

    parent_path: str
    paths: list[str]
