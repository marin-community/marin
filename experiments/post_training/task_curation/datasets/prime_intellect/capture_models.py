# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical wire models for complete Multi-SWE image artifact captures."""

from dataclasses import dataclass

from marin.execution.artifact import Artifact
from pydantic import BaseModel, ConfigDict, Field


class ImageBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    original_references: tuple[str, ...] = Field(min_length=1)
    pinned_reference: str = Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    task_ids: tuple[str, ...] = Field(min_length=1)


class CaptureInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    dataset_id: str
    revision: str
    taskset_revision: str
    images: tuple[ImageBinding, ...] = Field(min_length=1)
    raw_files: dict[str, str]


class CapturedArtifacts(Artifact):
    source_id: str
    dataset_id: str
    revision: str
    input_sha256: str
    images: int
    tasks: int
    manifest_uri: str
    manifest_sha256: str
    placement_uri: str


@dataclass(frozen=True)
class CaptureConfig:
    input_path: str
    input_sha256: str
    cluster: str
    concurrency: int
    license_receipt_json: str
    output_path: str
