# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen source-population selection shared by review and release workflows."""

from pydantic import BaseModel, Field


class RetentionPlan(BaseModel):
    source_id: str
    source_dataset_id: str
    source_revision: str
    converter_revision: str
    converter_version: str
    input_artifact: str
    task_specs_sha256: str
    raw_count: int = Field(ge=0)
    combined_drop_count: int = Field(ge=0)
    retained_count: int = Field(ge=0)
    combined_drop_manifest_uri: str
    lexical_use: str
