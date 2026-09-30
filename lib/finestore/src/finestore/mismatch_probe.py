# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed archive contract for same-token trainer and sampler mismatch probes.

The training callback writes these rows; Marin's report reads the same models.
Scorings add rows to ``scores`` instead of columns, so new trainer modes do not
change the table contract. Route tensors carry explicit
shape and dtype beside their bytes to preserve exact values through Parquet.
"""

from __future__ import annotations

from enum import StrEnum

import pyarrow as pa
from pydantic import BaseModel, model_validator

from finestore.layout import OnConflict
from finestore.schema import arrow_schema
from finestore.store import DataStore

SCHEMA_VERSION = 2
PROBE_TABLE = "probe"
SCORES_TABLE = "scores"
MANIFEST_TABLE = "manifest"


class ProbeRow(BaseModel):
    """One frozen answer, including token identity and generation-time routes."""

    schema_version: int = SCHEMA_VERSION
    probe_hash: str
    sample_id: str
    prompt_id: str
    prompt_token_ids: list[int]
    trainer_prompt_ids: list[int]
    vllm_output_ids: list[int]
    trainer_input_ids: list[int]
    response_mask: list[bool]
    loss_mask: list[bool]
    reward: float | None = None
    advantage: float | None = None
    request_seed: int
    batch_position: int
    routed_experts: bytes | None = None
    routed_experts_shape: list[int] | None = None
    routed_experts_dtype: str | None = None
    route_valid_mask: list[list[bool]] | None = None

    @model_validator(mode="after")
    def validate_token_lengths(self) -> ProbeRow:
        response_length = len(self.vllm_output_ids)
        if any(
            len(values) != response_length for values in (self.trainer_input_ids, self.response_mask, self.loss_mask)
        ):
            raise ValueError("probe response IDs and masks must have the same length")
        if (self.routed_experts is None) != (self.routed_experts_shape is None):
            raise ValueError("probe routes require both bytes and shape")
        if (self.routed_experts is None) != (self.routed_experts_dtype is None):
            raise ValueError("probe routes require both bytes and dtype")
        if (self.routed_experts is None) != (self.route_valid_mask is None):
            raise ValueError("probe routes require an explicit token-and-layer validity mask")
        if self.routed_experts_shape is not None and self.route_valid_mask is not None:
            if (
                len(self.routed_experts_shape) != 3
                or self.routed_experts_shape[0] != response_length
                or len(self.route_valid_mask) != response_length
            ):
                raise ValueError("probe route validity must match [response, layer, expert] routes")
            if any(len(token_layers) != self.routed_experts_shape[1] for token_layers in self.route_valid_mask):
                raise ValueError("probe route validity must cover every captured layer")
        return self


class ScoreRow(BaseModel):
    """One scorer's logprob vector for one frozen answer at one update."""

    schema_version: int = SCHEMA_VERSION
    probe_hash: str
    sample_id: str
    scorer: str
    mode: str = ""
    update: int
    weights_hash: str
    cache_mode: str | None = None
    logprobs: list[float]
    forward_seconds: float | None = None
    expert_choices: bytes | None = None
    expert_choices_shape: list[int] | None = None
    expert_choices_dtype: str | None = None
    replacement_mask: bytes | None = None

    @model_validator(mode="after")
    def validate_route_shape(self) -> ScoreRow:
        if (self.expert_choices is None) != (self.expert_choices_shape is None):
            raise ValueError("scorer route observations require both bytes and shape")
        if (self.expert_choices is None) != (self.expert_choices_dtype is None):
            raise ValueError("scorer route observations require both bytes and dtype")
        if self.replacement_mask is not None and self.expert_choices is None:
            raise ValueError("replacement mask requires scorer route observations")
        return self


class ArchiveStatus(StrEnum):
    BUILDING = "building"
    COMPLETE = "complete"


class ManifestRow(BaseModel):
    """Provenance and completion state for one mismatch archive."""

    schema_version: int = SCHEMA_VERSION
    archive: str
    status: ArchiveStatus
    probe_hash: str
    starting_weights_hash: str
    source_probe_archive: str | None = None
    tokenizer_fingerprint: str
    starting_global_step: int
    scored_updates: list[int]
    scored_global_steps: list[int]
    architecture: str
    vllm_enforce_eager: bool
    optimizer_steps_per_update: int
    seed: int
    bootstrap_seed: int
    created_at_utc: str
    config_json: str
    software_json: str
    hardware_json: str
    batch_layout_json: str
    timing_json: str
    step_metrics_json: str


ROW_MODELS: dict[str, type[BaseModel]] = {
    PROBE_TABLE: ProbeRow,
    SCORES_TABLE: ScoreRow,
    MANIFEST_TABLE: ManifestRow,
}

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    PROBE_TABLE: ("probe_hash", "sample_id"),
    SCORES_TABLE: ("probe_hash", "sample_id", "scorer", "update", "mode", "cache_mode"),
    MANIFEST_TABLE: ("archive",),
}


def mismatch_schema(table: str) -> pa.Schema:
    """Pin score vectors to float32 and derive all other columns from row models."""
    schema = arrow_schema(ROW_MODELS[table])
    if table == SCORES_TABLE:
        index = schema.get_field_index("logprobs")
        return schema.set(index, pa.field("logprobs", pa.list_(pa.float32())))
    return schema


def register_mismatch_tables(store: DataStore) -> None:
    """Register the three tables with their exact keys and Arrow contracts."""
    for table, primary_key in PRIMARY_KEYS.items():
        store.table(
            table,
            primary_key=primary_key,
            schema=mismatch_schema(table),
            on_conflict=OnConflict.SUPERSEDE if table == MANIFEST_TABLE else OnConflict.ERROR,
        )
