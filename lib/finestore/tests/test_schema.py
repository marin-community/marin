# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for deriving an arrow schema from a pydantic model or a dataclass."""

from __future__ import annotations

import dataclasses
import enum

import pyarrow as pa
import pytest
from finestore.rl.source_identity import SourceId
from finestore.schema import arrow_schema
from pydantic import BaseModel


class Color(enum.StrEnum):
    RED = "red"


class Inner(BaseModel):
    a: int
    b: str | None = None


class Sample(BaseModel):
    name: str
    score: float | None = None
    tags: list[str]
    metrics: dict[str, float]
    inner: Inner | None = None
    color: Color


@dataclasses.dataclass
class Row:
    x: int
    ys: list[float] | None


def test_arrow_schema_from_pydantic():
    types = {field.name: field.type for field in arrow_schema(Sample)}
    assert types["name"] == pa.string()
    assert types["score"] == pa.float64()  # T | None unwrapped to nullable T
    assert types["tags"] == pa.list_(pa.string())
    assert types["metrics"] == pa.map_(pa.string(), pa.float64())  # dynamic-keyed column stays one type
    assert types["inner"] == pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.string())])
    assert types["color"] == pa.string()  # StrEnum


def test_every_field_is_nullable_for_schema_evolution():
    assert all(field.nullable for field in arrow_schema(Sample))


def test_arrow_schema_from_dataclass():
    types = {field.name: field.type for field in arrow_schema(Row)}
    assert types["x"] == pa.int64()
    assert types["ys"] == pa.list_(pa.float64())


def test_pa_schema_passes_through_unchanged():
    raw = pa.schema([("a", pa.int64())])
    assert arrow_schema(raw) is raw


def test_unsupported_input_raises():
    with pytest.raises(TypeError):
        arrow_schema(42)


class AliasedSourceRow(BaseModel):
    source_id: SourceId | None
    related: list[SourceId]


def test_annotated_optional_source_ids_preserve_the_arrow_storage_contract():
    rows = [{"source_id": "MarinSkyRL:aime24", "related": ["MarinSkyRL:svamp"]}, {"source_id": None, "related": []}]
    table = pa.Table.from_pylist(rows, schema=arrow_schema(AliasedSourceRow))
    assert table.to_pylist() == rows
    decoded = [AliasedSourceRow.model_validate(row) for row in table.to_pylist()]
    assert decoded[0].source_id == "HuggingFaceH4:aime24"
    assert decoded[0].related == ["ChilleD:svamp"]
    assert decoded[1].source_id is None
