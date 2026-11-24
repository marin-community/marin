# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for deriving an arrow schema from a pydantic model or a dataclass."""

from __future__ import annotations

import dataclasses
import enum
from typing import Annotated, Literal

import pyarrow as pa
import pytest
from finestore.schema import arrow_schema
from pydantic import BaseModel, BeforeValidator


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


TrimmedId = Annotated[str, BeforeValidator(str.strip)]


class AnnotatedRow(BaseModel):
    source_id: TrimmedId | None
    related: list[TrimmedId]


def test_annotated_optional_source_ids_preserve_the_arrow_storage_contract():
    rows = [{"source_id": " first ", "related": [" second "]}, {"source_id": None, "related": []}]
    table = pa.Table.from_pylist(rows, schema=arrow_schema(AnnotatedRow))
    assert table.to_pylist() == rows
    decoded = [AnnotatedRow.model_validate(row) for row in table.to_pylist()]
    assert decoded[0].source_id == "first"
    assert decoded[0].related == ["second"]
    assert decoded[1].source_id is None


class VersionedPolicy(BaseModel):
    version: Literal[1]
    execution_required: Literal[False]
    status: Literal["retained", "excluded"]


def test_literal_policy_fields_round_trip_through_arrow():
    policy = VersionedPolicy(version=1, execution_required=False, status="retained")
    table = pa.Table.from_pylist([policy.model_dump()], schema=arrow_schema(VersionedPolicy))
    assert table.schema == pa.schema(
        [("version", pa.int64()), ("execution_required", pa.bool_()), ("status", pa.string())]
    )
    assert [VersionedPolicy.model_validate(row) for row in table.to_pylist()] == [policy]
