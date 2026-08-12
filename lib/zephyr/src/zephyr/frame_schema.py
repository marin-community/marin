# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relaxed schema unification helpers for Polars frames."""

from collections.abc import Sequence
from typing import Any, Literal

import polars as pl

ConcatHow = Literal["vertical_relaxed", "diagonal_relaxed"]


def unified_schema(
    frames: Sequence[pl.DataFrame | pl.LazyFrame],
    *,
    how: ConcatHow,
) -> pl.Schema:
    """Return the schema of ``pl.concat(..., how=how)`` over empty/limit-0 frames.

    Raises whatever ``pl.concat`` raises when *how* cannot unify the schemas.
    """
    if not frames:
        raise ValueError("unified_schema requires at least one frame")
    empties: list[pl.DataFrame | pl.LazyFrame] = []
    for frame in frames:
        if isinstance(frame, pl.LazyFrame):
            empties.append(frame.limit(0))
        else:
            empties.append(frame.clear())
    out = pl.concat(empties, how=how)
    if isinstance(out, pl.LazyFrame):
        return out.collect_schema()
    return out.schema


def is_safe_sort_key_dtype_pair(left: Any, right: Any) -> bool:
    """Whether two sort-key field dtypes can be unified without reordering risk.

    Allows identical types, ``Null`` ↔ T, integer family widenings, float family
    widenings, and Utf8/String equivalence. Rejects cross-family coercions that
    ``vertical_relaxed`` would still accept (e.g. Int64 ↔ Utf8 → String).
    """
    if left == right:
        return True
    if left == pl.Null or right == pl.Null:
        return True
    if left.is_integer() and right.is_integer():
        return True
    if left.is_float() and right.is_float():
        return True
    stringish = (pl.Utf8, pl.String)
    if left in stringish and right in stringish:
        return True
    if left == pl.Binary and right == pl.Binary:
        return True
    return False


def sort_key_field_dtype(schema: pl.Schema, sort_key_col: str, field: str) -> Any | None:
    """Return dtype of ``schema[sort_key_col].struct.field``, or ``None`` if absent."""
    if sort_key_col not in schema:
        return None
    dtype = schema[sort_key_col]
    if not isinstance(dtype, pl.Struct):
        return None
    for struct_field in dtype.fields:
        if struct_field.name == field:
            return struct_field.dtype
    return None


def assert_compatible_sort_key_dtypes(
    schemas: Sequence[pl.Schema],
    *,
    sort_key_col: str,
    field: str = "key",
) -> None:
    """Raise ``ValueError`` if *field* dtypes across *schemas* are not merge-safe."""
    dtypes: list[Any] = []
    for schema in schemas:
        dt = sort_key_field_dtype(schema, sort_key_col, field)
        if dt is not None:
            dtypes.append(dt)
    for i, left in enumerate(dtypes):
        for right in dtypes[i + 1 :]:
            if not is_safe_sort_key_dtype_pair(left, right):
                raise ValueError(
                    f"incompatible scatter sort-key field {field!r} dtypes across mapper "
                    f"chunks: {left} vs {right}. Columnar group_by(col(...)) keys must "
                    f"share a stable dtype (null/integer/float widenings only); "
                    f"refusing to cast before merge_sorted."
                )
