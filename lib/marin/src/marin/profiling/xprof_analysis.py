# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for converting and comparing xprof XPlane op data."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

GroupBy = Literal["program", "category", "name"]

_DECODER_ADD_SHELL_RE = re.compile(r"(^|/)add(_any)?(?::|/|$)")
_LAYOUT_SHELL_RE = re.compile(r"(^|/)(reshape|transpose|bitcast)(?::|/|$)")
_AD_WRAPPER_SHELL_RE = re.compile(
    r"(transpose\(jvp\(|^jvp\(|(^|/)(select_n|scatter-add|scatter_add|select-and-scatter|select-and-gather)(?::|/|$))"
)
_DISPATCH_SHARD_SHELL_RE = re.compile(
    r"(shard_map/pallas_call|closed_call/shard_map|shard_map/psum|all-gather|all_gather)"
)


@dataclass(frozen=True)
class XprofDeltaRow:
    name: str
    before: float
    after: float
    delta: float
    normalized_ms: float | None = None
    family: str | None = None


def _load_xprof_converter():
    try:
        from xprof.convert import raw_to_tool_data  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "xprof raw conversion is unavailable in this environment. "
            "Run on a Linux host with xprof installed and force use_saved_result=False."
        ) from exc
    return raw_to_tool_data


def convert_xplane_tool_data(
    xplane_paths: list[Path],
    *,
    tool: str,
    params: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    raw_to_tool_data = _load_xprof_converter()
    tool_params = dict(params or {})
    tool_params.setdefault("use_saved_result", False)
    data, content_type = raw_to_tool_data.xspace_to_tool_data([str(path) for path in xplane_paths], tool, tool_params)
    if data is None:
        raise RuntimeError(f"xprof conversion returned no data for tool={tool!r}")
    if isinstance(data, bytes):
        return json.loads(data), content_type
    if isinstance(data, str):
        return json.loads(data), content_type
    return data, content_type


def parse_op_profile_children(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    root_key = next(iter(payload))
    tree = payload[root_key]
    children = tree.get("children", []) if isinstance(tree, dict) else []
    result: dict[str, dict[str, Any]] = {}
    for child in children:
        if not isinstance(child, dict):
            continue
        name = child.get("name")
        if isinstance(name, str):
            result[name] = child
    return result


def parse_framework_op_stats_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        return []
    table = payload[0]
    if not isinstance(table, dict):
        return []
    cols = [col.get("id") for col in table.get("cols", []) if isinstance(col, dict)]
    rows: list[dict[str, Any]] = []
    for row in table.get("rows", []):
        if not isinstance(row, dict):
            continue
        cells = row.get("c", [])
        values = []
        for cell in cells:
            if isinstance(cell, dict):
                values.append(cell.get("v"))
            else:
                values.append(None)
        rows.append(dict(zip(cols, values, strict=False)))
    return rows


def _normalize_positive_rows(
    rows: list[XprofDeltaRow], normalize_positive_deltas_ms: float | None
) -> list[XprofDeltaRow]:
    if normalize_positive_deltas_ms is None:
        return rows
    positive_total = sum(row.delta for row in rows if row.delta > 0)
    if positive_total <= 0:
        return rows
    normalized: list[XprofDeltaRow] = []
    for row in rows:
        norm = None
        if row.delta > 0:
            norm = (row.delta / positive_total) * normalize_positive_deltas_ms
        normalized.append(
            XprofDeltaRow(
                name=row.name,
                before=row.before,
                after=row.after,
                delta=row.delta,
                normalized_ms=norm,
                family=row.family,
            )
        )
    return normalized


def compare_xprof_named_rows(
    before: dict[str, float],
    after: dict[str, float],
    *,
    top_k: int = 20,
    normalize_positive_deltas_ms: float | None = None,
    family_classifier: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    rows: list[XprofDeltaRow] = []
    family_totals: dict[str, float] = {}
    for name in sorted(set(before) | set(after)):
        before_value = float(before.get(name, 0.0))
        after_value = float(after.get(name, 0.0))
        family = family_classifier(name) if family_classifier is not None else None
        rows.append(
            XprofDeltaRow(
                name=name,
                before=before_value,
                after=after_value,
                delta=after_value - before_value,
                family=family,
            )
        )
        if family is not None and after_value - before_value > 0:
            family_totals[family] = family_totals.get(family, 0.0) + (after_value - before_value)
    positive = sorted((row for row in rows if row.delta > 0), key=lambda row: (-row.delta, row.name))
    negative = sorted((row for row in rows if row.delta < 0), key=lambda row: (row.delta, row.name))
    positive = _normalize_positive_rows(positive, normalize_positive_deltas_ms)[:top_k]
    negative = _normalize_positive_rows(negative, None)[:top_k]

    family_rows: list[dict[str, Any]] = []
    positive_family_total = sum(family_totals.values())
    for family, delta in sorted(family_totals.items(), key=lambda item: (-item[1], item[0])):
        normalized_ms = None
        if normalize_positive_deltas_ms is not None and positive_family_total > 0:
            normalized_ms = (delta / positive_family_total) * normalize_positive_deltas_ms
        family_rows.append({"family": family, "delta": delta, "normalized_ms": normalized_ms})

    return {
        "positive_deltas": [row.__dict__ for row in positive],
        "negative_deltas": [row.__dict__ for row in negative],
        "family_positive_deltas": family_rows,
        "positive_delta_total": sum(row.delta for row in rows if row.delta > 0),
        "negative_delta_total": sum(row.delta for row in rows if row.delta < 0),
    }


def _normalized_metric_map(
    rows: list[dict[str, Any]],
    *,
    key_field: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        key = row.get(key_field)
        normalized_ms = row.get("normalized_ms")
        if isinstance(key, str) and isinstance(normalized_ms, (int, float)):
            result[key] = float(normalized_ms)
    return result


def canonical_hybrid_shell_family(path: str) -> str | None:
    normalized = path.strip()
    if not normalized:
        return None
    if _DECODER_ADD_SHELL_RE.search(normalized) is not None:
        return "residual_add_shell"
    if _LAYOUT_SHELL_RE.search(normalized) is not None and not normalized.startswith("transpose(jvp("):
        return "layout_shell"
    if "transpose(jvp(" in normalized and "closed_call/shard_map" in normalized and "pallas_call" not in normalized:
        return "ad_wrapper_shell"
    if _AD_WRAPPER_SHELL_RE.search(normalized) is not None and not _DISPATCH_SHARD_SHELL_RE.search(normalized):
        return "ad_wrapper_shell"
    if _DISPATCH_SHARD_SHELL_RE.search(normalized) is not None:
        return "dispatch_shard_shell"
    if _AD_WRAPPER_SHELL_RE.search(normalized) is not None:
        return "ad_wrapper_shell"
    return None


def framework_rows_to_self_time_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        operation = row.get("operation")
        total_self_time = row.get("total_self_time")
        if isinstance(operation, str) and isinstance(total_self_time, (int, float)):
            result[operation] = float(total_self_time)
    return result


def op_profile_rows_to_raw_time_map(children: dict[str, dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, child in children.items():
        metrics = child.get("metrics") if isinstance(child, dict) else None
        raw_time = metrics.get("rawTime") if isinstance(metrics, dict) else None
        if isinstance(raw_time, (int, float)):
            result[name] = float(raw_time)
    return result


def build_xprof_comparison_report(
    *,
    before_xplane: Path,
    after_xplane: Path,
    top_k: int = 20,
    normalize_positive_deltas_ms: float | None = None,
) -> dict[str, Any]:
    before_framework, _ = convert_xplane_tool_data([before_xplane], tool="framework_op_stats")
    after_framework, _ = convert_xplane_tool_data([after_xplane], tool="framework_op_stats")
    before_framework_rows = parse_framework_op_stats_rows(before_framework)
    after_framework_rows = parse_framework_op_stats_rows(after_framework)
    framework_compare = compare_xprof_named_rows(
        framework_rows_to_self_time_map(before_framework_rows),
        framework_rows_to_self_time_map(after_framework_rows),
        top_k=top_k,
        normalize_positive_deltas_ms=normalize_positive_deltas_ms,
        family_classifier=canonical_hybrid_shell_family,
    )

    before_category, _ = convert_xplane_tool_data([before_xplane], tool="op_profile", params={"group_by": "category"})
    after_category, _ = convert_xplane_tool_data([after_xplane], tool="op_profile", params={"group_by": "category"})
    category_compare = compare_xprof_named_rows(
        op_profile_rows_to_raw_time_map(parse_op_profile_children(before_category)),
        op_profile_rows_to_raw_time_map(parse_op_profile_children(after_category)),
        top_k=top_k,
        normalize_positive_deltas_ms=normalize_positive_deltas_ms,
    )

    framework_family_normalized_ms = _normalized_metric_map(
        framework_compare["family_positive_deltas"],
        key_field="family",
    )
    op_profile_category_normalized_ms = _normalized_metric_map(
        category_compare["positive_deltas"],
        key_field="name",
    )

    return {
        "before_xplane": str(before_xplane),
        "after_xplane": str(after_xplane),
        "normalize_positive_deltas_ms": normalize_positive_deltas_ms,
        "framework_op_stats": framework_compare,
        "op_profile_category": category_compare,
        "derived_metrics": {
            "framework_family_normalized_ms": framework_family_normalized_ms,
            "op_profile_category_normalized_ms": op_profile_category_normalized_ms,
            "idle_normalized_ms": op_profile_category_normalized_ms.get("IDLE"),
        },
    }
