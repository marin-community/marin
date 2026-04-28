# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile trace-masked eval outputs into compact comparison tables."""

import csv
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import fsspec

from marin.execution.executor import ExecutorStep, OutputName
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)

TRACE_MASKED_PREFIX = "trace_masked_eval"
PATCH_PREFIX_PATTERN = re.compile(r"^(?P<base>.+)_patch_prefix_(?P<prefix>\d+)$")
LOCAL_BPB_FIELDS = {
    "assistant": "assistant",
    "assistant_text": "assistant_text",
    "final_assistant": "final_assistant",
    "tool_call": "tool",
    "observation": "obs",
}
OUTCOME_BPB_FIELDS = {
    "patch": "patch",
}
OUTCOME_AUC_FIELDS = {
    "outcome_contrastive/normalized_auroc": "auc",
    "outcome_contrastive/prefix_25/normalized_auroc": "auc25",
    "outcome_contrastive/prefix_50/normalized_auroc": "auc50",
    "outcome_contrastive/prefix_75/normalized_auroc": "auc75",
    "outcome_contrastive/prefix_100/normalized_auroc": "auc100",
}
PATCH_PREFIX_FIELDS = {
    0: "patch_00",
    50: "patch_50",
}
COMPACT_COLUMNS = (
    "model",
    "assistant_text",
    "final_assistant",
    "patch",
    "tool",
    "obs",
    "patch_gain",
)
EXPANDED_COLUMNS = (
    "model",
    "assistant",
    "assistant_text",
    "final_assistant",
    "tool",
    "obs",
    "patch",
    "auc25",
    "auc50",
    "auc75",
    "auc100",
    "patch_00",
    "patch_50",
    "patch_gain",
)


@dataclass(frozen=True)
class TraceMaskedResultInput:
    """One model/result pair to include in the compiled summary."""

    model_name: str
    results_path: str


def _load_results(path: str) -> dict[str, Any]:
    with fsspec.open(path, "r") as f:
        results = json.load(f)
    if not isinstance(results, dict):
        raise ValueError(f"Expected mapping in {path!r}, got {type(results)}")
    return results


def _completed_dataset_results(results: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    datasets = results.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("Trace-masked results missing 'datasets' mapping")

    completed: dict[str, Mapping[str, Any]] = {}
    for dataset_name, dataset_result in datasets.items():
        if not isinstance(dataset_name, str) or not isinstance(dataset_result, Mapping):
            continue
        metrics = dataset_result.get("metrics")
        metadata = dataset_result.get("metadata")
        if not isinstance(metrics, Mapping) or not isinstance(metadata, Mapping):
            continue
        completed[dataset_name] = dataset_result
    return completed


def _metric_value(metrics: Mapping[str, Any], dataset_name: str, suffix: str) -> float | None:
    metric_name = f"{TRACE_MASKED_PREFIX}/{dataset_name}/{suffix}"
    value = metrics.get(metric_name)
    if isinstance(value, int | float):
        return float(value)
    return None


def _macro_average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _base_patch_name(dataset_name: str) -> tuple[str, int] | None:
    match = PATCH_PREFIX_PATTERN.fullmatch(dataset_name)
    if match is None:
        return None
    return match.group("base"), int(match.group("prefix"))


def _format_value(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _render_markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---:" if column != "model" else "---" for column in columns) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(row["model"] if column == "model" else _format_value(row.get(column)) for column in columns)
            + " |"
        )
    return "\n".join([header, align, *body])


def _render_text_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    formatted_rows: list[list[str]] = []
    widths = [len(column) for column in columns]
    for row in rows:
        formatted = []
        for index, column in enumerate(columns):
            value = row["model"] if column == "model" else _format_value(row.get(column))
            widths[index] = max(widths[index], len(value))
            formatted.append(value)
        formatted_rows.append(formatted)

    header = "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    lines = [header]
    for formatted in formatted_rows:
        line_parts = []
        for index, value in enumerate(formatted):
            justify = str.ljust if columns[index] == "model" else str.rjust
            line_parts.append(justify(value, widths[index]))
        lines.append("  ".join(line_parts))
    return "\n".join(lines)


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with open_url(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_text(path: str, text: str) -> None:
    with open_url(path, "w") as f:
        f.write(text)
        f.write("\n")


def _summarize_single_model(model_name: str, results: Mapping[str, Any]) -> dict[str, Any]:
    dataset_results = _completed_dataset_results(results)
    row: dict[str, Any] = {"model": model_name}

    local_metric_values = {output_name: [] for output_name in LOCAL_BPB_FIELDS.values()}
    outcome_metric_values = {output_name: [] for output_name in OUTCOME_BPB_FIELDS.values()}
    outcome_auc_values = {output_name: [] for output_name in OUTCOME_AUC_FIELDS.values()}
    patch_prefix_values: dict[int, list[float]] = {prefix: [] for prefix in PATCH_PREFIX_FIELDS}
    patch_by_dataset: dict[str, float] = {}
    patch_prefix_by_dataset: dict[int, dict[str, float]] = {prefix: {} for prefix in PATCH_PREFIX_FIELDS}

    for dataset_name, dataset_result in dataset_results.items():
        metadata = dataset_result["metadata"]
        metrics = dataset_result["metrics"]
        assert isinstance(metadata, Mapping)
        assert isinstance(metrics, Mapping)

        contrastive_outcome = bool(metadata.get("contrastive_outcome"))
        row_prefix_fraction = metadata.get("row_prefix_fraction")

        for metric_name, output_name in LOCAL_BPB_FIELDS.items():
            value = _metric_value(metrics, dataset_name, f"{metric_name}/bpb")
            if value is None or contrastive_outcome or row_prefix_fraction is not None:
                continue
            local_metric_values[output_name].append(value)

        for metric_name, output_name in OUTCOME_BPB_FIELDS.items():
            value = _metric_value(metrics, dataset_name, f"{metric_name}/bpb")
            if value is None:
                continue
            if contrastive_outcome and row_prefix_fraction is None:
                outcome_metric_values[output_name].append(value)
                if output_name == "patch":
                    patch_by_dataset[dataset_name] = value
                continue

            patch_prefix_info = _base_patch_name(dataset_name)
            if output_name != "patch" or patch_prefix_info is None:
                continue
            base_dataset_name, prefix_percent = patch_prefix_info
            if prefix_percent in patch_prefix_values:
                patch_prefix_values[prefix_percent].append(value)
                patch_prefix_by_dataset[prefix_percent][base_dataset_name] = value

        if contrastive_outcome and row_prefix_fraction is None:
            for metric_name, output_name in OUTCOME_AUC_FIELDS.items():
                value = _metric_value(metrics, dataset_name, metric_name)
                if value is not None:
                    outcome_auc_values[output_name].append(value)

    for output_name, values in local_metric_values.items():
        row[output_name] = _macro_average(values)
    for output_name, values in outcome_metric_values.items():
        row[output_name] = _macro_average(values)
    for output_name, values in outcome_auc_values.items():
        row[output_name] = _macro_average(values)

    for prefix_percent, output_name in PATCH_PREFIX_FIELDS.items():
        row[output_name] = _macro_average(patch_prefix_values[prefix_percent])

    patch_gains = []
    patch_prefix_zero = patch_prefix_by_dataset.get(0, {})
    for dataset_name, patch_value in patch_by_dataset.items():
        patch_prefix_value = patch_prefix_zero.get(dataset_name)
        if patch_prefix_value is None:
            continue
        patch_gains.append(patch_prefix_value - patch_value)
    row["patch_gain"] = _macro_average(patch_gains)

    return row


def compile_trace_masked_results_fn(config: dict[str, Any]) -> None:
    """Aggregate trace-masked results and write compact/expanded summaries."""

    inputs = config.get("inputs")
    output_path = config.get("output_path")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("compile_trace_masked_results_fn requires a non-empty 'inputs' list")
    if not isinstance(output_path, str):
        raise ValueError("compile_trace_masked_results_fn requires string 'output_path'")

    rows = []
    for input_entry in inputs:
        if not isinstance(input_entry, Mapping):
            raise ValueError(f"Expected mapping input entry, got {type(input_entry)}")
        model_name = input_entry.get("model_name")
        results_path = input_entry.get("results_path")
        if not isinstance(model_name, str) or not isinstance(results_path, str):
            raise ValueError(f"Invalid trace-masked input entry {input_entry!r}")
        rows.append(_summarize_single_model(model_name, _load_results(results_path)))

    compact_markdown = _render_markdown_table(rows, COMPACT_COLUMNS)
    compact_text = _render_text_table(rows, COMPACT_COLUMNS)
    expanded_markdown = _render_markdown_table(rows, EXPANDED_COLUMNS)
    expanded_text = _render_text_table(rows, EXPANDED_COLUMNS)

    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)

    summary_path = os.path.join(output_path, "compiled_results.json")
    summary = {
        "rows": rows,
        "compact_columns": list(COMPACT_COLUMNS),
        "expanded_columns": list(EXPANDED_COLUMNS),
        "compact_markdown": compact_markdown,
        "compact_text": compact_text,
        "expanded_markdown": expanded_markdown,
        "expanded_text": expanded_text,
    }
    with open_url(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    _write_csv(os.path.join(output_path, "compiled_results_compact.csv"), rows, COMPACT_COLUMNS)
    _write_csv(os.path.join(output_path, "compiled_results_expanded.csv"), rows, EXPANDED_COLUMNS)
    _write_text(os.path.join(output_path, "compiled_results_compact.md"), compact_markdown)
    _write_text(os.path.join(output_path, "compiled_results_compact.txt"), compact_text)
    _write_text(os.path.join(output_path, "compiled_results_expanded.md"), expanded_markdown)
    _write_text(os.path.join(output_path, "compiled_results_expanded.txt"), expanded_text)

    logger.info("Trace-masked compact summary:\n%s", compact_text)
    logger.info("Trace-masked expanded summary:\n%s", expanded_text)


def compile_trace_masked_results(
    *,
    name: str,
    steps: Sequence[TraceMaskedResultInput],
) -> ExecutorStep[dict[str, Any]]:
    """Create a compile step for a family of trace-masked eval runs."""

    return ExecutorStep(
        name=f"analysis/trace_masked_eval/{name}/compile",
        fn=compile_trace_masked_results_fn,
        config={
            "inputs": [
                {
                    "model_name": step.model_name,
                    "results_path": step.results_path,
                }
                for step in steps
            ],
            "output_path": OutputName("compiled_results"),
        },
        description="Compile trace-masked evaluation results into compact model comparison tables.",
    )
