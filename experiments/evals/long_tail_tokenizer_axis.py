# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer-axis diagnostics for whitespace-sensitive long-tail slices.

First pass for issue #5079:
https://github.com/marin-community/marin/issues/5079

This module intentionally reports tokenizer-driven statistics (tokens/byte and
byte composition) and does not report perplexity, since PPL is coupled to the
model+tokenizer pair.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import os
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import fsspec
from datasets import load_dataset
from rigging.filesystem import open_url

from levanter.tokenizers import load_tokenizer
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import LONG_TAIL_PPL_REGISTRY, LongTailPplFamily
from experiments.evals.long_tail_ppl_runnable import RUNNABLE_LONG_TAIL_PPL_SLICES

logger = logging.getLogger(__name__)

TOKENIZER_AXIS_ISSUE = 5079
TOKENIZER_AXIS_PARENT_ISSUE = 5005


class TokenizerAxisBackend(StrEnum):
    LEVANTER_HF = "levanter_hf"
    TIKTOKEN = "tiktoken"


class TokenizerAxisSliceAvailability(StrEnum):
    RUNNABLE_HF = "runnable_hf"
    PLANNED_RAW_MIRROR = "planned_raw_mirror"


@dataclass(frozen=True)
class TokenizerAxisTokenizerSpec:
    """Tokenizer selection for axis sweeps."""

    name: str
    tokenizer_id: str
    backend: TokenizerAxisBackend
    required: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "tokenizer_id": self.tokenizer_id,
            "backend": self.backend.value,
            "required": self.required,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TokenizerAxisSliceSpec:
    """A long-tail slice for tokenizer-axis diagnostics."""

    registry_key: str
    family: LongTailPplFamily
    source_url: str
    availability: TokenizerAxisSliceAvailability
    tags: tuple[str, ...]
    notes: str = ""
    hf_dataset: HfDatasetSpec | None = None
    text_key: str = "text"
    split: str = "validation"
    raw_relative_path: str | None = None

    @property
    def is_runnable(self) -> bool:
        return self.availability == TokenizerAxisSliceAvailability.RUNNABLE_HF and self.hf_dataset is not None

    def to_dict(self) -> dict[str, str | list[str] | None]:
        return {
            "registry_key": self.registry_key,
            "family": self.family.value,
            "source_url": self.source_url,
            "availability": self.availability.value,
            "tags": list(self.tags),
            "notes": self.notes,
            "hf_dataset_id": self.hf_dataset.id if self.hf_dataset is not None else None,
            "hf_dataset_name": self.hf_dataset.name if self.hf_dataset is not None else None,
            "text_key": self.text_key,
            "split": self.split,
            "raw_relative_path": self.raw_relative_path,
        }


@dataclass(frozen=True)
class SliceByteStats:
    total_bytes: int
    whitespace_bytes: int
    punctuation_bytes: int
    non_ascii_bytes: int

    def to_share(self, value: int) -> float | None:
        if self.total_bytes <= 0:
            return None
        return value / self.total_bytes


@dataclass(frozen=True)
class TokenizerAxisSliceSample:
    """A bounded text sample for one slice."""

    slice_spec: TokenizerAxisSliceSpec
    texts: tuple[str, ...]
    docs_scanned: int
    docs_used: int
    docs_skipped_missing_text_key: int
    docs_skipped_non_string: int
    byte_stats: SliceByteStats


@dataclass(frozen=True)
class TokenizerAxisSliceMetrics:
    """Tokenizer metrics for one tokenizer-slice pair."""

    tokenizer_name: str
    tokenizer_id: str
    tokenizer_backend: TokenizerAxisBackend
    slice_registry_key: str
    family: LongTailPplFamily
    docs_used: int
    docs_scanned: int
    docs_skipped_missing_text_key: int
    docs_skipped_non_string: int
    token_count: int
    byte_count: int
    tokens_per_byte: float | None
    bytes_per_token: float | None
    whitespace_byte_share: float | None
    punctuation_byte_share: float | None
    non_ascii_byte_share: float | None

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_backend": self.tokenizer_backend.value,
            "slice_registry_key": self.slice_registry_key,
            "family": self.family.value,
            "docs_used": self.docs_used,
            "docs_scanned": self.docs_scanned,
            "docs_skipped_missing_text_key": self.docs_skipped_missing_text_key,
            "docs_skipped_non_string": self.docs_skipped_non_string,
            "token_count": self.token_count,
            "byte_count": self.byte_count,
            "tokens_per_byte": self.tokens_per_byte,
            "bytes_per_token": self.bytes_per_token,
            "whitespace_byte_share": self.whitespace_byte_share,
            "punctuation_byte_share": self.punctuation_byte_share,
            "non_ascii_byte_share": self.non_ascii_byte_share,
        }


@dataclass(frozen=True)
class TokenizerInflationCorrelation:
    tokenizer_name: str
    baseline_tokenizer_name: str
    points: int
    mean_tokens_per_byte_delta: float | None
    whitespace_vs_tpb_delta_pearson: float | None

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "baseline_tokenizer_name": self.baseline_tokenizer_name,
            "points": self.points,
            "mean_tokens_per_byte_delta": self.mean_tokens_per_byte_delta,
            "whitespace_vs_tpb_delta_pearson": self.whitespace_vs_tpb_delta_pearson,
        }


@dataclass(frozen=True)
class TokenizerLoadFailure:
    tokenizer_name: str
    tokenizer_id: str
    backend: TokenizerAxisBackend
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_id": self.tokenizer_id,
            "backend": self.backend.value,
            "error": self.error,
        }


@dataclass(frozen=True)
class LongTailTokenizerAxisConfig:
    """Executor config for the #5079 tokenizer-axis first pass."""

    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    max_docs_per_slice: int = 512
    max_doc_bytes: int = 32_768
    baseline_tokenizer_name: str = "llama3_1_8b"
    include_planned_raw_slices: bool = True
    include_o200k_base: bool = False
    include_byte_reference: bool = False


def default_tokenizer_axis_step(
    *,
    name: str = "long-tail-runnable-first-pass",
    max_docs_per_slice: int = 512,
    max_doc_bytes: int = 32_768,
    baseline_tokenizer_name: str = "llama3_1_8b",
    include_planned_raw_slices: bool = True,
    include_o200k_base: bool = False,
    include_byte_reference: bool = False,
) -> ExecutorStep:
    """Build a reusable step for tokenizer-axis long-tail diagnostics."""

    return ExecutorStep(
        name=f"analysis/tokenizer_axis/{name}",
        fn=run_long_tail_tokenizer_axis,
        config=LongTailTokenizerAxisConfig(
            output_path=this_output_path(),
            max_docs_per_slice=max_docs_per_slice,
            max_doc_bytes=max_doc_bytes,
            baseline_tokenizer_name=baseline_tokenizer_name,
            include_planned_raw_slices=include_planned_raw_slices,
            include_o200k_base=include_o200k_base,
            include_byte_reference=include_byte_reference,
        ),
    )


def default_tokenizer_axis_tokenizers(
    *,
    include_o200k_base: bool = False,
    include_byte_reference: bool = False,
) -> tuple[TokenizerAxisTokenizerSpec, ...]:
    tokenizers: list[TokenizerAxisTokenizerSpec] = [
        TokenizerAxisTokenizerSpec(
            name="llama3_1_8b",
            tokenizer_id="meta-llama/Llama-3.1-8B",
            backend=TokenizerAxisBackend.LEVANTER_HF,
        ),
        TokenizerAxisTokenizerSpec(
            name="qwen3_8b",
            tokenizer_id="Qwen/Qwen3-8B",
            backend=TokenizerAxisBackend.LEVANTER_HF,
        ),
        TokenizerAxisTokenizerSpec(
            name="gemma2_9b",
            tokenizer_id="google/gemma-2-9b",
            backend=TokenizerAxisBackend.LEVANTER_HF,
        ),
    ]

    if include_o200k_base:
        tokenizers.append(
            TokenizerAxisTokenizerSpec(
                name="o200k_base",
                tokenizer_id="o200k_base",
                backend=TokenizerAxisBackend.TIKTOKEN,
                required=False,
                notes="Optional; enabled only if tiktoken is available.",
            )
        )

    if include_byte_reference:
        tokenizers.append(
            TokenizerAxisTokenizerSpec(
                name="byt5_small",
                tokenizer_id="google/byt5-small",
                backend=TokenizerAxisBackend.LEVANTER_HF,
                required=False,
                notes="Optional byte-level tokenizer reference.",
            )
        )

    return tuple(tokenizers)


def default_tokenizer_axis_slices(
    *,
    include_planned_raw_slices: bool = True,
) -> tuple[TokenizerAxisSliceSpec, ...]:
    slices: list[TokenizerAxisSliceSpec] = []
    for runnable_slice in RUNNABLE_LONG_TAIL_PPL_SLICES:
        slices.append(
            TokenizerAxisSliceSpec(
                registry_key=runnable_slice.registry_key,
                family=runnable_slice.family,
                source_url=runnable_slice.source_url,
                availability=TokenizerAxisSliceAvailability.RUNNABLE_HF,
                tags=runnable_slice.tags,
                notes=runnable_slice.notes,
                hf_dataset=runnable_slice.hf_dataset,
                text_key=runnable_slice.text_key,
                split=runnable_slice.split,
            )
        )

    if include_planned_raw_slices:
        for registry_key in _PLANNED_SYMBOLIC_SLICE_KEYS:
            raw_slice = LONG_TAIL_PPL_REGISTRY[registry_key]
            slices.append(
                TokenizerAxisSliceSpec(
                    registry_key=raw_slice.registry_key,
                    family=raw_slice.family,
                    source_url=raw_slice.source_url,
                    availability=TokenizerAxisSliceAvailability.PLANNED_RAW_MIRROR,
                    tags=raw_slice.tags,
                    notes=(
                        "Planned raw mirror slice; not runnable in this first pass."
                        f" Expected under raw/long_tail_ppl/{raw_slice.raw_relative_path}."
                    ),
                    raw_relative_path=raw_slice.raw_relative_path,
                )
            )

    return tuple(slices)


def run_long_tail_tokenizer_axis(config: LongTailTokenizerAxisConfig) -> None:
    slices = default_tokenizer_axis_slices(include_planned_raw_slices=config.include_planned_raw_slices)
    runnable_slices = tuple(slice_spec for slice_spec in slices if slice_spec.is_runnable)
    planned_slices = tuple(slice_spec for slice_spec in slices if not slice_spec.is_runnable)

    tokenizers = default_tokenizer_axis_tokenizers(
        include_o200k_base=config.include_o200k_base,
        include_byte_reference=config.include_byte_reference,
    )

    if config.include_o200k_base and not _is_tiktoken_available():
        logger.warning("tiktoken is not installed; o200k_base tokenizer will be skipped.")

    samples = tuple(
        load_tokenizer_axis_slice_sample(
            slice_spec,
            max_docs_per_slice=config.max_docs_per_slice,
            max_doc_bytes=config.max_doc_bytes,
        )
        for slice_spec in runnable_slices
    )

    metrics: list[TokenizerAxisSliceMetrics] = []
    failures: list[TokenizerLoadFailure] = []
    for tokenizer_spec in tokenizers:
        try:
            encode = load_text_encoder(tokenizer_spec)
        except Exception as exc:
            if tokenizer_spec.required:
                raise RuntimeError(
                    f"Required tokenizer failed to load ({tokenizer_spec.name} / {tokenizer_spec.tokenizer_id})."
                ) from exc
            failures.append(
                TokenizerLoadFailure(
                    tokenizer_name=tokenizer_spec.name,
                    tokenizer_id=tokenizer_spec.tokenizer_id,
                    backend=tokenizer_spec.backend,
                    error=str(exc),
                )
            )
            logger.warning("Skipping optional tokenizer %s: %s", tokenizer_spec.name, exc)
            continue

        for sample in samples:
            metrics.append(compute_tokenizer_axis_slice_metrics(sample, tokenizer_spec=tokenizer_spec, encode=encode))

    correlations = compute_whitespace_inflation_correlations(
        metrics=metrics,
        baseline_tokenizer_name=config.baseline_tokenizer_name,
    )
    write_tokenizer_axis_report(
        output_path=config.output_path,
        tokenizers=tokenizers,
        runnable_slices=runnable_slices,
        planned_slices=planned_slices,
        metrics=metrics,
        correlations=correlations,
        load_failures=failures,
    )


def load_tokenizer_axis_slice_sample(
    slice_spec: TokenizerAxisSliceSpec,
    *,
    max_docs_per_slice: int,
    max_doc_bytes: int,
) -> TokenizerAxisSliceSample:
    if not slice_spec.is_runnable or slice_spec.hf_dataset is None:
        raise ValueError(f"Slice {slice_spec.registry_key} is not runnable.")

    dataset = load_dataset(
        slice_spec.hf_dataset.id,
        name=slice_spec.hf_dataset.name,
        split=slice_spec.split,
        streaming=True,
    )

    texts: list[str] = []
    docs_scanned = 0
    docs_skipped_missing_text_key = 0
    docs_skipped_non_string = 0
    for record in dataset:
        if len(texts) >= max_docs_per_slice:
            break
        docs_scanned += 1
        if slice_spec.text_key not in record:
            docs_skipped_missing_text_key += 1
            continue
        value = record[slice_spec.text_key]
        if not isinstance(value, str):
            docs_skipped_non_string += 1
            continue
        if max_doc_bytes > 0:
            value = truncate_to_utf8_bytes(value, max_doc_bytes)
        texts.append(value)

    byte_stats = summarize_slice_byte_stats(texts)
    return TokenizerAxisSliceSample(
        slice_spec=slice_spec,
        texts=tuple(texts),
        docs_scanned=docs_scanned,
        docs_used=len(texts),
        docs_skipped_missing_text_key=docs_skipped_missing_text_key,
        docs_skipped_non_string=docs_skipped_non_string,
        byte_stats=byte_stats,
    )


def compute_tokenizer_axis_slice_metrics(
    sample: TokenizerAxisSliceSample,
    *,
    tokenizer_spec: TokenizerAxisTokenizerSpec,
    encode: Callable[[str], Sequence[int]],
) -> TokenizerAxisSliceMetrics:
    token_count = sum(len(encode(text)) for text in sample.texts)
    byte_count = sample.byte_stats.total_bytes
    tokens_per_byte = token_count / byte_count if byte_count > 0 else None
    bytes_per_token = byte_count / token_count if token_count > 0 else None

    return TokenizerAxisSliceMetrics(
        tokenizer_name=tokenizer_spec.name,
        tokenizer_id=tokenizer_spec.tokenizer_id,
        tokenizer_backend=tokenizer_spec.backend,
        slice_registry_key=sample.slice_spec.registry_key,
        family=sample.slice_spec.family,
        docs_used=sample.docs_used,
        docs_scanned=sample.docs_scanned,
        docs_skipped_missing_text_key=sample.docs_skipped_missing_text_key,
        docs_skipped_non_string=sample.docs_skipped_non_string,
        token_count=token_count,
        byte_count=byte_count,
        tokens_per_byte=tokens_per_byte,
        bytes_per_token=bytes_per_token,
        whitespace_byte_share=sample.byte_stats.to_share(sample.byte_stats.whitespace_bytes),
        punctuation_byte_share=sample.byte_stats.to_share(sample.byte_stats.punctuation_bytes),
        non_ascii_byte_share=sample.byte_stats.to_share(sample.byte_stats.non_ascii_bytes),
    )


def compute_whitespace_inflation_correlations(
    *,
    metrics: Sequence[TokenizerAxisSliceMetrics],
    baseline_tokenizer_name: str,
) -> tuple[TokenizerInflationCorrelation, ...]:
    by_tokenizer: dict[str, dict[str, TokenizerAxisSliceMetrics]] = {}
    for row in metrics:
        by_tokenizer.setdefault(row.tokenizer_name, {})[row.slice_registry_key] = row

    baseline_rows = by_tokenizer.get(baseline_tokenizer_name, {})
    if not baseline_rows:
        return ()

    correlations: list[TokenizerInflationCorrelation] = []
    for tokenizer_name, tokenizer_rows in by_tokenizer.items():
        if tokenizer_name == baseline_tokenizer_name:
            continue

        whitespace_shares: list[float] = []
        deltas: list[float] = []
        for slice_key, baseline_row in baseline_rows.items():
            current_row = tokenizer_rows.get(slice_key)
            if current_row is None:
                continue
            if baseline_row.tokens_per_byte is None or current_row.tokens_per_byte is None:
                continue
            if baseline_row.whitespace_byte_share is None:
                continue
            whitespace_shares.append(baseline_row.whitespace_byte_share)
            deltas.append(current_row.tokens_per_byte - baseline_row.tokens_per_byte)

        correlations.append(
            TokenizerInflationCorrelation(
                tokenizer_name=tokenizer_name,
                baseline_tokenizer_name=baseline_tokenizer_name,
                points=len(deltas),
                mean_tokens_per_byte_delta=(sum(deltas) / len(deltas)) if deltas else None,
                whitespace_vs_tpb_delta_pearson=pearson_correlation(whitespace_shares, deltas),
            )
        )

    correlations.sort(key=lambda row: row.tokenizer_name)
    return tuple(correlations)


def summarize_slice_byte_stats(texts: Iterable[str]) -> SliceByteStats:
    total_bytes = 0
    whitespace_bytes = 0
    punctuation_bytes = 0
    non_ascii_bytes = 0

    for text in texts:
        for char in text:
            char_bytes = len(char.encode("utf-8"))
            total_bytes += char_bytes
            if char.isspace():
                whitespace_bytes += char_bytes
            if unicodedata.category(char).startswith("P"):
                punctuation_bytes += char_bytes
            if not char.isascii():
                non_ascii_bytes += char_bytes

    return SliceByteStats(
        total_bytes=total_bytes,
        whitespace_bytes=whitespace_bytes,
        punctuation_bytes=punctuation_bytes,
        non_ascii_bytes=non_ascii_bytes,
    )


def load_text_encoder(tokenizer_spec: TokenizerAxisTokenizerSpec) -> Callable[[str], Sequence[int]]:
    if tokenizer_spec.backend == TokenizerAxisBackend.LEVANTER_HF:
        tokenizer = load_tokenizer(tokenizer_spec.tokenizer_id)
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    if tokenizer_spec.backend == TokenizerAxisBackend.TIKTOKEN:
        if not _is_tiktoken_available():
            raise RuntimeError("tiktoken is not installed")
        import tiktoken

        encoding = tiktoken.get_encoding(tokenizer_spec.tokenizer_id)
        return encoding.encode_ordinary

    raise ValueError(f"Unsupported tokenizer backend: {tokenizer_spec.backend}")


def truncate_to_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numer / math.sqrt(denom_x * denom_y)


def write_tokenizer_axis_report(
    *,
    output_path: str,
    tokenizers: Sequence[TokenizerAxisTokenizerSpec],
    runnable_slices: Sequence[TokenizerAxisSliceSpec],
    planned_slices: Sequence[TokenizerAxisSliceSpec],
    metrics: Sequence[TokenizerAxisSliceMetrics],
    correlations: Sequence[TokenizerInflationCorrelation],
    load_failures: Sequence[TokenizerLoadFailure],
) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)

    summary = {
        "issue": TOKENIZER_AXIS_ISSUE,
        "parent_issue": TOKENIZER_AXIS_PARENT_ISSUE,
        "analysis_type": "tokenizer_only_first_pass",
        "note": (
            "Perplexity is intentionally excluded because it is coupled to model+tokenizer."
            " This report is designed to be joined later with existing PPL gap outputs."
        ),
        "tokenizers": [tokenizer.to_dict() for tokenizer in tokenizers],
        "runnable_slices": [slice_spec.to_dict() for slice_spec in runnable_slices],
        "planned_non_runnable_slices": [slice_spec.to_dict() for slice_spec in planned_slices],
        "metrics": [row.to_dict() for row in metrics],
        "whitespace_inflation_correlations": [row.to_dict() for row in correlations],
        "tokenizer_load_failures": [failure.to_dict() for failure in load_failures],
    }

    summary_path = os.path.join(output_path, "summary.json")
    with open_url(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    metrics_path = os.path.join(output_path, "slice_tokenizer_metrics.csv")
    with open_url(metrics_path, "w") as f:
        fieldnames = (
            list(metrics[0].to_dict().keys())
            if metrics
            else [
                "tokenizer_name",
                "tokenizer_id",
                "tokenizer_backend",
                "slice_registry_key",
                "family",
                "docs_used",
                "docs_scanned",
                "docs_skipped_missing_text_key",
                "docs_skipped_non_string",
                "token_count",
                "byte_count",
                "tokens_per_byte",
                "bytes_per_token",
                "whitespace_byte_share",
                "punctuation_byte_share",
                "non_ascii_byte_share",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.to_dict())

    report_path = os.path.join(output_path, "report.md")
    with open_url(report_path, "w") as f:
        f.write(render_tokenizer_axis_report_markdown(summary))

    logger.info("Wrote tokenizer-axis report to %s", output_path)


def render_tokenizer_axis_report_markdown(summary: dict) -> str:
    metric_rows = summary.get("metrics", [])
    correlation_rows = summary.get("whitespace_inflation_correlations", [])
    planned_rows = summary.get("planned_non_runnable_slices", [])

    lines = [
        "# Long-tail tokenizer-axis report",
        "",
        f"- issue: #{summary['issue']}",
        "- analysis: tokenizer-only first pass",
        f"- runnable slices: {len(summary.get('runnable_slices', []))}",
        f"- planned/non-runnable slices: {len(planned_rows)}",
        "",
        "## Slice tokenizer metrics",
    ]

    if not metric_rows:
        lines.append("")
        lines.append("No metrics produced.")
    else:
        lines.extend(
            [
                "",
                (
                    "| tokenizer | slice | tokens/byte | bytes/token | whitespace share | "
                    "punctuation share | non-ascii share |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in metric_rows:
            lines.append(
                "| "
                f"{row['tokenizer_name']} | {row['slice_registry_key']} | "
                f"{_fmt_float(row['tokens_per_byte'])} | {_fmt_float(row['bytes_per_token'])} | "
                f"{_fmt_float(row['whitespace_byte_share'])} | {_fmt_float(row['punctuation_byte_share'])} | "
                f"{_fmt_float(row['non_ascii_byte_share'])} |"
            )

    lines.append("")
    lines.append("## Whitespace inflation correlations")
    if not correlation_rows:
        lines.append("")
        lines.append("No correlation rows.")
    else:
        lines.extend(
            [
                "",
                "| tokenizer | baseline | points | mean delta tokens/byte | corr(whitespace_share, delta_tpb) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in correlation_rows:
            lines.append(
                "| "
                f"{row['tokenizer_name']} | {row['baseline_tokenizer_name']} | {row['points']} | "
                f"{_fmt_float(row['mean_tokens_per_byte_delta'])} | "
                f"{_fmt_float(row['whitespace_vs_tpb_delta_pearson'])} |"
            )

    lines.append("")
    lines.append("## Planned slices")
    if not planned_rows:
        lines.append("")
        lines.append("None.")
    else:
        lines.append("")
        for row in planned_rows:
            lines.append(f"- `{row['registry_key']}` ({row['family']}): {row['notes']}")

    lines.append("")
    return "\n".join(lines)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _is_tiktoken_available() -> bool:
    return importlib.util.find_spec("tiktoken") is not None


_PLANNED_SYMBOLIC_SLICE_KEYS = (
    "long_tail_ppl/formal_hardware/smtlib",
    "long_tail_ppl/formal_hardware/dimacs_cnf",
    "long_tail_ppl/formal_hardware/verilogeval",
    "long_tail_ppl/game_music/lichess_pgn",
    "long_tail_ppl/game_music/kernscores_humdrum",
    "long_tail_ppl/game_music/abc_notation",
)
