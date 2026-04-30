# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 wiring for binary / network / security artifact PPL slices (issue #5057).

This module is the bridge between:

* The renderers in :mod:`marin.transform.security_artifacts` (deterministic
  text formatters for hex dumps and Zeek TSV logs).
* The raw-text evaluation entry point
  :func:`marin.evaluation.perplexity_gap.raw_text_dataset`.

Phase 1 lands the shared pipeline and the cheapest corpus — a Zeek
connection-log renderer that reads records from a HuggingFace-hosted dataset,
re-serializes them as canonical Zeek TSV log blocks, and exposes the result
as a raw-text eval slice.

Phase 2 (separate PR) introduces ``tshark`` / ``objdump`` renderers behind an
Iris Docker image. Phase 3 (separate PR) extends the gap report with
regex-bucket tagging.
"""

from __future__ import annotations

from collections.abc import Iterable

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, this_output_path
from marin.transform.security_artifacts.zeek_to_dolma import (
    ZeekToDolmaConfig,
    convert_zeek_to_dolma,
)

from experiments.defaults import default_download

# Canonical Zeek conn.log field order as documented at
# https://docs.zeek.org/en/master/logs/conn.html. Kept as a tuple so it can be
# passed directly to a frozen dataclass.
ZEEK_CONN_FIELDS: tuple[str, ...] = (
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
)

# Matching Zeek types per conn.log field.
ZEEK_CONN_TYPES: tuple[str, ...] = (
    "time",
    "string",
    "addr",
    "port",
    "addr",
    "port",
    "enum",
    "string",
    "interval",
    "count",
    "count",
    "string",
    "bool",
    "bool",
    "count",
    "string",
    "count",
    "count",
    "count",
    "count",
    "set[string]",
)

# Slice-size guardrails per the issue ("small and region-local"; eval-only).
# With records_per_block=64 and max_blocks_per_file=64 we get at most ~4k
# records per input shard, which keeps each rendered slice comfortably under
# 10-20 MB even before compression.
DEFAULT_ZEEK_RECORDS_PER_BLOCK = 64
DEFAULT_ZEEK_MAX_BLOCKS_PER_FILE = 64


def zeek_conn_transform_step(
    *,
    name: str,
    raw_download_step: ExecutorStep,
    source_label: str,
    input_format: str = "parquet",
    fields: Iterable[str] = ZEEK_CONN_FIELDS,
    types: Iterable[str] | None = ZEEK_CONN_TYPES,
    records_per_block: int = DEFAULT_ZEEK_RECORDS_PER_BLOCK,
    max_blocks_per_file: int | None = DEFAULT_ZEEK_MAX_BLOCKS_PER_FILE,
    input_glob: str | None = None,
) -> ExecutorStep:
    """Build an ExecutorStep that renders Zeek records to Dolma-format JSONL.

    Args:
        name: Executor step name (becomes the output prefix).
        raw_download_step: An upstream step whose output directory contains
            the Zeek-record files (parquet or JSONL) to render.
        source_label: The ``source`` tag written into each Dolma record.
        input_format: ``"parquet"`` or ``"jsonl"``.
        fields: Zeek field names in canonical order.
        types: Optional matching Zeek types; pass ``None`` to skip the
            ``#types`` header.
        records_per_block: Number of Zeek records grouped under one
            ``#fields`` header per emitted Dolma record.
        max_blocks_per_file: Cap on the number of blocks emitted per input
            file. Keeps eval slices small per the issue's size budget.
        input_glob: Optional override for the input file glob.
    """
    return ExecutorStep(
        name=name,
        fn=convert_zeek_to_dolma,
        config=ZeekToDolmaConfig(
            input_path=raw_download_step,
            output_path=this_output_path(),
            zeek_path="conn",
            fields=tuple(fields),
            source_label=source_label,
            input_format=input_format,
            input_glob=input_glob,
            types=tuple(types) if types is not None else None,
            records_per_block=records_per_block,
            max_blocks_per_file=max_blocks_per_file,
        ),
    )


def zeek_conn_raw_text_slice(
    *,
    name: str,
    hf_dataset_id: str,
    revision: str,
    source_label: str | None = None,
    input_format: str = "parquet",
    input_glob: str | None = None,
    records_per_block: int = DEFAULT_ZEEK_RECORDS_PER_BLOCK,
    max_blocks_per_file: int | None = DEFAULT_ZEEK_MAX_BLOCKS_PER_FILE,
    tags: tuple[str, ...] = (),
) -> tuple[ExecutorStep, RawTextEvaluationDataset]:
    """Compose download → Zeek-to-Dolma → ``raw_text_dataset`` for one slice.

    Returns a pair ``(transform_step, raw_text_dataset)`` where
    ``transform_step`` is the ExecutorStep that produces the rendered slice
    and the dataset is ready to slot into
    :func:`marin.evaluation.perplexity_gap.default_model_perplexity_gap`.
    """
    download_step = default_download(
        name=f"raw/security_artifacts/{name}",
        hf_dataset_id=hf_dataset_id,
        revision=revision,
    )

    resolved_source = source_label or f"{hf_dataset_id}@{revision}"
    transform_step = zeek_conn_transform_step(
        name=f"documents/security_artifacts/{name}",
        raw_download_step=download_step,
        source_label=resolved_source,
        input_format=input_format,
        input_glob=input_glob,
        records_per_block=records_per_block,
        max_blocks_per_file=max_blocks_per_file,
    )
    dataset = raw_text_dataset(
        transform_step,
        tags=("security_artifacts", "zeek-tsv", *tags),
    )
    return transform_step, dataset


__all__ = [
    "DEFAULT_ZEEK_MAX_BLOCKS_PER_FILE",
    "DEFAULT_ZEEK_RECORDS_PER_BLOCK",
    "ZEEK_CONN_FIELDS",
    "ZEEK_CONN_TYPES",
    "zeek_conn_raw_text_slice",
    "zeek_conn_transform_step",
]
