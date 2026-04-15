# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Consolidate takes a set of documents with corresponding attributes and writes
out a subset of the documents based on various filters defined with respect to
the attributes.  Handles two cases:
- Span removal produces attributes (e.g., duplicate_text spans). Remove text spans.
- Document removal via attribute produced by deduplication.

Joins documents with their attribute files via Zephyr's ``sorted_merge_join``:
the datakit convention guarantees that attribute files share the input file
partitioning (1:1 file pairing, sorted by id), so each shard pairs with its
corresponding attribute shard without a shuffle. Multiple filters are chained
as successive left joins.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from fray import ResourceConfig
from marin.utils import (
    fsspec_exists,
    fsspec_glob,
    rebase_file_path,
)
from zephyr import Dataset, ZephyrContext, ZephyrExecutionResult


class FilterType(StrEnum):
    REMOVE_SPANS = "remove_spans"
    REMOVE_DOC = "remove_docs"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterConfig:
    """Config for filtering operation on Marin data"""

    type: FilterType
    """The type of filter to apply."""

    attribute_path: str
    """Base path where the files with the attributes are stored."""

    name: str
    """Name of attribute to use for filtering."""

    attribute_filetype: str | None = None
    """File extension for attribute files (e.g. 'jsonl.gz', 'vortex'). If None, uses the input filetype."""

    keep_if_missing: bool = False
    """If True, keep docs that have no attribute entry. If False (default), reject them."""


def _remove_spans_from_doc(doc: dict, filt: FilterConfig, attributes: dict) -> dict:
    def _remove_spans(text: str, spans: list[list[int]]) -> str:
        """Return ``text`` with ``spans`` removed.

        Example: text = "hello", spans = [[1, 4]], returns "ho"
        """
        # Sort spans in reverse order to avoid index shifting
        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
        for span in sorted_spans:
            start, end = span[0], span[1]
            text = text[:start] + text[end:]

        return text

    spans = attributes[filt.name]
    new_text = _remove_spans(doc["text"], spans)
    return {**doc, "text": new_text}


def _resolve_attribute_path(input_base: str, input_path: str, filt: FilterConfig, filetype: str) -> str | None:
    """Map an input file path to its attribute file path, with glob fallback for compression suffixes."""
    new_extension = f".{filt.attribute_filetype}" if filt.attribute_filetype else f".{filetype}"
    attr_path = rebase_file_path(
        input_base,
        input_path,
        filt.attribute_path,
        new_extension=new_extension,
        old_extension=f".{filetype}",
    )
    if fsspec_exists(attr_path):
        return attr_path
    candidates = fsspec_glob(f"{attr_path}.*")
    if candidates:
        return candidates[0]
    return None


def _attribute_paths_for_filter(input_base: str, input_paths: list[str], filt: FilterConfig, filetype: str) -> list[str]:
    """Resolve the 1:1 input→attribute paths for a filter.

    Raises if any shard's attribute file is missing — the datakit invariant is
    that all attribute files exist. ``keep_if_missing`` governs missing *rows*
    within a file, not missing files.
    """
    resolved = []
    for inp in input_paths:
        path = _resolve_attribute_path(input_base, inp, filt, filetype)
        if path is None:
            raise FileNotFoundError(
                f"No attribute file for filter '{filt.name}' corresponding to input {inp} "
                f"under {filt.attribute_path}"
            )
        resolved.append(path)
    return resolved


def _make_filter_combiner(filt: FilterConfig) -> Callable[[dict, dict | None], dict | None]:
    """Build a combiner for one filter.

    Called by ``sorted_merge_join`` with the current doc (``left``) and the
    matching attribute row or ``None`` (``right``). Returns the doc (possibly
    with mutated text for ``REMOVE_SPANS``) or ``None`` to drop it.
    """

    def combine(left: dict, right: dict | None) -> dict | None:
        if right is None:
            return left if filt.keep_if_missing else None

        attrs = right["attributes"]
        if filt.type == FilterType.REMOVE_DOC:
            return left if not attrs.get(filt.name, False) else None
        assert filt.type == FilterType.REMOVE_SPANS
        mutated = _remove_spans_from_doc(left, filt, attrs)
        return mutated if mutated.get("text") else None

    return combine


def consolidate(
    *,
    input_path: str,
    output_path: str,
    filters: list[FilterConfig],
    filetype: str = "jsonl.gz",
    worker_resources: ResourceConfig | None = None,
) -> ZephyrExecutionResult:
    """Consolidate documents by applying filters based on attributes.

    Joins each input file with its (co-partitioned, sorted) attribute files via
    chained ``sorted_merge_join`` ops — one left join per filter, with the
    filter's keep/mutate/drop logic encoded in its combiner.

    Args:
        input_path: Directory (recursively) containing input documents.
        output_path: Destination directory for filtered Parquet output.
        filters: List of filters to apply (see :class:`FilterConfig`).
        filetype: Extension of the input documents (default: ``"jsonl.gz"``).
        worker_resources: Optional Zephyr worker resource config (defaults to Zephyr defaults).
    """
    input_paths = sorted(fsspec_glob(os.path.join(input_path, f"**/*.{filetype}")))
    if not input_paths:
        raise ValueError(f"No input files matched {input_path}/**/*.{filetype}")
    logger.info(f"Consolidating {len(input_paths)} document files via {len(filters)} filters")

    # Resolve attribute paths up front so the plan can be built before execution.
    filter_attr_paths = [
        (filt, _attribute_paths_for_filter(input_path, input_paths, filt, filetype)) for filt in filters
    ]

    ds = Dataset.from_list(input_paths).load_parquet()
    for filt, attr_paths in filter_attr_paths:
        attrs = Dataset.from_list(attr_paths).load_parquet(columns=["id", "attributes"])
        ds = ds.sorted_merge_join(
            attrs,
            left_key=lambda r: r["id"],
            right_key=lambda r: r["id"],
            combiner=_make_filter_combiner(filt),
            how="left",
        )
        # Drop rejected docs before the next join so its key extractor never sees None.
        ds = ds.filter(lambda r: r is not None)

    ctx = ZephyrContext(
        name="consolidate-filter",
        **({"resources": worker_resources} if worker_resources else {}),
    )
    return ctx.execute(ds.write_parquet(f"{output_path}/part-{{shard:05d}}-of-{{total:05d}}.parquet"))
