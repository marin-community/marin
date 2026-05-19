# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download OEIS integer sequences for eval-only perplexity probes."""

from __future__ import annotations

import hashlib
import logging
import posixpath
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from rigging.filesystem import open_url
from zephyr import Dataset, ZephyrContext

from marin.execution.executor import THIS_OUTPUT_PATH
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

OEIS_STRIPPED_URL = "https://oeis.org/stripped.gz"
OEIS_NAMES_URL = "https://oeis.org/names.gz"
DEFAULT_MAX_SEQUENCES = 50_000
DEFAULT_RECORDS_PER_DOC = 16


@dataclass(frozen=True)
class OeisEvalConfig:
    """Configuration for the OEIS eval-only materialization step."""

    output_path: str = THIS_OUTPUT_PATH
    stripped_url: str = OEIS_STRIPPED_URL
    names_url: str = OEIS_NAMES_URL
    max_sequences: int = DEFAULT_MAX_SEQUENCES
    records_per_doc: int = DEFAULT_RECORDS_PER_DOC


def _iter_name_rows(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sequence_id, _, name = line.partition(" ")
        if not name:
            continue
        yield sequence_id, name


def _iter_sequence_rows(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sequence_id, _, terms = line.partition(" ")
        terms = terms.strip().strip(",")
        if not terms:
            continue
        yield sequence_id, terms


def _format_record(sequence_id: str, name: str, terms: str) -> str:
    return f"OEIS ID: {sequence_id}\nName: {name}\nTerms:\n{terms}"


def _doc_id(records: list[str], index: int) -> str:
    digest = hashlib.sha1("\n".join(records).encode("utf-8")).hexdigest()[:16]
    return f"oeis-{index:06d}-{digest}"


def _pack_records(records: Iterable[str], *, records_per_doc: int) -> Iterator[dict[str, str]]:
    if records_per_doc <= 0:
        raise ValueError("records_per_doc must be positive.")

    batch: list[str] = []
    doc_index = 0
    for record in records:
        batch.append(record)
        if len(batch) < records_per_doc:
            continue
        yield {"id": _doc_id(batch, index=doc_index), "text": "\n\n---\n\n".join(batch), "source": "oeis"}
        doc_index += 1
        batch = []

    if batch:
        yield {"id": _doc_id(batch, index=doc_index), "text": "\n\n---\n\n".join(batch), "source": "oeis"}


def _open_gzip_text(url: str) -> Iterator[str]:
    with open_url(url, mode="rt", compression="gzip", encoding="utf-8") as src:
        yield from src


def _iter_oeis_records(cfg: OeisEvalConfig) -> Iterator[str]:
    names = dict(_iter_name_rows(_open_gzip_text(cfg.names_url)))
    for index, (sequence_id, terms) in enumerate(_iter_sequence_rows(_open_gzip_text(cfg.stripped_url))):
        if index >= cfg.max_sequences:
            break
        yield _format_record(sequence_id, names.get(sequence_id, ""), terms)


def download_oeis_eval(cfg: OeisEvalConfig) -> dict[str, object]:
    """Stream OEIS bulk dumps into a Dolma-style Parquet eval shard."""

    logger.info("Streaming up to %d OEIS sequences from %s", cfg.max_sequences, cfg.stripped_url)
    output_pattern = posixpath.join(str(cfg.output_path), "oeis_integer_sequences-{shard:05d}-of-{total:05d}.parquet")
    pipeline = Dataset.from_iterable(_pack_records(_iter_oeis_records(cfg), records_per_doc=cfg.records_per_doc))
    results = ZephyrContext(name="download-oeis-eval").execute(pipeline.write_parquet(output_pattern)).results
    return {
        "stripped_url": cfg.stripped_url,
        "names_url": cfg.names_url,
        "max_sequences": cfg.max_sequences,
        "records_per_doc": cfg.records_per_doc,
        "files": list(results),
    }


def oeis_eval_step(
    *,
    max_sequences: int = DEFAULT_MAX_SEQUENCES,
    records_per_doc: int = DEFAULT_RECORDS_PER_DOC,
    stripped_url: str = OEIS_STRIPPED_URL,
    names_url: str = OEIS_NAMES_URL,
) -> StepSpec:
    """Create the eval-only OEIS download/materialization step."""

    def _run(output_path: str) -> dict[str, object]:
        return download_oeis_eval(
            OeisEvalConfig(
                output_path=output_path,
                stripped_url=stripped_url,
                names_url=names_url,
                max_sequences=max_sequences,
                records_per_doc=records_per_doc,
            )
        )

    return StepSpec(
        name="raw/oeis/integer_sequences_eval",
        fn=_run,
        hash_attrs={
            "stripped_url": stripped_url,
            "names_url": names_url,
            "max_sequences": max_sequences,
            "records_per_doc": records_per_doc,
        },
    )
