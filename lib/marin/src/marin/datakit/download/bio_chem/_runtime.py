# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared streaming runtime for bio/chem notation slices.

A "slice" is one (source URL, format, sampling cap) tuple. It produces a single
parquet file with ``{id, text, source}`` rows where ``text`` is one packed
document containing one or more original records preserved verbatim.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import posixpath
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rigging.filesystem import open_url
from zephyr import Dataset, ZephyrContext

from marin.execution.executor import THIS_OUTPUT_PATH
from marin.execution.step_spec import StepSpec
from marin.transform.bio_chem.splitters import (
    SamplingCap,
    iter_fasta_records,
    iter_gff_blocks,
    iter_mmcif_blocks,
    iter_sdf_records,
    iter_smiles_records,
    iter_uniprot_dat_records,
    pack_records_into_docs,
    take_until_cap,
)

logger = logging.getLogger(__name__)


class NotationFormat(enum.StrEnum):
    """The record splitter to apply to a streamed source."""

    FASTA = enum.auto()
    GFF = enum.auto()
    SMILES = enum.auto()
    SDF = enum.auto()
    MMCIF = enum.auto()
    UNIPROT_DAT = enum.auto()


@dataclass(frozen=True)
class PackingConfig:
    """How to bundle short records into one document for ICL evaluation."""

    target_doc_chars: int = 8192
    max_records_per_doc: int = 64
    record_separator: str = ""


# Default packing per format. Self-delimiting formats use no separator;
# line-oriented ones use a newline so consecutive records remain readable.
_DEFAULT_PACKING: dict[NotationFormat, PackingConfig] = {
    NotationFormat.FASTA: PackingConfig(target_doc_chars=8192, max_records_per_doc=32),
    NotationFormat.GFF: PackingConfig(target_doc_chars=8192, max_records_per_doc=8),
    NotationFormat.SMILES: PackingConfig(target_doc_chars=4096, max_records_per_doc=128, record_separator="\n"),
    NotationFormat.SDF: PackingConfig(target_doc_chars=16384, max_records_per_doc=4),
    NotationFormat.MMCIF: PackingConfig(target_doc_chars=32768, max_records_per_doc=1),
    NotationFormat.UNIPROT_DAT: PackingConfig(target_doc_chars=8192, max_records_per_doc=8),
}


@dataclass(frozen=True)
class NotationSliceSpec:
    """One streamed slice. Multiple URLs are concatenated into one logical stream
    (useful for sources like RCSB where each entry is its own small file)."""

    name: str
    """Short slot name used as the output filename stem (no extension)."""

    urls: tuple[str, ...]
    """One or more source URLs read in order. ``http(s)``, ``ftp``, ``hf://``
    and ``gs://`` all work via fsspec."""

    fmt: NotationFormat

    source_label: str
    """String written into the ``source`` column of the parquet output."""

    compression: str | None = "infer"
    """``"infer"`` lets fsspec auto-detect ``.gz``; pass ``None`` to disable."""

    encoding: str = "utf-8"
    """Text decoding for the source. PDB/mmCIF and most bio formats are ASCII."""

    sampling: SamplingCap = field(default_factory=SamplingCap)

    packing: PackingConfig | None = None
    """If ``None``, use the per-format default from ``_DEFAULT_PACKING``."""

    skip_header_lines: int = 0
    """Number of leading lines to drop before splitting (e.g. CSV headers).
    Applied per URL."""


@dataclass
class BioChemSliceConfig:
    """Top-level runtime config for one materialization step."""

    output_path: str = THIS_OUTPUT_PATH
    slices: tuple[NotationSliceSpec, ...] = ()


def _stream_lines(spec: NotationSliceSpec) -> Iterator[str]:
    """Iterate text lines across the spec's URLs without buffering whole files."""
    open_kwargs: dict[str, Any] = {"mode": "rt", "encoding": spec.encoding}
    if spec.compression is not None:
        open_kwargs["compression"] = spec.compression
    for url in spec.urls:
        with open_url(url, **open_kwargs) as src:
            for _ in range(spec.skip_header_lines):
                line = src.readline()
                if not line:
                    break
            yield from src


def _records_for(spec: NotationSliceSpec) -> Iterator[str]:
    lines = _stream_lines(spec)
    if spec.fmt is NotationFormat.FASTA:
        return iter_fasta_records(lines)
    if spec.fmt is NotationFormat.GFF:
        return iter_gff_blocks(lines)
    if spec.fmt is NotationFormat.SMILES:
        return iter_smiles_records(lines)
    if spec.fmt is NotationFormat.SDF:
        return iter_sdf_records(lines)
    if spec.fmt is NotationFormat.MMCIF:
        return iter_mmcif_blocks(lines)
    if spec.fmt is NotationFormat.UNIPROT_DAT:
        return iter_uniprot_dat_records(lines)
    raise ValueError(f"Unhandled notation format: {spec.fmt!r}")


def _doc_id(slice_name: str, doc_text: str, index: int) -> str:
    digest = hashlib.sha1(doc_text.encode("utf-8")).hexdigest()[:16]
    return f"{slice_name}-{index:06d}-{digest}"


def _docs_for_slice(spec: NotationSliceSpec) -> Iterator[dict[str, str]]:
    packing = spec.packing or _DEFAULT_PACKING[spec.fmt]
    raw_records = take_until_cap(_records_for(spec), spec.sampling)
    docs = pack_records_into_docs(
        raw_records,
        target_doc_chars=packing.target_doc_chars,
        max_records_per_doc=packing.max_records_per_doc,
        record_separator=packing.record_separator,
    )
    for index, text in enumerate(docs):
        if not text:
            continue
        yield {"id": _doc_id(spec.name, text, index), "text": text, "source": spec.source_label}


def run_notation_slice(spec: NotationSliceSpec, output_dir: str) -> dict[str, Any]:
    """Stream one slice and write it as ``<output_dir>/<spec.name>.parquet``.

    Returns a small summary so callers can write a manifest.
    """
    output_pattern = posixpath.join(output_dir, f"{spec.name}-{{shard:05d}}-of-{{total:05d}}.parquet")
    pipeline = Dataset.from_iterable(_docs_for_slice(spec)).write_parquet(output_pattern)
    ctx = ZephyrContext(name=f"bio-chem-slice-{spec.name}")
    results = ctx.execute(pipeline).results
    return {
        "name": spec.name,
        "source_label": spec.source_label,
        "format": spec.fmt.value,
        "urls": list(spec.urls),
        "files": list(results),
    }


def run_bio_chem_slices(cfg: BioChemSliceConfig) -> dict[str, Any]:
    """Run every slice in ``cfg`` sequentially.

    Each slice is small enough to stream on a single worker; running them in
    a loop keeps memory and HTTP concurrency low. If you need parallelism
    across slices, run multiple StepSpecs.
    """
    summaries: list[dict[str, Any]] = []
    for spec in cfg.slices:
        logger.info("Streaming bio/chem slice %s from %d url(s)", spec.name, len(spec.urls))
        summaries.append(run_notation_slice(spec, str(cfg.output_path)))
    return {"slices": summaries}


def bio_chem_slice_step(
    *,
    name: str,
    slices: tuple[NotationSliceSpec, ...],
) -> StepSpec:
    """Build a StepSpec that streams all ``slices`` into one output dir.

    The step writes one parquet shard per slice (named ``<spec.name>-*.parquet``)
    so downstream code can glob ``<step>/<spec.name>-*.parquet`` per slice.
    """
    return StepSpec(
        name=name,
        fn=lambda output_path: run_bio_chem_slices(BioChemSliceConfig(output_path=output_path, slices=slices)),
        hash_attrs={
            "slice_names": [slice_spec.name for slice_spec in slices],
            "slice_specs": [
                {
                    "name": slice_spec.name,
                    "urls": list(slice_spec.urls),
                    "fmt": slice_spec.fmt.value,
                    "source_label": slice_spec.source_label,
                    "compression": slice_spec.compression,
                    "encoding": slice_spec.encoding,
                    "sampling": {
                        "max_records": slice_spec.sampling.max_records,
                        "max_bytes": slice_spec.sampling.max_bytes,
                    },
                    "packing": (
                        None
                        if slice_spec.packing is None
                        else {
                            "target_doc_chars": slice_spec.packing.target_doc_chars,
                            "max_records_per_doc": slice_spec.packing.max_records_per_doc,
                            "record_separator": slice_spec.packing.record_separator,
                        }
                    ),
                    "skip_header_lines": slice_spec.skip_header_lines,
                }
                for slice_spec in slices
            ],
        },
    )
