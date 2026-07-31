# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- diff the ported OCR feature extractor against the vendored original on real PDFs.

DELETE this module and :mod:`experiments.build_pdf_source._finepdfs_reference` once the port is
validated. Nothing in the pipeline imports either.

Synthetic PyMuPDF-built documents already agree on all 124 features across nine structural classes,
including inline images and Form XObjects. What they cannot cover is the real corpus: broken font
encodings, truncated object streams, unusual colorspaces, generator quirks. This reads PDFs the
fetch step already wrote into the marin prefix -- so no Common Crawl egress -- and reports how often
the two implementations disagree.

Page sampling is aligned before each comparison by seeding the global ``random`` that the original
draws from with the same seed the port uses, so both pick the same pages. Padding slots are excluded
from the strict diff because the original fills them from the global ``numpy`` generator and is
genuinely nondeterministic there; ``ocr_prob`` is therefore compared only on documents that filled
all eight slots with real pages.

Run it on the cluster, where the fetch output and the model already live::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name validate-ocr-port \\
        -- python -m experiments.build_pdf_source._validate_ocr_port
"""

import logging
import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import StoragePath, url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import (
    OCR_PROBABILITY_THRESHOLD,
    _document_seed,
    model_step,
)
from experiments.build_pdf_source.common import OcrModelData, PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.ocr_feature_names import DOC_FEATURE_NAMES, FEATURE_NAMES, FEATURE_PAGES
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Enough documents to surface a systematic divergence without reading much of the corpus. Shards are
# ~244 MiB, so this reads a handful of them.
TARGET_DOCUMENTS = 3_000
MAX_SHARDS = 12
_RELATIVE_TOLERANCE = 1e-4

_RESOURCES = ResourceConfig(cpu=4, ram="16g", disk="16g")


class ValidationReport(BaseModel):
    """How often the ported extractor and the vendored original disagreed."""

    version: str = "v1"
    documents: int
    both_readable: int
    port_rejected: int
    reference_failed: int
    feature_mismatch_documents: int
    feature_mismatch_by_feature: dict[str, int]
    full_sample_documents: int
    probability_mismatch: int
    routing_mismatch: int
    max_probability_delta: float


@dataclass
class Tally:
    """Running comparison counts."""

    documents: int = 0
    both_readable: int = 0
    feature_mismatch_documents: int = 0
    feature_mismatch_counts: dict[str, int] = None
    port_rejected: int = 0
    reference_failed: int = 0
    full_sample_documents: int = 0
    probability_mismatch: int = 0
    routing_mismatch: int = 0
    max_probability_delta: float = 0.0

    def __post_init__(self):
        if self.feature_mismatch_counts is None:
            self.feature_mismatch_counts = {}


def _reference_features(pdf: bytes, seed: int) -> dict | None:
    """Run the vendored original, with its page draw forced onto the port's seed."""
    import pymupdf  # noqa: PLC0415

    from experiments.build_pdf_source._finepdfs_reference import (  # noqa: PLC0415
        PDFFeatureExtractor,
        flatten_per_page_features,
    )

    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        random.seed(seed)
        # The vendored original draws from both legacy global generators, so those are what have to
        # be seeded to align its page sample with the port's.
        np.random.seed(seed % (2**32))  # noqa: NPY002
        chunks = PDFFeatureExtractor(num_chunks=1, num_pages_to_sample=FEATURE_PAGES).extract_features(doc)
        if not chunks:
            return None
        return flatten_per_page_features(chunks[0], sample_to_k_page_features=FEATURE_PAGES)
    finally:
        doc.close()


def _compare_document(pdf: bytes, seed: int, booster: "xgboost.Booster", tally: Tally) -> None:  # noqa: F821
    import pymupdf  # noqa: PLC0415

    from experiments.build_pdf_source.ocr_features import CorruptPdf, document_features  # noqa: PLC0415

    tally.documents += 1

    try:
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            ported = document_features(doc, seed=seed)
    except CorruptPdf:
        tally.port_rejected += 1
        return
    except Exception:
        tally.port_rejected += 1
        logger.debug("port could not read a document", exc_info=True)
        return

    try:
        reference = _reference_features(pdf, seed)
    except Exception:
        tally.reference_failed += 1
        logger.debug("reference could not read a document", exc_info=True)
        return
    if reference is None:
        tally.reference_failed += 1
        return

    tally.both_readable += 1
    ported_values = dict(zip(FEATURE_NAMES, ported.vector().tolist(), strict=True))

    # Compare the document features and only the page slots backed by a real page; the original's
    # padding is drawn from a generator we cannot align.
    real_slots = ported.num_pages_successfully_sampled
    comparable = [
        name for name in FEATURE_NAMES if name in DOC_FEATURE_NAMES or int(name.rpartition("_page")[2]) <= real_slots
    ]

    mismatched = []
    for name in comparable:
        expected = float(reference[name])
        actual = float(ported_values[name])
        if abs(expected - actual) > _RELATIVE_TOLERANCE * max(1.0, abs(expected)):
            mismatched.append(name)
            base = name.rpartition("_page")[0] or name
            tally.feature_mismatch_counts[base] = tally.feature_mismatch_counts.get(base, 0) + 1
    if mismatched:
        tally.feature_mismatch_documents += 1

    if real_slots < FEATURE_PAGES:
        return

    # With every slot real, both implementations see identical inputs, so probabilities must match.
    tally.full_sample_documents += 1
    reference_row = np.array([[float(reference[name]) for name in FEATURE_NAMES]], dtype=np.float32)
    reference_probability = float(booster.inplace_predict(reference_row, validate_features=False)[0])
    ported_probability = float(booster.inplace_predict(ported.vector().reshape(1, -1), validate_features=False)[0])

    delta = abs(reference_probability - ported_probability)
    tally.max_probability_delta = max(tally.max_probability_delta, delta)
    if delta > 1e-6:
        tally.probability_mismatch += 1

    reference_route = reference_probability >= OCR_PROBABILITY_THRESHOLD or float(reference["garbled_text_ratio"]) > 0.0
    ported_route = ported_probability >= OCR_PROBABILITY_THRESHOLD or ported.garbled_text_ratio > 0.0
    if reference_route != ported_route:
        tally.routing_mismatch += 1


def validate(output_path: str, source_output_path: str, model_output_path: str) -> ValidationReport:
    """Compare both extractors over real fetched PDFs and log the agreement report."""
    import pymupdf  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    model = read_artifact(model_output_path, OcrModelData)

    booster = xgb.Booster()
    booster.load_model(bytearray(StoragePath(model.model_path).read_bytes()))
    booster.set_param({"nthread": 1})
    pymupdf.TOOLS.mupdf_display_errors(False)

    filesystem, path = url_to_fs(source.main_output_dir)
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:MAX_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetch output under {source.main_output_dir}")
    logger.info("Comparing against %d of the fetch step's shards", len(shards))

    tally = Tally()
    for shard in shards:
        if tally.documents >= TARGET_DOCUMENTS:
            break
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=["pdf", "warc_filename", "warc_record_offset", "content_digest"])
        for row in table.to_pylist():
            if tally.documents >= TARGET_DOCUMENTS:
                break
            seed = _document_seed(row["content_digest"], row["warc_filename"], row["warc_record_offset"])
            _compare_document(row["pdf"], seed, booster, tally)
        logger.info(
            "After %s: %d documents, %d feature mismatches",
            shard.rsplit("/", 1)[-1],
            tally.documents,
            tally.feature_mismatch_documents,
        )

    report = ValidationReport(
        documents=tally.documents,
        both_readable=tally.both_readable,
        port_rejected=tally.port_rejected,
        reference_failed=tally.reference_failed,
        feature_mismatch_documents=tally.feature_mismatch_documents,
        feature_mismatch_by_feature=dict(sorted(tally.feature_mismatch_counts.items())),
        full_sample_documents=tally.full_sample_documents,
        probability_mismatch=tally.probability_mismatch,
        routing_mismatch=tally.routing_mismatch,
        max_probability_delta=tally.max_probability_delta,
    )
    logger.info("=== OCR PORT VALIDATION REPORT ===")
    for key, value in report.model_dump().items():
        logger.info("  %s: %s", key, value)
    agreed = tally.both_readable - tally.feature_mismatch_documents
    if tally.both_readable:
        logger.info(
            "  feature agreement: %d/%d (%.4f%%)",
            agreed,
            tally.both_readable,
            100.0 * agreed / tally.both_readable,
        )
    return report


def validation_step(source_output_path: str, model: StepSpec) -> StepSpec:
    """Build the comparison step.

    The fetch step is deliberately *not* a dependency. It is named only by its resolved output path,
    so a missing or half-finished fetch makes :func:`validate` fail on ``read_artifact`` instead of
    making ``StepRunner`` helpfully launch a 411 GiB download.
    """
    return StepSpec(
        name="data/datakit/validate/finepdfs_ocr_port",
        deps=[model],
        hash_attrs={
            "source_output_path": source_output_path,
            "target_documents": TARGET_DOCUMENTS,
            "max_shards": MAX_SHARDS,
            "attempt": 1,
        },
        fn=remote(
            partial(
                validate,
                source_output_path=source_output_path,
                model_output_path=model.output_path,
            ),
            resources=_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    # fetch_step is constructed only to resolve its content-addressed output path; it is not run.
    fetch = fetch_step(plan_step())
    model = model_step()
    logger.info("Reading fetched PDFs from %s", fetch.output_path)
    StepRunner().run([model, validation_step(fetch.output_path, model)])


if __name__ == "__main__":
    main()
