# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3: route each fetched PDF to OCR or to text extraction, with the FinePDFs XGBoost model.

Two steps. :func:`model_step` stages the trained booster into the marin prefix once, pinned by
content hash so a run's predictions are attributable to an exact set of weights. :func:`classify_step`
maps over the fetch step's Parquet shards and scores every PDF.

The output is a routing table, not a copy of the corpus: one narrow row per PDF keyed by
``(warc_filename, warc_record_offset)``, which is unique where ``content_digest`` is not (the crawl
holds ~9.8% exact-duplicate PDFs). At ~30 MB for a 10% sample it fits in a task's memory, so
extraction (#7618) can broadcast it and join in a map, rather than paying to have 411 GiB of PDF
bytes copied forward through this step.

Inference is batched. The upstream implementation built a one-row ``pandas.DataFrame`` per document
and called ``predict_proba`` on it; here a Parquet row group's documents are scored in one
:meth:`xgboost.Booster.inplace_predict` over a float32 matrix, which also drops pandas and the
scikit-learn ``XGBClassifier`` wrapper from the hot path. The per-document cost that remains is
PyMuPDF parsing -- see :mod:`experiments.build_pdf_source.ocr_features` for what was removed there.

PyMuPDF, XGBoost and :mod:`~experiments.build_pdf_source.ocr_features` are imported inside the
functions that use them, not at module scope. They live in marin-core's ``datakit`` extra, which the
workers get via ``pip_dependency_groups`` but the entrypoint job does not: its ``uv sync`` carries no
extras. Since :mod:`~experiments.build_pdf_source.pipeline` imports this module to build its steps,
a module-scope ``import pymupdf`` here kills the driver before it submits anything. This mirrors how
:mod:`~experiments.build_pdf_source.fetch` defers ``warcio``. The booster's feature contract lives in
:mod:`~experiments.build_pdf_source.ocr_feature_names`, which is pure data and always importable.
"""

import hashlib
import logging
from collections.abc import Iterator
from functools import cache, partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source.common import (
    DOWNLOAD_CHUNK_BYTES,
    REQUEST_TIMEOUT,
    USER_AGENT,
    OcrModelData,
    PdfClassificationData,
    PdfSourceData,
    session,
)
from experiments.build_pdf_source.ocr_feature_names import FEATURE_NAMES

logger = logging.getLogger(__name__)

# The trained OCR router from https://github.com/huggingface/finepdfs, pinned to the commit the
# weights were read from. GitHub serves Git-LFS content from the ``media.`` host; the plain
# ``raw.`` host returns the 131-byte pointer file instead.
OCR_MODEL_REPO = "huggingface/finepdfs"
OCR_MODEL_REVISION = "15aa381fae828c8acd6efa058fa0b41d2c602ede"
OCR_MODEL_PATH_IN_REPO = "models/xgb_ocr_classifier/xgb_classifier.ubj"
OCR_MODEL_SHA256 = "b44c818261a246de84c35d3ba80b6e8d5f17f1ce376adb25922a80c9a2415227"
OCR_MODEL_FILENAME = "xgb_classifier.ubj"

# FinePDFs' routing rule. A document goes to OCR when the model is at least this confident, or when
# any sampled page produced a replacement character -- garbled text means the embedded text layer
# is unusable however confident the model is that one exists. See the upstream training notebook
# (``models/model_prep_code/ocr_xgb_classifier_train``) for the threshold's derivation.
OCR_PROBABILITY_THRESHOLD = 0.2

_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]
# What an extraction step reads back out of the routing table -- see :func:`routing_keys`.
_ROUTING_COLUMNS = ["warc_filename", "warc_record_offset", "needs_ocr"]

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_record_offset", pa.int64(), nullable=False),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("needs_ocr", pa.bool_(), nullable=True),
        pa.field("ocr_prob", pa.float32(), nullable=True),
        pa.field("num_pages", pa.int32(), nullable=True),
        pa.field("num_pages_sampled", pa.int32(), nullable=True),
        pa.field("garbled_text_ratio", pa.float32(), nullable=True),
        pa.field("is_form", pa.bool_(), nullable=True),
        pa.field("is_scanner_produced", pa.bool_(), nullable=True),
        pa.field("pdf_bytes", pa.int64(), nullable=False),
        # Null unless the PDF could not be read; the routing columns are null when it is set.
        pa.field("classification_error", pa.string(), nullable=True),
    ]
)

_MODEL_RESOURCES = ResourceConfig(cpu=1, ram="2g", disk="2g")
_DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
# PyMuPDF parsing is single-threaded and CPU-bound, so tasks are costed at one CPU and multiplex
# eight-deep per worker. Task disk is unused -- shards stream from object storage.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="6g", disk="2g")
_MAX_WORKERS = 28
_HEARTBEAT_TIMEOUT = 15 * 60


def ocr_model_url() -> str:
    """Return the LFS media URL for the pinned booster."""
    return f"https://media.githubusercontent.com/media/{OCR_MODEL_REPO}/{OCR_MODEL_REVISION}/{OCR_MODEL_PATH_IN_REPO}"


def stage_ocr_model(output_path: str) -> OcrModelData:
    """Copy the pinned booster into the marin prefix, refusing anything that fails its hash."""
    url = ocr_model_url()
    with session().get(url, headers={"user-agent": USER_AGENT}, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        payload = b"".join(response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES))

    digest = hashlib.sha256(payload).hexdigest()
    if digest != OCR_MODEL_SHA256:
        raise ValueError(f"{url} hashed to {digest}, expected {OCR_MODEL_SHA256}")

    model_path = prefix_join(output_path, OCR_MODEL_FILENAME)
    StoragePath(model_path).write_bytes(payload)
    logger.info("Staged OCR router (%d bytes) at %s", len(payload), model_path)
    return OcrModelData(model_path=model_path, revision=OCR_MODEL_REVISION, sha256=digest)


@cache
def load_booster(model_path: str) -> "xgboost.Booster":  # noqa: F821
    """Load the booster once per worker process and pin it to one thread.

    Loading through :class:`xgboost.Booster` rather than ``XGBClassifier`` keeps scikit-learn out of
    the inference path -- constructing the sklearn wrapper imports it. One thread is right because
    Zephyr costs each map task at one CPU and runs several per worker; the default would have every
    task claim every core.
    """
    import pymupdf  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415

    booster = xgb.Booster()
    booster.load_model(bytearray(StoragePath(model_path).read_bytes()))
    booster.set_param({"nthread": 1})

    if tuple(booster.feature_names or ()) != FEATURE_NAMES:
        raise ValueError(
            f"{model_path} expects {len(booster.feature_names or ())} features in a different order "
            f"than ocr_features.FEATURE_NAMES provides ({len(FEATURE_NAMES)})"
        )
    pymupdf.TOOLS.mupdf_display_errors(False)
    return booster


def _document_seed(content_digest: str, warc_filename: str, warc_record_offset: int) -> int:
    """Derive the page-sampling seed, so a re-run of a shard reproduces its predictions.

    Keyed on ``content_digest`` where present so identical PDFs sample identical pages wherever in
    the crawl they appear, and on the WARC coordinates otherwise.
    """
    key = content_digest or f"{warc_filename}:{warc_record_offset}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _classify_row(row: dict) -> tuple[dict, "DocumentFeatures | None"]:  # noqa: F821
    """Return the output row for one PDF, and its features when they could be extracted.

    Every column is present on every row, so an unreadable PDF is a fully-formed row whose routing
    columns are null and whose ``classification_error`` says why. ``ocr_prob`` and ``needs_ocr`` stay
    null until :func:`classify_batch` scores the batch.
    """
    import pymupdf  # noqa: PLC0415

    from experiments.build_pdf_source.ocr_features import CorruptPdf, document_features  # noqa: PLC0415

    output = {
        "warc_filename": row["warc_filename"],
        "warc_record_offset": row["warc_record_offset"],
        "content_digest": row["content_digest"],
        "url": row["url"],
        "pdf_bytes": len(row["pdf"]),
        "needs_ocr": None,
        "ocr_prob": None,
        "num_pages": None,
        "num_pages_sampled": None,
        "garbled_text_ratio": None,
        "is_form": None,
        "is_scanner_produced": None,
        "classification_error": None,
    }
    try:
        with pymupdf.open(stream=row["pdf"], filetype="pdf") as doc:
            features = document_features(
                doc,
                seed=_document_seed(row["content_digest"], row["warc_filename"], row["warc_record_offset"]),
            )
    except CorruptPdf as error:
        counters.pipeline.update_counter("focus_crawl_pdf/unreadable", 1)
        return output | {"classification_error": str(error)}, None
    except Exception as error:
        # PyMuPDF raises a wide range of types on damaged input; a document we cannot parse is data,
        # not a pipeline failure, so it is recorded and the shard carries on.
        counters.pipeline.update_counter("focus_crawl_pdf/parse_failed", 1)
        logger.warning("Could not read %s: %s", row["url"], error)
        return output | {"classification_error": f"{type(error).__name__}: {error}"}, None

    return (
        output
        | {
            "num_pages": features.num_pages,
            "num_pages_sampled": features.num_pages_successfully_sampled,
            "garbled_text_ratio": features.garbled_text_ratio,
            "is_form": features.is_form,
            "is_scanner_produced": features.creator_or_producer_is_known_scanner,
        },
        features,
    )


def classify_batch(batch: pa.RecordBatch, model_path: str) -> Iterator[dict]:
    """Score one Parquet row group's PDFs, extracting features per document but predicting once."""
    rows: list[dict] = []
    matrix: list[np.ndarray] = []
    scored: list[int] = []

    for row in batch.to_pylist():
        output, features = _classify_row(row)
        if features is not None:
            scored.append(len(rows))
            matrix.append(features.vector())
        rows.append(output)

    if matrix:
        booster = load_booster(model_path)
        probabilities = booster.inplace_predict(np.vstack(matrix), validate_features=False)
        for index, probability in zip(scored, probabilities.tolist(), strict=True):
            rows[index]["ocr_prob"] = probability
            rows[index]["needs_ocr"] = (
                probability >= OCR_PROBABILITY_THRESHOLD or rows[index]["garbled_text_ratio"] > 0.0
            )

    counters.pipeline.update_counter("focus_crawl_pdf/classified", len(scored))
    counters.pipeline.update_counter("focus_crawl_pdf/needs_ocr", sum(1 for row in rows if row.get("needs_ocr")))
    yield from rows


def classify_pdfs(output_path: str, source_output_path: str, model_output_path: str) -> PdfClassificationData:
    """Score every fetched PDF and write the routing table to ``output_path``."""
    source = read_artifact(source_output_path, PdfSourceData)
    model = read_artifact(model_output_path, OcrModelData)
    logger.info("Classifying %s with %s (%s)", source.main_output_dir, model.model_path, model.revision[:8])

    output_dir = prefix_join(output_path, "outputs/main")
    pipeline = (
        Dataset.from_files(prefix_join(source.main_output_dir, "*.parquet"))
        .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
        .flat_map(partial(classify_batch, model_path=model.model_path))
        .write_parquet(
            prefix_join(output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_OUTPUT_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name="focus-crawl-pdf-classify",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        map_task_resources=_MAP_TASK_RESOURCES,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline)
    return PdfClassificationData(main_output_dir=output_dir, counters=dict(outcome.counters))


def routing_keys(classification_dir: str, needs_ocr: bool) -> frozenset[tuple[str, int]]:
    """Read the routing table and return the keys of the documents on one route.

    The routing decision is a join, and this is the cheap side of it. The table is narrow -- about
    30 MB for the whole sample -- precisely so an extraction step can hold one route's keys in
    memory and filter as it reads, rather than paying to have hundreds of GiB of PDF bytes shuffled
    into two datasets.

    ``needs_ocr`` is null for documents the classifier could not read at all, and those are on
    neither route: a PDF that PyMuPDF could not open to extract features will not open for
    extraction and will not render either.
    """
    filesystem, path = url_to_fs(classification_dir)
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))
    if not shards:
        raise RuntimeError(f"No routing table under {classification_dir}")

    keys: set[tuple[str, int]] = set()
    for shard in shards:
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=_ROUTING_COLUMNS)
        for row in table.to_pylist():
            if row["needs_ocr"] is needs_ocr:
                keys.add((row["warc_filename"], row["warc_record_offset"]))
    route = "OCR" if needs_ocr else "text-extractable"
    logger.info("The %s route holds %d documents across %d routing shards", route, len(keys), len(shards))
    return frozenset(keys)


def model_step() -> StepSpec:
    """Build the step that stages the FinePDFs OCR router."""
    return StepSpec(
        name="data/datakit/model/finepdfs_xgb_ocr_classifier",
        hash_attrs={
            "repo": OCR_MODEL_REPO,
            "revision": OCR_MODEL_REVISION,
            "path_in_repo": OCR_MODEL_PATH_IN_REPO,
            "sha256": OCR_MODEL_SHA256,
        },
        fn=remote(stage_ocr_model, resources=_MODEL_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def classify_step(source: StepSpec, model: StepSpec) -> StepSpec:
    """Build the OCR routing step for a fetched PDF source and a staged model."""
    return StepSpec(
        name="data/datakit/classify/common_crawl_focus_2026_22_pdf_ocr",
        deps=[source, model],
        hash_attrs={
            "ocr_probability_threshold": OCR_PROBABILITY_THRESHOLD,
            "feature_count": len(FEATURE_NAMES),
            "schema_version": 1,
        },
        fn=remote(
            partial(
                classify_pdfs,
                source_output_path=source.output_path,
                model_output_path=model.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
