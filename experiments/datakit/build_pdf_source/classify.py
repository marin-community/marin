# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3: route each fetched PDF to the VLM or to Docling, on measured extraction agreement.

Two steps. :func:`model_step` stages the trained booster into the marin prefix once, pinned by
content hash so a run's predictions are attributable to an exact set of weights. :func:`classify_step`
maps over the fetch step's Parquet shards and scores every PDF.

**What this model is asked, and why it is not the FinePDFs one.** This step previously ran the
FinePDFs XGBoost router, which is trained to answer *is this a scan?* The decision actually being
made is different: *will Docling read this document the way the VLM would?* A born-digital paper
with a broken ToUnicode CMap, an invisible OCR layer over a bitmap, a two-column layout, or a page
of equations has healthy "not a scan" statistics and extracts badly. Measured against VLM output on
a 100,000-document sample, the FinePDFs rule sent 31.4% of documents to the VLM while leaving 28.7%
of the whole corpus on the Docling route with materially degraded text, catching only 42% of the
documents Docling cannot read. The booster here is trained directly on Docling-versus-VLM agreement
using :mod:`~experiments.datakit.build_pdf_source.quality.route_features`; at a matched budget it removes
about a fifth of that silent loss, and most of the gain comes from the features rather than from
retraining -- see ``experiments/datakit/build_pdf_source/pdf-extraction-routing.md``.

**The operating point is 50% of documents to the VLM, and it was chosen from the cost/quality
frontier rather than inherited.** Below ~35% nearly every additional document sent to the VLM is one
Docling would have botched, so stopping earlier leaves cheap quality unbought; past ~50% the
marginal document costs more than two VLM runs per document actually rescued. See
:data:`DOCLING_CONFIDENCE_THRESHOLD`.

The output is a routing table, not a copy of the corpus: one narrow row per PDF keyed by
``(warc_filename, warc_record_offset)``, which is unique where ``content_digest`` is not (the crawl
holds ~9.8% exact-duplicate PDFs). At full-crawl scale (3.17M PDFs) the table is ~300 MB of parquet
and one route's in-memory key set is ~0.6 GB (see :func:`routing_keys`), so extraction (#7618) can
broadcast it and join in a map, rather than paying to have ~4.4 TiB of PDF bytes copied forward
through this step.

Inference is batched: a Parquet row group's documents are scored in one
:meth:`xgboost.Booster.inplace_predict` over a float32 matrix. The per-document cost that remains is
PyMuPDF parsing, about 35 ms per page against Docling's ~1 s, so the router stays far cheaper than
the extraction it decides to skip.

PyMuPDF, XGBoost and :mod:`~experiments.datakit.build_pdf_source.quality.route_features` are imported inside
the functions that use them, not at module scope. They live in marin-core's ``pdf`` extra, which
the workers get via ``pip_dependency_groups`` but the entrypoint job does not: its ``uv sync`` carries
no extras. Since :mod:`~experiments.datakit.build_pdf_source.pipeline` imports this module to build its
steps, a module-scope ``import pymupdf`` here kills the driver before it submits anything. This
mirrors how :mod:`~experiments.datakit.build_pdf_source.fetch` defers ``warcio``.
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
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.common import (
    PdfClassificationData,
    PdfSourceData,
    StagedModelData,
)

logger = logging.getLogger(__name__)

# The routing booster, trained by :mod:`experiments.datakit.build_pdf_source.quality.fit_route_booster` on
# the 100k oracle sample and staged content-addressed. Pinned by hash rather than by path so a run's
# predictions are attributable to exact weights; regenerate with that module, not by editing this.
ROUTE_MODEL_SOURCE = "s3://marin-us-east-02a/marin/data/datakit/model/pdf_route_classifier_00757366/route_classifier.ubj"
ROUTE_MODEL_SHA256 = "007573661947c3d3ff33fcc76f918382791cb4bed036cb1f25163b92aa3eecb8"
ROUTE_MODEL_FILENAME = "route_classifier.ubj"

# The booster predicts P(Docling reads this document as well as the VLM does). A document goes to
# the VLM below this confidence. The value is the 50th percentile of the model's own output on the
# scoring corpus -- calibrated, not tuned: the score is a probability of a *proxy* label (bigram
# recall against VLM text), so only its rank carries meaning and the threshold is a quantile.
# Recalibrating for a different budget on a new corpus is therefore a quantile, not a retrain.
#
# 50% of documents to the VLM sits just past the frontier's knee (45.5%, by maximum distance from
# the endpoint chord over the study table; experiments/datakit/build_pdf_source/pdf-extraction-routing.md). Held out on
# documents from domains the model never saw, this point catches 79% of the documents Docling reads
# badly and leaves 10.8% of the corpus mis-routed to Docling, against 30.3% for the FinePDFs rule
# at its own 29% budget on the same held-out split and 26.8% for that rule rethresholded to spend
# the same 50%. (The 28.7% quoted in the module docstring is the same FinePDFs mis-routing measured
# over the full 100k sample rather than the held-out split.)
#
# The 50% budget itself lives with the calibration that derives this threshold from it:
# ``fit_route_booster.TARGET_VLM_FRACTION``.
DOCLING_CONFIDENCE_THRESHOLD = 0.542031

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
        # P(Docling matches the VLM); ``needs_ocr`` is this against DOCLING_CONFIDENCE_THRESHOLD.
        pa.field("docling_confidence", pa.float32(), nullable=True),
        pa.field("num_pages", pa.int32(), nullable=True),
        pa.field("num_pages_sampled", pa.int32(), nullable=True),
        # Kept on the row because they are the signals a human reaches for first when auditing a
        # routing decision, and reading them back out of the model is not possible.
        pa.field("replacement_ratio", pa.float32(), nullable=True),
        pa.field("invisible_char_ratio", pa.float32(), nullable=True),
        pa.field("column_count", pa.float32(), nullable=True),
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


def stage_route_model(output_path: str) -> StagedModelData:
    """Copy the pinned booster into the step's output prefix, refusing anything that fails its hash."""
    payload = StoragePath(ROUTE_MODEL_SOURCE).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ROUTE_MODEL_SHA256:
        raise ValueError(f"{ROUTE_MODEL_SOURCE} hashed to {digest}, expected {ROUTE_MODEL_SHA256}")

    model_path = prefix_join(output_path, ROUTE_MODEL_FILENAME)
    StoragePath(model_path).write_bytes(payload)
    logger.info("Staged routing booster (%d bytes) at %s", len(payload), model_path)
    return StagedModelData(model_path=model_path, revision=ROUTE_MODEL_SHA256[:12], sha256=digest)


@cache
def load_booster(model_path: str) -> "xgboost.Booster":  # noqa: F821
    """Load the booster once per worker process and pin it to one thread.

    Loading through :class:`xgboost.Booster` rather than ``XGBClassifier`` keeps scikit-learn out of
    the inference path -- constructing the sklearn wrapper imports it. One thread is right because
    Zephyr costs each map task at one CPU and runs several per worker; the default would have every
    task claim every core.

    The booster's own ``feature_names`` are checked against the feature module, because a booster
    whose feature order has moved would score confident nonsense rather than fail.
    """
    import xgboost as xgb  # noqa: PLC0415

    from experiments.datakit.build_pdf_source.quality.route_feature_names import FEATURE_NAMES  # noqa: PLC0415

    booster = xgb.Booster()
    booster.load_model(bytearray(StoragePath(model_path).read_bytes()))
    booster.set_param({"nthread": 1})

    trained_on = tuple(booster.feature_names or ())
    if trained_on != tuple(FEATURE_NAMES):
        missing = set(trained_on) ^ set(FEATURE_NAMES)
        raise ValueError(
            f"{model_path} expects {len(trained_on)} features in a different order or set than "
            f"route_feature_names.FEATURE_NAMES provides ({len(FEATURE_NAMES)}); differing: {sorted(missing)[:8]}"
        )
    return booster


@cache
def _silence_mupdf() -> None:
    """Stop MuPDF writing a diagnostic to stderr for every damaged page.

    Called before any document is opened rather than alongside the booster load, which happens
    after a whole batch has already been parsed: damaged PDFs are ordinary input here, and at
    corpus scale their per-page complaints are the bulk of the log.
    """
    import pymupdf  # noqa: PLC0415

    pymupdf.TOOLS.mupdf_display_errors(False)


def _document_seed(content_digest: str, warc_filename: str, warc_record_offset: int) -> int:
    """Derive the page-sampling seed, so a re-run of a shard reproduces its predictions.

    Keyed on ``content_digest`` where present so identical PDFs sample identical pages wherever in
    the crawl they appear, and on the WARC coordinates otherwise.
    """
    key = content_digest or f"{warc_filename}:{warc_record_offset}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _classify_row(row: dict) -> tuple[dict, "DocumentSignals | None"]:  # noqa: F821
    """Return the output row for one PDF, and its signals when they could be extracted.

    Every column is present on every row, so an unreadable PDF is a fully-formed row whose routing
    columns are null and whose ``classification_error`` says why. ``docling_confidence`` and
    ``needs_ocr`` stay null until :func:`classify_batch` scores the batch.
    """
    import pymupdf  # noqa: PLC0415

    from experiments.datakit.build_pdf_source.quality.route_features import CorruptPdf, signals_for_routing  # noqa: PLC0415

    _silence_mupdf()
    output = {
        "warc_filename": row["warc_filename"],
        "warc_record_offset": row["warc_record_offset"],
        "content_digest": row["content_digest"],
        "url": row["url"],
        "pdf_bytes": len(row["pdf"]),
        "needs_ocr": None,
        "docling_confidence": None,
        "num_pages": None,
        "num_pages_sampled": None,
        "replacement_ratio": None,
        "invisible_char_ratio": None,
        "column_count": None,
        "classification_error": None,
    }
    try:
        with pymupdf.open(stream=row["pdf"], filetype="pdf") as doc:
            signals = signals_for_routing(
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
            "num_pages": signals.page_count,
            "num_pages_sampled": signals.pages_sampled,
            "replacement_ratio": signals.maximum["replacement_ratio"],
            "invisible_char_ratio": signals.maximum["invisible_char_ratio"],
            "column_count": signals.maximum["column_count"],
        },
        signals,
    )


def classify_batch(batch: pa.RecordBatch, model_path: str) -> Iterator[dict]:
    """Score one Parquet row group's PDFs, extracting signals per document but predicting once."""
    from experiments.datakit.build_pdf_source.quality.route_feature_names import FEATURE_NAMES  # noqa: PLC0415

    rows: list[dict] = []
    matrix: list[list[float]] = []
    scored: list[int] = []

    for row in batch.to_pylist():
        output, signals = _classify_row(row)
        if signals is not None:
            vector = signals.feature_vector()
            scored.append(len(rows))
            matrix.append([vector[name] for name in FEATURE_NAMES])
        rows.append(output)

    if matrix:
        booster = load_booster(model_path)
        confidences = booster.inplace_predict(np.asarray(matrix, dtype=np.float32), validate_features=False)
        for index, confidence in zip(scored, confidences.tolist(), strict=True):
            rows[index]["docling_confidence"] = confidence
            rows[index]["needs_ocr"] = confidence < DOCLING_CONFIDENCE_THRESHOLD

    counters.pipeline.update_counter("focus_crawl_pdf/classified", len(scored))
    counters.pipeline.update_counter("focus_crawl_pdf/needs_ocr", sum(1 for row in rows if row.get("needs_ocr")))
    yield from rows


def classify_pdfs(output_path: str, source_output_path: str, model_output_path: str) -> PdfClassificationData:
    """Score every fetched PDF and write the routing table to ``output_path``."""
    source = read_artifact(source_output_path, PdfSourceData)
    model = read_artifact(model_output_path, StagedModelData)
    logger.info("Classifying %s with %s (%s)", source.main_output_dir, model.model_path, model.revision)

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
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
    return PdfClassificationData(main_output_dir=output_dir, counters=dict(outcome.counters))


def routing_keys(classification_dir: str, needs_ocr: bool) -> frozenset[tuple[str, int]]:
    """Read the routing table and return the keys of the documents on one route.

    The routing decision is a join, and this is the cheap side of it. The table is narrow
    precisely so an extraction step can hold one route's keys in memory and filter as it reads,
    rather than paying to have TiB of PDF bytes shuffled into two datasets. At full-crawl scale
    that is 3.17M classified PDFs: one route's ~1.6M keys cost ~350 bytes each as Python objects
    (a ~100-char WARC path string, an int, the tuple, and the set slot), so the broadcast set is
    ~0.6 GB. Against the sender tasks that hold it -- 6 GB on the converter fleet, 4 GB on the OCR
    route, each also holding roughly 1-1.5 GB of in-flight PDF or page payloads -- that leaves a
    couple of GB of headroom, so the task shapes need no bump for the full crawl.

    ``needs_ocr`` is null for documents the classifier could not read at all, and those are on
    neither route: a PDF that PyMuPDF could not open to extract signals will not open for
    extraction and will not render either.
    """
    shards = sorted(StoragePath(prefix_join(classification_dir, "*.parquet")).glob(), key=str)
    if not shards:
        raise RuntimeError(f"No routing table under {classification_dir}")

    keys: set[tuple[str, int]] = set()
    for shard in shards:
        with shard.open("rb") as stream:
            table = pq.read_table(stream, columns=_ROUTING_COLUMNS)
        for row in table.to_pylist():
            if row["needs_ocr"] is needs_ocr:
                keys.add((row["warc_filename"], row["warc_record_offset"]))
    route = "OCR" if needs_ocr else "text-extractable"
    logger.info("The %s route holds %d documents across %d routing shards", route, len(keys), len(shards))
    return frozenset(keys)


def model_step() -> StepSpec:
    """Build the step that stages the PDF routing booster."""
    return StepSpec(
        name="data/datakit/model/pdf_route_classifier",
        hash_attrs={"source": ROUTE_MODEL_SOURCE, "sha256": ROUTE_MODEL_SHA256},
        fn=remote(stage_route_model, resources=_MODEL_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def classify_step(source: StepSpec, model: StepSpec) -> StepSpec:
    """Build the routing step for a fetched PDF source and a staged model."""
    return StepSpec(
        name="data/datakit/classify/common_crawl_focus_2026_22_pdf_ocr",
        deps=[source, model],
        hash_attrs={
            "docling_confidence_threshold": DOCLING_CONFIDENCE_THRESHOLD,
            "model_sha256": ROUTE_MODEL_SHA256,
            "schema_version": 2,
        },
        fn=remote(
            partial(
                classify_pdfs,
                source_output_path=source.output_path,
                model_output_path=model.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            # The scoring tasks import pymupdf and xgboost at runtime; both live in the ``pdf``
            # extra, not in ``datakit``.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )
