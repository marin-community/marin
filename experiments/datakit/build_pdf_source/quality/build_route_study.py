# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the routing study table: cheap signals, the incumbent's score, and route agreement.

One Zephyr map over the 100k oracle sample
(:mod:`~experiments.datakit.build_pdf_source.quality.build_oracle_sample`), emitting one narrow row per
document that carries everything the routing question needs in one place:

* the incumbent FinePDFs router's own features and probability, recomputed here so its decisions
  can be scored rather than assumed -- the sample's stored ``needs_ocr`` came from the Docling
  route's copy of the classification and is null wherever that route dropped the document;
* the candidate signals from :mod:`~experiments.datakit.build_pdf_source.quality.route_features`;
* the Docling-versus-VLM agreement numbers from
  :mod:`~experiments.datakit.build_pdf_source.quality.route_agreement`;
* the sample's own extraction bookkeeping -- truncation, loop repair, legibility, page counts --
  which bounds how much of an observed disagreement is the VLM's fault rather than Docling's.

Both feature sets read the *same* sampled pages, drawn with the incumbent's per-document seed, so a
comparison between them is a comparison of the features and not of which pages each happened to
look at.

The output is ~100k rows of a few hundred float columns: small enough to model on a laptop, which
is the point. The 435 GB of PDF bytes are read once, here, next to the storage.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-route-study \\
        --cpu 8 --memory 32GB --disk 32GB \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_route_study
"""

import logging
import random
import time
from functools import partial

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
SAMPLE_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_sample100k"
OUTPUT_PREFIX = f"s3://{BUCKET}/marin/data/pdf_quality/cc_focus_2026_22_route_study"

# The incumbent router's weights, as staged by :mod:`experiments.datakit.build_pdf_source.classify` for
# the run that produced this corpus -- the same booster whose decisions the study is scoring.
OCR_MODEL_PATH = f"s3://{BUCKET}/marin/data/datakit/model/finepdfs_xgb_ocr_classifier_ac813d19/xgb_classifier.ubj"

# Columns carried through from the sample. The two text columns and the PDF bytes are read but
# never written out; everything else is bookkeeping the study needs to interpret a disagreement.
PASSTHROUGH_COLUMNS = (
    "source_id",
    "url",
    "num_pages",
    "extraction_status",
    "pages_ocred",
    "pages_failed",
    "pages_truncated",
    "pages_unrendered",
    "mean_render_dpi",
    "pages_below_legibility_floor",
    "completion_tokens",
    "loop_chars_dropped",
    "docling_num_pages",
    "docling_extraction_status",
    "layout_backend",
    "needs_ocr",
    "doc_tokens",
    "edu_score_v2_begin",
    "edu_score_v2_middle",
    "edu_score_v2_end",
)
READ_COLUMNS = (
    *PASSTHROUGH_COLUMNS,
    "warc_filename",
    "warc_record_offset",
    "content_digest",
    "text",
    "page_offsets",
    "docling_text",
    "docling_page_offsets",
    "pdf",
)

_TASK_RESOURCES = ResourceConfig(cpu=1, ram="8g", disk="4g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="32g")
_COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=False)
_MAX_WORKERS = 24
_HEARTBEAT_TIMEOUT = 30 * 60


def storage() -> fsspec.AbstractFileSystem:
    configure_coreweave_s3()
    return fsspec.filesystem("s3")


# Per-process memo for the incumbent booster, keyed by path. A plain dict rather than
# ``functools.cache`` because this module runs as ``__main__``: cloudpickle sends its functions
# to task subprocesses by value, and a cache-wrapped closure does not survive that trip.
_INCUMBENT_BOOSTERS: dict[str, "xgboost.Booster"] = {}  # noqa: F821


def _load_incumbent_booster(model_path: str) -> "xgboost.Booster":  # noqa: F821
    """Load the FinePDFs booster once per worker process, validating its own feature contract.

    Deliberately not :func:`classify.load_booster`: that loader validates against the NEW router's
    74-name contract (``route_feature_names.FEATURE_NAMES``) and would refuse the incumbent's
    124-feature model outright. The study exists to score the incumbent, so this loader checks
    against ``ocr_feature_names.FEATURE_NAMES`` -- the contract the incumbent was trained on.
    """
    import xgboost as xgb  # noqa: PLC0415

    from experiments.datakit.build_pdf_source.ocr_feature_names import FEATURE_NAMES  # noqa: PLC0415

    booster = _INCUMBENT_BOOSTERS.get(model_path)
    if booster is not None:
        return booster

    booster = xgb.Booster()
    booster.load_model(bytearray(StoragePath(model_path).read_bytes()))
    booster.set_param({"nthread": 1})
    trained_on = tuple(booster.feature_names or ())
    if trained_on != tuple(FEATURE_NAMES):
        missing = set(trained_on) ^ set(FEATURE_NAMES)
        raise ValueError(
            f"{model_path} expects {len(trained_on)} features in a different order or set than "
            f"ocr_feature_names.FEATURE_NAMES provides ({len(FEATURE_NAMES)}); differing: {sorted(missing)[:8]}"
        )
    _INCUMBENT_BOOSTERS[model_path] = booster
    return booster


def _score_incumbent(features, model_path: str) -> float:
    """The FinePDFs booster's OCR probability for one document's feature vector."""
    booster = _load_incumbent_booster(model_path)
    return float(booster.inplace_predict(features.vector().reshape(1, -1), validate_features=False)[0])


def study_row(row: dict, model_path: str) -> dict:
    """Everything the study knows about one document.

    A PDF that will not open is not a failure: it is the clearest possible routing label, since a
    document PyMuPDF cannot parse is one Docling cannot extract either. It comes back with null
    feature columns and its ``feature_error`` set.
    """
    import pymupdf  # noqa: PLC0415

    from experiments.datakit.build_pdf_source.classify import _document_seed  # noqa: PLC0415
    from experiments.datakit.build_pdf_source.ocr_features import (  # noqa: PLC0415
        CorruptPdf,
        document_features,
        sample_page_indices,
    )
    from experiments.datakit.build_pdf_source.quality import route_agreement, route_features  # noqa: PLC0415

    output = {name: row[name] for name in PASSTHROUGH_COLUMNS}
    output["pdf_bytes"] = len(row["pdf"])
    output["docling_missing"] = row["docling_text"] is None
    output["feature_error"] = None
    output["feature_seconds"] = 0.0
    output["ocr_prob"] = None
    output["garbled_text_ratio"] = None
    output["is_form"] = None
    output["is_scanner_produced"] = None
    output.update(dict.fromkeys(route_features.FEATURE_NAMES, None))

    if row["docling_text"] is None:
        output.update(route_agreement.empty_agreement())
    else:
        output.update(
            route_agreement.document_agreement(
                row["docling_text"], row["docling_page_offsets"], row["text"], row["page_offsets"]
            )
        )

    started = time.perf_counter()
    try:
        with pymupdf.open(stream=row["pdf"], filetype="pdf") as doc:
            seed = _document_seed(row["content_digest"], row["warc_filename"], row["warc_record_offset"])
            # The incumbent's own sampling, reused so both feature sets describe the same pages.
            page_indices = sample_page_indices(len(doc), random.Random(seed))
            incumbent = document_features(doc, seed=seed)
            candidate = route_features.document_signals(doc, page_indices)
            output.update(
                ocr_prob=_score_incumbent(incumbent, model_path),
                garbled_text_ratio=incumbent.garbled_text_ratio,
                is_form=incumbent.is_form,
                is_scanner_produced=incumbent.creator_or_producer_is_known_scanner,
            )
            output.update(candidate.feature_vector())
    except CorruptPdf as error:
        counters.pipeline.update_counter("route_study/corrupt_pdf", 1)
        output["feature_error"] = f"CorruptPdf: {error}"
    except Exception as error:
        counters.pipeline.update_counter("route_study/feature_failed", 1)
        logger.warning("Features failed for %s: %s", row["url"], error)
        output["feature_error"] = f"{type(error).__name__}: {error}"
    output["feature_seconds"] = time.perf_counter() - started
    return output


def study_shard(work: tuple[int, str], *, model_path: str) -> int:
    """Emit the study rows for one sample shard, or skip it if its output already exists."""
    index, shard = work
    fs = storage()
    output = f"{OUTPUT_PREFIX}/part-{index:05d}.parquet"
    if fs.exists(output):
        return 0

    with fs.open(shard, "rb") as stream:
        table = pq.read_table(stream, columns=list(READ_COLUMNS))

    rows = [study_row(row, model_path) for row in table.to_pylist()]
    with fs.open(output, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows), stream, compression="zstd", compression_level=1)
    counters.pipeline.update_counter("route_study/documents", len(rows))
    logger.info("shard %d: wrote %d rows", index, len(rows))
    return len(rows)


def shards() -> list[tuple[int, str]]:
    paths = sorted(str(path) for path in StoragePath(f"{SAMPLE_PREFIX}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"no sample shards under {SAMPLE_PREFIX}")
    return list(enumerate(paths))


def main() -> None:
    configure_logging(logging.INFO)
    storage()
    work = shards()
    logger.info("route study: %d shards -> %s", len(work), OUTPUT_PREFIX)

    outcome = ZephyrContext(
        name="pdf-route-study",
        resources=_WORKER_RESOURCES,
        coordinator_resources=_COORDINATOR_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(
        Dataset.from_list(work).map(partial(study_shard, model_path=OCR_MODEL_PATH)),
        map_task_resources=_TASK_RESOURCES,
    )
    logger.info("route study: done, counters %s", dict(outcome.counters))


if __name__ == "__main__":
    main()
