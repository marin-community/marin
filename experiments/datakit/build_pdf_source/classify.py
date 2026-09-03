# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4: decide which extracted PDFs to escalate to the VLM, on signals the extraction produced.

Router v2 runs *after* extraction (see ``pdf-router-v2.md`` on the ``mark/pdf_pipeline`` campaign
branch for more).

Two steps. :func:`model_step` stages the booster and its calibration sidecar into the marin prefix,
each pinned by content hash. :func:`classify_step` maps over
:mod:`~experiments.datakit.build_pdf_source.extract_inspector`'s output -- reading only the signal
columns, never ``text`` -- and writes one narrow row per document, one routing shard per extraction
shard and named after it, so the routing table stays co-partitioned with the fetch. The two
consumers, the OCR route and the union, read their decisions shard by shard through
:func:`shard_routing` instead of holding the corpus-wide table.

**Three decisions, and only one of them is the model's.**

* pdf-inspector returned no text -> escalate, always. Exact and validated: every one of the 2,054
  labelled such documents was escalated by the judge, a rate of 1.000, and they are 12.4% of the
  corpus. The score never sees them, so it is neither trained nor calibrated on them.
* Nothing can open the document for rendering -> keep it. The VLM route rasterises through the same
  library the geometry pass used, so escalating a document that cannot be rendered spends a routing
  slot on a page that will never reach the model.
* Everything else -> the booster, thresholded at :data:`ESCALATION_THRESHOLD`.

The legibility floor is deliberately **not** a fourth decision. What the floor earns instead is a
*render* policy: a flagged document is escalated like any other and rendered at
:data:`~experiments.datakit.build_pdf_source.ocr_extract.render.RAISED_MAX_VISUAL_TOKENS` rather
than being skipped. See :func:`render_budget`.

XGBoost is imported inside the functions that use it. It lives in marin-core's ``pdf`` extra, which
the Zephyr workers get via ``pip_dependency_groups`` but the entrypoint job does not: its ``uv sync``
carries no extras. Since :mod:`~experiments.datakit.build_pdf_source.pipeline` imports this module to
build its steps, a module-scope ``import xgboost`` here would kill the driver before it submitted
anything.
"""

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache, lru_cache, partial

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source import route_v2_features as contract
from experiments.datakit.build_pdf_source.common import (
    MAIN_OUTPUT_SUBDIR,
    SHARD_PATTERN,
    PdfClassificationData,
    PdfDocumentsData,
    StagedModelData,
)
from experiments.datakit.build_pdf_source.extract_inspector import SIGNAL_COLUMNS
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    DEFAULT_LEGIBILITY_FLOOR_DPI,
    DEFAULT_MAX_VISUAL_TOKENS,
    RAISED_MAX_VISUAL_TOKENS,
)

logger = logging.getLogger(__name__)

# The shipped booster and the sidecar that calibrates its threshold, both pinned by content hash;
# regenerate them with `quality/fit_route_v2.py` on the `mark/pdf_processing` campaign branch.
MODEL_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/model/pdf_route_v2"
ROUTE_MODEL_FILENAME = "route_v2_classifier.ubj"
ROUTE_MODEL_SOURCE = prefix_join(MODEL_PREFIX, ROUTE_MODEL_FILENAME)
ROUTE_MODEL_SHA256 = "5edde108ec41c50680f34802a09a368309e7d8c68c470f7a922a04d5a78f37c6"
ROUTE_SIDECAR_FILENAME = "route_v2_classifier.json"
ROUTE_SIDECAR_SOURCE = prefix_join(MODEL_PREFIX, ROUTE_SIDECAR_FILENAME)
ROUTE_SIDECAR_SHA256 = "d30ce417098bead8071b020405ab0098802ebcc5b6035313fb42c59b70387372"

# The escalation threshold, restated from the sidecar so that the value this pipeline routes on is
# visible in the code and re-keys the step when it moves. :func:`load_router` refuses a sidecar that
# disagrees, so the two cannot drift silently.
#
# It is a **quantile of the model's own output over the whole corpus**, not a tuned cut. The score is
# a probability of a judged preference, so its absolute value means nothing operationally and only
# its rank does; recalibrating for a different corpus or a different budget is a quantile, not a
# retrain, and every other budget's threshold is already tabulated in the sidecar's
# ``threshold_by_budget``. The budget it spends -- 89.59% of routable documents, 91.65% of their
# pages -- is the minimum of *total* misroute on the fit's own held-out frontier, which is the metric
# that has a minimum: one-sided quality loss is monotone in budget under a preference label and would
# always recommend escalating everything. At this point the held-out numbers are 0.0262 quality loss
# per page, 0.2086 misroute per page, 96.6% of the pages that want the VLM caught, on 4,428 documents
# from 633 unseen domains.
ESCALATION_THRESHOLD = 0.5809440612792969

_COUNTER_PREFIX = "focus_crawl_pdf_route"
_ROUTING_COLUMNS = ["warc_filename", "warc_record_offset", "needs_ocr", "render_visual_tokens"]
# Routing shards a worker process keeps decoded. A task reads one shard's documents, so one entry
# serves every row group of the shard; a couple more cover a process that interleaves shards.
_ROUTING_SHARDS_CACHED = 4

# Why a document went where it did. Stored per row because a routing decision that cannot be
# explained cannot be audited, and because two of the three are gates whose correctness is checkable
# by arithmetic rather than by rerunning the model.
GATE_NO_TEXT = "no_text"
GATE_UNRENDERABLE = "unrenderable"
DECIDED_BY_SCORE = "score"

ROUTING_SCHEMA = pa.schema(
    [
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_record_offset", pa.int64(), nullable=False),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("needs_ocr", pa.bool_(), nullable=False),
        pa.field("route_reason", pa.string(), nullable=False),
        # P(a judge prefers the VLM). Null on the gated rows, which the model never scored.
        pa.field("escalation_score", pa.float32(), nullable=True),
        # The render policy's output, and the only column the OCR route reads besides the key.
        pa.field("render_visual_tokens", pa.int32(), nullable=False),
        # The gates' own inputs, kept on the row because they are what a human reaches for first when
        # auditing a decision, and because reading them back out of the model is not possible.
        pa.field("inspector_markdown_chars", pa.int64(), nullable=True),
        pa.field("mean_render_dpi", pa.float32(), nullable=True),
        pa.field("num_pages", pa.int32(), nullable=True),
    ]
)

_MODEL_RESOURCES = ResourceConfig(cpu=1, ram="2g", disk="2g")
_DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
# Scoring is a column projection and one prediction per row group: no PDF is parsed and no page is
# rendered, because the extraction already did all of that. Tasks are correspondingly small.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="2g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="32g", disk="32g")
_MAX_WORKERS = 16
_HEARTBEAT_TIMEOUT = 15 * 60


def stage_route_model(output_path: str) -> StagedModelData:
    """Copy the pinned booster and its sidecar into the step's output prefix, refusing bad hashes."""
    staged: dict[str, str] = {}
    for source, filename, expected in (
        (ROUTE_MODEL_SOURCE, ROUTE_MODEL_FILENAME, ROUTE_MODEL_SHA256),
        (ROUTE_SIDECAR_SOURCE, ROUTE_SIDECAR_FILENAME, ROUTE_SIDECAR_SHA256),
    ):
        payload = StoragePath(source).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"{source} hashed to {digest}, expected {expected}")
        staged[filename] = prefix_join(output_path, filename)
        StoragePath(staged[filename]).write_bytes(payload)
        logger.info("Staged %s (%d bytes) at %s", filename, len(payload), staged[filename])

    # ``model_path`` is the directory, because this step stages two files that only mean anything
    # together: a booster without its calibration is a ranking with no cut in it.
    return StagedModelData(model_path=output_path, revision=ROUTE_MODEL_SHA256[:12], sha256=ROUTE_MODEL_SHA256)


def router_threshold(trained_on: tuple[str, ...], sidecar: dict) -> float:
    """Check the booster, its calibration and this module against each other, and return the cut.

    Three things are checked, because each would otherwise produce confident nonsense rather than an
    error. XGBoost scores a bare float matrix by position, so a booster whose ``feature_names`` have
    moved reads every column as a different feature. A sidecar naming a different feature list means
    the threshold was calibrated on a different model. And a sidecar threshold that disagrees with
    :data:`ESCALATION_THRESHOLD` means the pipeline routes at a budget its own step hash does not
    describe -- so the run's identity would not name what it did.
    """
    if trained_on != contract.ROUTER_FEATURES:
        differing = set(trained_on) ^ set(contract.ROUTER_FEATURES)
        raise ValueError(
            f"the booster expects {len(trained_on)} features in a different order or set than "
            f"route_v2_features.ROUTER_FEATURES provides ({len(contract.ROUTER_FEATURES)}); "
            f"differing: {sorted(differing)[:8]}"
        )
    if tuple(sidecar["features"]) != contract.ROUTER_FEATURES:
        raise ValueError(f"{ROUTE_SIDECAR_FILENAME} calibrates a different feature set than the booster declares")
    threshold = float(sidecar["escalation_threshold"])
    if threshold != ESCALATION_THRESHOLD:
        raise ValueError(
            f"{ROUTE_SIDECAR_FILENAME} thresholds at {threshold!r} but this step is keyed on "
            f"{ESCALATION_THRESHOLD!r}; restage the model or update the constant, not both silently"
        )
    return threshold


@cache
def load_router(model_dir: str) -> tuple["xgboost.Booster", float]:  # noqa: F821
    """Load the booster and its calibrated threshold once per worker process."""
    import xgboost as xgb  # noqa: PLC0415

    booster = xgb.Booster()
    booster.load_model(bytearray(StoragePath(prefix_join(model_dir, ROUTE_MODEL_FILENAME)).read_bytes()))
    booster.set_param({"nthread": 1})
    sidecar = json.loads(StoragePath(prefix_join(model_dir, ROUTE_SIDECAR_FILENAME)).read_bytes())

    threshold = router_threshold(tuple(booster.feature_names or ()), sidecar)
    logger.info(
        "Router v2 loaded: arm %r, %d features, escalating at %.6f (%.2f%% of routable documents)",
        sidecar["arm"],
        len(contract.ROUTER_FEATURES),
        threshold,
        100 * sidecar["target_document_budget"],
    )
    return booster, threshold


def render_budget(mean_render_dpi: float | None, floor_dpi: float) -> int:
    """The visual-token budget an escalated document is rendered at.

    A document whose mean render DPI falls below the legibility floor at the default budget is
    rendered at :data:`RAISED_MAX_VISUAL_TOKENS` instead. A document with no geometry gets the
    default budget: it is not going to be rendered at all.
    """
    if mean_render_dpi is None or mean_render_dpi >= floor_dpi:
        return DEFAULT_MAX_VISUAL_TOKENS
    return RAISED_MAX_VISUAL_TOKENS


def gate(row: dict) -> str | None:
    """The arithmetic decision for a document, or ``None`` if the score has to make it."""
    if not row.get("inspector_markdown_chars"):
        return GATE_NO_TEXT
    if row.get("mean_render_dpi") is None:
        return GATE_UNRENDERABLE
    return None


def route_batch(batch: pa.RecordBatch, model_dir: str, floor_dpi: float) -> Iterator[dict]:
    """Route one Parquet row group: gate what is arithmetic, score the rest in one prediction.

    The feature frame is built through :func:`route_v2_features.with_derived`, the same function the
    fit ran, so a document is scored on arithmetic identical to the arithmetic it was fit on. Gated
    rows are excluded from the matrix rather than scored and overridden: the model was neither
    trained nor calibrated on them, so a score for them would be an extrapolation presented as a
    probability.
    """
    rows = batch.to_pylist()
    if not rows:
        return

    gates = [gate(row) for row in rows]
    scorable = [index for index, reason in enumerate(gates) if reason is None]
    score_by_index: dict[int, float] = {}
    # The same number the sidecar carries -- :func:`router_threshold` refuses a model whose
    # calibration disagrees -- so a batch with nothing to score need not load the booster to route.
    threshold = ESCALATION_THRESHOLD
    if scorable:
        booster, threshold = load_router(model_dir)
        frame = contract.with_derived(pl.from_arrow(batch.take(scorable)))
        matrix = frame.select(contract.ROUTER_FEATURES).to_numpy().astype(np.float32)
        predictions = booster.inplace_predict(matrix, validate_features=False).tolist()
        score_by_index = dict(zip(scorable, predictions, strict=True))

    for index, (row, reason) in enumerate(zip(rows, gates, strict=True)):
        score = score_by_index.get(index)
        needs_ocr = (reason == GATE_NO_TEXT) if reason is not None else (score >= threshold)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{reason or DECIDED_BY_SCORE}", 1)
        if needs_ocr:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/needs_ocr", 1)
        tokens = render_budget(row["mean_render_dpi"], floor_dpi) if needs_ocr else DEFAULT_MAX_VISUAL_TOKENS
        if tokens != DEFAULT_MAX_VISUAL_TOKENS:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/raised_render_budget", 1)
        yield {
            "warc_filename": row["warc_filename"],
            "warc_record_offset": row["warc_record_offset"],
            "content_digest": row["content_digest"],
            "url": row["url"],
            "needs_ocr": needs_ocr,
            "route_reason": reason or DECIDED_BY_SCORE,
            "escalation_score": score,
            "render_visual_tokens": tokens,
            "inspector_markdown_chars": row["inspector_markdown_chars"],
            "mean_render_dpi": row["mean_render_dpi"],
            "num_pages": row["num_pages"],
        }


def classify_pdfs(output_path: str, extraction_output_path: str, model_output_path: str) -> PdfClassificationData:
    """Route every extracted document and write the routing table to ``output_path``.

    The output is a routing table, not a copy of the corpus: one narrow row per document keyed by
    ``(warc_filename, warc_record_offset)``, which is unique where ``content_digest`` is not (the
    crawl holds ~9.8% exact-duplicate PDFs). It is written as one shard per extraction shard, named
    after it, over the extraction's sorted listing -- the same 1:1 map the extraction ran over the
    fetch -- so the decisions for a fetched shard sit in the routing shard of the same name and a
    consumer reads them by name as it reads the shard.
    """
    extraction = read_artifact(extraction_output_path, PdfDocumentsData)
    model = read_artifact(model_output_path, StagedModelData)
    shards = sorted(str(shard) for shard in StoragePath(prefix_join(extraction.main_output_dir, "*.parquet")).glob())
    if not shards:
        raise RuntimeError(f"No extraction shards under {extraction.main_output_dir}")
    logger.info(
        "Routing %d shards under %s with %s (%s)",
        len(shards),
        extraction.main_output_dir,
        model.model_path,
        model.revision,
    )

    output_dir = prefix_join(output_path, MAIN_OUTPUT_SUBDIR)
    pipeline = (
        Dataset.from_list(shards)
        # Column projection is what makes this step cheap: the extraction's rows carry the corpus's
        # text and this reads none of it.
        .load_parquet(columns=SIGNAL_COLUMNS, batch_mode=True)
        .flat_map(partial(route_batch, model_dir=model.model_path, floor_dpi=DEFAULT_LEGIBILITY_FLOOR_DPI))
        .write_parquet(prefix_join(output_dir, SHARD_PATTERN), schema=ROUTING_SCHEMA, skip_existing=True)
    )
    outcome = ZephyrContext(
        name="focus-crawl-pdf-route",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
    return PdfClassificationData(main_output_dir=output_dir, counters=dict(outcome.counters))


@dataclass(frozen=True)
class RouteDecision:
    """What the router decided for one document: the route, and the budget it is rendered at."""

    needs_ocr: bool
    render_visual_tokens: int


@lru_cache(maxsize=_ROUTING_SHARDS_CACHED)
def shard_routing(classification_dir: str, shard_basename: str) -> dict[tuple[str, int], RouteDecision]:
    """The routing decisions for the documents of one fetched shard, keyed by WARC record.

    The routing table is co-partitioned with the fetch: the extraction maps the fetched shards 1:1
    and names its outputs after them, and this step does the same over the extraction, so the
    decisions for ``part-00012-of-01773.parquet`` sit in the routing shard of that name. A consumer
    reads that one shard as it reads the documents, and never holds the corpus-wide table.

    A fetched shard with no routing shard is an error, not an empty route: a name missing here
    means the step outputs are not the co-partitioned set this pipeline was built from.
    """
    path = StoragePath(prefix_join(classification_dir, shard_basename))
    if not path.exists():
        raise FileNotFoundError(
            f"No routing shard {shard_basename} under {classification_dir}: the routing table is "
            "not co-partitioned with the shard being read"
        )
    with path.open("rb") as stream:
        table = pq.read_table(stream, columns=_ROUTING_COLUMNS)
    return {
        (row["warc_filename"], row["warc_record_offset"]): RouteDecision(
            needs_ocr=row["needs_ocr"], render_visual_tokens=row["render_visual_tokens"]
        )
        for row in table.to_pylist()
    }


def model_step() -> StepSpec:
    """Build the step that stages the router v2 booster and its calibration sidecar."""
    return StepSpec(
        name="data/pdf_quality/model/pdf_route_v2",
        hash_attrs={
            "source": MODEL_PREFIX,
            "model_sha256": ROUTE_MODEL_SHA256,
            "sidecar_sha256": ROUTE_SIDECAR_SHA256,
        },
        fn=remote(stage_route_model, resources=_MODEL_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def classify_step(extraction: StepSpec, model: StepSpec) -> StepSpec:
    """Build the routing step over an extracted corpus and a staged router."""
    return StepSpec(
        name="data/datakit/classify/common_crawl_focus_2026_22_pdf_route_v2",
        deps=[extraction, model],
        hash_attrs={
            "escalation_threshold": ESCALATION_THRESHOLD,
            "model_sha256": ROUTE_MODEL_SHA256,
            "sidecar_sha256": ROUTE_SIDECAR_SHA256,
            "features": list(contract.ROUTER_FEATURES),
            "legibility_floor_dpi": DEFAULT_LEGIBILITY_FLOOR_DPI,
            "raised_max_visual_tokens": RAISED_MAX_VISUAL_TOKENS,
            "schema_version": 1,
        },
        fn=remote(
            partial(
                classify_pdfs,
                extraction_output_path=extraction.output_path,
                model_output_path=model.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            # The scoring tasks import xgboost at runtime; it lives in the ``pdf`` extra.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )
