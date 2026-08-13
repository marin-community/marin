# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Price pdf-inspector against the shipped router, as a feature source and as the cheap route itself.

Reads the two study tables -- the routing study
(:mod:`~experiments.datakit.build_pdf_source.quality.build_route_study`) and the pdf-inspector study
(:mod:`~experiments.datakit.build_pdf_source.quality.build_inspector_study`) -- joins them on
``source_id`` and answers two questions that need different labels.

**Question A: is pdf-inspector a better feature source?** The label stays ``docling_ok`` and the
comparison is the published one: domain-disjoint held-out documents, cost-matched at 50% of the
corpus routed to the VLM, against the shipped route-features booster's 0.1230 quality loss
(``pdf-extraction-routing.md``). The arms are the shipped features, pdf-inspector's own cheap-path
rule untrained, pdf-inspector's signals retrained as a booster, and both feature sets together. The
precedent for expecting nothing from the last is the incumbent's ``ocr_prob``, which ranked 7th by
gain when the model could use everything and moved the frontier by 0.0019.

Each pdf-inspector arm is run twice, because the library has two cheap paths and they are not the
same price. ``detect_pdf_bytes`` costs 0.441 ms/page and reports ``pdf_type``, ``confidence``,
``page_count`` and the pages it wants OCR'd with reasons; the four layout signals it also declares
are constant over all 100,000 documents and are excluded here rather than counted as features.
Those four exist only from ``extract_pages_markdown_bytes``, at 4.656 ms/page -- ~10x -- so the
extract tier has to earn that on the frontier.

**Question B: is pdf-inspector a better cheap route?** Stage 1 measured it reading closer to the VLM
than Docling does (bigram recall 0.7721 against 0.7469, share above the label floor 0.6916 against
0.5973). If it replaces Docling on the cheap side, the routing label changes with it: ``inspector_ok``
is the identical construction -- bigram recall at :data:`~...analyze_route_study.RECALL_FLOOR` with a
page-level floor, false wherever the route produced nothing -- against pdf-inspector's own agreement
columns. The headline is the VLM budget the pipeline needs to reach the quality the Docling pipeline
reaches at 50%, converted to crawl-wide GPU hours.

**Docling-versus-VLM is read from the inspector study for Question B, not from the route study.** The
normalizer gained two rules between the two passes, so ``docling_ok`` as published and
``inspector_ok`` are not measured with the same ruler. Question A keeps the published label so its
numbers stay comparable to the published table; Question B uses the inspector study's recomputed
``docling_vlm_*`` columns for both sides, so the two cheap routes are compared under one metric.

Everything is restricted to the VLM-trustworthy subset the published report uses: no truncated,
failed or unrendered pages, no loop repair, no page below the legibility floor. On rows where the
VLM extraction is itself damaged a disagreement measures the VLM's failure rather than the cheap
route's. Every arm is evaluated on one shared row set and one shared domain-disjoint split, so the
arms differ only in what they were allowed to read.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdf-inspector-routing --extra pdf \\
        --cpu 16 --memory 48GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.analyze_inspector_routing

``--extra pdf`` carries ``xgboost-cpu``. Writes a JSON result document next to the study tables and
prints the tables the report quotes.
"""

import json
import logging
from dataclasses import asdict, dataclass, replace

import fsspec
import numpy as np
import polars as pl
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.quality.analyze_route_study import (
    RoutePoint,
    label,
    read_table,
    route_frontier,
    route_ok,
)
from experiments.datakit.build_pdf_source.quality.route_feature_names import FEATURE_NAMES
from experiments.datakit.build_pdf_source.quality.train_route_model import (
    Split,
    fit,
    importances,
    split_by,
    vlm_scores,
)

logger = logging.getLogger(__name__)

ROUTE_STUDY_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_route_study"
INSPECTOR_STUDY_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_inspector_study"
RESULT_PATH = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_inspector_routing.json"

# The published operating points: what the shipped router spends, and what the incumbent spent.
SHIPPED_BUDGET = 0.50
INCUMBENT_BUDGET = 0.29
# The published benchmark this work has to beat, from pdf-extraction-routing.md.
SHIPPED_QUALITY_LOSS = 0.1230

REPORTED_BUDGETS = (0.10, 0.20, 0.25, 0.29, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
# Budgets are set by quantile of each score's own output, the same calibration `fit_route_booster`
# uses, so every arm is evaluated at the budget asked for rather than at the nearest point of a
# fixed threshold grid -- which matters most for the arms whose scores are clumped.
BUDGET_GRID = np.round(np.arange(0.002, 1.0, 0.002), 4)
# Marginal precision is the local slope of the frontier: the share of the next documents added to
# the VLM's queue that the cheap route would genuinely have botched. Read over a +-2.5% window of
# corpus so one tied clump does not turn the derivative into a step.
MARGINAL_WINDOW = 0.025

# Crawl-wide scale, for turning a page budget into GPU time.
CRAWL_PAGES = 56_000_000
VLM_GPU_HOURS_PER_MILLION_PAGES = 15.6

# Per-page cost of each path, from Stage 0/1 and the published report (ms per page).
DETECT_MS_PER_PAGE = 0.441
EXTRACT_MS_PER_PAGE = 4.656
ROUTE_FEATURES_MS_PER_PAGE = 35.0
DOCLING_MS_PER_PAGE = 1000.0

# Every reason ``detect_pdf_bytes`` gives for wanting OCR on a page, carried as one count column
# each. Unioned with whatever the table holds, so a reason the crate adds later is measured rather
# than silently dropped.
OCR_REASONS = ("no_text", "scanned", "suspected_garbled_text", "vector_text")
PDF_TYPES = ("text_based", "scanned", "image_based", "mixed")

# Signals ``detect_pdf_bytes`` reports but does not populate: constant over all 100,000 documents
# (Stage 1). Asserted rather than assumed, and excluded from the cheap feature set -- a constant
# column is not a feature, and carrying it would make the cheap arm look richer than it is.
CONSTANT_DETECT_SIGNALS = (
    "inspector_has_encoding_issues",
    "inspector_detect_is_complex_layout",
    "inspector_detect_pages_with_tables",
    "inspector_detect_pages_with_columns",
)

# What a router may read for 0.441 ms/page. ``inspector_library_milliseconds`` is deliberately not
# here: it is wall time on whichever machine ran the study, not a property of the document.
DETECT_FEATURES = (
    "inspector_confidence",
    "inspector_page_count",
    "inspector_has_title",
    "inspector_detect_pages_needing_ocr",
    "inspector_detect_ocr_page_fraction",
    *(f"inspector_type_{name}" for name in PDF_TYPES),
    *(f"inspector_reason_{name}" for name in OCR_REASONS),
)

# What the same router reads for 4.656 ms/page, ~10x: the four signals detect declares and does not
# deliver, plus the extraction's own page count and text yield.
EXTRACT_FEATURES = (
    "inspector_extract_is_complex_layout",
    "inspector_extract_pages_needing_ocr",
    "inspector_extract_pages_with_tables",
    "inspector_extract_pages_with_columns",
    "inspector_extract_table_page_fraction",
    "inspector_extract_column_page_fraction",
    "inspector_extracted_pages",
    "inspector_extract_page_deficit",
    "inspector_markdown_chars",
    "inspector_markdown_chars_per_page",
)

# Feature columns that arrive as booleans and have to reach XGBoost as numbers.
BOOLEAN_FEATURES = ("inspector_has_title", "inspector_extract_is_complex_layout")


# ---------------------------------------------------------------------------
# Reading and labelling
# ---------------------------------------------------------------------------


def joined(fs: fsspec.AbstractFileSystem) -> pl.DataFrame:
    """The two study tables joined on ``source_id``, with all three routing labels attached.

    ``trustworthy`` and the published ``docling_ok`` come from the route study; ``inspector_ok`` and
    the recomputed ``docling_ok_same_metric`` come from the inspector study, which measured both
    cheap routes against the VLM under one normalizer.
    """
    route = label(read_table(ROUTE_STUDY_PREFIX, fs))
    inspector = read_table(INSPECTOR_STUDY_PREFIX, fs)
    logger.info("route study %d rows, inspector study %d rows", route.height, inspector.height)

    frame = route.drop("url", "num_pages", "pdf_bytes", "docling_missing").join(inspector, on="source_id", how="inner")
    logger.info("joined %d rows", frame.height)

    for name in CONSTANT_DETECT_SIGNALS:
        distinct = frame[name].drop_nulls().n_unique()
        if distinct > 1:
            raise ValueError(f"{name} was constant in Stage 1 and is not any more ({distinct} values)")

    inspector_missing = pl.col("inspector_vlm_bigram_recall_mean").is_null()
    docling_missing = pl.col("docling_vlm_bigram_recall_mean").is_null()
    return frame.with_columns(
        inspector_ok=route_ok("inspector_vlm", inspector_missing),
        docling_ok_same_metric=route_ok("docling_vlm", docling_missing),
        inspector_ok_unigram=route_ok("inspector_vlm", inspector_missing, metric="unigram"),
        docling_ok_unigram=route_ok("docling_vlm", docling_missing, metric="unigram"),
    )


def ocr_reason_names(frame: pl.DataFrame) -> tuple[str, ...]:
    """Every OCR reason the table actually carries, unioned with the declared vocabulary."""
    seen: set[str] = set()
    for payload in frame["inspector_ocr_reasons"].drop_nulls().unique().to_list():
        seen.update(json.loads(payload))
    unexpected = sorted(seen - set(OCR_REASONS))
    if unexpected:
        logger.warning("ocr reasons outside the declared vocabulary: %s", unexpected)
    return tuple(sorted(set(OCR_REASONS) | seen))


def with_features(frame: pl.DataFrame, reasons: tuple[str, ...]) -> pl.DataFrame:
    """Expand pdf-inspector's categorical and JSON signals into model input columns."""
    pages = pl.col("inspector_page_count")
    return frame.with_columns(
        **{f"inspector_type_{name}": (pl.col("inspector_pdf_type") == name).cast(pl.Float64) for name in PDF_TYPES},
        **{
            f"inspector_reason_{name}": (
                pl.col("inspector_ocr_reasons").str.json_path_match(f"$.{name}").cast(pl.Float64).fill_null(0.0)
            )
            for name in reasons
        },
        **{name: pl.col(name).cast(pl.Float64) for name in BOOLEAN_FEATURES},
        inspector_detect_ocr_page_fraction=pl.col("inspector_detect_pages_needing_ocr") / pages,
        inspector_extract_table_page_fraction=pl.col("inspector_extract_pages_with_tables") / pages,
        inspector_extract_column_page_fraction=pl.col("inspector_extract_pages_with_columns") / pages,
        inspector_extract_page_deficit=pages - pl.col("inspector_extracted_pages"),
        inspector_markdown_chars_per_page=pl.col("inspector_markdown_chars") / pages,
    )


def usable(frame: pl.DataFrame, route_features: list[str]) -> pl.DataFrame:
    """The rows every arm is evaluated on -- one row set, so the arms are comparable.

    Trustworthy VLM extraction, route features that computed, and a real domain to split on. A row
    where pdf-inspector failed outright is kept rather than dropped: its signals come back null,
    which XGBoost reads as missing, and dropping it would quietly credit pdf-inspector for documents
    it lost.
    """
    return frame.filter(pl.col("trustworthy") & pl.col("feature_error").is_null() & (pl.col("domain") != "")).drop_nulls(
        subset=route_features
    )


# ---------------------------------------------------------------------------
# The frontier, read at exact budgets
# ---------------------------------------------------------------------------


def frontier(scores: np.ndarray, ok: np.ndarray) -> list[RoutePoint]:
    """The cost/quality frontier, swept by quantile of the score's own output.

    Quantiles rather than a fixed threshold grid because the arms' scores live on different scales,
    and because a clumped score has whole budget ranges no threshold can express -- which is a
    property of the score worth seeing rather than one to hide behind a grid. This is the
    calibration the shipped router uses: the threshold is whatever quantile spends the target
    budget, so only the score's rank ever matters.
    """
    thresholds = np.quantile(scores, 1.0 - BUDGET_GRID)
    return route_frontier(scores, ok, thresholds)


def at_budget(points: list[RoutePoint], budget: float) -> RoutePoint:
    return min(points, key=lambda point: abs(point.vlm_fraction - budget))


def marginal_precision(points: list[RoutePoint], budget: float) -> float:
    """The share of the next documents sent to the VLM that the cheap route would have botched.

    The local slope of quality loss against VLM spend, read over :data:`MARGINAL_WINDOW` either
    side. Its reciprocal is how many VLM runs the pipeline buys per document actually rescued.
    """
    low = at_budget(points, budget - MARGINAL_WINDOW)
    high = at_budget(points, budget + MARGINAL_WINDOW)
    span = high.vlm_fraction - low.vlm_fraction
    if span <= 0:
        return float("nan")
    return (low.quality_loss - high.quality_loss) / span


def budget_for_quality(points: list[RoutePoint], target_loss: float) -> RoutePoint | None:
    """The cheapest operating point whose quality loss is at or below *target_loss*."""
    affordable = [point for point in points if point.quality_loss <= target_loss]
    return min(affordable, key=lambda point: point.vlm_fraction) if affordable else None


def clumping(scores: np.ndarray) -> dict:
    """Whether a score can rank at all, or piles documents onto a handful of values.

    The incumbent FinePDFs rule pinned 17.4% of documents at exactly 1.0, which made its frontier
    locally degenerate: inside that band it could not order documents at all, so the marginal
    document it added was no more likely to need the VLM than one drawn at random. Any rule-based
    score is a candidate for the same pathology, and an AUC computed over one is not informative.
    """
    values, counts = np.unique(np.round(scores.astype(np.float64), 9), return_counts=True)
    return {
        "distinct_values": int(values.size),
        "largest_clump_share": float(counts.max() / scores.size),
        "largest_clump_value": float(values[counts.argmax()]),
        "tied_share": float(counts[counts > 1].sum() / scores.size),
        "share_at_max": float(counts[-1] / scores.size),
    }


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One candidate router: a name, the features it may read, and what reading them costs."""

    name: str
    features: tuple[str, ...]
    routing_ms_per_page: float


@dataclass(frozen=True)
class ArmResult:
    name: str
    features: int
    routing_ms_per_page: float
    documents_held_out: int
    points: dict[float, dict]
    marginal: dict[float, float]
    clumping: dict
    gains: list[tuple[str, float]]


def read_out(
    name: str, arm_features: int, ms_per_page: float, scores: np.ndarray, truth: np.ndarray, gains
) -> ArmResult:
    """Everything reported about one score on one held-out set."""
    points = frontier(scores, truth)
    return ArmResult(
        name=name,
        features=arm_features,
        routing_ms_per_page=ms_per_page,
        documents_held_out=int(truth.size),
        points={budget: asdict(at_budget(points, budget)) for budget in REPORTED_BUDGETS},
        marginal={budget: marginal_precision(points, budget) for budget in REPORTED_BUDGETS},
        clumping=clumping(scores),
        gains=gains,
    )


def evaluate(split: Split, arm: Arm) -> ArmResult:
    """Train the arm on one side of the shared domain-disjoint split, read it on the other."""
    features = list(arm.features)
    booster = fit(split, features)
    scores = vlm_scores(booster, split.test, features)
    logger.info("%s: %s", arm.name, split.describe())
    return read_out(
        arm.name,
        len(features),
        arm.routing_ms_per_page,
        scores,
        split.test[split.label].to_numpy(),
        importances(booster, top=15),
    )


def evaluate_rule(split: Split, name: str, column: str) -> ArmResult:
    """The same read-out for a score nobody trained: a raw library signal used as a routing rule.

    Read on the same held-out documents as the trained arms, so the comparison is not also a
    comparison of row sets, even though an untrained score has no training side to hold out from.
    """
    scores = np.nan_to_num(split.test[column].to_numpy().astype(np.float64))
    return read_out(name, 0, DETECT_MS_PER_PAGE, scores, split.test[split.label].to_numpy(), [])


# ---------------------------------------------------------------------------
# Question B: pdf-inspector as the cheap route
# ---------------------------------------------------------------------------


def page_weighted_budget(scores: np.ndarray, pages: np.ndarray, budget: float) -> float:
    """The share of *pages* a document-level budget actually spends.

    GPU time is charged per rendered page and page counts are heavily skewed (p50 6, p90 38,
    p99 207), so a document budget and a page budget are not the same number. Everything downstream
    of GPU hours reads this one.
    """
    threshold = float(np.quantile(scores, 1.0 - budget))
    return float(pages[scores >= threshold].sum() / pages.sum())


def gpu_hours(page_fraction: float) -> float:
    """Crawl-wide VLM GPU hours implied by a page budget."""
    return page_fraction * CRAWL_PAGES / 1e6 * VLM_GPU_HOURS_PER_MILLION_PAGES


@dataclass(frozen=True)
class CheapRoute:
    """One cheap route evaluated as the pipeline's default: its label and its per-page cost."""

    name: str
    label_column: str
    cheap_ms_per_page: float


@dataclass
class RouteResult:
    """A cheap route under one router: the frontier, in documents, pages and GPU hours."""

    route: CheapRoute
    router: str
    split: Split
    scores: np.ndarray
    truth: np.ndarray
    pages: np.ndarray
    points: list[RoutePoint]

    def summary(self) -> dict:
        return {
            "route": self.route.name,
            "router": self.router,
            "label": self.route.label_column,
            "cheap_ms_per_page": self.route.cheap_ms_per_page,
            "documents_held_out": int(self.truth.size),
            "positive_rate": float(self.truth.mean()),
            "loss_without_router": float(1.0 - self.truth.mean()),
            "points": {budget: asdict(at_budget(self.points, budget)) for budget in REPORTED_BUDGETS},
            "page_budget": {
                budget: page_weighted_budget(self.scores, self.pages, budget) for budget in REPORTED_BUDGETS
            },
        }


def evaluate_cheap_route(base: Split, route: CheapRoute, arm: Arm) -> RouteResult:
    """Fit a router for one cheap route's own label and read its frontier on unseen domains.

    The split is handed in and only its label is swapped, so the two cheap routes are compared on
    exactly the same held-out documents rather than on two draws of the same procedure.
    """
    features = list(arm.features)
    split = replace(base, label=route.label_column)
    booster = fit(split, features)
    scores = vlm_scores(booster, split.test, features)
    truth = split.test[route.label_column].to_numpy()
    pages = split.test["num_pages"].to_numpy().astype(np.float64)
    logger.info("cheap route %s: %s", route.name, split.describe())
    return RouteResult(route, arm.name, split, scores, truth, pages, frontier(scores, truth))


def equal_quality(result: RouteResult, target_loss: float) -> dict | None:
    """What this cheap route has to spend to reach a given quality loss, in documents and GPU hours."""
    point = budget_for_quality(result.points, target_loss)
    if point is None:
        return None
    page_fraction = page_weighted_budget(result.scores, result.pages, point.vlm_fraction)
    return {
        "target_quality_loss": target_loss,
        "document_budget": point.vlm_fraction,
        "quality_loss": point.quality_loss,
        "page_budget": page_fraction,
        "gpu_hours": gpu_hours(page_fraction),
    }


def per_page_routing(result: RouteResult, budget: float) -> dict:
    """What page-level routing on ``pages_needing_ocr`` could buy over routing whole documents.

    Marin routes whole documents: a 200-page report with three scanned inserts costs either a full
    VLM pass or three silently lost pages. pdf-inspector names the pages it cannot read, so the
    obvious refinement is to send those pages and leave the rest on the cheap route.

    The page budget each scheme spends is measured exactly. The quality it costs is *bracketed*
    rather than measured, because this table carries per-document agreement and not per-page: within
    a document with ``destroyed`` badly-read pages and ``flagged`` pages the library wants OCR'd,
    the best case is that the flagged pages cover the destroyed ones and the worst case is that the
    two sets are disjoint. A point estimate needs per-page agreement columns the study does not
    carry, and the bracket is wide enough that the conclusion has to survive both ends of it.

    Three schemes over the same documents, all charged in pages:

    * whole document, what ships today -- every page of a routed document is rendered;
    * hybrid -- the document router still selects, but only the flagged pages of a selected document
      are rendered;
    * flagged only -- no document router at all, every flagged page in the corpus is rendered.
    """
    test = result.split.test
    pages = result.pages
    # ``pages_compared`` is the alignment's page count, which is what the destroyed-page fraction is
    # a fraction of; page budgets are charged against the PDF's own page count.
    compared = np.nan_to_num(test["inspector_vlm_pages_compared"].cast(pl.Float64).to_numpy())
    destroyed = np.nan_to_num(test["inspector_vlm_frac_pages_bigram_below_50"].cast(pl.Float64).to_numpy()) * compared
    flagged = np.minimum(np.nan_to_num(test["inspector_detect_pages_needing_ocr"].cast(pl.Float64).to_numpy()), pages)

    threshold = float(np.quantile(result.scores, 1.0 - budget))
    to_vlm = result.scores >= threshold
    kept = ~to_vlm
    total = pages.sum()
    unrescued_best = np.maximum(destroyed - flagged, 0.0)
    unrescued_worst = np.minimum(destroyed, pages - flagged)

    return {
        "document_budget": float(to_vlm.mean()),
        "page_budget_whole_document": float(pages[to_vlm].sum() / total),
        "page_budget_hybrid": float(flagged[to_vlm].sum() / total),
        "page_budget_flagged_only": float(flagged.sum() / total),
        "destroyed_page_share": float(destroyed.sum() / total),
        "destroyed_left_whole_document": float(destroyed[kept].sum() / total),
        "destroyed_left_hybrid_best_case": float((destroyed[kept].sum() + unrescued_best[to_vlm].sum()) / total),
        "destroyed_left_hybrid_worst_case": float((destroyed[kept].sum() + unrescued_worst[to_vlm].sum()) / total),
        "destroyed_left_flagged_only_best_case": float(unrescued_best.sum() / total),
        "destroyed_left_flagged_only_worst_case": float(unrescued_worst.sum() / total),
        "flagged_destroyed_correlation": float(
            np.corrcoef(flagged / np.maximum(pages, 1), destroyed / np.maximum(pages, 1))[0, 1]
        ),
        "page_count_quantiles": {str(q): float(np.quantile(pages, q)) for q in (0.5, 0.9, 0.99, 1.0)},
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def frontier_table(result: ArmResult) -> str:
    lines = [f"  {'budget':>7} {'loss':>8} {'catches':>8} {'marginal':>9} {'per rescue':>11}"]
    for budget in REPORTED_BUDGETS:
        point = result.points[budget]
        precision = result.marginal[budget]
        rescue = 1.0 / precision if precision and precision > 0 else float("inf")
        lines.append(
            f"  {point['vlm_fraction']:>7.1%} {point['quality_loss']:>8.4f} "
            f"{point['recall_of_bad']:>8.1%} {precision:>9.2f} {rescue:>11.1f}"
        )
    return "\n".join(lines)


def report(results: dict) -> str:
    lines: list[str] = [
        f"documents: {results['documents']} usable of {results['joined']} joined, " f"{results['domains']} domains",
        "label base rates: " + ", ".join(f"{name} {value:.4f}" for name, value in results["base_rates"].items()),
        "",
        "== Question A: pdf-inspector as a feature source (label docling_ok) ==",
        f"{'arm':<40} {'ms/pg':>7} {'@29%':>8} {'@50%':>8} {'vs 0.1230':>10} {'clump':>7}",
    ]
    for arm in results["question_a"]:
        loss29 = arm["points"][INCUMBENT_BUDGET]["quality_loss"]
        loss50 = arm["points"][SHIPPED_BUDGET]["quality_loss"]
        lines.append(
            f"{arm['name']:<40} {arm['routing_ms_per_page']:>7.2f} {loss29:>8.4f} {loss50:>8.4f} "
            f"{1 - loss50 / SHIPPED_QUALITY_LOSS:>+9.1%} {arm['clumping']['largest_clump_share']:>7.1%}"
        )
    for arm in results["question_a"]:
        lines.append(f"\n{arm['name']} ({arm['documents_held_out']} held-out documents)")
        lines.append(frontier_table(ArmResult(**arm)))
        lines.append(f"  clumping: {arm['clumping']}")
        if arm["gains"]:
            lines.append("  gain: " + ", ".join(f"{i + 1}. {n}" for i, (n, _) in enumerate(arm["gains"][:12])))

    lines.append("\n== Question B: pdf-inspector as the cheap route ==")
    for entry in results["question_b"]["routes"]:
        lines.append(
            f"\n{entry['route']} cheap route ({entry['label']}), router {entry['router']}: "
            f"positive rate {entry['positive_rate']:.4f}, unrouted loss {entry['loss_without_router']:.4f}"
        )
        for budget in REPORTED_BUDGETS:
            point = entry["points"][budget]
            page = entry["page_budget"][budget]
            lines.append(
                f"  {point['vlm_fraction']:>6.1%} docs / {page:>6.1%} pages -> "
                f"loss {point['quality_loss']:.4f}, {gpu_hours(page):>8.0f} GPU-h"
            )
    lines.append("\nequal-quality comparison:")
    lines.append(json.dumps(results["question_b"]["equal_quality"], indent=2))
    lines.append("\nper-page routing:")
    lines.append(json.dumps(results["question_b"]["per_page"], indent=2))
    return "\n".join(lines)


def main() -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")

    frame = joined(fs)
    reasons = ocr_reason_names(frame)
    logger.info("ocr reasons: %s", reasons)
    frame = with_features(frame, reasons)

    route_signals = [name for name in FEATURE_NAMES if name in frame.columns]
    detect = list(DETECT_FEATURES) + [
        f"inspector_reason_{name}" for name in reasons if f"inspector_reason_{name}" not in DETECT_FEATURES
    ]
    extract = list(EXTRACT_FEATURES)

    rows = usable(frame, route_signals)
    logger.info("usable %d rows, %d domains", rows.height, rows["domain"].n_unique())

    arms = [
        Arm("route_features (shipped)", tuple(route_signals), ROUTE_FEATURES_MS_PER_PAGE),
        Arm("inspector detect", tuple(detect), DETECT_MS_PER_PAGE),
        Arm("inspector detect + extract", tuple(detect + extract), EXTRACT_MS_PER_PAGE),
        Arm("route_features + detect", tuple(route_signals + detect), ROUTE_FEATURES_MS_PER_PAGE + DETECT_MS_PER_PAGE),
        Arm(
            "route_features + detect + extract",
            tuple(route_signals + detect + extract),
            ROUTE_FEATURES_MS_PER_PAGE + EXTRACT_MS_PER_PAGE,
        ),
    ]

    # One split for every Question A arm, so the arms differ only in what they may read.
    split = split_by(rows, "domain", "docling_ok")
    question_a = [
        asdict(evaluate_rule(split, "rule: detect pages_needing_ocr fraction", "inspector_detect_ocr_page_fraction")),
        *(asdict(evaluate(split, arm)) for arm in arms),
    ]

    # Question B routes the pipeline on the cheap route's own label, with the cheapest router that
    # Question A justifies -- route features plus the detect tier.
    router = Arm(
        "route_features + detect", tuple(route_signals + detect), ROUTE_FEATURES_MS_PER_PAGE + DETECT_MS_PER_PAGE
    )
    routes = [
        CheapRoute("docling", "docling_ok_same_metric", DOCLING_MS_PER_PAGE),
        CheapRoute("inspector", "inspector_ok", EXTRACT_MS_PER_PAGE),
        # The same two under the unigram floor, reported for contrast only: reading order is one of
        # the properties the router exists to protect, so the decision is made on the bigram pair.
        CheapRoute("docling (unigram floor)", "docling_ok_unigram", DOCLING_MS_PER_PAGE),
        CheapRoute("inspector (unigram floor)", "inspector_ok_unigram", EXTRACT_MS_PER_PAGE),
    ]
    evaluated = {cheap.name: evaluate_cheap_route(split, cheap, router) for cheap in routes}

    docling = evaluated["docling"]
    inspector = evaluated["inspector"]
    docling_point = at_budget(docling.points, SHIPPED_BUDGET)
    docling_pages = page_weighted_budget(docling.scores, docling.pages, SHIPPED_BUDGET)
    unigram_target = at_budget(evaluated["docling (unigram floor)"].points, SHIPPED_BUDGET).quality_loss

    results = {
        "documents": rows.height,
        "joined": frame.height,
        "domains": rows["domain"].n_unique(),
        "base_rates": {
            name: float(rows[name].mean())
            for name in (
                "docling_ok",
                "docling_ok_same_metric",
                "inspector_ok",
                "docling_ok_unigram",
                "inspector_ok_unigram",
                "trustworthy",
            )
        },
        "question_a": question_a,
        "question_b": {
            "routes": [result.summary() for result in evaluated.values()],
            "equal_quality": {
                "docling_at_shipped_budget": {
                    "document_budget": docling_point.vlm_fraction,
                    "quality_loss": docling_point.quality_loss,
                    "page_budget": docling_pages,
                    "gpu_hours": gpu_hours(docling_pages),
                },
                "inspector_at_equal_quality": equal_quality(inspector, docling_point.quality_loss),
                "inspector_at_shipped_budget": {
                    "document_budget": at_budget(inspector.points, SHIPPED_BUDGET).vlm_fraction,
                    "quality_loss": at_budget(inspector.points, SHIPPED_BUDGET).quality_loss,
                },
                "inspector_at_equal_quality_unigram": equal_quality(
                    evaluated["inspector (unigram floor)"], unigram_target
                ),
            },
            "per_page": {
                "at_equal_quality": per_page_routing(
                    inspector,
                    (equal_quality(inspector, docling_point.quality_loss) or {}).get("document_budget", SHIPPED_BUDGET),
                ),
                "at_shipped_budget": per_page_routing(inspector, SHIPPED_BUDGET),
            },
        },
    }

    with fs.open(RESULT_PATH, "w") as stream:
        json.dump(results, stream, indent=2, default=float)
    print(report(results))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
