# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate router v2: escalate to the VLM, or keep the document on pdf-inspector.

Router v1 asked whether Docling would read a document the way the VLM would, and answered it with an
agreement metric. Both halves of that have been retired. Docling is gone -- it costs 278 CPU core-h
per million pages against pdf-inspector's 2.1, at corpus-wide quality parity -- and the agreement
label did not survive blind adjudication: on documents ``docling_ok`` called fine, the other route
still won 41-43% of head-to-heads, and the label separated preference by 0.015, which is noise. The
target here is instead a pairwise preference against the rendered page
(:mod:`~experiments.datakit.build_pdf_source.quality.judge_preference_set`), which is the only
ground truth that can rank two extractions rather than measure the distance between them.

**The frontier is drawn in CPU core-hours.** This cluster is CPU-constrained and GPU-rich, so a
GPU-denominated frontier optimizes the resource that is spare. The cost model is
:mod:`~experiments.datakit.build_pdf_source.quality.route_v2_features`: pdf-inspector 2.1 core-h per
million pages, the PyMuPDF router pass 3.4, the VLM's render-and-encode feed path 17.8 on top of
15.6 GPU-h. GPU hours are reported beside every point, and never optimized.

Escalation is charged **per page**, because the feed path and the model are. Page counts are heavily
skewed -- p50 6, p90 38, p99 207 -- so a document budget and a page budget are different numbers and
only one of them is money.

**No budget is assumed.** The published v1 report picked 50% from a frontier whose knee was 45.5%;
this one reports the curve, its knee, and the marginal precision at every point, and leaves the
operating point to be chosen from them. The marginal slope has a direct reading: the share of the
next documents escalated that the VLM would genuinely have read better, and its reciprocal is how
many VLM runs the pipeline buys per document actually rescued.

**Free features are separated from paid ones, and that separation is the experiment.** pdf-inspector
runs on every document regardless, so everything its extraction reports -- and everything measurable
on the text it produced (:mod:`~...build_inspector_output_study`) -- costs the router nothing. The
PyMuPDF pass costs 3.4 core-h per million pages. If the free groups match it, the router pass should
be deleted, and that is worth more than any threshold choice on this curve.

**Two gates are arithmetic and stay out of the model.** A page rendered below the legibility floor
cannot be read by the VLM whatever a score says, and a document pdf-inspector failed outright has no
cheap route to keep. :func:`gate_report` prices the first against the alternative of raising the
visual-token budget instead of skipping the page.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-route-v2 --extra pdf \\
        --cpu 16 --memory 48GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.analyze_route_v2
"""

import json
import logging
from dataclasses import asdict, dataclass

import fsspec
import numpy as np
import polars as pl
import xgboost as xgb
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.ocr_extract.render import DEFAULT_MAX_VISUAL_TOKENS
from experiments.datakit.build_pdf_source.quality import route_v2_features as contract
from experiments.datakit.build_pdf_source.quality.analyze_route_study import read_table
from experiments.datakit.build_pdf_source.quality.build_inspector_output_study import (
    OUTPUT_PREFIX as INSPECTOR_OUTPUT_PREFIX,
)
from experiments.datakit.build_pdf_source.quality.build_inspector_study import (
    OUTPUT_PREFIX as INSPECTOR_STUDY_PREFIX,
)
from experiments.datakit.build_pdf_source.quality.judge_preference_set import (
    CONFIDENCE_COLUMN,
    ESCALATE_COLUMN,
    LABELS_PATH,
)
from experiments.datakit.build_pdf_source.quality.train_route_model import Split, fit, matrix, split_by

logger = logging.getLogger(__name__)

ROUTE_STUDY_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_route_study"
RESULT_PATH = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_route_v2.json"

# Escalation budgets the report quotes, as a share of documents. Swept finely underneath so a
# quantile lands where it is asked for rather than at the nearest point of a coarse grid.
REPORTED_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
BUDGET_GRID = np.round(np.arange(0.002, 1.0, 0.002), 4)
# Marginal precision is a local slope; read over a +-2.5% window of corpus so one tied clump of
# scores does not turn the derivative into a step.
MARGINAL_WINDOW = 0.025

# Page-weighted quality-loss targets the arms are compared at. Holding quality fixed and reading off
# CPU is what prices the router pass; holding budget fixed and reading off quality does not, because
# the arms do not cost the same to run.
EQUAL_QUALITY_TARGETS = (0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01)

# Seeds the domain-disjoint split is redrawn under, to measure how much of a difference between two
# arms is the arms and how much is which domains landed in the test half. The published v1 pass put
# that noise floor at ~0.012 on 89k rows; this label set is smaller, so it is measured again rather
# than inherited.
NOISE_SEEDS = (0, 1, 2, 3, 4)

# Visual-token budgets priced against the legibility gate, as GPU-hours per million pages.
#
# Up to 8192 these are measured (`ocr-budget-sweep.md`, packable lean config). 16384 is
# extrapolated from that curve and flagged as such in the output: the sweep skipped it because the
# 300-DPI upscale cap already binds at 8192 for ordinary paper, so a 16384 arm would have rendered
# near-identical payloads. That reasoning does not hold for the pages this gate is about, which are
# large-format sheets the cap never reaches -- 16384 is the largest budget the render path can
# express at all, since 16384 x 1024 pixels is exactly `render.MAX_PIXELS`.
BUDGET_SWEEP = {1024: 16.2, 2048: 18.3, 4096: 22.8, 8192: 25.2, 16384: 28.0}
EXTRAPOLATED_BUDGETS = (16384,)


# ---------------------------------------------------------------------------
# Reading and labelling
# ---------------------------------------------------------------------------


def joined(fs: fsspec.AbstractFileSystem) -> pl.DataFrame:
    """Every study table joined on ``source_id``, with the preference label attached where it exists.

    A left join on the labels, not an inner one: the label set is 20,000 documents and the corpus is
    100,000, and the unlabelled rows are still needed. The shipped threshold is a quantile of the
    score over the *whole* corpus, and the gate arithmetic is a corpus-wide count.
    """
    route = read_table(ROUTE_STUDY_PREFIX, fs)
    inspector = read_table(INSPECTOR_STUDY_PREFIX, fs)
    output = read_table(INSPECTOR_OUTPUT_PREFIX, fs)
    with fs.open(LABELS_PATH, "rb") as stream:
        labels = pl.read_parquet(stream)
    logger.info(
        "route %d, inspector %d, output %d, labels %d rows",
        route.height,
        inspector.height,
        output.height,
        labels.height,
    )

    # Each table carries its own copy of the document's identity columns; the join keeps one of
    # each. ``domain`` comes from the inspector study because that is where it is derived, and
    # ``num_pages`` and ``pdf_bytes`` from the route study, which is where the render statistics
    # they sit beside were measured.
    frame = (
        route.drop("url", "docling_missing", strict=False)
        .join(inspector.drop("num_pages", "pdf_bytes", "url", strict=False), on="source_id", how="inner")
        .join(output, on="source_id", how="left")
        .join(labels.drop("domain", "trustworthy", strict=False), on="source_id", how="left")
    )
    logger.info(
        "joined %d rows, %d labelled, %d with pdf-inspector output statistics",
        frame.height,
        frame[ESCALATE_COLUMN].is_not_null().sum(),
        frame["inspector_output_alpha_ratio"].is_not_null().sum(),
    )
    return contract.with_derived(frame)


def trainable(frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """The labelled rows a router may learn from, after the arithmetic gates have taken theirs.

    Gated documents are removed rather than labelled because their decision is not the model's to
    make: pdf-inspector producing no text means escalate whatever the score says, and a document the
    VLM cannot resolve means keep. Leaving them in would let the model spend capacity re-deriving
    two rules that are already exact, and would flatter whichever feature set happens to encode
    them -- ``inspector_markdown_chars == 0`` is a single split that captures ~9% of the corpus, and
    an arm that owns that column would take credit for a decision arithmetic already made.
    """
    return frame.filter(
        pl.col(ESCALATE_COLUMN).is_not_null()
        & (pl.col("domain") != "")
        & pl.col("inspector_error").is_null()
        & (pl.col("inspector_markdown_chars") > 0)
        & contract.legible_at_budget(DEFAULT_MAX_VISUAL_TOKENS)
    ).drop_nulls(subset=features)


# ---------------------------------------------------------------------------
# The frontier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutePoint:
    """One operating point, priced in CPU core-hours per million crawl pages.

    Quality loss is reported both ways on purpose. Page-weighted is the corpus number and shares
    units with the cost axis: a 200-page report read badly damages a hundred times more of the
    corpus than a 2-page flyer does. Document-weighted is what the published v1 frontier reported,
    and is carried so the two curves can be read against each other.
    """

    threshold: float
    document_budget: float
    page_budget: float
    quality_loss_pages: float
    quality_loss_documents: float
    wasted_escalation: float
    recall_of_escalations: float
    cpu_core_hours: float
    gpu_hours: float


def frontier(
    scores: np.ndarray, escalate: np.ndarray, pages: np.ndarray, router_core_hours: float, needs_inspector: bool
) -> list[RoutePoint]:
    """Sweep a score by quantile of its own output and price every point.

    Quantiles rather than a fixed threshold grid because the arms' scores live on different scales,
    and because a clumped score has whole budget ranges no threshold can express -- a property of
    the score worth seeing rather than one to hide behind a grid. This is the calibration the
    shipped router uses: the threshold is whatever quantile spends the target budget, so only rank
    ever matters.

    ``needs_inspector`` is the difference between a router that reads pdf-inspector's output and one
    that does not. A router built on the extraction's own signals cannot skip the extraction on a
    document it is about to escalate, so it pays 2.1 core-h on every page; one built only on the
    PyMuPDF pass can skip it, and its escalated pages cost 2.1 less. That is where the ~15.7 core-h
    marginal escalation cost comes from, against 17.8 for the arms that must run pdf-inspector first.
    """
    total_pages = pages.sum()
    bad_pages = pages[escalate].sum()
    points = []
    for budget in BUDGET_GRID:
        threshold = float(np.quantile(scores, 1.0 - budget))
        to_vlm = scores >= threshold
        kept = ~to_vlm
        page_budget = float(pages[to_vlm].sum() / total_pages)
        inspector_pages = 1.0 if needs_inspector else 1.0 - page_budget
        points.append(
            RoutePoint(
                threshold=threshold,
                document_budget=float(to_vlm.mean()),
                page_budget=page_budget,
                quality_loss_pages=float(pages[kept & escalate].sum() / total_pages),
                quality_loss_documents=float((kept & escalate).mean()),
                wasted_escalation=float(pages[to_vlm & ~escalate].sum() / total_pages),
                recall_of_escalations=float(pages[to_vlm & escalate].sum() / max(bad_pages, 1.0)),
                cpu_core_hours=(
                    inspector_pages * contract.INSPECTOR_CORE_HOURS
                    + router_core_hours
                    + page_budget * contract.VLM_FEED_CORE_HOURS
                ),
                gpu_hours=page_budget * contract.VLM_GPU_HOURS,
            )
        )
    return points


def at_budget(points: list[RoutePoint], budget: float) -> RoutePoint:
    return min(points, key=lambda point: abs(point.document_budget - budget))


def marginal_precision(points: list[RoutePoint], budget: float) -> float:
    """The share of the next pages escalated that pdf-inspector would genuinely have read worse."""
    low = at_budget(points, budget - MARGINAL_WINDOW)
    high = at_budget(points, budget + MARGINAL_WINDOW)
    span = high.page_budget - low.page_budget
    if span <= 0:
        return float("nan")
    return (low.quality_loss_pages - high.quality_loss_pages) / span


def knee(points: list[RoutePoint]) -> RoutePoint:
    """The frontier's knee: the point furthest from the chord joining its two endpoints.

    Computed on the CPU-cost axis rather than the document-budget one, because that is the axis this
    router is being bought on and the two curves bend in different places.
    """
    first, last = points[0], points[-1]
    span_x = last.cpu_core_hours - first.cpu_core_hours
    span_y = last.quality_loss_pages - first.quality_loss_pages
    scale = np.hypot(span_x, span_y) or 1.0

    def distance(point: RoutePoint) -> float:
        return (
            abs(
                span_x * (first.quality_loss_pages - point.quality_loss_pages)
                - (first.cpu_core_hours - point.cpu_core_hours) * span_y
            )
            / scale
        )

    return max(points, key=distance)


def clumping(scores: np.ndarray) -> dict:
    """Whether a score can rank at all, or piles documents onto a handful of values.

    Two prior scores failed exactly here. The incumbent FinePDFs rule pinned 17.4% of documents at
    exactly 1.0, which made its frontier locally degenerate -- inside that band the marginal document
    it added was no likelier to need the VLM than one drawn at random. The untrained
    ``pages_needing_ocr`` rule scored 0.4201 at *every* budget because 91.3% of documents tie at
    exactly 0.0. A frontier over a clumped score is not a frontier, so every arm reports this.
    """
    values, counts = np.unique(np.round(scores.astype(np.float64), 9), return_counts=True)
    return {
        "distinct_values": int(values.size),
        "largest_clump_share": float(counts.max() / scores.size),
        "largest_clump_value": float(values[counts.argmax()]),
        "tied_share": float(counts[counts > 1].sum() / scores.size),
        "share_at_max": float(counts[-1] / scores.size),
    }


def confusion(points: list[RoutePoint], scores: np.ndarray, escalate: np.ndarray, budget: float) -> dict:
    """The four outcomes at one operating point, in documents.

    The two errors are not symmetric and the report should not present them as if they were. An
    escalation the cheap route would have handled costs CPU and GPU and is recoverable. A document
    left on pdf-inspector that the VLM would have read better lands in training data degraded, and
    nothing downstream flags it.
    """
    threshold = at_budget(points, budget).threshold
    to_vlm = scores >= threshold
    return {
        "budget": float(to_vlm.mean()),
        "escalated_correctly": int((to_vlm & escalate).sum()),
        "escalated_wastefully": int((to_vlm & ~escalate).sum()),
        "kept_correctly": int((~to_vlm & ~escalate).sum()),
        "kept_and_degraded": int((~to_vlm & escalate).sum()),
        "precision": float(escalate[to_vlm].mean()) if to_vlm.any() else float("nan"),
        "recall": float(to_vlm[escalate].mean()) if escalate.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# What the gates take before the score sees anything
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusShare:
    """How the corpus divides between the two gates and the documents the score actually ranks.

    The learned score is trained and evaluated on the routable remainder, so a budget quoted against
    it is a share of that remainder and not of the corpus. That distinction is not cosmetic here:
    the documents pdf-inspector returns no text for are ~12% of the corpus and every one of them is
    a certain escalation, so they spend VLM budget before the score allocates a page.
    """

    documents: int
    corpus_pages: float
    forced_escalate_documents: int
    forced_escalate_page_share: float
    forced_keep_documents: int
    forced_keep_page_share: float
    unreadable_by_either_documents: int
    routable_documents: int
    routable_page_share: float


def corpus_share(frame: pl.DataFrame) -> CorpusShare:
    """Split the corpus into forced escalations, forced keeps, and the routable remainder."""
    pages = frame["num_pages"].cast(pl.Float64)
    total = float(pages.sum())
    no_text = pl.col("inspector_error").is_not_null() | (pl.col("inspector_markdown_chars") == 0)
    illegible = contract.legible_at_budget(DEFAULT_MAX_VISUAL_TOKENS).not_()

    # Legibility is checked first, and the order is the decision. A document can be both -- ~600 of
    # them are -- and the two gates then disagree: one says escalate because there is no cheap text,
    # the other says the VLM cannot read the page either. The second is the stronger statement, so
    # it wins. Escalating a document rendered below the floor buys a transcription of a blur at full
    # render and GPU price. Neither route can read these; they are candidates for tiling or for
    # exclusion from the corpus, and that decision sits above the router.
    keep_forced = frame.filter(illegible)
    escalate_forced = frame.filter(illegible.not_() & no_text)
    routable = frame.filter(illegible.not_() & no_text.not_())
    return CorpusShare(
        documents=frame.height,
        corpus_pages=total,
        forced_escalate_documents=escalate_forced.height,
        forced_escalate_page_share=float(escalate_forced["num_pages"].cast(pl.Float64).sum()) / total,
        forced_keep_documents=keep_forced.height,
        forced_keep_page_share=float(keep_forced["num_pages"].cast(pl.Float64).sum()) / total,
        unreadable_by_either_documents=frame.filter(illegible & no_text).height,
        routable_documents=routable.height,
        routable_page_share=float(routable["num_pages"].cast(pl.Float64).sum()) / total,
    )


def corpus_cost(point: RoutePoint, share: CorpusShare, router_core_hours: float) -> dict:
    """One held-out operating point restated against the whole corpus rather than the remainder.

    ``needs_inspector`` does not appear, and its absence is a result. The gate that removes ~12% of
    the corpus from the score's reach is "pdf-inspector produced no text", which is only knowable
    after running pdf-inspector. Any pipeline using that gate therefore pays the 2.1 core-h on every
    page whatever its router reads, so the marginal cost of escalating a page is the full 17.8 core-h
    of the feed path, not the 17.8 - 2.1 = 15.7 that a router able to skip the extraction would pay.
    Dropping to 15.7 means giving up both the free feature groups and the empty-extraction gate.
    """
    escalated = share.forced_escalate_page_share + point.page_budget * share.routable_page_share
    return {
        "learned_budget_of_routable": point.document_budget,
        "corpus_page_budget": escalated,
        "forced_share_of_escalation": share.forced_escalate_page_share / escalated if escalated else 0.0,
        "cpu_core_hours": contract.INSPECTOR_CORE_HOURS + router_core_hours + escalated * contract.VLM_FEED_CORE_HOURS,
        "gpu_hours": escalated * contract.VLM_GPU_HOURS,
        "crawl_core_hours": (
            (contract.INSPECTOR_CORE_HOURS + router_core_hours + escalated * contract.VLM_FEED_CORE_HOURS)
            * contract.CRAWL_PAGES
            / 1e6
        ),
    }


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One candidate router: the feature groups it may read, and what reading them costs."""

    name: str
    groups: tuple[str, ...]

    @property
    def features(self) -> list[str]:
        return contract.columns_for(self.groups)

    @property
    def core_hours(self) -> float:
        return contract.cost_of(self.groups)

    @property
    def needs_inspector(self) -> bool:
        """Whether escalated documents still have to be run through pdf-inspector first."""
        return any(group.startswith("inspector") for group in self.groups)


ARMS = (
    Arm("free (inspector output + extract + shape)", ("inspector_extract", "inspector_output", "document_shape")),
    Arm("free + detect", ("inspector_extract", "inspector_output", "document_shape", "inspector_detect")),
    Arm("paid page_signals only", ("page_signals",)),
    Arm("free + page_signals", ("inspector_extract", "inspector_output", "document_shape", "page_signals")),
    Arm("everything", tuple(group.name for group in contract.GROUPS)),
)


def escalation_scores(booster: xgb.Booster, frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Probability that a document should be escalated -- the label's own direction, unflipped."""
    return booster.predict(xgb.DMatrix(matrix(frame, features), feature_names=features))


def group_gains(booster: xgb.Booster, features: list[str]) -> dict:
    """Total gain per feature and per priced group.

    The per-group total is the number that decides whether the PyMuPDF pass stays: it is the share
    of the model's explanatory work done by signals that cost 3.4 core-h per million pages, against
    signals that cost nothing.
    """
    gains = {name: float(gain) for name, gain in booster.get_score(importance_type="total_gain").items()}
    total = sum(gains.values()) or 1.0
    by_group: dict[str, float] = {}
    for group in contract.GROUPS:
        share = sum(gains.get(column, 0.0) for column in group.columns)
        if any(column in features for column in group.columns):
            by_group[group.name] = share / total
    return {
        "top_features": sorted(gains.items(), key=lambda item: item[1], reverse=True)[:20],
        "share_by_group": dict(sorted(by_group.items(), key=lambda item: item[1], reverse=True)),
        "free_share": sum(share for name, share in by_group.items() if contract.GROUPS_BY_NAME[name].free),
        "paid_share": sum(share for name, share in by_group.items() if not contract.GROUPS_BY_NAME[name].free),
    }


@dataclass(frozen=True)
class ArmResult:
    name: str
    features: int
    router_core_hours: float
    needs_inspector: bool
    documents_held_out: int
    domains_held_out: int
    points: dict[float, dict]
    marginal: dict[float, float]
    knee: dict
    clumping: dict
    confusion: dict[float, dict]
    gains: dict
    corpus: dict[float, dict]


def read_out(arm: Arm, split: Split, scores: np.ndarray, gains: dict, share: CorpusShare) -> ArmResult:
    """Everything reported about one score on the shared held-out set.

    Reported twice: against the routable remainder the score actually ranks, and against the whole
    corpus once the gates' forced escalations are added back. The second is what the pipeline pays.
    """
    escalate = split.test[ESCALATE_COLUMN].to_numpy().astype(bool)
    pages = split.test["num_pages"].to_numpy().astype(np.float64)
    points = frontier(scores, escalate, pages, arm.core_hours, arm.needs_inspector)
    return ArmResult(
        name=arm.name,
        features=len(arm.features),
        router_core_hours=arm.core_hours,
        needs_inspector=arm.needs_inspector,
        documents_held_out=int(escalate.size),
        domains_held_out=split.test["domain"].n_unique(),
        points={budget: asdict(at_budget(points, budget)) for budget in REPORTED_BUDGETS},
        marginal={budget: marginal_precision(points, budget) for budget in REPORTED_BUDGETS},
        knee=asdict(knee(points)),
        clumping=clumping(scores),
        confusion={budget: confusion(points, scores, escalate, budget) for budget in (0.25, 0.5, 0.75)},
        gains=gains,
        corpus={budget: corpus_cost(at_budget(points, budget), share, arm.core_hours) for budget in REPORTED_BUDGETS},
    )


def evaluate(split: Split, arm: Arm, share: CorpusShare) -> ArmResult:
    """Train the arm on one side of the shared domain-disjoint split, read it on the other."""
    features = arm.features
    booster = fit(split, features)
    logger.info("%s: %d features, %s", arm.name, len(features), split.describe())
    return read_out(arm, split, escalation_scores(booster, split.test, features), group_gains(booster, features), share)


def evaluate_rule(split: Split, name: str, column: str, groups: tuple[str, ...], share: CorpusShare) -> ArmResult:
    """The same read-out for a score nobody trained: a raw signal used directly as a routing rule.

    Read on the same held-out documents as the trained arms, so the comparison is not also a
    comparison of row sets. The point of including these is the clumping column: an untrained rule
    tends to tie most of the corpus at one value, and a frontier over that is a fiction.

    ``groups`` prices the rule rather than describing what it may read. The incumbent's probability
    is not free -- it needs its own PyMuPDF feature pass -- so it is charged as if it were the
    ``page_signals`` pass. That is an approximation and an upper bound on the incumbent's cost; the
    conclusion it feeds does not turn on it, because the incumbent loses on quality at every budget.
    """
    arm = Arm(name, groups)
    scores = np.nan_to_num(split.test[column].cast(pl.Float64).to_numpy())
    empty = {"top_features": [], "share_by_group": {}, "free_share": 0.0, "paid_share": 0.0}
    return read_out(arm, split, scores, empty, share)


def cheapest_at_quality(result: ArmResult, target_loss: float) -> dict | None:
    """The least corpus CPU this arm can spend and still hold page-weighted loss at or below target.

    This is the comparison that prices the router pass, and it is the one a gain ranking cannot
    make. Gain says how much of a model's explanatory work a feature group did; it says nothing
    about whether the same quality was reachable without the group at a lower budget. Two arms
    quoted at equal quality and unequal CPU answer that directly.
    """
    affordable = [
        (budget, point, result.corpus[budget])
        for budget, point in result.points.items()
        if point["quality_loss_pages"] <= target_loss
    ]
    if not affordable:
        return None
    budget, point, corpus = min(affordable, key=lambda item: item[2]["cpu_core_hours"])
    return {
        "target_quality_loss": target_loss,
        "learned_budget": budget,
        "quality_loss_pages": point["quality_loss_pages"],
        "corpus_page_budget": corpus["corpus_page_budget"],
        "cpu_core_hours": corpus["cpu_core_hours"],
        "gpu_hours": corpus["gpu_hours"],
        "crawl_core_hours": corpus["crawl_core_hours"],
    }


def equal_quality_table(arms: list[ArmResult], targets: tuple[float, ...]) -> dict:
    """Every arm's cheapest corpus CPU at each quality target, so the arms differ only in cost."""
    return {target: {arm.name: cheapest_at_quality(arm, target) for arm in arms} for target in targets}


# ---------------------------------------------------------------------------
# How much of a difference is the split rather than the arm
# ---------------------------------------------------------------------------


def noise_floor(rows: pl.DataFrame, arm: Arm, budget: float) -> dict:
    """Refit one arm under several domain splits and report how far its result moves.

    Any difference between two arms smaller than this is a difference between two draws of the same
    procedure. The published v1 pass measured ~0.012 on 89,000 rows; this label set is a fifth of
    that size, so the floor is measured here rather than inherited.
    """
    losses, budgets = [], []
    for seed in NOISE_SEEDS:
        split = split_by(rows, "domain", ESCALATE_COLUMN, seed=20260826 + seed)
        booster = fit(split, arm.features)
        scores = escalation_scores(booster, split.test, arm.features)
        escalate = split.test[ESCALATE_COLUMN].to_numpy().astype(bool)
        pages = split.test["num_pages"].to_numpy().astype(np.float64)
        point = at_budget(frontier(scores, escalate, pages, arm.core_hours, arm.needs_inspector), budget)
        losses.append(point.quality_loss_pages)
        budgets.append(point.page_budget)
    return {
        "arm": arm.name,
        "document_budget": budget,
        "splits": len(NOISE_SEEDS),
        "quality_loss_mean": float(np.mean(losses)),
        "quality_loss_sd": float(np.std(losses, ddof=1)),
        "quality_loss_range": [float(min(losses)), float(max(losses))],
        # The floor a difference has to clear to mean anything: the spread of the same arm over
        # splits, read as the full range rather than as a standard error.
        "noise_floor": float(max(losses) - min(losses)),
        "page_budget_mean": float(np.mean(budgets)),
    }


# ---------------------------------------------------------------------------
# The arithmetic gates
# ---------------------------------------------------------------------------


def gate_report(frame: pl.DataFrame) -> dict:
    """Price the legibility gate against the alternative of rendering those pages bigger.

    A page below :data:`~...render.DEFAULT_LEGIBILITY_FLOOR_DPI` is one the VLM cannot read, so
    escalating it buys a transcription of a blur. Skipping it is one answer. Raising the visual-token
    budget is the other, and it is tempting here because GPU is the resource this cluster has spare
    -- but the budget is global, so rescuing the illegible pages pays the throughput cost on every
    page in the corpus. Both are priced, plus the third option nobody has costed: raising the budget
    only for the documents that need it, which is what makes the trade affordable.
    """
    pages = frame["num_pages"].cast(pl.Float64)
    total_pages = float(pages.sum())
    below = frame["pages_below_legibility_floor"].cast(pl.Float64)
    # The gated set is the one the gate's own arithmetic names, not the one the extraction's
    # per-page counter found. They are close but not identical -- a document can have one oversized
    # insert below the floor and a mean above it -- and using the counter as the baseline while
    # using the mean to decide what a larger budget rescues would report documents as "rescued" at
    # the budget they are already gated out of. The discrepancy is reported rather than smoothed.
    affected = frame.filter(contract.legible_at_budget(DEFAULT_MAX_VISUAL_TOKENS).not_())
    affected_pages = float(affected["num_pages"].cast(pl.Float64).sum())

    baseline_gpu = BUDGET_SWEEP[DEFAULT_MAX_VISUAL_TOKENS]
    options = {}
    for budget, gpu_hours in sorted(BUDGET_SWEEP.items()):
        rescued = affected.filter(contract.legible_at_budget(budget))
        rescued_pages = float(rescued["num_pages"].cast(pl.Float64).sum())
        options[budget] = {
            "median_dpi": contract.dpi_at_budget(146.0, budget),
            "throughput_measured": budget not in EXTRAPOLATED_BUDGETS,
            "gpu_hours_per_million_pages": gpu_hours,
            "documents_made_legible": rescued.height,
            "pages_made_legible": rescued_pages,
            "share_of_corpus_pages_rescued": rescued_pages / total_pages,
            # Raising the budget globally charges every escalated page the higher rate.
            "gpu_cost_multiplier_global": gpu_hours / baseline_gpu,
            # Raising it only for the documents that need it charges the higher rate on their pages
            # alone, which is what makes the option worth separating from the global one.
            "gpu_cost_multiplier_targeted": 1.0 + (gpu_hours / baseline_gpu - 1.0) * affected_pages / total_pages,
        }

    # Whether the gate is right is checkable rather than assertable. A document the VLM cannot
    # resolve should be one a judge, looking at the same rendered page, declined to escalate; and a
    # document pdf-inspector lost outright should be one the judge escalated. Both gates are stated
    # as facts about the pipeline, so they are worth confirming against the label that disagrees
    # with them freely.
    illegible_labelled = frame.filter(
        (pl.col("pages_below_legibility_floor") > 0)
        & contract.legible_at_budget(DEFAULT_MAX_VISUAL_TOKENS).not_()
        & pl.col(ESCALATE_COLUMN).is_not_null()
    )
    no_text = pl.col("inspector_error").is_not_null() | (pl.col("inspector_markdown_chars") == 0)
    failed_labelled = frame.filter(no_text & pl.col(ESCALATE_COLUMN).is_not_null())

    return {
        "documents": frame.height,
        "corpus_pages": total_pages,
        "documents_gated_illegible": affected.height,
        "gated_page_share": affected_pages / total_pages,
        "documents_with_any_page_below_floor": int((frame["pages_below_legibility_floor"] > 0).sum()),
        "documents_entirely_below_floor": (
            frame.filter(
                (pl.col("pages_below_legibility_floor") > 0)
                & (pl.col("pages_below_legibility_floor") >= pl.col("num_pages"))
            ).height
        ),
        "pages_below_floor": float(below.sum()),
        "pages_below_floor_share": float(below.sum()) / total_pages,
        "inspector_library_failures": int(frame["inspector_error"].is_not_null().sum()),
        "inspector_no_text": int(frame.filter(no_text).height),
        "gate_agreement": {
            "illegible_documents_labelled": illegible_labelled.height,
            "illegible_escalate_rate": (
                float(illegible_labelled[ESCALATE_COLUMN].mean()) if illegible_labelled.height else float("nan")
            ),
            "inspector_no_text_documents_labelled": failed_labelled.height,
            "inspector_no_text_escalate_rate": (
                float(failed_labelled[ESCALATE_COLUMN].mean()) if failed_labelled.height else float("nan")
            ),
        },
        "budget_options": options,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def frontier_table(result: ArmResult) -> str:
    lines = [
        f"  {'docs':>6} {'pages':>7} {'loss/pg':>8} {'loss/doc':>9} {'catches':>8} {'marg':>6} "
        f"{'per resc':>8} | {'corpus pg':>10} {'core-h/M':>9} {'GPU-h/M':>8} {'crawl core-h':>13}"
    ]
    for budget in REPORTED_BUDGETS:
        point = result.points[budget]
        corpus = result.corpus[budget]
        precision = result.marginal[budget]
        rescue = 1.0 / precision if precision and precision > 0 else float("inf")
        lines.append(
            f"  {point['document_budget']:>6.1%} {point['page_budget']:>7.1%} "
            f"{point['quality_loss_pages']:>8.4f} {point['quality_loss_documents']:>9.4f} "
            f"{point['recall_of_escalations']:>8.1%} {precision:>6.2f} {rescue:>8.1f} | "
            f"{corpus['corpus_page_budget']:>10.1%} {corpus['cpu_core_hours']:>9.1f} "
            f"{corpus['gpu_hours']:>8.1f} {corpus['crawl_core_hours']:>13.0f}"
        )
    return "\n".join(lines)


def report(results: dict) -> str:
    lines = [
        f"labelled {results['labelled_documents']} documents over {results['labelled_domains']} domains; "
        f"trainable {results['trainable_documents']} over {results['trainable_domains']}",
        f"escalation base rate {results['escalate_rate']:.4f} (page-weighted {results['escalate_rate_pages']:.4f})"
        " over the routable remainder",
        "corpus split: "
        f"{results['corpus_share']['forced_escalate_page_share']:.2%} of pages forced to the VLM "
        f"({results['corpus_share']['forced_escalate_documents']} documents pdf-inspector read nothing from), "
        f"{results['corpus_share']['forced_keep_page_share']:.2%} forced to stay "
        f"({results['corpus_share']['forced_keep_documents']} below the legibility floor), "
        f"{results['corpus_share']['routable_page_share']:.2%} routable",
        f"noise floor: {results['noise_floor']['noise_floor']:.4f} page-weighted quality loss "
        f"over {results['noise_floor']['splits']} domain splits at "
        f"{results['noise_floor']['document_budget']:.0%} of documents",
        "",
        f"{'arm':<46} {'core-h':>7} {'@25%':>8} {'@50%':>8} {'@70%':>8} {'clump':>7} {'free gain':>10}",
    ]
    for arm in results["arms"]:
        lines.append(
            f"{arm['name']:<46} {arm['router_core_hours']:>7.2f} "
            f"{arm['points'][0.25]['quality_loss_pages']:>8.4f} "
            f"{arm['points'][0.50]['quality_loss_pages']:>8.4f} "
            f"{arm['points'][0.70]['quality_loss_pages']:>8.4f} "
            f"{arm['clumping']['largest_clump_share']:>7.1%} {arm['gains']['free_share']:>10.1%}"
        )
    for arm in results["arms"]:
        lines.append(f"\n{arm['name']} ({arm['documents_held_out']} documents, {arm['domains_held_out']} domains)")
        lines.append(frontier_table(ArmResult(**arm)))
        lines.append(
            f"  knee: {arm['knee']['document_budget']:.1%} of documents / "
            f"{arm['knee']['page_budget']:.1%} of pages, "
            f"{arm['knee']['cpu_core_hours']:.1f} core-h/M, loss {arm['knee']['quality_loss_pages']:.4f}"
        )
        lines.append(f"  clumping: {arm['clumping']}")
        if arm["gains"]["share_by_group"]:
            lines.append(
                "  gain by group: "
                + ", ".join(f"{name} {share:.1%}" for name, share in arm["gains"]["share_by_group"].items())
            )
            lines.append("  top: " + ", ".join(name for name, _ in arm["gains"]["top_features"][:12]))
    lines.append("\n== equal quality, priced in corpus CPU core-hours per million pages ==")
    for target, byarm in results["equal_quality"].items():
        reached = {name: row for name, row in byarm.items() if row}
        if not reached:
            continue
        best = min(reached.values(), key=lambda row: row["cpu_core_hours"])["cpu_core_hours"]
        lines.append(f"  loss <= {target:.2f}:")
        for name, row in sorted(reached.items(), key=lambda item: item[1]["cpu_core_hours"]):
            lines.append(
                f"    {name:<46} {row['cpu_core_hours']:>7.2f} core-h/M "
                f"({row['cpu_core_hours'] - best:+.2f}), {row['corpus_page_budget']:>6.1%} of pages, "
                f"{row['crawl_core_hours']:>7.0f} crawl core-h"
            )
        unreached = sorted(name for name, row in byarm.items() if not row)
        if unreached:
            lines.append(f"    cannot reach at any budget: {', '.join(unreached)}")
    lines.append("\n== gates ==")
    lines.append(json.dumps(results["gates"], indent=2))
    return "\n".join(lines)


def main() -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")

    frame = joined(fs)
    every_feature = contract.columns_for(tuple(group.name for group in contract.GROUPS))
    present = [name for name in every_feature if name in frame.columns]
    missing = sorted(set(every_feature) - set(present))
    if missing:
        raise RuntimeError(f"feature contract declares columns the study tables do not carry: {missing}")

    labelled = frame.filter(pl.col(ESCALATE_COLUMN).is_not_null())
    rows = trainable(frame, present)
    logger.info("trainable %d rows, %d domains", rows.height, rows["domain"].n_unique())

    share = corpus_share(frame)
    logger.info(
        "corpus: %d documents, %.1f%% forced-escalate pages (no text), %.1f%% forced-keep pages "
        "(below the legibility floor), %.1f%% routable",
        share.documents,
        100 * share.forced_escalate_page_share,
        100 * share.forced_keep_page_share,
        100 * share.routable_page_share,
    )

    split = split_by(rows, "domain", ESCALATE_COLUMN)
    arms = [
        evaluate_rule(split, "rule: incumbent FinePDFs ocr_prob", "ocr_prob", ("page_signals",), share),
        evaluate_rule(
            split,
            "rule: inspector pages_needing_ocr fraction",
            "inspector_extract_ocr_page_fraction",
            ("inspector_extract",),
            share,
        ),
        *(evaluate(split, arm, share) for arm in ARMS),
    ]

    pages = rows["num_pages"].cast(pl.Float64).to_numpy()
    escalate = rows[ESCALATE_COLUMN].to_numpy().astype(bool)
    results = {
        "labelled_documents": labelled.height,
        "labelled_domains": labelled["domain"].n_unique(),
        "trainable_documents": rows.height,
        "trainable_domains": rows["domain"].n_unique(),
        # Coverage of the output-statistics table, because the free feature groups depend on it and
        # a partial table silently shrinks the trainable set rather than failing.
        "inspector_output_coverage": float(frame["inspector_output_alpha_ratio"].is_not_null().mean()),
        "escalate_rate": float(escalate.mean()),
        "escalate_rate_pages": float(pages[escalate].sum() / pages.sum()),
        "label_sources": dict(labelled["label_source"].value_counts().iter_rows()),
        "graded_target_available": int(labelled[CONFIDENCE_COLUMN].is_not_null().sum()),
        "cost_model": {
            "inspector_core_hours": contract.INSPECTOR_CORE_HOURS,
            "route_features_core_hours": contract.ROUTE_FEATURES_CORE_HOURS,
            "vlm_feed_core_hours": contract.VLM_FEED_CORE_HOURS,
            "vlm_gpu_hours": contract.VLM_GPU_HOURS,
        },
        "corpus_share": asdict(share),
        "arms": [asdict(arm) for arm in arms],
        "noise_floor": noise_floor(rows, ARMS[-1], 0.50),
        # Quality targets spanning the useful part of the curve, set from the arms' own reach rather
        # than from round numbers, so no target is unreachable for every arm at once.
        "equal_quality": equal_quality_table(arms, EQUAL_QUALITY_TARGETS),
        "gates": gate_report(frame),
    }
    with fs.open(RESULT_PATH, "w") as stream:
        json.dump(results, stream, indent=2, default=float)
    print(report(results))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
