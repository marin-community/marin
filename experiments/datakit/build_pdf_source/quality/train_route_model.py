# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit and score a candidate router against the incumbent on the routing study table.

Two things make this more than a call to XGBoost.

**The split is by content, not by row.** The crawl holds ~9.8% exact-duplicate PDFs and many more
near-duplicates from the same publisher, so a random row split leaks: the same document lands in
both halves and the test score measures memorization. Rows are split on ``content_digest`` and, for
the stricter variant, on the URL's registered domain -- a router that only works on domains it has
seen is not a router.

**The comparison is cost-matched.** The incumbent and the candidate are compared at the same VLM
budget rather than at their own preferred thresholds, because a router that routes more documents
to the VLM will always look better on quality and always cost more. See
:mod:`~experiments.datakit.build_pdf_source.quality.analyze_route_study` for the frontier this reads from.

The candidate is deliberately a small, shallow booster. The feature set is ~70 numbers per document
derived from 8 sampled pages, the label is noisy, and the decision is a threshold on one
probability; depth here buys overfitting rather than routing accuracy.
"""

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import polars as pl
import xgboost as xgb

from experiments.datakit.build_pdf_source.quality.analyze_route_study import (
    INCUMBENT_THRESHOLD,
    RoutePoint,
    point_at_budget,
    route_frontier,
)

logger = logging.getLogger(__name__)

TEST_FRACTION = 0.25
SPLIT_SEED = 20260806

BOOSTER_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "nthread": 8,
}
BOOST_ROUNDS = 400
EARLY_STOPPING_ROUNDS = 30

THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 3)


def registered_domain(url: str | None) -> str:
    """The host of a URL, which is the unit near-duplicate documents cluster in."""
    if not url:
        return ""
    return (urlparse(url).hostname or "").lower()


@dataclass(frozen=True)
class Split:
    """A content-disjoint train/test split of the study table."""

    train: pl.DataFrame
    test: pl.DataFrame
    key: str

    def describe(self) -> str:
        return (
            f"{self.key}-disjoint split: {self.train.height} train / {self.test.height} test, "
            f"positive rate {self.train['docling_ok'].mean():.3f} / {self.test['docling_ok'].mean():.3f}"
        )


def split_by(frame: pl.DataFrame, key: str, seed: int = SPLIT_SEED) -> Split:
    """Split so that no value of *key* appears on both sides."""
    values = frame[key].unique().to_list()
    rng = np.random.default_rng(seed)
    held_out = set(rng.permutation(values)[: int(len(values) * TEST_FRACTION)].tolist())
    mask = frame[key].is_in(list(held_out))
    return Split(train=frame.filter(~mask), test=frame.filter(mask), key=key)


def matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    return frame.select(features).to_numpy().astype(np.float32)


def fit(split: Split, features: list[str]) -> xgb.Booster:
    """Train the candidate, holding out part of the training side to stop early."""
    inner = split_by(split.train, split.key, seed=SPLIT_SEED + 1)
    train = xgb.DMatrix(
        matrix(inner.train, features), label=inner.train["docling_ok"].to_numpy(), feature_names=features
    )
    validation = xgb.DMatrix(
        matrix(inner.test, features), label=inner.test["docling_ok"].to_numpy(), feature_names=features
    )
    return xgb.train(
        BOOSTER_PARAMS,
        train,
        num_boost_round=BOOST_ROUNDS,
        evals=[(validation, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )


def vlm_scores(booster: xgb.Booster, frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Probability that a document *needs* the VLM, so the score points the same way as the incumbent's."""
    predicted = booster.predict(xgb.DMatrix(matrix(frame, features), feature_names=features))
    return 1.0 - predicted


@dataclass(frozen=True)
class Comparison:
    """Incumbent versus candidate at a matched VLM budget on held-out documents."""

    budget: float
    incumbent: RoutePoint
    candidate: RoutePoint

    @property
    def quality_loss_reduction(self) -> float:
        """Fraction of the incumbent's silent quality loss the candidate removes at equal cost."""
        if self.incumbent.quality_loss == 0:
            return 0.0
        return 1.0 - self.candidate.quality_loss / self.incumbent.quality_loss

    def summary(self) -> str:
        return (
            f"at {self.budget:.1%} of documents routed to the VLM: "
            f"quality loss {self.incumbent.quality_loss:.4f} -> {self.candidate.quality_loss:.4f} "
            f"({self.quality_loss_reduction:+.1%}), "
            f"wasted VLM {self.incumbent.wasted_vlm_fraction:.4f} -> {self.candidate.wasted_vlm_fraction:.4f}"
        )


def compare(test: pl.DataFrame, candidate: np.ndarray) -> list[Comparison]:
    """Compare the two routers across a range of VLM budgets on the same held-out rows."""
    docling_ok = test["docling_ok"].to_numpy()
    incumbent_score = test["ocr_prob"].to_numpy()
    # The incumbent's shipped rule overrides the probability when text came back garbled, so the
    # comparison has to score that rule rather than the bare probability.
    incumbent_score = np.where(test["garbled_text_ratio"].to_numpy() > 0.0, 1.0, incumbent_score)

    incumbent_points = route_frontier(incumbent_score, docling_ok, THRESHOLDS)
    candidate_points = route_frontier(candidate, docling_ok, THRESHOLDS)
    shipped = point_at_budget(incumbent_points, float((incumbent_score >= INCUMBENT_THRESHOLD).mean()))

    budgets = sorted({shipped.vlm_fraction, 0.1, 0.2, 0.3, 0.4, 0.5})
    return [
        Comparison(
            budget=budget,
            incumbent=point_at_budget(incumbent_points, budget),
            candidate=point_at_budget(candidate_points, budget),
        )
        for budget in budgets
    ]


def importances(booster: xgb.Booster, top: int = 25) -> list[tuple[str, float]]:
    """The features the candidate actually leans on, by total gain."""
    gains = {name: float(gain) for name, gain in booster.get_score(importance_type="total_gain").items()}
    return sorted(gains.items(), key=lambda item: item[1], reverse=True)[:top]
