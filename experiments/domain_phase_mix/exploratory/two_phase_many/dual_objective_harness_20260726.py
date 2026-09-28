# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a surrogate on goodness of fit and on optimum prediction quality at once.

The two objectives disagree. Four times in this experiment fit-panel CV has chosen
a configuration that fits better and proposes worse, so a harness reporting only one
of them will keep selecting the wrong model. This module reports both from a single
call and keeps every selection step inside the fit panel, so a candidate stays
comparable with the Observatory baselines.

Three arms, each chosen because the obvious alternative leaks.

*Fit quality* is grouped out-of-fold prediction on the 300M fit panel, the 280 rows
that are the deployed training budget. Same protocol as the Observatory baselines,
so the numbers are directly comparable.

*Optimum quality* is measured by top-censored fitting. The best rows of the fit panel
by observed target are hidden, the model is fitted and its hyperparameters selected
on what remains, and it is then scored on the rows it never saw. This asks the
question a proposer actually faces: can the model recognize a policy better than
anything in its training data? Ranking the observed archive cannot answer it, because
every Observatory baseline already places the archive best in its top five at every
KL radius from 0.05 to 0.5, so that metric is saturated and blind.

Censored bias is the operative number. It is positive for every existing baseline,
from +0.0014 to +0.0206 BPB, meaning they all under-rate unseen good policies and so
propose conservatively into a region they cannot see. Censored Spearman asks the
companion question of whether the good policies are ordered correctly among
themselves, where the baselines reach only 0.20 to 0.74.

*Phase decision skill* is the second optimum-quality arm. It uses the 238
exposure-matched pairs at 300M, where
``Delta = L(a, d) - L(a, 0)`` is observed directly. The model is fitted at 60M for
this arm. Fitting at 300M would score the pairs partly in sample, and the 300M fit
panel has 280 groups for 280 rows so grouped K-fold degenerates to plain K-fold over
one design series and flatters high-capacity models. Crossing the scale gap removes
both problems.

*3e18 is never read here.* It is the final arbiter and using it to choose among
candidates would burn it. The sealed ``targeted_pairwise`` panel is never read at all.

One caveat that constrains interpretation: the 60M and 300M fit panels are the same
mixtures at two scales, 99.6 percent matching exactly in aggregate. The phase arm is
therefore scale transfer on shared policies, not generalization to new policies. The
out-of-design arm is the one that tests new policies.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_order_spine_20260725 import load_paired_panel  # noqa: E402
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import (  # noqa: E402
    TABLE9,
    UNCHEATABLE,
    Model,
    Panel,
    fit_head,
    from_link,
    grouped_splits,
    link_floor,
    load_scale,
    to_link,
)

TARGETS = (UNCHEATABLE, TABLE9)
TIED_PREFIX_300M = r"^singleavg_"

# Run-to-run standard deviation from replicate runs of identical mixtures, used so
# regret is comparable between the two targets.
RUN_SIGMA = {UNCHEATABLE: 0.00096, TABLE9: 0.0031}

N_SPLITS = 5
SPLIT_SEED = 0
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
TOP_K = 5

# A heldout row counts as novel only if its aggregate is at least this far in total
# variation from every fit-panel aggregate. Six heldout rows sit within 0.001 and
# would otherwise be scored as generalization when they are near-duplicates.
MIN_NOVEL_TV = 0.005

# Fractions of the fit panel censored from the top, by observed target. 0.10 keeps
# the training set close to the deployed one; 0.20 is the stress case.
CENSOR_FRACTIONS = (0.10, 0.20)

BOOTSTRAP_DRAWS = 60
BOOTSTRAP_SEED = 20260726


@dataclass(frozen=True)
class Fitted:
    model: Model
    shape: dict
    l2: float
    intercept: float
    coefficients: np.ndarray
    floor: float

    def predict(self, panel: Panel) -> np.ndarray:
        eta = self.intercept + self.model.build(panel, self.shape).matrix @ self.coefficients
        return from_link(self.model, eta, self.floor)


def _penalty(model: Model, panel: Panel, shape: dict) -> np.ndarray | None:
    return None if model.penalty_scale is None else model.penalty_scale(panel, shape)


def aggregate_of(panel: Panel) -> np.ndarray:
    return panel.alpha * panel.phase0 + (1.0 - panel.alpha) * panel.phase1


def fit_on(panel: Panel, model: Model, target: str, shape: dict, l2: float, rows: np.ndarray | None = None) -> Fitted:
    """Fit the linear head for one shape and ridge, optionally on a row subset."""
    observed = panel.targets[target]
    keep = np.isfinite(observed) if rows is None else (rows & np.isfinite(observed))
    design = model.build(panel, shape).matrix[keep]
    floor = link_floor(model, shape, observed[keep])
    intercept, coefficients = fit_head(design, to_link(model, observed[keep], floor), l2, _penalty(model, panel, shape))
    return Fitted(model, shape, l2, intercept, coefficients, floor)


def out_of_fold_predictions(
    panel: Panel, model: Model, target: str, shape: dict, l2: float, rows: np.ndarray | None = None
) -> np.ndarray:
    """Grouped out-of-fold predictions, NaN where the target is missing or excluded."""
    observed = panel.targets[target]
    if rows is not None:
        observed = np.where(rows, observed, np.nan)
    design_all = model.build(panel, shape).matrix
    penalty = _penalty(model, panel, shape)
    prediction = np.full(len(observed), np.nan)
    for train, test in grouped_splits(panel, N_SPLITS, SPLIT_SEED):
        train = train & np.isfinite(observed)
        if train.sum() < 2:
            continue
        floor = link_floor(model, shape, observed[train])
        intercept, coefficients = fit_head(design_all[train], to_link(model, observed[train], floor), l2, penalty)
        prediction[test] = from_link(model, intercept + design_all[test] @ coefficients, floor)
    return prediction


def fit_metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Goodness of fit, including behaviour in the low-predicted tail a proposal is drawn from."""
    finite = np.isfinite(observed) & np.isfinite(prediction)
    observed, prediction = observed[finite], prediction[finite]
    residual = prediction - observed
    order = np.argsort(prediction)
    tail = order[: max(LOWER_TAIL_MIN_COUNT, int(LOWER_TAIL_FRACTION * len(order)))]
    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    return {
        "n": int(finite.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(np.corrcoef(ranks(prediction), ranks(observed))[0, 1]),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        # Positive means the model promises more than it delivers among its own best picks.
        "low_tail_optimism": float(np.mean(observed[tail] - prediction[tail])),
    }


def censored_metrics(panel: Panel, model: Model, target: str, criterion: str, fraction: float) -> dict[str, float]:
    """Hide the best rows, fit on the rest, and score the model on what it never saw.

    Hyperparameters are selected on the training rows only, so no property of the
    censored rows reaches the model. ``bias`` is the operative number: positive means
    the model rates unseen good policies worse than they are, which is exactly the
    failure that makes a proposer conservative in the region it should explore.
    """
    observed = panel.targets[target]
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    censored = np.zeros(len(observed), dtype=bool)
    censored[ordering[:n_censored]] = True
    train = available & ~censored

    shape, l2, _ = select_by(panel, model, target, criterion, rows=train)
    fitted = fit_on(panel, model, target, shape, l2, rows=train)
    prediction = fitted.predict(panel)[censored]
    truth = observed[censored]
    residual = prediction - truth
    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    return {
        "n_censored": int(n_censored),
        "worst_censored_bpb": float(truth.max()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
    }


@dataclass(frozen=True)
class PairedEvaluation:
    """Exposure-matched pairs as two panels, so any model can predict on both members."""

    two_phase_panel: Panel
    tied_panel: Panel
    observed_delta: dict[str, np.ndarray]


@dataclass(frozen=True)
class Benchmark:
    """Every panel and mask the harness needs, built once and reused by all candidates."""

    fit_60m: Panel
    fit_300m: Panel
    heldout_300m: Panel
    novel_rows: np.ndarray
    paired_300m: PairedEvaluation
    metadata: dict[str, Any] = field(default_factory=dict)


def _panel_like(reference: Panel, phase0: np.ndarray, phase1: np.ndarray, alpha: float, row_id: np.ndarray) -> Panel:
    rows = len(phase0)
    return Panel(
        scale=reference.scale,
        split="paired",
        alpha=alpha,
        buckets=reference.buckets,
        c0=reference.c0,
        c1=reference.c1,
        family_index=reference.family_index,
        family_names=reference.family_names,
        phase0=phase0,
        phase1=phase1,
        targets={key: np.full(rows, np.nan) for key in TARGETS},
        series=np.array(["paired"] * rows),
        policy_class=np.array(["two_phase"] * rows),
        group=np.arange(rows),
        row_id=row_id,
    )


def build_benchmark() -> Benchmark:
    """Load every panel once and precompute the novel-row mask."""
    fit_60m, _ = load_scale("60m")
    fit_300m, heldout_300m = load_scale("300m")

    distance = cdist(aggregate_of(heldout_300m), aggregate_of(fit_300m), metric="cityblock") / 2.0
    novel_rows = distance.min(axis=1) >= MIN_NOVEL_TV

    paired = load_paired_panel("300m", TIED_PREFIX_300M)
    assert paired.buckets == fit_300m.buckets, "paired panel bucket order differs from the fit panel"
    tied = paired.aggregate
    paired_evaluation = PairedEvaluation(
        two_phase_panel=_panel_like(fit_300m, paired.phase0, paired.phase1, paired.alpha, paired.row_id),
        tied_panel=_panel_like(fit_300m, tied, tied, paired.alpha, paired.row_id),
        observed_delta=dict(paired.delta),
    )

    metadata = {
        "n_fit_60m": len(fit_60m.row_id),
        "n_fit_300m": len(fit_300m.row_id),
        "n_heldout_300m": len(heldout_300m.row_id),
        "n_novel_300m": int(novel_rows.sum()),
        "min_novel_tv": MIN_NOVEL_TV,
        "n_pairs_300m": len(paired.row_id),
    }
    return Benchmark(fit_60m, fit_300m, heldout_300m, novel_rows, paired_evaluation, metadata)


def select_by(
    panel: Panel, model: Model, target: str, criterion: str, rows: np.ndarray | None = None
) -> tuple[dict, float, dict[str, Any]]:
    """Choose shape and ridge on the fit panel by the named out-of-fold criterion.

    ``rmse`` is what every Observatory baseline uses. ``low_tail_rmse`` scores only
    the policies the model itself ranks best, which is the region a proposal comes
    from. ``low_tail_honesty`` scores the absolute optimism in that same region,
    targeting the extrapolation bias directly rather than the error magnitude: a model
    is asked to be unbiased about its own best picks, not merely close to them. All
    three are out of fold on the fit panel alone, so any of them leaves the model
    identified by that panel.
    """
    observed = panel.targets[target]
    if rows is not None:
        observed = np.where(rows, observed, np.nan)
    best: tuple[float, dict, float] | None = None
    evaluated = 0
    for shape in model.shapes():
        for l2 in model.l2_grid:
            prediction = out_of_fold_predictions(panel, model, target, shape, l2, rows=rows)
            evaluated += 1
            metrics = fit_metrics(observed, prediction)
            score = abs(metrics["low_tail_optimism"]) if criterion == "low_tail_honesty" else metrics[criterion]
            if best is None or score < best[0]:
                best = (score, shape, l2)
    assert best is not None, "empty shape or ridge grid"
    score, shape, l2 = best
    return shape, l2, {"criterion": criterion, "evaluated": evaluated, "selected_score": score}


def phase_skill_from(benchmark: Benchmark, model: Model, target: str, criterion: str) -> dict[str, Any]:
    """Two-phase-versus-tied decision value, fitted at 60M so the 300M pairs are out of sample."""
    shape, l2, _ = select_by(benchmark.fit_60m, model, target, criterion)
    fitted = fit_on(benchmark.fit_60m, model, target, shape, l2)
    predicted_delta = fitted.predict(benchmark.paired_300m.two_phase_panel) - fitted.predict(
        benchmark.paired_300m.tied_panel
    )
    return {
        **phase_decision_skill(predicted_delta, benchmark.paired_300m.observed_delta[target]),
        "shape_60m": shape,
        "l2_60m": l2,
    }


def score_candidate(benchmark: Benchmark, model: Model, target: str, criterion: str = "rmse") -> dict[str, Any]:
    """Score one candidate on fit quality, censored extrapolation, and phase decision skill."""
    shape, l2, selection = select_by(benchmark.fit_300m, model, target, criterion)
    in_scale_oof = out_of_fold_predictions(benchmark.fit_300m, model, target, shape, l2)
    return {
        "model": model.name,
        "target": target,
        "criterion": criterion,
        "shape_300m": shape,
        "l2_300m": l2,
        "selection": selection,
        "fit": fit_metrics(benchmark.fit_300m.targets[target], in_scale_oof),
        "censored": {
            f"{fraction:.2f}": censored_metrics(benchmark.fit_300m, model, target, criterion, fraction)
            for fraction in CENSOR_FRACTIONS
        },
        "phase": phase_skill_from(benchmark, model, target, criterion),
    }


def summarize(result: dict[str, Any]) -> dict[str, float]:
    """Flatten the headline numbers used for ranking candidates."""
    censored = result["censored"]["0.10"]
    return {
        "oof_rmse": result["fit"]["rmse"],
        "oof_spearman": result["fit"]["spearman"],
        "oof_low_tail_rmse": result["fit"]["low_tail_rmse"],
        "cens_rmse": censored["rmse"],
        "cens_bias": censored["bias"],
        "cens_spearman": censored["spearman"],
        "phase_skill": result["phase"]["phase_skill_score"],
        "phase_accuracy": result["phase"]["decision_accuracy"],
    }
