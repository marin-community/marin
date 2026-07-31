# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the unridged power-deficit candidate against the hierarchical replay incumbent.

The link sweep found a broad region rather than a lucky cell: on Uncheatable, nine of
forty configurations beat the incumbent simultaneously on OOF RMSE, censored RMSE,
absolute censored bias, and censored Spearman, spanning lambda in {0, 0.25, 0.5, 0.75}
and floor fraction in {0.70, 0.85, 0.95}, all with the ridge forced to zero. Breadth is
the reason to take it seriously; a single winning cell in a forty-cell sweep would not
be.

Three things the sweep did not establish, and this script does.

*Phase decision skill.* The sweep scored fit and extrapolation only. A candidate that
improves both but degrades the two-phase-versus-tied call is not an improvement for this
project, so the 60M-to-300M phase arm is scored here.

*Paired uncertainty.* Point estimates on 280 rows move under resampling. Draws are
shared between candidate and incumbent and the reported statistic is the per-draw
difference, which cancels the variance common to both; marginal intervals on a shared
panel would be far too wide to separate anything.

*Selection honesty.* Picking lambda and the floor by looking at censored metrics would be
selecting on the evaluation set. Two arms are therefore reported: ``panel_selected``,
where lambda and the floor are chosen by OOF RMSE on the fit panel alone and the censored
result is whatever falls out, and ``fixed_default``, where lambda = 0.5 and floor = 0.95
are fixed a priori as the midpoint of the additive-to-multiplicative family and the floor
already established by earlier work. Only these two are legitimate; the sweep's best cell
is reported alongside purely as an upper bound on what the family can do.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_power_deficit_link_20260726 import censored_split, fit_predict, out_of_fold, select  # noqa: E402
from dual_objective_harness_20260726 import build_benchmark  # noqa: E402
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "deficit_link_validation_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
CENSOR_FRACTION = 0.10
BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 20260726

# The incumbent: hierarchical phase replay with a cross-validated ridge, additive response.
INCUMBENT = {"lam": 1.0, "floor": 0.95, "l2_grid": (0.0, 0.01, 0.1, 1.0)}
# Fixed a priori: lambda 0.5 is the midpoint of the additive-to-multiplicative family and
# 0.95 is the floor fraction established by the bounded-link work.
FIXED_DEFAULT = {"lam": 0.5, "floor": 0.95, "l2_grid": (0.0,)}
PANEL_SELECT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PANEL_SELECT_FLOORS = (0.70, 0.85, 0.95)


def hpr_model() -> Model:
    return Model("hpr", build_hierarchical_phase_replay, lambda: _state_shapes(True))


def evaluate_config(
    panel: Panel,
    model: Model,
    target: str,
    lam: float,
    floor: float,
    l2_grid: tuple[float, ...],
    train: np.ndarray,
    censored: np.ndarray,
    rows: np.ndarray | None = None,
    frozen: tuple[dict, float] | None = None,
) -> dict[str, float]:
    """Fit metrics from grouped OOF, censored metrics from the held-back best rows.

    ``frozen`` supplies a shape and ridge chosen once on the full training set. Bootstrap
    draws pass it so only the linear head is refitted per draw. Re-selecting the shape
    inside every draw would cost roughly five million head fits and, more importantly,
    would conflate head instability with selection instability. The interval this yields
    is therefore a statement about the head only, and so is a lower bound on the total
    uncertainty.
    """
    fit_rows = train if rows is None else (train & rows)
    shape, l2 = frozen if frozen is not None else select(panel, model, target, l2_grid, lam, floor, rows=fit_rows)
    observed = panel.targets[target]
    oof = out_of_fold(panel, model, target, shape, l2, lam, floor)
    finite = np.isfinite(oof) & np.isfinite(observed)
    prediction = fit_predict(panel, model, target, shape, l2, lam, floor, fit_rows)[censored]
    truth = observed[censored]
    residual = prediction - truth
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return {
        "oof_rmse": float(np.sqrt(np.mean((oof[finite] - observed[finite]) ** 2))),
        "oof_spearman": float(np.corrcoef(ranks(oof[finite]), ranks(observed[finite]))[0, 1]),
        "cens_rmse": float(np.sqrt(np.mean(residual**2))),
        "cens_bias": float(np.mean(residual)),
        "cens_spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
        "shape": shape,
        "l2": l2,
    }


def phase_skill(benchmark, model: Model, target: str, lam: float, floor: float, l2_grid) -> dict[str, float]:
    """Fit at 60M, score the two-phase call on the 238 exposure-matched 300M pairs."""
    panel = benchmark.fit_60m
    available = np.isfinite(panel.targets[target])
    shape, l2 = select(panel, model, target, l2_grid, lam, floor, rows=available)

    design_two = model.build(benchmark.paired_300m.two_phase_panel, shape).matrix
    design_tied = model.build(benchmark.paired_300m.tied_panel, shape).matrix
    observed = panel.targets[target]
    from audit_power_deficit_link_20260726 import from_link, to_link  # noqa: PLC0415
    from swarm39_harness_20260725 import fit_head  # noqa: PLC0415

    base = model.build(panel, shape).matrix[available]
    floor_value = floor * float(np.min(observed[available]))
    intercept, coefficients = fit_head(base, to_link(observed[available] - floor_value, lam), l2)
    predict = lambda design: floor_value + from_link(intercept + design @ coefficients, lam)  # noqa: E731
    delta = predict(design_two) - predict(design_tied)
    result = phase_decision_skill(delta, benchmark.paired_300m.observed_delta[target])
    return {"phase_skill": result["phase_skill_score"], "phase_accuracy": result["decision_accuracy"]}


def panel_selected_config(panel: Panel, model: Model, target: str, train: np.ndarray) -> tuple[float, float]:
    """Choose lambda and floor by OOF RMSE on the training rows only."""
    best: tuple[float, float, float] | None = None
    for lam in PANEL_SELECT_LAMBDAS:
        for floor in PANEL_SELECT_FLOORS:
            shape, l2 = select(panel, model, target, (0.0,), lam, floor, rows=train)
            observed = np.where(train, panel.targets[target], np.nan)
            scratch = dataclasses.replace(panel, targets={**panel.targets, target: observed})
            oof = out_of_fold(scratch, model, target, shape, l2, lam, floor)
            finite = np.isfinite(oof) & np.isfinite(observed)
            score = float(np.sqrt(np.mean((oof[finite] - observed[finite]) ** 2)))
            if best is None or score < best[0]:
                best = (score, lam, floor)
    assert best is not None
    return best[1], best[2]


def stratified_draw(series: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    drawn: list[int] = []
    for name in np.unique(series[rows]):
        pool = np.flatnonzero(rows & (series == name))
        drawn.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
    return np.asarray(drawn, dtype=int)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_300m
    model = hpr_model()
    point_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        train, censored = censored_split(panel.targets[target], CENSOR_FRACTION)
        lam_panel, floor_panel = panel_selected_config(panel, model, target, train)
        arms = {
            "incumbent": INCUMBENT,
            "fixed_default": FIXED_DEFAULT,
            "panel_selected": {"lam": lam_panel, "floor": floor_panel, "l2_grid": (0.0,)},
        }
        print(f"{target}: panel-selected lambda={lam_panel} floor={floor_panel}")

        for label, config in arms.items():
            scores = evaluate_config(
                panel, model, target, config["lam"], config["floor"], config["l2_grid"], train, censored
            )
            skill = phase_skill(benchmark, model, target, config["lam"], config["floor"], config["l2_grid"])
            point_rows.append(
                {
                    "target": target,
                    "arm": label,
                    "lambda": config["lam"],
                    "floor_fraction": config["floor"],
                    "l2": scores.pop("l2"),
                    **{k: v for k, v in scores.items() if k != "shape"},
                    **skill,
                }
            )

        frozen = {
            label: select(panel, model, target, config["l2_grid"], config["lam"], config["floor"], rows=train)
            for label, config in arms.items()
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for draw in range(BOOTSTRAP_DRAWS):
            indices = stratified_draw(panel.series, train, rng)
            resampled = np.zeros(len(panel.series), dtype=bool)
            resampled[np.unique(indices)] = True
            record: dict[str, Any] = {"target": target, "draw": draw}
            for label, config in arms.items():
                scores = evaluate_config(
                    panel,
                    model,
                    target,
                    config["lam"],
                    config["floor"],
                    config["l2_grid"],
                    train,
                    censored,
                    rows=resampled,
                    frozen=frozen[label],
                )
                for key in ("oof_rmse", "cens_rmse", "cens_bias", "cens_spearman"):
                    record[f"{label}:{key}"] = scores[key]
            draw_rows.append(record)
            if (draw + 1) % 50 == 0:
                print(f"  {target}: {draw + 1}/{BOOTSTRAP_DRAWS} draws")

    point_frame = pd.DataFrame(point_rows)
    draw_frame = pd.DataFrame(draw_rows)
    point_frame.to_csv(OUTPUT_DIR / "point_estimates.csv", index=False)
    draw_frame.to_csv(OUTPUT_DIR / "bootstrap_draws.csv", index=False)

    print("\n=== point estimates ===")
    print(point_frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== paired bootstrap: candidate minus incumbent (200 shared draws) ===")
    summary = []
    for target, group in draw_frame.groupby("target"):
        for label in ("fixed_default", "panel_selected"):
            for metric in ("oof_rmse", "cens_rmse", "cens_bias", "cens_spearman"):
                candidate = group[f"{label}:{metric}"].to_numpy()
                incumbent = group[f"incumbent:{metric}"].to_numpy()
                delta = candidate - incumbent
                if metric == "cens_spearman":
                    better = candidate > incumbent
                elif metric == "cens_bias":
                    better = np.abs(candidate) < np.abs(incumbent)
                else:
                    better = candidate < incumbent
                summary.append(
                    {
                        "target": target,
                        "arm": label,
                        "metric": metric,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": float(np.quantile(delta, 0.025)),
                        "ci95_high": float(np.quantile(delta, 0.975)),
                        "fraction_better": float(better.mean()),
                    }
                )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "bootstrap_summary.csv", index=False)
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "incumbent": INCUMBENT,
                "fixed_default": FIXED_DEFAULT,
                "censor_fraction": CENSOR_FRACTION,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "row, stratified by design series; censored set fixed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
