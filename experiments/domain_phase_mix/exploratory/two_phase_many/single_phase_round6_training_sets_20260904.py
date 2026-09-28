# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Alternative 280-row training sets drawn from sampled and intervention runs, scored on the model-optimum archive.

Eligible training rows are the 280-run panel plus registry coordinates that were not proposed by any surrogate or by
OLMix: the conditional epoch-dose runs (single-bucket dose ladders around a proportional-like anchor) and the baseline
mixture. A fixed random half of the eligible registry rows is held out as an
"interventions" evaluation stratum; every model-proposed archive coordinate (surrogate optima, OLMix outputs,
validation panels, cap sweeps) is evaluation-only and forms the "optima" stratum that carries the frontier and the
OLMix coordinate. Training designs stay within the 280-row budget: the panel itself, panel rows swapped for pool rows
(redundancy-pruned removals, coverage-greedy additions; random variants for uncertainty), a pure space-filling
selection, and a pool-first design. Each design is refitted per component for several models, scored by selection
value on the optima stratum with a paired coordinate bootstrap against the panel design, and, for the successor, its
cap-7 min-plus optimum is reported with its nearest measured neighbours. Nothing is launched.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_proposals_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_union_loso_20260903 as loso,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_candidates_20260904 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_remedies_20260904 as remedies,
)

BUDGET = 280
# Registry sources whose coordinates were sampled or designed as interventions rather than proposed as optima. The
# adversarial stress panel is not among them: its generator keeps only coordinates that a frozen surrogate predicted
# at or beyond the frontier (materialize_delphi_3e18_adversarial_heldout_panel.py), so it is model-proposed.
ELIGIBLE_SOURCES = (
    "conditional_epoch_dose_response",
    "archive::delphi_baseline_mixtures_issue6607_20260623",
)
DEFAULT_MODELS = (
    "weibull_softplus_unscaled",
    "weibull_softplus_unscaled@log_deficit_bounded_link",
    "dsp_total_exposure_concentration",
    "olmix_loglinear_taskwise",
)
SWAP_SIZES = (40, 80, 120)
RANDOM_SWAP = 80
RANDOM_SEEDS = (0, 1, 2)
HOLDOUT_SEED = 20_260_904
BOOTSTRAP_DRAWS = 1000
CAP = 7.0


@dataclasses.dataclass(frozen=True)
class Design:
    name: str
    train: np.ndarray  # union row indices


def total_variation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a[:, None, :] - b[None, :, :]).sum(axis=-1)


def prune_redundant(weights: np.ndarray, rows: np.ndarray, count: int) -> np.ndarray:
    """Remove ``count`` rows, each time the one closest (TV) to another remaining row; ties by row order."""
    keep = list(rows)
    for _ in range(count):
        block = weights[keep]
        distance = total_variation(block, block)
        np.fill_diagonal(distance, np.inf)
        nearest = distance.min(axis=1)
        keep.pop(int(np.argmin(nearest)))
    return np.array(keep)


def coverage_greedy(weights: np.ndarray, seed_rows: np.ndarray, pool: np.ndarray, count: int) -> np.ndarray:
    """Add ``count`` pool rows by max-min TV distance to the growing set (farthest-point sampling)."""
    chosen: list[int] = []
    reference = list(seed_rows)
    excluded = set(int(row) for row in seed_rows)
    available = [int(row) for row in pool if int(row) not in excluded]
    for _ in range(min(count, len(available))):
        candidates_left = np.array([row for row in available if row not in chosen])
        distance = (
            total_variation(weights[candidates_left], weights[reference]).min(axis=1)
            if reference
            else np.ones(len(candidates_left))
        )
        pick = int(candidates_left[np.argmax(distance)])
        chosen.append(pick)
        reference.append(pick)
    return np.array(chosen, dtype=int)


def build_designs(union: loso.Union, pool: np.ndarray) -> list[Design]:
    weights = union.features.weights
    panel_rows = np.where(union.is_panel())[0]
    designs = [Design("panel_280", panel_rows)]
    for size in sorted({min(size, len(pool)) for size in SWAP_SIZES}):  # sizes capped by the pool, no duplicates
        kept = prune_redundant(weights, panel_rows, size)
        added = coverage_greedy(weights, kept, pool, size)
        designs.append(Design(f"swap_pruned_coverage_{size}", np.concatenate([kept, added])))
    for seed in RANDOM_SEEDS:
        size = min(RANDOM_SWAP, len(pool))
        rng = np.random.default_rng(HOLDOUT_SEED + seed)
        kept = np.sort(rng.choice(panel_rows, size=len(panel_rows) - size, replace=False))
        added = np.sort(rng.choice(pool, size=size, replace=False))
        designs.append(Design(f"swap_random_{size}_seed{seed}", np.concatenate([kept, added])))
    everything = np.concatenate([panel_rows, pool])
    positive = np.where(weights[everything] > 0, weights[everything], 1.0)
    most_spread = everything[[int(np.argmax(np.exp(-(weights[everything] * np.log(positive)).sum(axis=1))))]]
    designs.append(
        Design(
            "coverage_280", np.concatenate([most_spread, coverage_greedy(weights, most_spread, everything, BUDGET - 1)])
        )
    )
    fill = coverage_greedy(weights, pool, panel_rows, BUDGET - len(pool))
    designs.append(Design("pool_first_280", np.concatenate([pool, fill])))
    for design in designs:
        if len(design.train) > BUDGET or len(set(design.train.tolist())) != len(design.train):
            raise ValueError(f"{design.name}: {len(design.train)} rows, budget {BUDGET}")
    # Over-budget reference: the panel plus every pool row, to show what the budget itself costs.
    designs.append(Design(f"panel_plus_pool_{len(panel_rows) + len(pool)}_over_budget", everything))
    return designs


def fit_rows(model_id: str, union: loso.Union, component_index: int, train: np.ndarray, test: np.ndarray):
    """Predictions on ``test`` and, for the identity-link successor layout, the fitted per-bucket curve."""
    panel = harness.load_panel(loso.PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(union.target)
    component = group.components[component_index]
    features = dataclasses.replace(registry.apply_transform(union.features, entry), component=str(component))
    model = entry.build(features)
    fitted = model.fit(
        features,
        union.outcomes[:, component_index],
        train,
        loso.inner_folds(features, train),
        harness._seed(harness.FitTask(model_id, loso.PANEL, union.target, component_index, component, 0, 0)),
    )
    prediction = np.asarray(model.predict(fitted, features, test), dtype=float)
    curve = None
    if isinstance(model, models.GridModel):
        spec = model.head_for(fitted.shape, models.LinkKind(str(fitted.diagnostics["link"])))
        design = model.design(features, fitted.shape)
        buckets = features.buckets
        expected = tuple(f"bucket_signal:{i}" for i in range(buckets)) + tuple(
            f"bucket_overexposure:{i}" for i in range(buckets)
        )
        if spec.link is models.LinkKind.IDENTITY and design.names == expected:
            curve = proposals.ComponentCurve(
                float(group.aggregation_weights[component_index]),
                fitted.head.intercept,
                fitted.head.coefficients[:buckets].copy(),
                fitted.head.coefficients[buckets:].copy(),
                dict(fitted.shape),
            )
    return prediction, curve


def eligible_mask(union: loso.Union) -> np.ndarray:
    return (
        np.array([any(source in membership for source in ELIGIBLE_SOURCES) for membership in union.memberships])
        & ~union.is_panel()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--targets", default="uncheatable,table9")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "training_sets_round6")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--designs", default="", help="comma-separated design-name prefixes to run (default: all)")
    args = parser.parse_args()
    wanted = [token.strip() for token in args.designs.split(",") if token.strip()]
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = harness.load_panel(loso.PANEL)
    model_ids = [token.strip() for token in args.models.split(",") if token.strip()]
    metric_rows, boot_rows, pick_rows, proposal_rows, design_rows, prediction_rows = [], [], [], [], [], []
    for target in [token.strip() for token in args.targets.split(",") if token.strip()]:
        union = loso.build_union(panel, target)
        group = panel.group(target)
        eligible = eligible_mask(union) & union.trainable
        optima = ~union.is_panel() & ~eligible_mask(union)
        rng = np.random.default_rng(HOLDOUT_SEED)
        eligible_rows = np.where(eligible)[0]
        holdout = np.zeros(len(union.aggregate), dtype=bool)
        for source in ELIGIBLE_SOURCES:
            members = np.array([row for row in eligible_rows if source in union.memberships[row]])
            if len(members) == 0:
                continue
            holdout[rng.choice(members, size=len(members) // 2, replace=False)] = True
        pool = np.array([row for row in eligible_rows if not holdout[row]])
        evaluation = np.where(optima | holdout)[0]
        strata = {
            "optima": np.where(optima)[0],
            "interventions_heldout": np.where(holdout)[0],
            "pooled": evaluation,
        }
        designs = build_designs(union, pool)
        if wanted:
            designs = [d for d in designs if any(d.name.startswith(prefix) for prefix in wanted)]
            if "panel_280" not in {d.name for d in designs}:
                designs.insert(0, Design("panel_280", np.where(union.is_panel())[0]))
        print(
            f"{target}: union {len(union.aggregate)} rows; eligible registry rows {eligible.sum()} "
            f"(pool {len(pool)}, held out {holdout.sum()}); optima stratum {optima.sum()}; designs {len(designs)}"
        )
        tolerance = harness.BASIN_TOLERANCE_SD * panel.repeat_sd.get(target, float("nan"))
        predictions: dict[tuple[str, str], np.ndarray] = {}
        for design in designs:
            exposures = union.features.exposures[design.train]
            beyond = (exposures > panel.features.exposures.max(axis=0)[None, :]).any(axis=1)
            cc_columns = [i for i, bucket in enumerate(panel.buckets) if bucket.startswith("dolma3_cc/")]
            cc_beyond = int(
                (exposures[:, cc_columns].max(axis=0) > panel.features.exposures.max(axis=0)[cc_columns]).sum()
            )
            train_weights = union.features.weights[design.train]
            train_positive = np.where(train_weights > 0, train_weights, 1.0)
            effective = np.exp(-(train_weights * np.log(train_positive)).sum(axis=1))
            design_rows.append(
                {
                    "target": target,
                    "design": design.name,
                    "rows": len(design.train),
                    "panel_rows": int(union.is_panel()[design.train].sum()),
                    "dose_rows": int(union.is_dose()[design.train].sum()),
                    "other_eligible_rows": int((~union.is_panel() & ~union.is_dose())[design.train].sum()),
                    "rows_beyond_panel_exposure": int(beyond.sum()),
                    "cc_buckets_beyond_panel_max": cc_beyond,
                    "min_effective_buckets": float(effective.min()),
                    "max_epochs": float(exposures.max()),
                    "mean_nearest_tv_within": float(
                        np.sort(
                            total_variation(union.features.weights[design.train], union.features.weights[design.train]),
                            axis=1,
                        )[:, 1].mean()
                    ),
                    "coordinate_ids": ";".join(union.coordinate_id[design.train]),
                }
            )
            for model_id in model_ids:
                with harness.parallel_config(backend="loky", inner_max_num_threads=1):
                    parts = Parallel(n_jobs=args.workers, verbose=0)(
                        delayed(fit_rows)(model_id, union, index, design.train, evaluation)
                        for index in range(len(group.components))
                    )
                matrix = np.stack([part[0] for part in parts], axis=1)
                guess = np.full(len(union.aggregate), np.nan)
                guess[evaluation] = matrix @ group.aggregation_weights
                predictions[(design.name, model_id)] = guess
                for stratum in ("optima", "interventions_heldout"):
                    for row in strata[stratum]:
                        prediction_rows.append(
                            {
                                "target": target,
                                "design": design.name,
                                "model": model_id,
                                "stratum": stratum,
                                "coordinate_id": union.coordinate_id[row],
                                "sources": ";".join(sorted(union.memberships[row])),
                                "measured": float(union.aggregate[row]),
                                "prediction": float(guess[row]),
                            }
                        )
                for stratum, rows in strata.items():
                    row = {"target": target, "design": design.name, "model": model_id, "stratum": stratum}
                    row.update(selection.selection_row(union.aggregate[rows], guess[rows], tolerance))
                    metric_rows.append(row)
                order = strata["optima"][np.argsort(guess[strata["optima"]], kind="stable")]
                pick_rows.append(
                    {
                        "target": target,
                        "design": design.name,
                        "model": model_id,
                        "pick": union.coordinate_id[order[0]],
                        "pick_measured": float(union.aggregate[order[0]]),
                        "pick_source": ";".join(sorted(union.memberships[order[0]])),
                        "frontier_measured": float(union.aggregate[strata["optima"]].min()),
                    }
                )
                curves = [part[1] for part in parts]
                if all(curve is not None for curve in curves):
                    curve_set = remedies.Curves(tuple(curves))
                    inventory = panel.features.inventory
                    upper = np.minimum(np.floor(CAP / inventory * candidates.BLOCKS).astype(int), candidates.BLOCKS)
                    weights = candidates.solve(
                        curve_set,
                        inventory,
                        np.zeros(len(inventory), dtype=int),
                        upper,
                        "plain",
                        panel.features.exposures.max(axis=0),
                    )
                    predicted = float(
                        curve_set.component_matrix(weights[None, :] * inventory[None, :]) @ group.aggregation_weights
                    )
                    bank_rows = np.where(~union.is_panel())[0]
                    distance = total_variation(weights[None, :], union.features.weights[bank_rows])[0]
                    nearest = bank_rows[np.argsort(distance)[:3]]
                    buckets = list(panel.buckets)
                    positive = np.where(weights > 0, weights, 1.0)
                    proposal_rows.append(
                        {
                            "target": target,
                            "design": design.name,
                            "model": model_id,
                            "predicted": predicted,
                            "share_synth_qa": float(weights[buckets.index("dolmino_synth_qa")]),
                            "share_olmocr": float(weights[buckets.index("dolmino_olmocr_pdfs_hq")]),
                            "share_stack": float(
                                weights[
                                    [buckets.index("dolma3_stack_edu"), buckets.index("dolmino_stack_edu_fim")]
                                ].sum()
                            ),
                            "share_cc": float(
                                sum(weights[i] for i, b in enumerate(buckets) if b.startswith("dolma3_cc/"))
                            ),
                            "effective_buckets": float(np.exp(-(weights * np.log(positive)).sum())),
                            "max_epochs": float((weights * inventory).max()),
                            "tv_to_nearest_panel_row": float(
                                total_variation(weights[None, :], panel.features.weights)[0].min()
                            ),
                            "nearest_measured": ";".join(
                                f"{union.coordinate_id[row][-8:]}:{nearest_distance:.3f}:{union.aggregate[row]:.4f}"
                                for row, nearest_distance in zip(nearest, np.sort(distance)[:3], strict=True)
                            ),
                            "weights": json.dumps(dict(zip(buckets, map(float, weights), strict=True))),
                        }
                    )
                latest = metric_rows[-3]
                print(
                    f"  {design.name:28s} {model_id:52s} optima regret@1 {latest['regret_at_1']:.4f}"
                    f" frontier rank {latest['frontier_predicted_rank']:.0f} bias {latest['bias']:+.4f}",
                    flush=True,
                )
        for model_id in model_ids:
            reference = "panel_280"
            for stratum in ("optima", "pooled"):
                rows = strata[stratum]
                bank = selection.Bank(
                    target,
                    union.coordinate_id[rows],
                    union.aggregate[rows],
                    np.array([";".join(sorted(m)) for m in np.array(union.memberships, dtype=object)[rows]]),
                    np.ones(len(rows), dtype=int),
                    union.distance[rows],
                    tolerance,
                )
                per_design = {design.name: predictions[(design.name, model_id)][rows] for design in designs}
                for row in selection.bootstrap_rows(
                    bank, per_design, reference, BOOTSTRAP_DRAWS, stratum, np.ones(len(rows), dtype=bool)
                ):
                    boot_rows.append({**row, "fit_model": model_id})
    provenance = {
        "registry_dir": str(harness.HELDOUT_DIR),
        "manifest_sha256": hashlib.sha256((harness.HELDOUT_DIR / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "eligible_sources": ELIGIBLE_SOURCES,
        "budget": BUDGET,
        "holdout_seed": HOLDOUT_SEED,
        "targets": args.targets,
        "models": model_ids,
        "designs_filter": wanted,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "selection_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(args.output_dir / "bootstrap.csv", index=False)
    pd.DataFrame(pick_rows).to_csv(args.output_dir / "picks.csv", index=False)
    pd.DataFrame(proposal_rows).to_csv(args.output_dir / "proposals.csv", index=False)
    pd.DataFrame(design_rows).to_csv(args.output_dir / "designs.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    metrics = pd.DataFrame(metric_rows)
    view = metrics[metrics["stratum"].eq("optima")].pivot_table(
        index=["target", "design"], columns="model", values=["regret_at_1", "frontier_predicted_rank", "bias"]
    )
    print(view.round(4).to_string())
    if proposal_rows:
        print(pd.DataFrame(proposal_rows).drop(columns=["weights"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
