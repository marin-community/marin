# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 39-bucket high-TPP 300M stage: component-level joint fitting against the HPR baseline.

The primary identification panel. 520 structured rows over 280 correspondence groups: 282 physically
tied policies, 238 asymmetric ones, and an exact aggregate-matched tied counterpart for every
asymmetric policy. None of the 238 asymmetric policies beats the best tied policy on either target, so
a near-tied optimum is the expected correct answer here and a large predicted gain is a failure.

Labels are constructed so that nothing is counted twice. Table-9's published macro is the unweighted
mean of 51 components; the packet carries 47 of them, so the remaining four MMLU buckets enter as one
derived mean and the macro becomes an exact linear functional of predictions rather than a label of its
own. Uncheatable's macro is the flat mean of 7 components, which exist only on the 280 two-phase rows;
the 240 rows that carry the macro alone contribute it as an observation of that mean.

WITHDRAWN. This stage produced no valid result and needs a redesign before it is run again. Two defects,
both found by independent review and both recorded in the round report:

The macro-only rows were never actually coupled to the component heads. Nested prediction fills a row
only where that label is observed, so the 240 rows carrying the Uncheatable macro but not its components
left every component prediction unset, and the reconstruction averaged them into NaN. `macro_predictions`
now refuses rather than propagating that, but refusing is not a fix -- the macro has to enter as a linear
constraint on the component heads, which is what the preregistration described and what the code does not
do.

The comparison target was wrong. The published HPR reference is computed on `eval_uncheatable_eval_bpb`
while this stage reconstructs `eval_uncheatable_eval_macro_bpb`; they differ by up to 0.004774 BPB, more
than the gate's own slack. And the frozen pair gate asks for improvement over HPR beyond clustered
bootstrap uncertainty, which needs an HPR residual vector this harness never builds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    interference_evidence_model_20260806 as ile,
)

# Only the recency law reaches this panel. The other two are already closed on WSD80 by structural
# arguments that do not depend on the panel -- the absolute law is not tied-neutral so its rate is
# identified from aggregate curvature, and the one-sided law provably cannot produce a fixed-aggregate
# gain on a simplex. Rerunning them here would cost hours and could not change either conclusion.
LAWS = {ile.InterferenceLaw.RECENCY_EXPOSURE: ile.CURVATURE_GRID}
NUM_TABLE9_COMPONENTS = 51
NUM_MMLU_BUCKETS = 4
DERIVED_MMLU = "derived/table9_mmlu_bucket_mean"
UNCHEATABLE_MACRO = "eval_uncheatable_eval_macro_bpb"
UNCHEATABLE_BLOCK_SHARE = 0.5
LOWER_TAIL_FRACTION = 0.15


def build_targets(frame: pd.DataFrame) -> harness.MultiTarget:
    """Component labels for both suites, with the preregistered equal-block weighting."""
    leaves = sorted(c for c in frame.columns if c.startswith("olmo_base_eval/"))
    if len(leaves) != NUM_TABLE9_COMPONENTS - NUM_MMLU_BUCKETS:
        raise ValueError(f"expected {NUM_TABLE9_COMPONENTS - NUM_MMLU_BUCKETS} Table-9 leaves, found {len(leaves)}")
    mmlu = (
        NUM_TABLE9_COMPONENTS * frame["table9_macro_bpb"].to_numpy(dtype=float) - frame[leaves].sum(axis=1).to_numpy()
    ) / NUM_MMLU_BUCKETS

    uncheatable = sorted(
        c
        for c in frame.columns
        if c.startswith("eval_uncheatable_eval_")
        and c.endswith("_bpb")
        and "macro" not in c
        and c != "eval_uncheatable_eval_bpb"
    )
    has_components = frame[uncheatable].notna().all(axis=1).to_numpy()

    names = [*leaves, DERIVED_MMLU, *uncheatable, UNCHEATABLE_MACRO]
    values = np.column_stack(
        [
            frame[leaves].to_numpy(dtype=float),
            mmlu,
            frame[uncheatable].to_numpy(dtype=float),
            frame[UNCHEATABLE_MACRO].to_numpy(dtype=float),
        ]
    )
    observed = np.ones(values.shape, dtype=bool)
    for offset in range(len(uncheatable)):
        observed[:, len(leaves) + 1 + offset] = has_components
    # The macro is only a label where its components are missing, so no row supplies both.
    observed[:, -1] = ~has_components
    values = np.nan_to_num(values, nan=0.0)

    family = tuple(["table9"] * (len(leaves) + 1) + ["uncheatable"] * (len(uncheatable) + 1))
    share = np.array(
        [
            (
                (1.0 - UNCHEATABLE_BLOCK_SHARE) / (len(leaves) + 1)
                if suite == "table9"
                else UNCHEATABLE_BLOCK_SHARE / (len(uncheatable) + 1)
            )
            for suite in family
        ]
    )
    return harness.MultiTarget(
        names=tuple(names),
        values=values,
        observed=observed,
        family=family,
        family_share=share,
    )


def macro_predictions(targets: harness.MultiTarget, predictions: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruct both published macros as exact linear functionals of the component predictions."""
    leaves = [j for j, name in enumerate(targets.names) if name.startswith("olmo_base_eval/")]
    mmlu = targets.index(DERIVED_MMLU)
    table9 = (predictions[:, leaves].sum(axis=1) + NUM_MMLU_BUCKETS * predictions[:, mmlu]) / NUM_TABLE9_COMPONENTS
    components = [
        j
        for j, name in enumerate(targets.names)
        if name.startswith("eval_uncheatable_eval_") and name != UNCHEATABLE_MACRO
    ]
    uncheatable = predictions[:, components].mean(axis=1)
    # A NaN here means some component prediction was never filled, which happens when a row carries the
    # macro but not its components. Averaging those silently produced a macro that was NaN on 240 of 520
    # rows in the first version of this stage, so the reconstruction now refuses rather than propagates.
    if not np.all(np.isfinite(uncheatable)):
        missing = int(np.sum(~np.isfinite(uncheatable)))
        raise ValueError(
            f"Uncheatable macro is undefined on {missing} rows: those rows observe the macro but not its "
            "components, so the macro must constrain the component heads rather than be reconstructed "
            "from them. See report.md section 0."
        )
    return {"table9": table9, "uncheatable": uncheatable}


def score_target(
    observed: np.ndarray,
    prediction: np.ndarray,
    tied: np.ndarray,
) -> dict:
    residual = prediction - observed
    order = np.argsort(prediction)
    ranked = observed[order]
    best = float(np.min(observed))
    tail = observed <= np.quantile(observed, LOWER_TAIL_FRACTION)
    slope = float(np.polyfit(prediction, observed, 1)[0])
    return {
        "all_rmse": float(np.sqrt(np.mean(residual**2))),
        "tied_rmse": float(np.sqrt(np.mean(residual[tied] ** 2))),
        "asymmetric_rmse": float(np.sqrt(np.mean(residual[~tied] ** 2))),
        "lower_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "bias": float(np.mean(residual)),
        "calibration_slope": slope,
        "spearman": float(pd.Series(prediction).corr(pd.Series(observed), method="spearman")),
        "regret_at_1": float(ranked[0] - best),
        "regret_at_3": float(np.min(ranked[:3]) - best),
        "regret_at_5": float(np.min(ranked[:5]) - best),
    }


def exact_pair_metrics(frame: pd.DataFrame, observed: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict:
    """Phase effect on every exact aggregate-matched asymmetric/tied pair."""
    indexed = frame.reset_index(drop=True)
    keys = indexed["phase_correspondence_key"].astype(str)
    family = indexed["policy_family"]
    lookup = {}
    for position, (key, kind) in enumerate(zip(keys, family, strict=True)):
        lookup[(key, kind)] = position
    shared = sorted(
        {key for key, kind in lookup if kind == "two_phase"} & {key for key, kind in lookup if kind == "single_phase"}
    )
    asymmetric = np.array([lookup[(key, "two_phase")] for key in shared])
    counterpart = np.array([lookup[(key, "single_phase")] for key in shared])
    # The published HPR control scores 238 genuinely asymmetric pairs. Intersecting policy families also
    # admits two rows that are labelled two-phase but are physically tied; their near-zero residuals
    # would shift both the pair RMSE and the bootstrap distribution away from the reference.
    keep = ~np.isclose(weights[asymmetric, 0, :], weights[asymmetric, 1, :]).all(axis=1)
    asymmetric, counterpart = asymmetric[keep], counterpart[keep]
    observed_delta = observed[asymmetric] - observed[counterpart]
    predicted_delta = prediction[asymmetric] - prediction[counterpart]
    residual = predicted_delta - observed_delta
    return {
        "n_pairs": len(asymmetric),
        "delta_rmse": float(np.sqrt(np.mean(residual**2))),
        "delta_bias": float(np.mean(residual)),
        "delta_spearman": float(pd.Series(predicted_delta).corr(pd.Series(observed_delta), method="spearman")),
        "sign_accuracy": float(np.mean(np.sign(predicted_delta) == np.sign(observed_delta))),
        "pair_residual": residual,
        "asymmetric_rows": asymmetric,
    }


def run(output_dir: Path, draws: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = expanded.load_300m("uncheatable")
    frame = dataset.frame.reset_index(drop=True)
    targets = build_targets(frame)
    base = expanded.geometry_300m(dataset)
    geo = ile.Geometry(
        c0=dataset.c0,
        c1=dataset.c1,
        phase_1_fraction=1.0 - base.phase_0_fraction,
        family_index=dataset.family_index,
    )
    tied = np.isclose(dataset.weights[:, 0, :], dataset.weights[:, 1, :]).all(axis=1)
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()

    outer = expanded.grouped_folds(frame, harness.PANEL_OUTER_SEED, harness.PANEL_OUTER_SPLITS)

    def inner_for(fold_id: int, train: np.ndarray):
        local = frame.iloc[train].reset_index(drop=True)
        folds = expanded.grouped_folds(local, harness.PANEL_INNER_SEED_BASE + fold_id, harness.PANEL_INNER_SPLITS)
        return tuple((train[a], train[b]) for a, b in folds)

    macro_rows, pair_rows, component_rows, trace_rows = [], [], [], []
    pair_store, macro_store = {}, {}
    observed_macros = {"table9": frame["table9_macro_bpb"].to_numpy(float), "uncheatable": None}
    observed_macros["uncheatable"] = macro_predictions(targets, np.where(targets.observed, targets.values, np.nan))[
        "uncheatable"
    ]
    # Rows without Uncheatable components fall back to their published macro.
    fallback = np.isnan(observed_macros["uncheatable"])
    observed_macros["uncheatable"][fallback] = frame[UNCHEATABLE_MACRO].to_numpy(float)[fallback]

    for law, curvature_grid in LAWS.items():
        shapes = ile.shape_grid(law=law, curvature_grid=curvature_grid)
        blind_shapes = tuple(shape for shape in shapes if shape.interference == 0.0)
        for modes, grid_shapes in ((("joint", "independent"), shapes), (("phase_blind",), blind_shapes)):
            fitting = tuple("independent" if mode == "phase_blind" else mode for mode in modes)
            predicted, trace = harness.nested_predictions(
                dataset.weights, geo, targets, outer, inner_for, grid_shapes, ile.HEAD_RIDGE_GRID, fitting
            )
            rename = dict(zip(fitting, modes, strict=True))
            for record in trace:
                record["mode"] = rename[record["mode"]]
                record["law"] = str(law)
            trace_rows.extend(trace)

            for fitting_mode, mode in rename.items():
                predictions = predicted[fitting_mode]
                macros = macro_predictions(targets, predictions)
                for suite, prediction in macros.items():
                    observed = observed_macros[suite]
                    row: dict[str, object] = {"law": str(law), "mode": mode, "target": suite}
                    row.update(score_target(observed, prediction, tied))
                    reference = harness.HPR_REFERENCE[suite]
                    row["hpr_all_rmse"] = reference["all_rmse"]
                    ratio = float(row["all_rmse"]) / reference["all_rmse"]
                    row["core_rmse_ratio_vs_hpr"] = ratio
                    row["core_rmse_passes"] = bool(ratio <= harness.CORE_RMSE_RATIO_LIMIT)
                    row["regret_passes"] = bool(
                        float(row["regret_at_1"]) <= reference["regret_at_1"] + harness.REGRET_SLACK
                    )
                    macro_rows.append(row)
                    macro_store[(law, mode, suite)] = prediction

                    pair = exact_pair_metrics(frame, observed, prediction, dataset.weights)
                    pair_store[(law, mode, suite)] = pair
                    pair_rows.append(
                        {
                            "law": str(law),
                            "mode": mode,
                            "target": suite,
                            "n_pairs": pair["n_pairs"],
                            "delta_rmse": pair["delta_rmse"],
                            "delta_bias": pair["delta_bias"],
                            "delta_spearman": pair["delta_spearman"],
                            "sign_accuracy": pair["sign_accuracy"],
                            "hpr_delta_rmse": reference["pair_delta_rmse"],
                        }
                    )

                for j, name in enumerate(targets.names):
                    mask = targets.observed[:, j]
                    residual = predictions[mask, j] - targets.values[mask, j]
                    component_rows.append(
                        {
                            "law": str(law),
                            "mode": mode,
                            "component": name,
                            "suite": targets.family[j],
                            "n": int(mask.sum()),
                            "rmse": float(np.sqrt(np.mean(residual**2))),
                        }
                    )

    macro_frame = pd.DataFrame(macro_rows)
    pair_frame = pd.DataFrame(pair_rows)
    component_frame = pd.DataFrame(component_rows)
    trace_frame = pd.DataFrame(trace_rows)
    macro_frame.to_csv(output_dir / "panel300m_macro_metrics.csv", index=False)
    pair_frame.to_csv(output_dir / "panel300m_pair_metrics.csv", index=False)
    component_frame.to_csv(output_dir / "panel300m_component_metrics.csv", index=False)
    trace_frame.to_csv(output_dir / "panel300m_selection_trace.csv", index=False)

    comparisons = []
    for law in LAWS:
        for suite in ("uncheatable", "table9"):
            observed = observed_macros[suite]
            joint = macro_store[(law, "joint", suite)] - observed
            independent = macro_store[(law, "independent", suite)] - observed
            blind = macro_store[(law, "phase_blind", suite)] - observed
            for label, other in (("joint_minus_independent", independent), ("joint_minus_phase_blind", blind)):
                record = harness.paired_bootstrap_difference(joint, other, groups, draws=draws)
                record.update({"law": str(law), "target": suite, "contrast": label, "scope": "all_rows"})
                comparisons.append(record)
            pair_joint = pair_store[(law, "joint", suite)]
            pair_independent = pair_store[(law, "independent", suite)]
            record = harness.paired_bootstrap_difference(
                pair_joint["pair_residual"],
                pair_independent["pair_residual"],
                groups[pair_joint["asymmetric_rows"]],
                draws=draws,
            )
            record.update(
                {"law": str(law), "target": suite, "contrast": "joint_minus_independent", "scope": "exact_pairs"}
            )
            comparisons.append(record)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "panel300m_joint_vs_independent.csv", index=False)

    harness.write_json(
        output_dir / "panel300m_summary.json",
        {
            "protocol": harness.protocol_hash({"stage": "panel300m"}),
            "n_rows": len(frame),
            "n_targets": targets.n_targets,
            "n_correspondence_groups": int(pd.Series(groups).nunique()),
            "macro": macro_frame.to_dict(orient="records"),
            "pairs": pair_frame.to_dict(orient="records"),
        },
    )

    pd.set_option("display.width", 260)
    print("=== selected transition, nested folds ===")
    print(trace_frame.groupby(["law", "mode"])[["rho", "interference"]].median().to_string())
    print()
    print("=== macro metrics versus HPR ===")
    print(
        macro_frame[
            [
                "law",
                "mode",
                "target",
                "all_rmse",
                "hpr_all_rmse",
                "core_rmse_ratio_vs_hpr",
                "core_rmse_passes",
                "tied_rmse",
                "asymmetric_rmse",
                "regret_at_1",
                "regret_passes",
            ]
        ].to_string(index=False)
    )
    print()
    print("=== exact aggregate-matched pairs ===")
    print(pair_frame.to_string(index=False))
    print()
    print("=== joint versus independent ===")
    print(comparison_frame.to_string(index=False))
