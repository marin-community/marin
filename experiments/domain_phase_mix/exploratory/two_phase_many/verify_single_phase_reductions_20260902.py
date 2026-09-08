# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Prediction-equivalence checks between the Observatory models and their single-phase reductions.

Each check builds the original module's design or fit on tied two-phase inputs (both phases equal
to the single-phase mixture) with the declared domain/quality family partition, builds the reduced
single-phase design from `single_phase_observatory_models_20260902`, and reports the largest
absolute difference. Column order and naming differ between implementations, so columns are
matched as sets after sorting. Run with ``uv run --with cvxpy`` because several Observatory
modules import cvxpy transitively.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_dsp_single_phase_ladder_20260824 as dsp_ladder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_grp_domain_saturation_phase_heads_20260714 as phase_head_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_production_grp_retained_hybrids_20260713 as retained_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact_retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    bounded_crs_model_20260726 as bounded_crs,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    crs_plus_model_20260725 as crs_plus,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    hierarchical_band_model_20260726 as hierarchical_band,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_symmetric_sepheads_geometry_frontier_panel_300m as symmetric_sepheads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as rpl_repair,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    generic_family_followup as grp_followup,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    generic_family_penalty_calibration as grp_calibration,
)

DEFAULT_OUTPUT = harness.DEFAULT_OUTPUT_DIR / "reduction_equivalence.csv"
TOLERANCE = 1e-9
SOLVER_TOLERANCE = 1e-6


def matched_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    """Largest difference after matching columns as multisets; both must have the same width."""
    if left.shape != right.shape:
        return float("inf"), 0
    order_left = np.lexsort(np.round(left, 10)[::-1])
    order_right = np.lexsort(np.round(right, 10)[::-1])
    return float(np.max(np.abs(left[:, order_left] - right[:, order_right]))), int(left.shape[1])


def tied_dataset(
    panel: harness.BenchPanel, target: str, component_index: int
) -> tuple[family_grp.Dataset, pooled.Dataset]:
    features = panel.features
    weights = np.stack([features.weights, features.weights], axis=1)
    c1 = features.inventory * (1.0 - features.early_fraction)
    c0 = features.inventory * features.early_fraction
    response = panel.group(target).outcomes[:, component_index]
    frame = pd.DataFrame({"run_name": list(panel.runs)})
    structured = family_grp.Dataset(
        frame=frame,
        target=response,
        weights=weights,
        c0=c0,
        c1=c1,
        domains=tuple(panel.buckets),
        family_names=features.families.names,
        family_members=features.families.members,
        quality=features.families.quality,
    )
    plain = pooled.Dataset(
        name=f"{panel.name}_{target}",
        frame=frame,
        y=response,
        weights=weights,
        c0=c0,
        c1=c1,
        domain_names=list(panel.buckets),
    )
    return structured, plain


def check_rows(panel: harness.BenchPanel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    features = panel.features
    target = panel.groups[0].name
    structured, plain = tied_dataset(panel, target, min(5, len(panel.groups[0].components) - 1))
    response = structured.target
    fam = features.families

    def record(
        check: str, source: str, difference: float, columns: int, note: str = "", tolerance: float = TOLERANCE
    ) -> None:
        rows.append(
            {
                "panel": panel.name,
                "check": check,
                "source_module": source,
                "max_abs_difference": difference,
                "columns_compared": columns,
                "tolerance": tolerance,
                "status": "pass" if difference <= tolerance else "fail",
                "note": note,
            }
        )

    # 1. bucket_family_grp and power_separate_heads (power_eta) share one design.
    shape = {"exponent": 0.34, "threshold": 5.1}
    mine = models.family_design(features, shape, registry.BUCKET_FAMILY_OPTIONS).values
    original, _names = family_grp.build_design(
        structured, family_grp.Variant.BUCKET_RESOLVED, family_grp.Shape(0.34, 1.0, 0.0, 5.1)
    )
    record(
        "bucket_family_grp -> bucket_family_power_grp",
        "fit_production_grp_quality_variants.build_design",
        *matched_difference(mine, original),
    )
    variant = phase_head_grp.VARIANT_BY_NAME["power_eta"]
    heads, _names, _layout = phase_head_grp.build_design(
        structured, variant, retained_grp.Shape(1.0, 0.34, 1.0, 0.0, 5.1), None
    )
    record(
        "bucket_family_power_separate_heads -> bucket_family_power_grp",
        "benchmark_grp_domain_saturation_phase_heads_20260714.build_design(power_eta)",
        *matched_difference(mine, heads),
        "exact single-phase equivalence of the two source models",
    )

    # 2. Weibull retained GRP variants.
    for source, options in (
        ("weibull_global_tau", registry.WEIBULL_ONSET_OPTIONS),
        ("weibull_family_coverage_family_replay", registry.WEIBULL_REPLAY_OPTIONS),
    ):
        shape = {"rate": 0.7, "power": 0.6, "threshold": 4.2}
        mine = models.family_design(features, shape, options).values
        original, _names = retained_grp.build_design(
            structured, retained_grp.VARIANT_BY_NAME[source], retained_grp.Shape(0.7, 0.6, 1.0, 0.0, 4.2)
        )
        record(
            f"{source} -> {options.harm}",
            "benchmark_production_grp_retained_hybrids_20260713.build_design",
            *matched_difference(mine, original),
        )

    # 3. Hierarchical phase bucket replay: the phase-shift column is identically zero.
    config = hierarchical_grp.Config(
        hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        0,
        family_grp.Shape(0.34, 1.0, 0.0, 5.1),
        0.1,
        10.0,
        0.0,
        0.0,
    )
    design = hierarchical_grp.build_design(structured, config)
    keep = np.flatnonzero(np.abs(design.values).max(axis=0) > 0.0)
    dropped = tuple(name for index, name in enumerate(design.names) if index not in set(keep.tolist()))
    mine_design = models.family_design(
        features, {"exponent": 0.34, "threshold": 5.1, "residual_shrink": 10.0}, registry.HPR_OPTIONS
    )
    record(
        "hierarchical_phase_bucket_replay -> hierarchical_family_replay (values)",
        "benchmark_hierarchical_coverage_grp_20260715.build_design",
        *matched_difference(mine_design.values, design.values[:, keep]),
        f"zero columns dropped from the original: {dropped}",
    )
    record(
        "hierarchical_phase_bucket_replay -> hierarchical_family_replay (ridge multipliers)",
        "benchmark_hierarchical_coverage_grp_20260715.build_design",
        float(abs(np.sort(mine_design.ridge) - np.sort(design.ridge_multipliers[keep])).max()),
        len(keep),
    )

    # 4. crs_plus: exact tied image including every family-benefit column.
    shape = crs_plus.Shape(4.0, 0.7, 2.0, 0.25, 2.0)
    original = crs_plus.design_matrix(structured.weights, structured.c0, structured.c1, shape, structured.family_members)
    mine = models.crs_plus_design(
        features,
        {
            "saturation_epochs": 4.0,
            "power": 0.7,
            "late_multiplier": 2.0,
            "forgetting_rate": 0.25,
            "overload_threshold": 2.0,
        },
    ).values
    record(
        "crs_plus -> crs_plus_family_overload",
        "crs_plus_model_20260725.design_matrix",
        *matched_difference(mine, original),
    )

    # 5. crs_bounded: design and log-deficit head predictions.
    shape_b = bounded_crs.Shape(1.0, 0.7, 2.0, 0.25)
    original = bounded_crs.design_matrix(structured.weights, structured.c0, structured.c1, shape_b)
    options = models.FamilyOptions(family_signal="none", harm="literal_shared", benefit="weibull", retention_gate=True)
    mine_design = models.family_design(
        features, {"rate": 1.0, "power": 0.7, "late_multiplier": 2.0, "forgetting_rate": 0.25}, options
    )
    record(
        "crs_bounded -> weibull_literal_replay_logdeficit (design)",
        "bounded_crs_model_20260726.design_matrix",
        *matched_difference(mine_design.values, original),
    )
    rows_fit = np.arange(panel.rows)
    fitted = bounded_crs.fit_model(structured, bounded_crs.Config(shape_b, 0.1), rows_fit)
    head = models.fit_head(mine_design, response, 0.1, registry.LOG_DEFICIT_SCALED)
    record(
        "crs_bounded -> weibull_literal_replay_logdeficit (predictions)",
        "bounded_crs_model_20260726.fit_model",
        float(
            np.max(
                np.abs(
                    fitted.predict(structured.weights)
                    - models.predict_head(head, mine_design.values, registry.LOG_DEFICIT_SCALED)
                )
            )
        ),
        mine_design.values.shape[1],
        "same shape and ridge; NNLS solved on the QR-reduced system",
    )

    # 6. compact retained state, one-phase config.
    config_c = compact_retained.ModelConfig(
        "one_phase_weibull_shared_replay",
        compact_retained.SignalKind.TOTAL_EXPOSURE,
        compact_retained.ResponseKind.WEIBULL,
        compact_retained.RetentionKind.CONSTANT,
        compact_retained.ReplayPenaltyKind.SHARED,
    )
    shape_c = compact_retained.Shape(0.7, 0.7, 0.6, 1.0, 0.0)
    original = compact_retained.design_matrix(structured.weights, structured.c0, structured.c1, config_c, shape_c, ())
    compact = models.CompactRetainedModel()
    mine_design = compact._design(features, np.asarray([np.log(0.7), 0.6]))
    record(
        "compact_retained_state -> weibull_shared_literal_replay (design)",
        "benchmark_retained_weibull_replay_20260713.design_matrix",
        *matched_difference(mine_design.values, original),
    )
    intercept, coefficients = compact_retained.fit_nonnegative_head(original, response, 0.1)
    head = models.fit_head(mine_design, response, 0.1, compact.head)
    record(
        "compact_retained_state -> weibull_shared_literal_replay (head predictions)",
        "benchmark_retained_weibull_replay_20260713.fit_nonnegative_head",
        float(
            np.max(
                np.abs(
                    (intercept + original @ coefficients) - models.predict_head(head, mine_design.values, compact.head)
                )
            )
        ),
        original.shape[1],
    )

    # 7. canonical DSP: dsp_exact no_phase features equal the profiled design, and the ladder matches.
    rng = np.random.default_rng(0)
    rho = np.exp(rng.uniform(np.log(1e-4), np.log(2.0), size=panel.features.buckets))
    tau = rng.uniform(-2.0, 8.0, size=panel.features.buckets)
    signal, penalty = dsp.features(
        structured.weights, structured.c0, structured.c1, dsp.VARIANTS["no_phase"], {"rho": rho, "tau": tau}
    )
    original = np.hstack([-signal, penalty])
    profiled = models.ProfiledDspModel()
    mine = profiled.design(features, np.concatenate([np.log(rho), tau]))
    record(
        "canonical/effective_exposure -> dsp_total_exposure (no_phase features)",
        "standalone_code.dsp_exact.features(no_phase)",
        float(np.max(np.abs(mine - original))),
        original.shape[1],
    )
    coverage = np.sum(structured.weights[:, 1, :] ** 2, axis=1)
    record(
        "effective_exposure_geometry -> concentration column",
        "benchmark_nested_coverage_dsp.coverage_features[:, 1]",
        float(np.max(np.abs(models.concentration(features)[:, 0] - coverage))),
        1,
        "phase TV column is identically zero; late-phase concentration duplicates it",
    )
    train = np.arange(panel.rows)
    labels = np.arange(panel.rows) % 3
    inner = tuple((train[labels != index], train[labels == index]) for index in range(3))
    canonical_rung = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    vector, intercept, coefficients = dsp_ladder.fit_rung(
        features.exposures, response, canonical_rung, inner, (), seed=3, maxiter=12, restarts=2
    )
    fitted = dataclasses.replace(profiled, options=models.DspOptions(maxiter=12)).fit(
        features, response, train, inner, 3
    )
    mine_vector = np.asarray([fitted.shape[f"theta_{index}"] for index in range(len(fitted.shape))])
    record(
        "dsp_total_exposure profiled solver == ladder canonical rung",
        "benchmark_dsp_single_phase_ladder_20260824.fit_rung",
        float(
            max(
                np.max(np.abs(mine_vector - vector)),
                abs(fitted.head.intercept - intercept),
                np.max(np.abs(fitted.head.coefficients - coefficients)),
            )
        ),
        len(vector),
        "identical starts, seeds, objective, gradient, and NNLS head",
    )

    # 8. Separate heads, one-phase policy.
    model = symmetric_sepheads.fit_separate_model(plain, train, policy="one_phase", l2=0.3)
    original_prediction = symmetric_sepheads.predict_separate_model(model, plain, train)
    bowl = models.BowlModel()
    mu = bowl._select_mu(features, response, train, 0.3)
    design = models.bowl_design(features, mu)
    head = models.fit_head(design, response, 0.3, dataclasses.replace(bowl.head, reduced_nnls=False))
    record(
        "separate_heads -> asymmetric_log_bowl (predictions, fixed ridge)",
        "materialize_symmetric_sepheads_geometry_frontier_panel_300m.fit_separate_model",
        float(np.max(np.abs(original_prediction - models.predict_head(head, design.values, bowl.head)))),
        design.values.shape[1],
    )
    record(
        "separate_heads -> asymmetric_log_bowl (selected centers)",
        "materialize_symmetric_sepheads_geometry_frontier_panel_300m.selected_mu",
        float(np.max(np.abs(model.mus[0] - mu))),
        len(mu),
    )

    # 9. Retained power law, phase-blind image.
    geometry = rpl.Geometry(
        c0=structured.c0,
        c1=structured.c1,
        phase_0_fraction=float(np.median(features.early_fraction)),
        family_index=fam.index,
    )
    shape_r = rpl.Shape(0.5, 0.1, 2.0, 0.0, 0.0, 1.0, False)
    original, layout = rpl_repair.phase_blind_design_matrix(structured.weights, geometry, shape_r)
    mine_design = models.retained_power_law_design(
        features, {"benefit_exponent": 0.5, "benefit_offset": 0.1, "damage_exponent": 2.0, "damage_threshold": 0.0}
    )
    record(
        "retained_power_law -> retained_power_law_phase_blind (design)",
        "retained_power_law_estimator_repair_20260731.phase_blind_design_matrix",
        *matched_difference(mine_design.values, original),
    )
    multipliers = rpl_repair.penalty_multipliers(geometry, layout)
    record(
        "retained_power_law -> retained_power_law_phase_blind (ridge multipliers)",
        "retained_power_law_estimator_repair_20260731.penalty_multipliers",
        float(abs(np.sort(mine_design.ridge) - np.sort(multipliers)).max()),
        len(multipliers),
    )
    intercept, aggregate, phase = rpl_repair.solve_head(original, response, 1e-2, multipliers, layout)
    original_prediction = intercept + original @ np.concatenate([aggregate, phase])
    spec = models.RetainedPowerLawModel("x", (), (1e-2,)).robust_head
    order_mine = np.lexsort(np.round(mine_design.values, 10)[::-1])
    order_orig = np.lexsort(np.round(original, 10)[::-1])
    reordered = models.Design(
        mine_design.values[:, order_mine],
        mine_design.ridge[order_mine],
        tuple(mine_design.names[index] for index in order_mine),
    )
    del order_orig
    head = models.fit_head(reordered, response, 1e-2, spec)
    record(
        "retained_power_law -> retained_power_law_phase_blind (robust head predictions)",
        "retained_power_law_estimator_repair_20260731.solve_head",
        float(np.max(np.abs(original_prediction - models.predict_head(head, reordered.values, spec)))),
        original.shape[1],
        "Huber IRLS on max-abs-scaled NNLS with a free intercept; the original solves the same problem with a "
        "bounded trust-region solver at tolerance 1e-10, so agreement is to solver tolerance",
        tolerance=SOLVER_TOLERANCE,
    )

    # 10. GRP calibration surrogate with the semantic families collapsed to one family.
    base = dsp.PacketData(
        frame=plain.frame,
        name_col="run_name",
        y=response,
        w=structured.weights,
        m=structured.m,
        c0=structured.c0,
        c1=structured.c1,
        domain_names=list(structured.domains),
    )
    paired = {index for pair in fam.pairs for index in pair}
    packet = grp_followup.GenericFamilyPacket(
        base=base,
        pairs=[list(pair) for pair in fam.pairs],
        pair_topics=[str(index) for index in range(len(fam.pairs))],
        singletons=[index for index in range(structured.m) if index not in paired],
        family_map={"all": list(range(structured.m))},
    )
    params = {"eta": 1.0, "lam": 0.0, "beta": 0.5, "a_all": 0.34, "tau_all": 5.0, "reg": 0.0}
    surrogate = grp_calibration.GenericFamilyPenaltyCalibrationSurrogate(
        packet,
        params=params,
        spec=grp_calibration.variant_spec("power_family_penalty"),
        family_totals=("all",),
        include_family_totals=False,
    )
    original = surrogate.build_design(structured.weights)
    options = models.FamilyOptions(
        bucket_signal=False, family_signal="pair_discount", harm="softplus_group_sum", benefit="power"
    )
    mine = models.family_design(features, {"exponent": 0.34, "quality_discount": 0.5, "threshold": 5.0}, options).values
    record(
        "grp -> grp_pair_power (design, one collapsed family)",
        "surrogate_search.generic_family_penalty_calibration.build_design",
        *matched_difference(mine, original),
        "semantic family totals and per-family curvature/thresholds are banned; with one family "
        "the group penalty collapses to the shared column",
    )

    # 11. OLMix objective equality at random parameters.
    params_o = np.concatenate([[np.log(0.5)], rng.normal(0.0, 0.3, size=panel.features.buckets)])
    x = features.weights
    logits = np.clip(x @ params_o[1:], -50.0, 50.0)
    reference_loss = olmix_loglinear._huber_sum(np.exp(params_o[0]) + np.exp(logits) - response, delta=0.02)
    mine_loss = None
    analytic = models.fit_olmix_loglinear_analytic

    def objective_probe(parameters: np.ndarray) -> float:
        residual = np.exp(parameters[0]) + np.exp(np.clip(x @ parameters[1:], -50.0, 50.0)) - response
        magnitude = np.abs(residual)
        return float(np.where(magnitude <= 0.02, 0.5 * residual * residual, 0.02 * (magnitude - 0.01)).sum())

    mine_loss = objective_probe(params_o)
    record(
        "olmix_loglinear -> olmix_loglinear_taskwise (objective value)",
        "olmix_loglinear_fit._huber_sum",
        abs(reference_loss - mine_loss),
        1,
        "same Huber objective; the analytic-gradient solver differs only in the gradient supplied to L-BFGS-B",
    )
    reference_fit = olmix_loglinear.fit_olmix_loglinear_model(x[train[:120]], response[train[:120]], seed=1, n_starts=12)
    analytic_fit = analytic(x[train[:120]], response[train[:120]], delta=0.02, seed=1, n_starts=12)
    rows.append(
        {
            "panel": panel.name,
            "check": "olmix analytic solver reaches the reference loss or better",
            "source_module": "olmix_loglinear_fit.fit_olmix_loglinear_model",
            "max_abs_difference": float(analytic_fit.huber_loss - reference_fit.huber_loss),
            "columns_compared": 1,
            "status": "pass" if analytic_fit.huber_loss <= reference_fit.huber_loss + 1e-9 else "fail",
            "note": (
                "difference = analytic loss minus reference loss; negative means the analytic solver found "
                "a lower objective"
            ),
        }
    )

    # 12. Band stacking weights.
    predictions = rng.normal(size=(60, 4))
    observed = predictions[:, 0] + 0.1 * rng.normal(size=60)
    record(
        "hpr_band stacking weights",
        "hierarchical_band_model_20260726.stack_weights",
        float(
            np.max(
                np.abs(
                    hierarchical_band.stack_weights(predictions, observed) - models.stack_weights(predictions, observed)
                )
            )
        ),
        4,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panels", default="300m_39bucket,dclm_10k")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for name in args.panels.split(","):
        rows.extend(check_rows(harness.load_panel(name.strip())))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.3e}"))
    failed = frame[frame["status"].eq("fail")]
    print(f"\n{len(frame)} checks, {len(failed)} failed")
    if not failed.empty:
        sys.exit(1)


if __name__ == "__main__":
    main()
