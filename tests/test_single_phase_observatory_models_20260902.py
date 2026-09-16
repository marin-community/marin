# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import pathlib

import numpy as np
import pytest
from scipy.optimize import nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_dsp_single_phase_ladder_20260824 as dsp_ladder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

BUCKETS_39 = (
    "dolma3_cc/art_high",
    "dolma3_cc/art_low",
    "dolma3_cc/games_high",
    "dolma3_cc/games_low",
    "dolma3_arxiv",
    "dolmino_synth_qa",
)


def _features(seed: int = 0, rows: int = 60, buckets: tuple[str, ...] = BUCKETS_39) -> models.Features:
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(buckets)) * 2.0, size=rows)
    inventory = rng.uniform(2.0, 40.0, size=len(buckets))
    return models.features_from_panel(
        weights, inventory, buckets, early_fraction=np.full(len(buckets), 0.8), label="fixture"
    )


def _response(features: models.Features, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    exposure = features.exposures
    return (
        1.0
        - 0.02 * np.log1p(exposure).sum(axis=1)
        + 0.001 * np.maximum(exposure - 5.0, 0.0).sum(axis=1)
        + 0.002 * rng.normal(size=features.rows)
    )


def _folds(rows: int) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    train = np.arange(rows)
    labels = train % 3
    return train, tuple((train[labels != index], train[labels == index]) for index in range(3))


def test_families_from_buckets_pairs_cc_quality_splits_and_keeps_singletons():
    families = models.families_from_buckets(BUCKETS_39)

    assert families.names == ("dolma3_cc/art", "dolma3_cc/games", "dolma3_arxiv", "dolmino_synth_qa")
    assert families.pairs == ((0, 1), (2, 3))
    assert families.nonsingleton == (0, 1)
    assert families.singleton_buckets.tolist() == [4, 5]
    assert families.quality_ordered


def test_families_from_manifest_cluster_names_are_unordered():
    families = models.families_from_buckets(("c00_q0", "c00_q1", "c01_q0", "c01_q3"))

    assert families.names == ("c00", "c01")
    assert families.quality.tolist() == [0, 1, 0, 3]
    assert families.pairs == ()
    assert not families.quality_ordered


def test_shuffled_families_preserve_sizes_and_quality_labels():
    families = models.families_from_buckets(BUCKETS_39)
    shuffled = models.shuffled_families(families, seed=3)

    assert sorted(len(block) for block in shuffled.members) == sorted(len(block) for block in families.members)
    assert sorted(shuffled.quality.tolist()) == sorted(families.quality.tolist())
    assert np.array_equal(np.sort(np.concatenate(shuffled.members)), np.arange(len(BUCKETS_39)))


def test_reduced_nnls_matches_direct_solve():
    features = _features()
    response = _response(features)
    design = models.family_design(features, {"exponent": 0.4, "threshold": 3.0}, registry.BUCKET_FAMILY_OPTIONS)
    direct = models.fit_head(design, response, 0.01, models.HeadSpec(reduced_nnls=False))
    reduced = models.fit_head(design, response, 0.01, models.HeadSpec(reduced_nnls=True))

    assert np.allclose(direct.coefficients, reduced.coefficients, atol=1e-9)
    assert abs(direct.intercept - reduced.intercept) < 1e-9


def test_nonnegative_head_matches_reference_nnls_with_ridge_rows():
    features = _features()
    response = _response(features)
    design = models.family_design(features, {"exponent": 0.4, "threshold": 3.0}, registry.BUCKET_FAMILY_OPTIONS)
    head = models.fit_head(design, response, 0.1, models.HeadSpec())
    centered = design.values - design.values.mean(axis=0)
    rows = np.vstack([centered, np.sqrt(0.1) * np.eye(design.values.shape[1])])
    expected, _ = nnls(rows, np.concatenate([response - response.mean(), np.zeros(design.values.shape[1])]))

    assert np.allclose(head.coefficients, expected, atol=1e-9)
    assert np.all(head.coefficients >= 0.0)


def test_profiled_dsp_matches_ladder_canonical_rung():
    features = _features(rows=45)
    response = _response(features)
    train, inner = _folds(features.rows)
    model = models.ProfiledDspModel(options=models.DspOptions(maxiter=8))
    fitted = model.fit(features, response, train, inner, seed=5)
    canonical = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    vector, intercept, coefficients = dsp_ladder.fit_rung(
        features.exposures, response, canonical, inner, (), seed=5, maxiter=8, restarts=2
    )

    assert np.allclose([fitted.shape[f"theta_{index}"] for index in range(len(vector))], vector, atol=1e-12)
    assert abs(fitted.head.intercept - intercept) < 1e-12
    assert np.allclose(fitted.head.coefficients, coefficients, atol=1e-12)


def test_dsp_no_harm_option_drops_penalty_columns():
    features = _features()
    model = models.ProfiledDspModel(options=models.DspOptions(penalty="none"))
    design = model.design(features, np.zeros(features.buckets))

    assert design.shape == (features.rows, features.buckets)
    assert model.nonlinear_dof(features) == features.buckets


def test_weight_coordinate_transform_makes_exposure_equal_weight():
    features = _features().with_weight_coordinate()

    assert np.array_equal(features.exposures, features.weights)


def test_permuted_inventory_keeps_bucket_inventories_as_a_multiset():
    features = _features()
    permuted = features.with_permuted_inventory(seed=1)

    assert sorted(permuted.inventory.tolist()) == sorted(features.inventory.tolist())
    assert not np.array_equal(permuted.exposures, features.exposures)


def test_log_deficit_link_predictions_stay_above_floor():
    features = _features()
    response = _response(features)
    spec = models.HeadSpec(scale_columns=True, link=models.LinkKind.LOG_DEFICIT)
    design = models.family_design(
        features,
        {"rate": 1.0, "power": 0.7},
        models.FamilyOptions(family_signal="none", harm="literal_shared", benefit="weibull"),
    )
    head = models.fit_head(design, response, 0.1, spec)
    prediction = models.predict_head(head, design.values, spec)

    assert head.floor == pytest.approx(0.95 * response.min())
    assert np.all(prediction > head.floor)


def test_grid_model_two_stage_screen_agrees_with_exhaustive_search_when_optimum_is_separated():
    features = _features(rows=90)
    response = _response(features)
    train, inner = _folds(features.rows)
    entry = registry.PARENT_BY_ID["bucket_family_power_grp"]
    exhaustive = dataclasses.replace(entry.build(features), screen_top=10_000, model_id="exhaustive")
    staged = dataclasses.replace(entry.build(features), screen_top=8, model_id="staged")
    full = exhaustive.fit(features, response, train, inner, 0)
    short = staged.fit(features, response, train, inner, 0)

    assert full.shape == short.shape
    assert full.ridge == short.ridge


def test_every_registered_model_fits_and_predicts_on_a_small_panel():
    features = _features(rows=48)
    response = _response(features)
    train, inner = _folds(features.rows)
    slow = {"olmix_loglinear_taskwise", "olmix_loglinear_taskwise@reference_solver"}
    for entry in registry.PARENTS + registry.REFERENCES:
        if entry.model_id in slow:
            continue
        transformed = registry.apply_transform(features, entry)
        model = entry.build(transformed)
        if isinstance(model, models.ProfiledDspModel):
            model = dataclasses.replace(model, options=dataclasses.replace(model.options, maxiter=4))
        if isinstance(model, models.GridModel):
            model = dataclasses.replace(model, shapes=model.shapes[:6])
        fitted = model.fit(transformed, response, train, inner, 0)
        prediction = model.predict(fitted, transformed, train)
        assert prediction.shape == (len(train),), entry.model_id
        assert np.isfinite(prediction).all(), entry.model_id
        assert fitted.diagnostics["effective_rank"] <= fitted.diagnostics["columns"] or entry.model_id in {"fold_mean"}


def test_registry_covers_every_observatory_model_once():
    covered = [source for entry in registry.PARENTS for source in entry.source_model_ids]

    assert sorted(covered) == sorted(registry.OBSERVATORY_MODEL_IDS)
    assert all(entry.parent in registry.ENTRY_BY_ID for entry in registry.ABLATIONS)


def test_stack_weights_return_a_simplex_vector_that_prefers_the_better_member():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=80)
    predictions = np.column_stack([truth + 0.5 * rng.normal(size=80), truth + 0.05 * rng.normal(size=80)])
    weights = models.stack_weights(predictions, truth)

    assert weights.shape == (2,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] > weights[0]


def test_row_scrambled_harm_permutes_mixtures_while_column_scrambling_only_reorders_a_per_bucket_harm():
    features = _features()
    shape = {"rate": 0.5, "power": 0.5, "threshold": 2.0}
    base = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    plain = models.family_design(features, shape, base)
    columns = models.family_design(features, shape, dataclasses.replace(base, scrambled_harm=True))
    rows = models.family_design(features, shape, dataclasses.replace(base, row_scrambled_harm=True))
    harm = [index for index, name in enumerate(plain.names) if name.startswith("bucket_overexposure")]
    benefit = [index for index in range(len(plain.names)) if index not in harm]

    assert len(harm) == features.buckets
    assert np.allclose(np.sort(columns.values[:, harm], axis=1), np.sort(plain.values[:, harm], axis=1))
    assert not np.allclose(rows.values[:, harm], plain.values[:, harm])
    assert np.allclose(np.sort(rows.values[:, harm], axis=0), np.sort(plain.values[:, harm], axis=0))
    assert np.allclose(rows.values[:, benefit], plain.values[:, benefit])


def _constant_repr(value: object) -> str:
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}({{{', '.join(repr(item) for item in sorted(value))}}})"
    return repr(value)


def test_fit_helpers_change_only_with_a_design_revision_bump():
    """Cache acceptance compares built-model descriptions, which do not see helper bodies or default constants.

    A helper listed in the pin file may change its source only together with a higher DESIGN_REVISIONS entry
    (which changes every dependent description and refits the affected shards) or a deliberate pin refresh.
    """
    pins = json.loads(
        (pathlib.Path(__file__).parent / "data" / "single_phase_observatory_helper_pins.json").read_text()
    )["pins"]
    drifted = []
    for name, pin in pins.items():
        target = models
        for part in name.split("."):
            target = getattr(target, part)
        source = inspect.getsource(target)
        if hashlib.sha256(source.encode()).hexdigest() == pin["source_sha256"]:
            continue
        if models.DESIGN_REVISIONS.get(name, 1) > pin["design_revision"]:
            continue
        drifted.append(name)

    assert not drifted, f"helpers changed without a DESIGN_REVISIONS bump or pin refresh: {drifted}"

    constants = json.loads(
        (pathlib.Path(__file__).parent / "data" / "single_phase_observatory_helper_pins.json").read_text()
    )["constants"]
    # Set-valued constants are compared in sorted form: their repr order is hash-randomized per process.
    current = {name: _constant_repr(getattr(models, name)) for name in constants["values"]}
    assert current == constants["values"], "module constants read by the pinned helpers changed without a pin refresh"


def test_interaction_columns_come_in_signed_pairs():
    features = _features()
    shape = {"rate": 0.5, "power": 0.5, "threshold": 2.0}
    base = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    plain = models.family_design(features, shape, base)
    total = models.family_design(features, shape, dataclasses.replace(base, interaction="total_square"))
    pairs = models.family_design(features, shape, dataclasses.replace(base, interaction="family_products"))

    assert total.values.shape[1] == plain.values.shape[1] + 2
    assert np.allclose(total.values[:, -1], -total.values[:, -2])
    assert pairs.values.shape[1] == plain.values.shape[1] + 2 * len(features.families.pairs)


def test_quality_axis_pools_across_families_and_the_shuffled_control_differs():
    features = _features()
    shape = {"rate": 0.5, "power": 0.5, "threshold": 2.0}
    base = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    plain = models.family_design(features, shape, base)
    both = models.family_design(features, shape, dataclasses.replace(base, quality_axis="both"))
    shuffled = models.family_design(
        features, shape, dataclasses.replace(base, quality_axis="both", shuffled_quality=True)
    )
    levels = sorted({int(level) for level in features.families.quality if level >= 0})

    assert both.values.shape[1] == plain.values.shape[1] + 2 * len(levels)
    assert [name for name in both.names if name.startswith("quality_")] == [
        f"quality_benefit:{level}" for level in levels
    ] + [f"quality_harm:{level}" for level in levels]
    assert not np.allclose(both.values[:, -2 * len(levels) :], shuffled.values[:, -2 * len(levels) :])


def test_bounded_log_deficit_link_caps_extrapolated_predictions():
    features = _features()
    response = _response(features)
    design = models.family_design(
        features,
        {"rate": 0.5, "power": 0.5, "threshold": 2.0},
        models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull"),
    )
    spec = models.HeadSpec(kind=models.HeadKind.NNLS, link=models.LinkKind.LOG_DEFICIT_BOUNDED)
    head = models.fit_head(design, response, 0.01, spec)
    extreme = models.predict_head(head, design.values * 50.0, spec)

    assert np.isfinite(head.cap)
    assert np.all(extreme <= head.floor + np.exp(head.cap) + 1e-9)
    assert np.exp(head.cap) <= (response.max() - head.floor) * np.exp(models.LINK_CAP_MARGIN) + 1e-9


def test_grid_model_records_the_full_inner_cv_table():
    features = _features()
    response = _response(features)
    train, inner = _folds(features.rows)
    options = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    shapes = ({"rate": 0.5, "power": 0.5, "threshold": 2.0}, {"rate": 1.0, "power": 0.7, "threshold": 3.0})
    model = models.GridModel(
        "grid",
        lambda feats, shape: models.family_design(feats, shape, options),
        shapes,
        (0.0, 0.1),
        models.HeadSpec(),
        3,
    )
    fitted = model.fit(features, response, train, inner, 0)

    assert fitted.cv_table is not None and fitted.cv_table.shape == (2, 2)
    assert np.isfinite(fitted.cv_table).all()
    assert fitted.diagnostics["inner_cv_rmse"] == fitted.cv_table.min()


def test_component_ridge_prior_scales_only_the_named_component():
    features = dataclasses.replace(_features(), component="metric_a")
    shape = {"rate": 0.5, "power": 0.5, "threshold": 2.0}
    bucket = features.buckets_names[0]
    table = (("metric_a", ((bucket, 10.0),)),)
    options = models.FamilyOptions(
        family_signal="none", harm="softplus_bucket", benefit="weibull", component_ridge=table
    )
    prior = models.family_design(features, shape, options)
    other = models.family_design(dataclasses.replace(features, component="metric_b"), shape, options)

    assert prior.ridge[prior.names.index("bucket_signal:0")] == 10.0
    assert prior.ridge[prior.names.index("bucket_overexposure:0")] == 10.0
    assert prior.ridge[prior.names.index("bucket_signal:1")] == 1.0
    assert np.all(other.ridge == 1.0)


def test_refined_grid_model_never_scores_worse_than_its_grid_argmin():
    features = _features()
    response = _response(features)
    train, inner = _folds(features.rows)
    options = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    shapes = ({"rate": 0.5, "power": 0.5, "threshold": 2.0}, {"rate": 1.0, "power": 0.7, "threshold": 3.0})
    grid = models.GridModel(
        "grid",
        lambda feats, shape: models.family_design(feats, shape, options),
        shapes,
        (0.0, 0.1),
        models.HeadSpec(),
        3,
    )
    refined = dataclasses.replace(grid, refine=True, refine_evaluations=40)
    plain = grid.fit(features, response, train, inner, 0)
    better = refined.fit(features, response, train, inner, 0)

    assert better.diagnostics["inner_cv_rmse"] <= plain.diagnostics["inner_cv_rmse"]
    assert better.diagnostics["refine_evaluations"] > 0
    assert set(better.shape) == set(plain.shape)


def test_grid_model_selects_a_link_by_inner_cv_and_predicts_with_it():
    features = _features()
    response = _response(features)
    train, inner = _folds(features.rows)
    options = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    shapes = ({"rate": 0.5, "power": 0.5, "threshold": 2.0},)
    grid = models.GridModel(
        "grid",
        lambda feats, shape: models.family_design(feats, shape, options),
        shapes,
        (0.0, 0.1),
        models.HeadSpec(),
        3,
    )
    both = dataclasses.replace(grid, link_candidates=(models.LinkKind.IDENTITY, models.LinkKind.LOG_DEFICIT_BOUNDED))
    plain = grid.fit(features, response, train, inner, 0)
    chosen = both.fit(features, response, train, inner, 0)

    assert chosen.diagnostics["link"] in {"identity", "log_deficit_bounded"}
    assert chosen.diagnostics["inner_cv_rmse"] <= plain.diagnostics["inner_cv_rmse"]
    assert np.isfinite(both.predict(chosen, features, train)).all()


def test_features_from_panel_scales_exposures_by_pool_fraction():
    weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    inventory = np.array([2.0, 8.0])
    pools = np.array([[1.0, 1.0], [0.5, 1.0]])
    features = models.features_from_panel(
        weights, inventory, ("a", "b"), early_fraction=None, label="t", pool_fractions=pools
    )
    np.testing.assert_allclose(features.exposures, [[1.0, 4.0], [2.0, 4.0]])
    np.testing.assert_allclose(features.weights, weights)
    with pytest.raises(ValueError, match="pool_fractions"):
        models.features_from_panel(
            weights,
            inventory,
            ("a", "b"),
            early_fraction=None,
            label="t",
            pool_fractions=np.array([[1.0, 1.5], [1.0, 1.0]]),
        )


def test_fitted_floor_link_places_the_floor_below_the_training_minimum_and_caps_extrapolation():
    features = _features()
    response = _response(features)
    design = models.family_design(
        features,
        {"rate": 0.5, "power": 0.5, "threshold": 2.0},
        models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull"),
    )
    anchor = float(np.median(response))
    spec = models.HeadSpec(
        kind=models.HeadKind.NNLS, link=models.LinkKind.FITTED_FLOOR, floor_anchor=anchor, noise_sd=0.002
    )
    head = models.fit_head(design, response, 0.01, spec)
    fitted = models.predict_head(head, design.values, spec)
    extreme = models.predict_head(head, design.values * 50.0, spec)

    low, high = spec.floor_kappa_bounds
    assert low <= head.kappa <= high
    assert head.floor < response.min()
    assert np.isclose(head.floor, anchor - max(head.kappa * (anchor - response.min()), 3 * 0.002))
    assert np.all(fitted > head.floor)
    assert np.all(extreme <= head.floor + np.exp(head.cap) + 1e-9)
    # The response-space fit is at least as close in-sample as the log-space fit at the 0.95 floor.
    bounded = models.HeadSpec(kind=models.HeadKind.NNLS, link=models.LinkKind.LOG_DEFICIT_BOUNDED)
    reference = models.predict_head(models.fit_head(design, response, 0.01, bounded), design.values, bounded)
    assert np.sqrt(np.mean((fitted - response) ** 2)) <= 1.02 * np.sqrt(np.mean((reference - response) ** 2))


def test_fitted_floor_model_chooses_kappa_by_inner_cv_and_keeps_the_identity_link_when_it_wins():
    features = _features()
    train, inner = _folds(features.rows)
    options = models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    shapes = ({"rate": 0.5, "power": 0.5, "threshold": 2.0},)
    base = models.GridModel(
        "floor",
        lambda feats, shape: models.family_design(feats, shape, options),
        shapes,
        (0.01,),
        models.HeadSpec(kind=models.HeadKind.NNLS, link=models.LinkKind.KAPPA_FLOOR),
        3,
    )
    model = models.FittedFloorModel(base=base, kappa_bounds=(1.0, 8.0))

    # A response that is exactly additive in the design prefers the identity link.
    additive = _response(features)
    fit = model.fit(features, additive, train, inner, 0)
    assert fit.diagnostics["link"] == "identity"
    assert np.isnan(fit.diagnostics["kappa"])
    assert np.isfinite(model.predict(fit, features, train)).all()

    # A response that is a floor plus an exponential of the same design prefers the fitted floor.
    design = base.design(features, shapes[0])
    rng = np.random.default_rng(3)
    linear = -0.6 * design.values[:, : features.buckets].sum(axis=1) / features.buckets
    curved = 0.7 + np.exp(-1.0 + 2.0 * linear) + 0.002 * rng.normal(size=features.rows)
    fit = model.fit(features, curved, train, inner, 0)
    assert fit.diagnostics["link"] == "fitted_floor"
    assert 1.0 <= fit.diagnostics["kappa"] <= 8.0
    assert fit.diagnostics["floor"] < curved.min()
    assert fit.diagnostics["fitted_floor_inner_cv_rmse"] < fit.diagnostics["identity_inner_cv_rmse"]
    assert fit.diagnostics["fitted_floor_inner_cv_rmse"] <= fit.diagnostics["log_deficit_bounded_inner_cv_rmse"]


def test_per_bucket_design_with_uniform_shapes_equals_the_shared_successor_design():
    features = _features()
    shape = {"rate": 0.5, "power": 0.7, "threshold": 2.0}
    shared = models.family_design(features, shape, registry.SUCCESSOR_OPTIONS)
    per_bucket = models.per_bucket_weibull_softplus_design(features, models.per_bucket_shape(shape, features.buckets))

    assert per_bucket.names == shared.names
    np.testing.assert_array_equal(per_bucket.values, shared.values)
    np.testing.assert_array_equal(per_bucket.ridge, shared.ridge)


def test_per_bucket_shape_search_starts_at_the_shared_optimum_and_never_scores_worse():
    features = _features(rows=90)
    response = _response(features)
    train, inner = _folds(features.rows)
    shapes = (
        {"rate": 0.25, "power": 0.5, "threshold": 2.0},
        {"rate": 1.0, "power": 0.7, "threshold": 3.0},
        {"rate": 2.0, "power": 1.0, "threshold": 1.0},
    )
    shared = models.GridModel(
        "grid",
        lambda feats, shape: models.family_design(feats, shape, registry.SUCCESSOR_OPTIONS),
        shapes,
        (0.0, 0.1),
        models.HeadSpec(),
        3,
    )
    shared_fit = shared.fit(features, response, train, inner, 0)
    model = models.PerBucketShapeGridModel(shared=shared, head=shared.head)
    fitted = model.fit(features, response, train, inner, 0)

    assert set(fitted.shape) == {
        f"{key}:{index}" for key in models.PER_BUCKET_SHAPE_KEYS for index in range(features.buckets)
    }
    assert fitted.diagnostics["shared_inner_cv_rmse"] == shared_fit.diagnostics["inner_cv_rmse"]
    assert fitted.diagnostics["inner_cv_rmse"] <= shared_fit.diagnostics["inner_cv_rmse"] + 1e-12
    assert fitted.diagnostics["nonlinear_dof"] == 3 * features.buckets
    assert all(
        (fitted.shape[f"rate:{index}"], fitted.shape[f"power:{index}"], fitted.shape[f"threshold:{index}"])
        in {(s["rate"], s["power"], s["threshold"]) for s in shapes}
        for index in range(features.buckets)
    )
    predictions = model.predict(fitted, features, train)
    assert predictions.shape == (features.rows,) and np.isfinite(predictions).all()


def test_per_bucket_shape_entry_wraps_the_frozen_procedure_in_the_registry():
    entry = registry.ENTRY_BY_ID["weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_per_bucket_shape"]
    parent = registry.ENTRY_BY_ID["weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"]
    assert entry.mechanisms["sharing"] == "per_bucket"
    assert {k: v for k, v in entry.mechanisms.items() if k != "sharing"} == {
        k: v for k, v in parent.mechanisms.items() if k != "sharing"
    }
    model = entry.build(_features())
    assert isinstance(model, models.FittedFloorModel)
    assert isinstance(model.base, models.PerBucketShapeGridModel)
    assert model.flat_profile_kappa == 1.5 and model.kappa_bounds == (1.0, 6.0)


@pytest.mark.parametrize(
    "suffix",
    (
        "no_harm",
        "row_scrambled_harm",
        "permuted_inventory",
        "weight_coordinate",
        "common_inventory",
        "signed_head",
        "outcome_permutation",
    ),
)
def test_frozen_procedure_ablation_entries_build_and_fit(suffix):
    entry = registry.ENTRY_BY_ID[f"weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_{suffix}"]
    features = registry.apply_transform(_features(rows=60), entry)
    response = _response(features)
    train, inner = _folds(features.rows)
    model = entry.build(features)
    assert isinstance(model, models.FittedFloorModel)
    fitted = model.fit(features, response, train, inner, 0)
    predictions = model.predict(fitted, features, train)
    assert np.isfinite(predictions).all()
    if suffix == "no_harm":
        assert "threshold" not in fitted.shape
    if suffix == "signed_head":
        assert model.base.head.kind is models.HeadKind.RIDGE


def test_log_quadratic_design_recovers_an_additive_quadratic_in_log_epochs():
    features = _features(rows=200)
    u = np.log1p(features.exposures)
    response = 1.0 - 0.3 * u[:, 0] + 0.2 * u[:, 1] ** 2 + 0.1 * u[:, 2] - 0.05 * u[:, 3] ** 2
    design = models.log_quadratic_design(features, {})
    assert design.values.shape == (200, 2 * features.buckets)
    head = models.fit_head(design, response, 1e-10, models.HeadSpec(kind=models.HeadKind.RIDGE))
    prediction = models.predict_head(head, design.values, models.HeadSpec(kind=models.HeadKind.RIDGE))
    assert np.max(np.abs(prediction - response)) < 1e-6


def test_spline_knots_bound_at_build_time_carry_the_fitted_basis_to_a_single_query_row():
    features = dataclasses.replace(_features(rows=200), component="task")
    rows = np.arange(features.rows)
    response = np.log1p(features.exposures[:, 0]) - 0.1 * np.log1p(features.exposures[:, 1]) ** 2 + 1.0
    inner = tuple((rows[rows % 3 != k], rows[rows % 3 == k]) for k in range(3))
    model = registry.ENTRY_BY_ID[registry.SPLINE_LINKED_ID].build(features)
    fitted = model.fit(features, response, rows, inner, 0)
    whole = model.predict(fitted, features, rows)
    single = dataclasses.replace(
        features,
        exposures=features.exposures[7:8],
        weights=features.weights[7:8],
        label="query",
    )
    # A one-row query has no quartiles of its own; the prediction must still use the swarm's knots.
    assert model.predict(fitted, single, np.array([0]))[0] == pytest.approx(whole[7], abs=1e-9)
    assert models.spline_knots(single) == (None,) * features.buckets


def test_natural_spline_design_has_four_columns_per_bucket_and_is_linear_beyond_the_boundary_knots():
    features = _features(rows=200)
    design = models.natural_spline_design(features, {})
    assert design.values.shape == (200, 4 * features.buckets)
    knots = np.array([0.1, 0.3, 0.6, 0.9, 1.4])
    far = np.array([1.4, 2.4, 3.4, 4.4])
    basis = models.natural_cubic_basis(far, knots)
    differences = np.diff(basis, axis=0)
    # Natural constraint: every column is affine beyond the last knot, so consecutive differences are constant.
    assert np.allclose(differences[1:], differences[:-1], atol=1e-8)


def test_hellinger_kernel_ridge_predicts_a_smooth_function_out_of_fold():
    features = _features(seed=1, rows=160)
    response = np.sin(6.0 * features.weights[:, 0]) + 4.0 * features.weights[:, 1] ** 2
    train = np.arange(120)
    inner = tuple((train[np.arange(120) % 3 != k], train[np.arange(120) % 3 == k]) for k in range(3))
    model = models.HellingerKernelRidgeModel()
    fitted = model.fit(features, response, train, inner, seed=0)
    prediction = model.predict(fitted, features, np.arange(120, 160))
    residual = response[120:] - prediction
    assert np.sqrt(np.mean(residual**2)) < 0.5 * np.std(response[120:])
    assert (
        fitted.head.train_weights.shape == (120, features.buckets)
        and fitted.shape["gamma"] in models.HELLINGER_GAMMA_GRID
    )
    # Predictions must not depend on the row layout of the feature set they are asked about.
    shuffled = np.random.default_rng(3).permutation(160)
    other = dataclasses.replace(features, weights=features.weights[shuffled], exposures=features.exposures[shuffled])
    assert np.allclose(model.predict(fitted, other, np.arange(160)), model.predict(fitted, features, shuffled))


def test_single_amplitude_and_hinge_harm_designs_have_the_expected_columns():
    features = _features(rows=80)
    shape = {"rate": 0.25, "power": 1.0, "threshold": 2.0}
    single = models.family_design(
        features, shape, models.FamilyOptions(family_signal="none", harm="softplus_bucket_sum", benefit="weibull")
    )
    hinge = models.family_design(
        features, shape, models.FamilyOptions(family_signal="none", harm="softplus_bucket_hinge", benefit="weibull")
    )
    full = models.family_design(
        features, shape, models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    )
    assert single.values.shape[1] == features.buckets + 1
    assert hinge.values.shape[1] == full.values.shape[1] == 2 * features.buckets
    # The summed column equals the sum of the per-bucket squared-softplus harms; the hinge is their square root.
    assert np.allclose(single.values[:, -1], full.values[:, features.buckets :].sum(axis=1))
    assert np.allclose(hinge.values[:, features.buckets :] ** 2, full.values[:, features.buckets :])


def test_nonparametric_comparators_fit_and_predict_out_of_fold():
    pytest.importorskip("lightgbm")
    features = _features(seed=2, rows=160)
    response = (
        1.0
        + 2.0 * features.weights[:, 0]
        - features.weights[:, 1] ** 2
        + 0.5 * features.weights[:, 2] * features.weights[:, 3]
    )
    train = np.arange(120)
    inner = tuple((train[np.arange(120) % 3 != k], train[np.arange(120) % 3 == k]) for k in range(3))
    for model in (models.LightGBMModel(), models.MLPModel()):
        fitted = model.fit(features, response, train, inner, seed=0)
        prediction = model.predict(fitted, features, np.arange(120, 160))
        assert prediction.shape == (40,) and np.isfinite(prediction).all()
        assert np.corrcoef(prediction, response[120:])[0, 1] > 0.5, model.model_id
        assert fitted.head.active == 120


def test_olmix_taskwise_log_epoch_coordinate_is_log1p_of_exposures():
    features = _features(rows=40)
    model = models.OlmixTaskwiseModel(model_id="x", coordinate="log_epoch")
    assert np.allclose(model._matrix(features), np.log1p(features.exposures))
    with pytest.raises(ValueError):
        models.OlmixTaskwiseModel(model_id="x", coordinate="epochs")._matrix(features)
