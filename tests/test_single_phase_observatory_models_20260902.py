# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

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
