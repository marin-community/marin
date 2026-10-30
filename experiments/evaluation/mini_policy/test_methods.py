# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for identity preservation and subset response modeling."""

from itertools import product

import numpy as np
from scipy.special import expit
from sklearn.model_selection import GroupKFold

from experiments.evaluation.mini_policy.calibration import tournament_bounds
from experiments.evaluation.mini_policy.experiment import retained_benchmark_indices, retained_responses, tournament
from experiments.evaluation.mini_policy.greedy import greedy_cv_order, prediction_loss
from experiments.evaluation.mini_policy.identities import document_identity
from experiments.evaluation.mini_policy.joint_intervals import vector_tournament
from experiments.evaluation.mini_policy.methods import (
    Mirt,
    estimate_ability,
    fit_mirt,
    information_order,
    predict_panel,
    random_order,
    ridge_prediction,
)
from experiments.evaluation.mini_policy.svd_baseline import fit_pca, predict


def test_dataset_ids_survive_archive_schema_changes():
    # Observed newer TriviaQA manifest ID for official row 5309.
    hashed = "0004e3d4bd8c8311a0f4542e69ef68568e422b988f08d396b7b2a1e86d0236c5"
    document = {"question": "A repeated question", "answer": {"value": "A"}}
    assert document_identity("triviaqa", document, hashed) == document_identity("triviaqa", document, "5309")
    assert document_identity("triviaqa", document, "5309") != document_identity("triviaqa", document, "5310")
    old = {"task_id": "HumanEval/87", "language": "python", "prompt": "code"}
    new = {"task_id": "HumanEval/87", "prompt": "code"}
    assert document_identity("humanevalplus", old, "87") == document_identity("humanevalplus", new, "opaque-id")
    assert document_identity("cruxeval", {"task_id": "sample_582"}, "582") == document_identity(
        "cruxeval", {"id": "sample_582"}, "opaque-id"
    )


def test_harbor_ids_preserve_repo_and_remove_only_hf_namespace():
    left = document_identity("swebench-verified", {}, "astropy__astropy-12907")
    assert left == "astropy__astropy-12907"
    assert document_identity("ds-1000-local", {}, "103") == document_identity("ds-1000-local", {}, "ds-1000/103")
    assert left != document_identity("swebench-verified", {}, "other__astropy-12907")
    assert document_identity("bixbench-pi", {}, "futurehouse/bixbench__bix-1-q1") == document_identity(
        "bixbench-pi", {}, "bix-1-q1"
    )


def test_random_prefix_is_nested_and_balances_benchmark_budget():
    tasks = [{"benchmark": f"b{b}", "strata": {"domain": j % 3}} for b in range(21) for j in range(60)]
    order = random_order(np.arange(len(tasks)), tasks, 42)
    micro = order[:210]
    mini = order[:840]
    assert len(set(mini)) == 840
    assert set(micro) <= set(mini)
    assert all(sum(tasks[j]["benchmark"] == f"b{b}" for j in micro) == 10 for b in range(21))
    assert all(sum(tasks[j]["benchmark"] == f"b{b}" for j in mini) == 40 for b in range(21))


def test_mirt_recovers_new_abilities_from_probes_with_missing_answers():
    # An independent logistic response generator supplies known new-model means.
    rng = np.random.default_rng(32)
    ability = np.linspace(-3, 3, 16)
    discrimination = rng.uniform(0.6, 1.8, 120)
    difficulty = rng.uniform(-2, 2, 120)
    response = expit(ability[:, None] * discrimination + difficulty)
    counts = np.full(response.shape, 10)
    model = fit_mirt(response, counts, np.array(["math"] * 120), np.zeros(120, dtype=bool), 1, 17, maxiter=300)
    predictions = []
    for theta in [-2, 2]:
        observed = expit(theta * discrimination + difficulty)
        observed[::7] = np.nan
        inferred = estimate_ability(model, np.arange(60), observed, np.full(120, 10))
        predicted = predict_panel(model, inferred, np.array(["math"] * 120), ["math"], np.ones(120, dtype=bool))[0]
        expected = float(np.mean(expit(theta * discrimination + difficulty)))
        assert abs(predicted - expected) < 0.08
        predictions.append(predicted)
    assert predictions[1] > predictions[0] + 0.3


def test_static_and_adaptive_information_agree_at_same_ability():
    # With no observed answers and an ability distribution concentrated at zero,
    # the continuous and binary likelihoods have the same information at p=.5.
    a = np.array([[1.0, 0.0], [0.0, 0.9], [0.7, 0.7], [0.1, 0.1]])
    model = Mirt(np.zeros((3, 2)), a, np.zeros(4), np.array([False, True, False, False]), np.ones(4), True, 0, 0.0, 0.0)
    static, _ = information_order(model, np.arange(4), 3, 1)
    adaptive, _ = information_order(model, np.arange(4), 3, 1, np.full(4, np.nan), np.ones(4))
    np.testing.assert_array_equal(static, adaptive)


def test_greedy_validation_prefers_cheaper_duplicate_signal():
    ability = np.linspace(0.05, 0.95, 15)
    response = np.column_stack([ability, ability, np.full(15, 0.5)])
    target = ability[:, None]
    groups = np.repeat(np.arange(5), 3)
    order = greedy_cv_order(
        response, target, np.full(target.shape, 0.01), groups, np.arange(3), 1, 42, np.array([100.0, 1.0, 1.0])
    )
    assert order.tolist() == [1]


def test_cost_information_respects_nested_capability_coverage():
    domains = np.repeat(["reasoning", "coding", "instruction", "long_context", "interactive"], 100)
    model = Mirt(
        np.zeros((3, 1)), np.ones((500, 1)), np.zeros(500), np.zeros(500, dtype=bool), np.ones(500), True, 0, 0.0, 0.0
    )
    costs = np.where(domains == "coding", 1.0, 100.0)
    order, _ = information_order(model, np.arange(500), 210, 42, costs=costs, domains=domains)
    assert len(set(order)) == 210
    assert all(np.sum(domains[order] == domain) >= 10 for domain in set(domains))


def test_greedy_rank_one_selection_matches_independent_ridge_refits():
    # Refit the complete ridge system for every candidate, without using the
    # incremental inverse or cross-products from the selector.
    rng = np.random.default_rng(591)
    response = rng.uniform(0, 1, (18, 12))
    response[::5, ::4] = np.nan
    target = rng.uniform(0.1, 0.9, (18, 4))
    sems = rng.uniform(0.01, 0.05, target.shape)
    groups = np.repeat(np.arange(6), 3)
    costs = rng.uniform(1, 10, response.shape[1])
    splits = list(GroupKFold(n_splits=3).split(response, groups=groups))
    selected = []
    for _ in range(7):
        candidates = sorted(set(range(response.shape[1])) - set(selected))
        losses = []
        baseline = []
        for train, valid in splits:
            candidate_predictions = np.stack(
                [
                    ridge_prediction(response[train], response[valid], target[train], [*selected, candidate])
                    for candidate in candidates
                ]
            )
            predicted_sem = np.median(sems[train], axis=0)
            losses.append(prediction_loss(candidate_predictions, target[valid], sems[valid], predicted_sem))
            current = ridge_prediction(response[train], response[valid], target[train], selected)
            baseline.append(prediction_loss(current[None], target[valid], sems[valid], predicted_sem)[0])
        loss = np.mean(losses, axis=0)
        gain = np.mean(baseline) - loss
        if (gain > 0).any():
            chosen = np.argmax(np.where(gain > 0, gain / costs[candidates], -np.inf))
        else:
            chosen = np.lexsort((costs[candidates], loss))[0]
        selected.append(candidates[chosen])
    actual = greedy_cv_order(response, target, sems, groups, np.arange(12), 7, 42, costs)
    np.testing.assert_array_equal(actual, selected)


def test_tournament_intervals_enclose_scores_with_uncertain_means_and_sems():
    lower = np.array([[0.35], [0.45], [0.55]])
    upper = np.array([[0.65], [0.55], [0.80]])
    sem_lower = np.array([[0.01], [0.03], [0.02]])
    sem_upper = np.array([[0.12], [0.08], [0.10]])
    low, high = tournament_bounds(lower, upper, sem_lower, sem_upper)
    for choices in product((0, 1), repeat=6):
        score = np.where(np.array(choices[:3])[:, None], upper, lower)
        sem = np.where(np.array(choices[3:])[:, None], sem_upper, sem_lower)
        actual = tournament(score, sem)
        assert np.all(actual >= low - 1e-12)
        assert np.all(actual <= high + 1e-12)
    point = (lower + upper) / 2
    sem = (sem_lower + sem_upper) / 2
    point_low, point_high = tournament_bounds(point, point, sem, sem)
    np.testing.assert_allclose(point_low, tournament(point, sem))
    np.testing.assert_allclose(point_high, tournament(point, sem))


def test_batched_tournament_matches_full_rule_for_each_draw():
    rng = np.random.default_rng(96)
    scores = rng.uniform(0, 1, (20, 4, 7))
    sems = rng.uniform(0.01, 0.1, scores.shape)
    actual = vector_tournament(scores, sems)
    expected = np.stack([tournament(score, sem) for score, sem in zip(scores, sems, strict=True)])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_pca_recovers_heldout_scores_from_partial_probe_responses():
    rng = np.random.default_rng(200)
    ability = np.linspace(-2, 2, 80)
    loading = rng.uniform(0.1, 0.2, 160)
    response = 0.5 + ability[:, None] * loading
    response[::9, ::7] = np.nan
    target = np.column_stack([0.5 + 0.15 * ability, 0.5 - 0.12 * ability])
    new_ability = np.array([-1.5, 1.5])
    heldout = 0.5 + new_ability[:, None] * loading
    heldout[:, ::11] = np.nan
    expected = np.column_stack([0.5 + 0.15 * new_ability, 0.5 - 0.12 * new_ability])
    pool = np.arange(160)
    fit = fit_pca(response, pool, np.repeat(["reasoning", "coding", "agents", "long_context"], 40), 1)
    point = predict(response, heldout, target, pool, pool[::4], fit)
    np.testing.assert_allclose(point, expected, atol=0.05)


def test_tournament_scores_preserve_exact_ties_under_model_permutation():
    rng = np.random.default_rng(2026)
    scores = rng.uniform(0, 1, (18, 23))
    sems = rng.uniform(0.01, 0.05, scores.shape)
    permutation = rng.permutation(len(scores))
    original = tournament(scores, sems)
    reordered = tournament(scores[permutation], sems[permutation])
    np.testing.assert_array_equal(original[permutation], reordered)
    numerator = np.rint(original * (2 * 17 * 23))
    np.testing.assert_array_equal(original, numerator / (2 * 17 * 23))


def test_excluded_benchmark_grades_cannot_change_irt_fit():
    rng = np.random.default_rng(42)
    response = rng.integers(0, 2, size=(8, 12)).astype(float)
    benchmarks = np.asarray(["math500"] * 4 + ["boolq"] * 4 + ["NUPA"] * 4)
    altered = response.copy()
    altered[:, 4:8] = 1 - altered[:, 4:8]
    fits = [
        fit_mirt(
            retained_responses(panel, benchmarks),
            np.ones(panel.shape),
            benchmarks,
            np.zeros(12, dtype=bool),
            1,
            7,
        )
        for panel in (response, altered)
    ]
    np.testing.assert_array_equal(fits[0].ability, fits[1].ability)
    np.testing.assert_array_equal(fits[0].discrimination, fits[1].discrimination)
    np.testing.assert_array_equal(fits[0].intercept, fits[1].intercept)


def test_predictor_scope_retains_triviaqa_and_excludes_the_same_five_targets():
    benchmarks = [f"retained-{i}" for i in range(20)] + [
        "financebench",
        "piqa",
        "triviaqa",
        "boolq",
        "winogrande",
        "SOTOPIA-hard",
    ]
    projected = [benchmarks[j] for j in retained_benchmark_indices(benchmarks)]
    assert projected == [*benchmarks[:20], "triviaqa"]
