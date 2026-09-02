# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)


def test_metric_row_selection_metrics_follow_the_predicted_minimum():
    observed = np.asarray([1.0, 0.9, 1.2, 0.95, 1.1])
    predicted = np.asarray([0.98, 1.05, 1.3, 0.90, 1.0])
    basin = np.asarray([False, True, False, True, False])

    row = harness.metric_row(observed, predicted, basin)

    assert row["regret_at_1"] == pytest.approx(0.95 - 0.9)
    assert row["regret_at_top_k"] == pytest.approx(0.0)
    assert row["selection_optimism"] == pytest.approx(0.95 - 0.90)
    assert row["basin_rmse"] == pytest.approx(np.sqrt(np.mean([(1.05 - 0.9) ** 2, (0.90 - 0.95) ** 2])))
    assert row["n_basin"] == 2


def test_metric_row_reports_failed_predictions_as_nan_with_a_count():
    row = harness.metric_row(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, np.nan, 3.0]), np.zeros(3, dtype=bool))

    assert np.isnan(row["rmse"])
    assert row["n_failed"] == 1


def test_random_ranking_expectations_match_order_statistics():
    loss = np.asarray([1.0, 1.1, 1.3, 1.6])
    expected = harness.random_ranking_expectations(loss, top_k=2)
    ordered = np.sort(loss) - loss.min()
    manual = sum(comb(4 - 1 - i, 1) / comb(4, 2) * ordered[i] for i in range(4))

    assert expected["random_regret_at_1"] == pytest.approx(ordered.mean())
    assert expected["random_best_of_2_regret"] == pytest.approx(manual)
    assert expected["random_best_of_10_regret"] == pytest.approx(0.0)


def test_corrected_contrast_uses_nadeau_bengio_variance():
    difference = np.asarray([0.1, 0.2, 0.15, 0.05, 0.1])
    row = harness.corrected_contrast(difference, test_train_ratio=0.25, folds_per_repeat=5, repeats=1)
    factor = 1.0 / 5 + 0.25

    assert row["mean"] == pytest.approx(difference.mean())
    assert row["corrected_se"] == pytest.approx(np.sqrt(factor * difference.var(ddof=1)))
    assert row["ci_low"] < row["mean"] < row["ci_high"]


def test_certify_plan_has_the_handoff_shard_count():
    plan = harness.tier_plan("certify")
    tasks = harness.plan_tasks(plan, ("fold_mean",))

    assert len(tasks) == 5 * (3 * 58 + 2 * 8 + 45)
    assert len({(task.panel, task.target, task.component, task.fold) for task in tasks}) == len(tasks)


def test_screen_plan_uses_the_six_anchors_and_eight_michael_tasks():
    plan = harness.tier_plan("screen")

    assert {
        len(components)
        for panel, targets in plan.components.items()
        if panel in harness.THIRTY_NINE_BUCKET_PANELS
        for components in targets.values()
    } == {2, 4}
    assert all(
        tuple(targets[harness.MICHAEL_TARGET]) == harness.MICHAEL_TASKS
        for panel, targets in plan.components.items()
        if panel in harness.MICHAEL_PANELS
    )
    assert len(plan.curves) == 45


def test_shard_validity_is_bound_to_protocol_hash_and_test_rows(tmp_path):
    path = tmp_path / "shard.npz"
    test = np.asarray([1, 4, 6])
    harness.atomic_save(path, {"protocol_hash": "abc", "component": "x", "test": test, "prediction": np.ones(3)})

    assert harness.valid_shard(path, ("abc",), "x", test)
    assert harness.valid_shard(path, ("new", "abc"), "x", test)
    assert not harness.valid_shard(path, ("other",), "x", test)
    assert not harness.valid_shard(path, ("abc",), "x", np.asarray([1, 4, 7]))


def test_split_fingerprint_is_independent_of_planned_repeats():
    single = harness.split_fingerprint("300m_39bucket", 0, 2)
    harness._panel_splits("300m_39bucket", 3)
    again = harness.split_fingerprint("300m_39bucket", 0, 2)
    other = harness.split_fingerprint("300m_39bucket", 1, 2)

    assert single == again
    assert single != other


def test_aggregate_reconstruction_requires_every_component():
    panel = harness.load_panel("300m_39bucket")
    group = panel.group("uncheatable")
    rows = []
    for component_index in range(len(group.components)):
        for row_index in range(panel.rows):
            rows.append(
                {
                    "model": "m",
                    "role": "parent",
                    "parent": "",
                    "panel": panel.name,
                    "panel_kind": "tabular",
                    "curve_family": "",
                    "target": "uncheatable",
                    "component_index": component_index,
                    "component": group.components[component_index],
                    "repeat": 0,
                    "fold": row_index % 5,
                    "row_index": row_index,
                    "run": panel.runs[row_index],
                    "observed": group.outcomes[row_index, component_index],
                    "prediction": group.outcomes[row_index, component_index],
                    "basin": False,
                }
            )
    complete = harness.aggregate_predictions(pd.DataFrame(rows))
    partial = harness.aggregate_predictions(pd.DataFrame([row for row in rows if row["component_index"] != 0]))

    assert len(complete) == panel.rows
    assert np.allclose(complete["prediction"], complete["observed"], atol=3e-6)
    assert partial.empty


def test_starcoder_curve_panel_exposures_follow_the_support_rule():
    panel = harness.load_panel("starcoder::fixed_model_wsd80_1b__endpoint")

    assert panel.buckets == ("nemotron_full", "starcoder")
    assert panel.features.inventory[1] == pytest.approx(26.4579, abs=1e-3)
    assert panel.features.inventory[0] == pytest.approx(1.0)
    assert panel.repeat_sd["programming_languages_bpb"] > 0.0


def test_second_cache_generation_accepts_only_unchanged_configurations(tmp_path):
    panel = harness.load_panel("300m_39bucket")
    entry = registry.ENTRY_BY_ID["dsp_total_exposure"]
    task = harness.FitTask(entry.model_id, panel.name, "uncheatable", 0, "x", 0, 0)
    key = f"{entry.model_id}|{panel.name}"
    generation = {"models_hash": "previous", "fit_path_hash": harness.fit_path_hash(), "entries": {}}
    path = tmp_path / "legacy_entry_descriptions_gen2.json"

    harness.cache_generations.cache_clear()
    primary = harness.task_protocol_hashes(task, entry, "legacy", panel, tmp_path)
    assert len(primary) == 1

    path.write_text(json.dumps({**generation, "entries": {key: harness.description_hash(entry, panel)}}))
    harness.cache_generations.cache_clear()
    accepted = harness.task_protocol_hashes(task, entry, "legacy", panel, tmp_path)
    assert accepted[0] == primary[0]
    assert len(accepted) == 2 and accepted[1] != primary[0]

    path.write_text(json.dumps({**generation, "entries": {key: "changed"}}))
    harness.cache_generations.cache_clear()
    assert harness.task_protocol_hashes(task, entry, "legacy", panel, tmp_path) == primary
