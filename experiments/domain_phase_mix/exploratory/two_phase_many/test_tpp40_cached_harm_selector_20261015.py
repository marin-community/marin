# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Toy checks for cached choice, the held-table boundary, and provenance guards."""

import copy
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_cached_harm_selector_20261015 as selector


def toy_fit(score, strength):
    return {
        "training_rows": ["a", "b", "c"],
        "inner_folds": [0, 1, 2],
        "alpha": 0.8,
        "selection_contract": "toy pooled inner BPB",
        "calibration_contract": "toy train-local floors",
        "inner_rmse": score,
        "parameters": {"lambda": strength},
        "shape": {"rate": 0.5, "power": 0.7, "threshold": 2},
        "ridge": 0.01,
        "kappa": 1.5,
        "head": {"floor": 0.5, "intercept": 0.1, "coefficients": [1, 2, 3, 4]},
        "selection_profile": [{"inner_rmse": score, "stage": "full_rescan"}],
        "search_space": {
            "shapes": [1],
            "ridges": [0.01],
            "gammas": [1.5],
            "parameters": [{"lambda": 0}, {"lambda": 1}],
            "complete_aggregate_source": {"inner_rmse": 0.3},
            "test_grid_override": False,
            "selected_trial_index": 0,
            "selected_from_failed_refinement": strength == 1,
            "refinement_results": [{"success": strength != 1, "status": int(strength == 1)}],
        },
    }


def toy_decisions():
    return [
        {
            "panel": selector.PANEL,
            "target": "toy",
            "repeat": 0,
            "outer_fold": fold,
            **selector.choose_context(toy_fit(p, 0), toy_fit(e, 1)),
        }
        for fold, p, e in ((0, 0.1, 0.2), (1, 0.2, 0.1))
    ]


def test_exact_tie_and_adjacent_float_preserve_status():
    physical, effective = toy_fit(0.1, 0), toy_fit(0.1, 1)
    tied = selector.choose_context(physical, effective)
    assert tied["source_family"] == selector.PHYSICAL
    assert tied["exact_score_tie"]
    assert tied["selected_status"]["null"]
    assert tied["candidate_statuses"][selector.EFFECTIVE]["selected_from_failed_refinement"]
    effective["inner_rmse"] = math.nextafter(0.1, -math.inf)
    effective["selection_profile"][0]["inner_rmse"] = effective["inner_rmse"]
    changed = selector.choose_context(physical, effective)
    assert changed["source_family"] == selector.EFFECTIVE
    assert not changed["exact_score_tie"]
    assert changed["absolute_score_margin"] > 0
    assert changed["selected_status"]["physical_strength_boundary"]
    assert changed["selected_status"]["refinement_results"] == [{"success": False, "status": 1}]


@pytest.mark.parametrize("key,value", [("training_rows", ["b", "a", "c"]), ("inner_folds", [1, 0, 2])])
def test_partition_mismatch_rejected(key, value):
    physical, effective = toy_fit(0.1, 0), toy_fit(0.01, 1)
    effective[key] = value
    with pytest.raises(ValueError, match=key):
        selector.choose_context(physical, effective)


def test_held_label_poisoning_cannot_change_durable_choices(tmp_path):
    current = {"toy_identity": "fixed"}
    decisions = toy_decisions()
    paths = {family: tmp_path / f"{family}.csv" for family in (selector.PHYSICAL, selector.EFFECTIVE)}
    # Even unavailable prediction paths cannot be reached before complete decisions.
    with pytest.raises(ValueError, match="committed"):
        selector.materialize(current, tmp_path, paths)
    selector.write_once(
        tmp_path / "decisions.json", {"identity": current, "decisions": decisions, "held_tables_parsed": False}
    )
    with pytest.raises(ValueError, match="committed"):
        selector.materialize(current, tmp_path, paths)
    selector.finish("selection", current, ["decisions.json"], tmp_path)
    decision_hash = selector.benchmark.sha256(tmp_path / "decisions.json")
    results = []
    for poisoned in (False, True):
        for family, prediction in ((selector.PHYSICAL, 1.0), (selector.EFFECTIVE, 3.0)):
            records = [
                {
                    "panel": selector.PANEL,
                    "target": "toy",
                    "repeat": 0,
                    "outer_fold": fold,
                    "row_id": f"row{fold}_{role}",
                    "role": role,
                    "model": family,
                    "predicted": prediction,
                    "observed": (3.0 if fold == 0 else 1.0) if poisoned else (1.0 if fold == 0 else 3.0),
                }
                for fold in (0, 1)
                for role in ("test", "tied_test")
            ]
            pd.DataFrame(records).to_csv(paths[family], index=False)
        results.append(selector.materialize(current, tmp_path, paths))
    assert results[0].source_family.tolist() == results[1].source_family.tolist()
    assert results[0].predicted.tolist() == results[1].predicted.tolist() == [1, 1, 3, 3]
    assert ((results[0].predicted - results[0].observed) ** 2).mean() == 0
    assert ((results[1].predicted - results[1].observed) ** 2).mean() == 4
    assert selector.benchmark.sha256(tmp_path / "decisions.json") == decision_hash
    corrupted = copy.deepcopy(decisions)
    corrupted[0]["source_family"] = selector.EFFECTIVE
    selector.benchmark.write_json(
        tmp_path / "decisions.json", {"identity": current, "decisions": corrupted, "held_tables_parsed": False}
    )
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        selector.materialize(current, tmp_path, paths)


def test_pinned_source_drift_and_reduced_scope_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    monkeypatch.setattr(selector, "source_paths", lambda: {source})
    registration = {
        **selector.SCOPE,
        "status": "registered",
        "environment": selector.benchmark.environment_identity(),
        "source_seals": selector.SOURCE_SEALS,
        "frozen_reference_inputs": {str(source): selector.benchmark.sha256(source)},
    }
    selector.verify_scope(registration)
    reduced = copy.deepcopy(registration)
    reduced["targets"] = reduced["targets"][:-1]
    with pytest.raises(ValueError, match="targets"):
        selector.verify_scope(reduced)
    source.write_text("version = 2\n")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        selector.verify_scope(registration)


def test_completed_selection_cannot_resume_under_different_identity(tmp_path):
    selector.write_once(tmp_path / "decisions.json", {"decisions": toy_decisions()})
    selector.finish("selection", {"version": 1}, ["decisions.json"], tmp_path)
    with pytest.raises(ValueError, match="identity or scope"):
        selector.cached_complete("selection", {"version": 2}, tmp_path)
