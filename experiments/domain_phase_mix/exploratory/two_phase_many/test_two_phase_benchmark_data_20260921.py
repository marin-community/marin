# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.two_phase_benchmark_data_20260921 import (
    BenchPanel,
    frozen_folds,
    split_coverage_audit,
    support_transfer_plan,
)


@pytest.fixture
def panel() -> BenchPanel:
    aggregates = np.linspace(0.15, 0.85, 15)
    first, second, groups, tied = [], [], [], []
    for index, aggregate in enumerate(aggregates):
        for contrast in (-0.1, 0.1, 0.0):
            first.append(aggregate - 0.2 * contrast)
            second.append(aggregate + 0.8 * contrast)
            groups.append(f"group_{index:02d}")
            tied.append(contrast == 0)
    # Two training seeds at one coordinate must remain in the same neighborhood.
    first.append(first[0])
    second.append(second[0])
    groups.append(groups[0])
    tied.append(False)
    n = len(first)
    rows = pd.DataFrame(
        {
            "row_id": [f"row_{i:03d}" for i in range(n)],
            "policy_id": [f"policy_{i:03d}" for i in range(n - 1)] + ["policy_000"],
            "group_id": groups,
            "source": "synthetic",
            "region": "synthetic",
            "tied": tied,
        }
    )
    return BenchPanel(
        name="synthetic",
        buckets=("text", "code"),
        phase0=np.column_stack([1 - np.asarray(first), first]),
        phase1=np.column_stack([1 - np.asarray(second), second]),
        alpha=0.8,
        c0=np.array([0.8, 8.0]),
        c1=np.array([0.2, 2.0]),
        targets={"loss": np.linspace(1, 2, n)},
        rows=rows,
        input_hashes={},
        metadata={"grouping_alpha": 0.8},
    )


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["outer_fold", "row_id"]).reset_index(drop=True)


def test_nested_folds_keep_counterparts_and_repeated_coordinates_together(panel: BenchPanel) -> None:
    folds = frozen_folds(panel)
    joined = folds.merge(panel.rows, on="row_id", validate="many_to_one")
    scored = joined.loc[joined.role.ne("train")]
    assert scored.row_id.value_counts().eq(1).all()
    assert set(scored.row_id) == set(panel.rows.row_id)
    assert not joined.loc[joined.role.eq("train"), "tied"].any()
    assert joined.loc[joined.tied, "role"].eq("tied_test").all()
    assert scored.groupby("group_id").outer_fold.nunique().eq(1).all()
    assert scored.groupby("policy_id").outer_fold.nunique().eq(1).all()
    for _, block in joined.groupby("outer_fold"):
        train, test = block.loc[block.role.eq("train")], block.loc[block.role.ne("train")]
        assert set(train.group_id).isdisjoint(test.group_id)
        assert train.inner_fold.ge(0).all()
        assert test.inner_fold.eq(-1).all()
        assert train.groupby("group_id").inner_fold.nunique().eq(1).all()
        for inner in train.inner_fold.unique():
            validation = train.loc[train.inner_fold.eq(inner)]
            fitting = train.loc[train.inner_fold.ne(inner)]
            assert set(validation.group_id).isdisjoint(fitting.group_id)


def test_fold_identities_ignore_outcomes_missingness_and_input_order(panel: BenchPanel) -> None:
    expected = _ordered(frozen_folds(panel, repeat=2))
    rng = np.random.default_rng(73)
    labels = rng.permutation(panel.targets["loss"])
    labels[::3] = np.nan
    changed = replace(panel, targets={"loss": labels, "unrelated": rng.normal(size=len(labels))})
    pd.testing.assert_frame_equal(expected, _ordered(frozen_folds(changed, repeat=2)))
    order = rng.permutation(len(panel.rows))
    reordered = replace(
        changed,
        phase0=changed.phase0[order],
        phase1=changed.phase1[order],
        rows=changed.rows.iloc[order].reset_index(drop=True),
        targets={key: values[order] for key, values in changed.targets.items()},
    )
    pd.testing.assert_frame_equal(expected, _ordered(frozen_folds(reordered, repeat=2)))


def test_inner_neighborhoods_are_rebuilt_from_outer_training_only(panel: BenchPanel) -> None:
    seed = 120
    outer = 2
    folds = frozen_folds(panel, outer_folds=5, inner_folds=3, seed=seed)
    training = folds.loc[folds.outer_fold.eq(outer) & folds.role.eq("train")]
    positions = panel.rows.set_index("row_id").index.get_indexer(training.row_id)
    restricted = replace(
        panel,
        phase0=panel.phase0[positions],
        phase1=panel.phase1[positions],
        rows=panel.rows.iloc[positions].reset_index(drop=True),
        targets={key: values[positions] for key, values in panel.targets.items()},
    )
    independently_rebuilt = frozen_folds(restricted, outer_folds=3, inner_folds=2, seed=seed + outer + 1)
    validation = independently_rebuilt.loc[independently_rebuilt.role.eq("test")].set_index("row_id")
    np.testing.assert_array_equal(validation.loc[training.row_id, "outer_fold"], training.inner_fold)


def test_isolated_tied_policy_never_creates_a_calibration_or_empty_scoring_fold(panel: BenchPanel) -> None:
    tied_index = int(np.flatnonzero(panel.tied)[0])
    phase0, phase1 = panel.phase0.copy(), panel.phase1.copy()
    phase0[tied_index] = phase1[tied_index] = [1.0, 0.0]
    rows = panel.rows.copy()
    rows.loc[tied_index, "group_id"] = "isolated_tie"
    isolated = replace(panel, phase0=phase0, phase1=phase1, rows=rows)
    folds = frozen_folds(isolated)
    diagnostic = folds.loc[folds.row_id.eq(rows.row_id.iloc[tied_index])]
    assert diagnostic.role.tolist() == ["tied_test"]
    assert folds.loc[folds.role.eq("test")].groupby("outer_fold").size().gt(0).all()


def test_split_audit_exposes_small_menus_without_changing_folds(panel: BenchPanel) -> None:
    folds = frozen_folds(panel)
    original = folds.copy(deep=True)
    audit = split_coverage_audit(
        panel,
        folds,
        outer_folds=5,
        inner_folds=3,
        minimum_asymmetric_test_rows=100,
        minimum_asymmetric_test_groups=2,
        expected_group_count=15,
    )
    assert audit["status"] == "insufficient"
    assert not audit["structural_errors"]
    assert all("asymmetric_test_rows_below_minimum" in cell["reasons"] for cell in audit["cells"])
    assert sum(cell["menus"]["test"]["rows"] for cell in audit["cells"]) == int((~panel.tied).sum())
    assert sum(cell["menus"]["tied_test"]["rows"] for cell in audit["cells"]) == int(panel.tied.sum())
    pd.testing.assert_frame_equal(folds, original)


def test_split_audit_reports_missing_challenge_for_insufficient_groups(panel: BenchPanel) -> None:
    selection = np.flatnonzero(panel.rows.group_id.eq("group_00"))
    restricted = replace(
        panel,
        phase0=panel.phase0[selection],
        phase1=panel.phase1[selection],
        rows=panel.rows.iloc[selection].reset_index(drop=True),
        targets={},
    )
    empty = pd.DataFrame(columns=["panel", "repeat", "outer_fold", "row_id", "role", "inner_fold"])
    audit = split_coverage_audit(
        restricted,
        empty,
        outer_folds=5,
        inner_folds=3,
        minimum_asymmetric_test_rows=10,
        minimum_asymmetric_test_groups=2,
    )
    assert audit["status"] == "insufficient"
    assert "asymmetric_aggregate_groups_fewer_than_outer_folds" in audit["missing_challenges"]
    assert not audit["cells"]


def test_support_transfer_excludes_asymmetric_aliases_in_outer_and_inner_fits(panel: BenchPanel) -> None:
    panels = []
    for support in ("m0125", "m025", "m050", "m100", "m200", "m400"):
        rows = panel.rows.copy()
        rows["checkpoint_id"] = [f"{support}_{row}" for row in rows.row_id]
        rows.loc[0, "checkpoint_id"] = "shared_asymmetric_checkpoint"
        rows.loc[3, "checkpoint_id"] = rows.loc[1, "checkpoint_id"]
        phase0, phase1 = panel.phase0.copy(), panel.phase1.copy()
        phase0[2] = phase1[2] = [1, 0]
        panels.append(
            replace(
                panel,
                name=f"support_{support}",
                buckets=("text", "starcoder"),
                rows=rows,
                phase0=phase0,
                phase1=phase1,
                targets={},
                metadata={"support": support, "horizon_tokens": 1000},
            )
        )
    plan = support_transfer_plan(panels)
    registry = plan["rows"]
    assert plan["status"] == "ready"
    for cell in plan["cells"]:
        training = [registry[index] for index in cell["train_rows"]]
        held = [registry[index] for index in cell["test_rows"] + cell["tied_test_rows"]]
        training_ids = [row["checkpoint_id"] for row in training]
        assert len(training_ids) == len(set(training_ids))
        assert set(training_ids).isdisjoint(row["checkpoint_id"] for row in held)
        assert "shared_asymmetric_checkpoint" not in training_ids
        assert not any(row["tied"] for row in training)
        assert not any(row["zero_starcoder"] for row in held)
        assert cell["counts"]["removed_held_checkpoint_aliases"] > 0
        assert cell["counts"]["removed_training_checkpoint_duplicates"] > 0
        for inner in cell["inner_folds"]:
            fitting_ids = [registry[index]["checkpoint_id"] for index in inner["train_rows"]]
            validation_ids = {registry[index]["checkpoint_id"] for index in inner["validation_rows"]}
            assert len(fitting_ids) == len(set(fitting_ids))
            assert set(fitting_ids).isdisjoint(validation_ids)
            assert set(inner["train_rows"]).issubset(cell["train_candidate_rows"])
