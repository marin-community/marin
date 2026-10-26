# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "pytest"]
# ///
"""Behavior checks for fold routing, sealed adoption, and cache ownership."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_known_floor_all_folds_20261015 as runner,
)


def test_nonzero_fold_routes_only_its_training_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = pd.DataFrame({"row_id": list("abcdefgh"), "tied": [False] * 7 + [True]})
    folds = pd.DataFrame(
        [
            {"outer_fold": fold, "row_id": rows.row_id[index], "role": role, "inner_fold": inner}
            for fold, train, held in ((0, [0, 1, 2], [3]), (3, [3, 4, 5], [6]))
            for role, indices in (("train", train), ("test", held), ("tied_test", [7]))
            for inner, index in enumerate(indices)
        ]
    )
    data = SimpleNamespace(
        rows=rows,
        early=np.arange(16).reshape(8, 2) / 20,
        late=np.arange(16).reshape(8, 2) / 30,
        alpha=0.8,
        total_scales=np.array([1.0, 5.0]),
    )
    prepared = tmp_path / "stage1/prepared"
    prepared.mkdir(parents=True)
    clean, noise = np.linspace(0.9, 1.2, 8), np.arange(8) * 0.001
    np.savez(prepared / "clean.npz", native_null=clean)
    np.savez(prepared / "noise.npz", clean=noise)
    np.savez(prepared / "matched_tied.npz", native_null=clean + 0.01)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "new")
    monkeypatch.setattr(runner, "STAGE1", prepared.parent)
    monkeypatch.setattr(runner, "execution_identity", lambda: {})
    monkeypatch.setattr(runner, "verify_adoption", lambda identity: None)
    monkeypatch.setattr(runner, "verify_cache_seed", lambda: None)
    monkeypatch.setattr(runner.oracle, "inputs", lambda: (data, folds))
    captured = []

    class FitReached(Exception):
        pass

    def capture(early, late, observed, labels, alpha, config, candidate, aggregate, cache):
        captured.append((early.copy(), late.copy(), observed.copy(), labels.copy(), candidate, aggregate))
        raise FitReached

    monkeypatch.setattr(runner.prior, "fit_family", capture)
    for poison in (False, True):
        if poison:
            clean[[0, 1, 2, 6, 7]] = [-8, 50, 30, 40, 60]
            noise[[0, 1, 2, 6, 7]] = 100
            np.savez(prepared / "clean.npz", native_null=clean)
            np.savez(prepared / "noise.npz", clean=noise)
        with pytest.raises(FitReached):
            runner.run_context((3, "native_null", "clean"))
    for left, right in zip(captured[0][:4], captured[1][:4], strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(captured[0][0], data.early[[3, 4, 5]])
    np.testing.assert_array_equal(captured[0][2], clean[[3, 4, 5]] + noise[[3, 4, 5]])
    np.testing.assert_array_equal(captured[0][3], [0, 1, 2])
    assert captured[0][4:] == ("aggregate_mariner", None)
    with pytest.raises(ValueError, match="Only registered folds"):
        runner.run_context((0, "native_null", "clean"))


def test_cache_seed_preserves_mutable_resume_and_rejects_unowned_bytes(tmp_path: Path):
    source, destination = tmp_path / "sealed.sqlite", tmp_path / "new/features.sqlite"
    source.write_bytes(b"closed-feature-cache")
    pin = runner.benchmark.sha256(source)
    record = runner.seed_cache(source, destination, pin)
    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"new-private-features")
    assert runner.seed_cache(source, destination, pin) == record
    assert destination.read_bytes() == b"new-private-features"
    assert source.read_bytes() == b"closed-feature-cache"
    (destination.parent / "cache_seed.json").unlink()
    with pytest.raises(ValueError, match="unexplained cache"):
        runner.seed_cache(source, destination, pin)
    source.write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="source changed"):
        runner.seed_cache(source, destination, pin)


def test_cache_seed_rejects_nonempty_source_wal_but_allows_harmless_shm(tmp_path: Path):
    source, destination = tmp_path / "sealed.sqlite", tmp_path / "new/features.sqlite"
    source.write_bytes(b"closed-feature-cache")
    pin = runner.benchmark.sha256(source)
    Path(str(source) + "-shm").write_bytes(b"harmless-lock-index")
    wal = Path(str(source) + "-wal")
    wal.write_bytes(b"uncheckpointed-content")
    with pytest.raises(ValueError, match="nonempty source WAL"):
        runner.seed_cache(source, destination, pin)
    assert not destination.exists()
    wal.write_bytes(b"")
    runner.seed_cache(source, destination, pin)
    assert destination.read_bytes() == source.read_bytes()


def test_adoption_preserves_old_identity_and_detects_cell_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "PRIOR", tmp_path)
    old_identity = {"original": "foreign-frozen-identity"}
    runner.benchmark.write_json(tmp_path / "execution_identity.json", old_identity)
    folders = []
    for world in runner.oracle.WORLDS:
        for noise in runner.NOISES:
            for candidate in runner.oracle.candidates(world):
                folder = runner.cell_folder(0, world["name"], noise, candidate)
                folder.mkdir(parents=True)
                (folder / "fit.json").write_text("{}")
                spec = {"world": world, "noise": noise, "candidate": candidate, "outer_fold": 0, "status": runner.STATUS}
                runner.oracle.finish(folder, spec, old_identity, ("fit.json",))
                folders.append(folder)
    records = runner.adopted_records()
    assert len(records) == 40
    assert len({record["source_complete"] for record in records}) == 40
    assert all(record["spec"]["outer_fold"] == 0 for record in records)
    assert json.loads((folders[0] / "complete.json").read_text())["identity"] == old_identity
    (folders[-1] / "fit.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError):
        runner.adopted_records()


def test_unregistered_source_cannot_load_arrays_or_seed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"status": "draft", "scope": runner.SCOPE}))
    for operation in (runner.freeze, runner.prepare, runner.fit_all, runner.summarize):
        with pytest.raises(ValueError, match="Register this separate all-fold"):
            operation()
    assert list(tmp_path.iterdir()) == [tmp_path / "registration.json"]


def test_combined_summary_keeps_old_identity_and_all_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "PRIOR", tmp_path / "old")
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "new")
    old_identity, new_identity = {"version": "old"}, {"version": "new"}
    runner.benchmark.write_json(runner.PRIOR / "execution_identity.json", old_identity)
    monkeypatch.setattr(runner, "execution_identity", lambda: new_identity)
    monkeypatch.setattr(runner, "verify_adoption", lambda identity: None)
    for name in ("adopted_outer0.json", "cache_seed.json"):
        runner.benchmark.write_json(runner.OUTPUT / name, {})
    frame = pd.DataFrame(
        {
            "role": ["asymmetric"] * 2 + ["matched_tied"] * 2 + ["actual_tied"],
            "row_id": ["a", "b", "a", "b", "c"],
            "truth": [1, 2, 1.1, 2.1, 3],
            "predicted": [1.5, 0.8, 1.7, 1.0, 4],
        }
    )
    saved_scores = {
        "asymmetric_rmse": 1.1,
        "matched_tied_rmse": 1.2,
        "actual_tied_rmse": 1.3,
        "gain_rmse": 0.4,
        "selected_mu": 30,
        "selected_upper_boundary": True,
        "clean_matching_recovery_reference_met": False,
        "scientific_promotion": False,
    }
    folders = []
    for fold, world_name, noise in runner.contexts((0, *runner.NEW_FOLDS)):
        world = next(item for item in runner.oracle.WORLDS if item["name"] == world_name)
        for candidate in runner.oracle.candidates(world):
            folder = runner.cell_folder(fold, world_name, noise, candidate)
            runner.benchmark.write_json(folder / "scores.json", saved_scores)
            frame.to_csv(folder / "predictions.csv", index=False)
            spec = {"world": world, "noise": noise, "candidate": candidate, "outer_fold": fold, "status": runner.STATUS}
            runner.oracle.finish(
                folder, spec, old_identity if fold == 0 else new_identity, ("scores.json", "predictions.csv")
            )
            folders.append(folder)
    old_receipt = (folders[0] / "complete.json").read_bytes()
    runner.summarize()
    summary = json.loads((runner.OUTPUT / "summary.json").read_text())
    assert summary["heads"] == 200 and summary["new_heads"] == 160 and summary["reused_heads"] == 40
    assert sum(row["origin"] == "reused_outer0" for row in summary["records"]) == 40
    assert all(
        row["selected_upper_boundary"] and row["clean_matching_recovery_reference_met"] is False
        for row in summary["records"]
    )
    assert all(row["menu_metrics"]["regret_at_1"] == 1 for row in summary["records"])
    assert all(row["menu_metrics"]["frontier_indices"] == [1] for row in summary["records"])
    assert summary["learned_floor_domain_ready"] is False and summary["scientific_promotion"] is False
    assert (folders[0] / "complete.json").read_bytes() == old_receipt
    (folders[-1] / "complete.json").unlink()
    with pytest.raises(ValueError, match="incomplete or failed"):
        runner.summarize()
