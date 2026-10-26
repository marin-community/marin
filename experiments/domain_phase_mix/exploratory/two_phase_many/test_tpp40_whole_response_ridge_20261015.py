# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compact auditability, exact adoption and train-only ridge selection on invented data."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_whole_response_fold3_noise_20261015 as original,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_whole_response_ridge_20261015 as runner


def invented_fit():
    t = np.linspace(0.05, 2.0, 21)
    matrix = np.column_stack((-t, t**2))
    observed = 0.6 + np.exp(matrix @ np.array([0.7, 0.08]))
    starts = [{"amplitudes": [0.4 + i * 0.08, 0.05]} for i in range(5)]
    fit = runner.model.fit_response(matrix, observed, None, starts, ridge=0.01, evaluation_cap=5000)
    return matrix, observed, fit


def test_compact_checkpoint_retains_selected_each_start_best_and_every_failure():
    matrix, _, fit = invented_fit()
    failure = {"trial_id": len(fit["trials"]), "start_index": 4, "status": "numerical_failure", "message": "invented"}
    fit["trials"].append(failure)
    compact = runner.compact_fit(fit)
    assert compact["selected"] == fit["selected"]
    assert compact["solves"] == fit["solves"]
    assert compact["starts"] == fit["starts"]
    retained = {row["trial_id"] for row in compact["trials"]}
    assert all(item["best_trial"] in retained for item in fit["solves"])
    assert failure in compact["trials"]
    assert compact["trial_storage"]["omitted_finite_trial_count"] > 0
    omitted = [row for row in fit["trials"] if row["trial_id"] not in retained]
    assert runner.prior.canonical_hash(omitted) == compact["trial_storage"]["omitted_finite_trials_sha256"]
    assert runner.prior.fit_diagnostics(compact) == runner.prior.fit_diagnostics(fit)
    decoded = json.loads(gzip.decompress(gzip.compress(json.dumps(compact, allow_nan=False).encode())))
    np.testing.assert_array_equal(
        runner.model.prior.predict(matrix, decoded["selected"]), runner.model.prior.predict(matrix, fit["selected"])
    )


def test_failed_nonwinning_starts_survive_compaction():
    t = np.linspace(0.1, 2.0, 12)
    matrix = np.column_stack((-t, t**2))
    observed = 0.6 + np.exp(-t)
    starts = [{"amplitudes": [0.3 + i * 0.1, 0.01]} for i in range(5)]
    fit = runner.model.fit_response(matrix, observed, None, starts, ridge=1, evaluation_cap=1)
    compact = runner.compact_fit(fit)
    assert len(compact["solves"]) == 5
    assert all(not item["success"] and item["termination"] == "evaluation_cap" for item in compact["solves"])
    assert runner.prior.fit_diagnostics(compact)["nonwinning_failed_start_count"] == 4
    assert compact["unresolved"]


def profile(loss, ridge_index, strength_index, shape_index, penalty):
    return {
        "inner_mse": loss,
        "eligible_for_selection": loss is not None,
        "penalized_training_score": penalty,
        "spec": {"ridge_index": ridge_index, "strength_index": strength_index, "shape_index": shape_index},
    }


def test_selection_uses_unpenalized_cv_and_registered_tie_order():
    rows = [profile(0.02, 0, 0, 0, 0.01), profile(0.01, 4, 6, 167, 100)]
    assert runner.choose(rows) is rows[1]
    rows += [profile(0.01, 1, 6, 167, 1000), profile(0.01, 1, 0, 10, 2000), profile(0.01, 1, 0, 2, 3000)]
    assert runner.choose(rows) is rows[-1]
    rows.append(profile(None, 0, 0, 0, 0))
    assert runner.choose(rows) is rows[-2]


def test_adoption_requires_exact_original_spec_input_identity_and_payload(tmp_path, monkeypatch):
    matrix, observed, fit = invented_fit()
    old_root = tmp_path / "old"
    spec = {
        "noise": "invented",
        "world": "invented",
        "candidate": "aggregate_mariner",
        "ridge": 0.0,
        "ridge_index": 0,
        "strength_index": 0,
        "shape_index": 0,
        "inner_fold": 0,
    }
    header = {
        "spec": spec,
        "identity_sha256": "new",
        "input_sha256": runner.prior.canonical_hash({"matrix": matrix.tolist(), "observed": observed.tolist()}),
    }
    old_spec = {k: v for k, v in spec.items() if k not in {"ridge", "ridge_index"}}
    source = old_root / "cells/outer3/invented/invented/aggregate_mariner/conditional/mu0_shape000/inner0.json.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "header": header | {"spec": old_spec, "identity_sha256": "old"},
                    "fit": fit,
                    "fit_sha256": runner.prior.canonical_hash(fit),
                }
            ).encode()
        )
    )
    target = tmp_path / "new"
    target.mkdir()
    manifest = {
        "old_identity_sha256": "old",
        "records": {str(source.relative_to(old_root)): runner.prior.digest(source)},
    }
    (target / "adoption_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "OLD_OUTPUT", old_root)
    monkeypatch.setattr(runner, "OUTPUT", target)
    runner.adoption_manifest.cache_clear()
    try:
        adopted, lineage = runner.adopted_fit(Path("inner0.json.gz"), header, spec)
        assert adopted == fit
        assert lineage["original_fit_sha256"] == runner.prior.canonical_hash(fit)
        checkpoint = target / "inner0.json.gz"
        compact = runner.compact_fit(fit)
        record = {
            "header": header,
            "fit": compact,
            "fit_sha256": runner.prior.canonical_hash(compact),
            "adopted_from": lineage,
        }
        checkpoint.write_bytes(gzip.compress(json.dumps(record).encode()))
        assert runner.conditional_fit(checkpoint, matrix, observed, spec, "new") == compact
        record["adopted_from"] = {"changed": True}
        checkpoint.write_bytes(gzip.compress(json.dumps(record).encode()))
        with pytest.raises(ValueError, match="lineage"):
            runner.conditional_fit(checkpoint, matrix, observed, spec, "new")
        with pytest.raises(ValueError, match="identical"):
            runner.adopted_fit(Path("inner0.json.gz"), header | {"input_sha256": "held-label-poison"}, spec)
    finally:
        runner.adoption_manifest.cache_clear()


def test_complete_native_reuse_requires_same_ridge_and_all_head_parameters():
    spec = {"ridge_index": 3, "ridge": 1.0, "strength_index": 0, "mu": 0, "shape_index": 4}
    native = {"selected_candidate": spec, "candidate": "aggregate_mariner", "selected": {"floor": 0.7}}
    reused = runner.backend.reuse_native_full(native, {"spec": spec}, "equal_group")
    assert reused["selected"] == native["selected"]
    assert reused["native_full_fit_reused"]
    with pytest.raises(ValueError, match="exact"):
        runner.backend.reuse_native_full(native, {"spec": spec | {"ridge_index": 2}}, "equal_group")


def test_private_scorer_keeps_original_global_root_and_procedure():
    assert original.OUTPUT == runner.OLD_OUTPUT
    assert original.CONTRACT["candidate_id"] == "TP40-046"
    assert runner.old.OUTPUT == runner.OUTPUT
    assert runner.old.CONTRACT["candidate_id"] == "TP40-049"
