# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark

PANEL = "starcoder_wsd80_dense"
REFERENCE = "effective_exposure_dsp"
CANDIDATE = "phase_candidate"
REFERENCES = (REFERENCE, "aggregate_mariner_train_median")


@dataclasses.dataclass(frozen=True)
class ScreenFiles:
    prepared: Path
    baseline: Path
    policies: Path
    candidate: Path
    evidence: Path


@pytest.fixture
def screen_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ScreenFiles:
    protocol_dir = tmp_path / "protocol"
    monkeypatch.setattr(benchmark, "PROTOCOL_DIR", protocol_dir)
    benchmark.write_json(protocol_dir / "data_contract.json", {"expected_group_counts": {PANEL: 60}})
    benchmark.write_json(protocol_dir / "protocol.json", {"protocol_id": "synthetic_promotion_contract"})

    prepared = tmp_path / "prepared"
    panel = prepared / PANEL
    panel.mkdir(parents=True)
    row_number = np.arange(60)
    rows = pd.DataFrame(
        {
            "row_id": [f"row_{index}" for index in row_number],
            "policy_id": [f"policy_{index}" for index in row_number],
            "group_id": [f"group_{index}" for index in row_number],
            "source": "synthetic_fixture",
            "region": "interior",
            "tied": row_number % 12 >= 10,
        }
    )
    rows.to_csv(panel / "rows.csv", index=False)
    targets = {target: 1.0 + row_number / 60 + index for index, target in enumerate(benchmark.QUICK_TASKS)}
    pd.DataFrame(targets).to_csv(panel / "outcomes.csv", index=False)
    phase0 = np.column_stack([np.linspace(0.1, 0.9, 60), np.linspace(0.9, 0.1, 60)])
    phase1 = phase0[:, ::-1].copy()
    phase1[rows.tied] = phase0[rows.tied]
    np.savez_compressed(
        panel / "policies.npz",
        phase0=phase0,
        phase1=phase1,
        c0=np.ones(2),
        c1=np.ones(2),
        phase0_epochs=phase0,
        phase1_epochs=phase1,
    )
    benchmark.write_json(
        panel / "metadata.json",
        {"buckets": ["a", "b"], "alpha": 0.8, "metadata": {}, "source_hashes": {}},
    )
    benchmark.write_json(
        panel / "split_audit.json",
        {
            "status": "sufficient",
            "cells": [
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "status": "sufficient",
                    "menus": {"test": {"rows": 10, "groups": 10}, "tied_test": {"rows": 2, "groups": 2}},
                }
                for repeat in range(5)
                for fold in range(5)
            ],
        },
    )
    manifest = prepared / "manifest.json"
    benchmark.write_json(
        manifest,
        {
            "identity": {"data_contract_sha256": benchmark.sha256(protocol_dir / "data_contract.json")},
            "sources": {},
            "outputs": {str(path.relative_to(prepared)): benchmark.sha256(path) for path in panel.iterdir()},
        },
    )
    predictions = pd.concat(
        [
            rows.drop(columns="tied").assign(
                model=model,
                panel=PANEL,
                target=target,
                repeat=repeat,
                outer_fold=row_number // 12,
                role=np.where(rows.tied, "tied_test", "test"),
                observed=observed,
                predicted=observed + 0.05,
            )
            for model in REFERENCES
            for target, observed in targets.items()
            for repeat in range(5)
        ],
        ignore_index=True,
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    predictions.to_csv(baseline / "predictions.csv", index=False)
    benchmark.write_json(
        baseline / "run.json",
        {
            "prepared_manifest": benchmark.sha256(manifest),
            "claim": "baseline_execution_only",
            "outputs": {"predictions.csv": benchmark.sha256(baseline / "predictions.csv")},
        },
    )
    policies = tmp_path / "policies"
    benchmark.freeze_screen_policies(prepared, [baseline], policies)
    candidate = tmp_path / "candidate.csv"
    predictions.loc[predictions.model.eq(REFERENCE) & predictions.repeat.eq(0)].assign(model=CANDIDATE).to_csv(
        candidate, index=False
    )
    policy = json.loads((policies / f"{REFERENCE}.json").read_text())
    receipts = {}
    for name in policy["required_evidence"]:
        path = tmp_path / "evidence" / f"{name}.json"
        benchmark.write_json(
            path,
            {
                "evidence": name,
                "protocol_id": policy["benchmark_protocol_id"],
                "candidate_model": CANDIDATE,
                "data_contract_sha256": policy["data_contract_sha256"],
                "status": "pass",
            },
        )
        receipts[name] = [benchmark.receipt(path)]
    evidence = tmp_path / "evidence.json"
    benchmark.write_json(evidence, receipts)
    return ScreenFiles(prepared, baseline, policies, candidate, evidence)


def test_policy_file_cannot_substitute_for_typed_candidate_evidence(screen_files: ScreenFiles) -> None:
    policy_path = screen_files.policies / f"{REFERENCE}.json"
    policy = json.loads(policy_path.read_text())
    benchmark.write_json(
        screen_files.evidence,
        {name: [benchmark.receipt(policy_path)] for name in policy["required_evidence"]},
    )
    verified = benchmark.verified_policy(policy_path, screen_files.evidence, CANDIDATE)
    inputs = benchmark.promotion_inputs(
        screen_files.candidate, screen_files.baseline / "predictions.csv", verified, REFERENCE
    )
    assert not any(verified["evidence"].values())
    assert benchmark.scoring.compare_models(inputs, REFERENCE, verified)["status"] == "incomplete"


@pytest.mark.parametrize("field", ["evidence", "protocol_id", "candidate_model", "data_contract_sha256", "status"])
def test_misidentified_evidence_cannot_advance_even_with_valid_hash(screen_files: ScreenFiles, field: str) -> None:
    receipts = json.loads(screen_files.evidence.read_text())
    name = "candidate_registration"
    path = Path(receipts[name][0]["path"])
    record = json.loads(path.read_text())
    record[field] = "wrong_identity_or_failed_status"
    benchmark.write_json(path, record)
    receipts[name] = [benchmark.receipt(path)]
    benchmark.write_json(screen_files.evidence, receipts)
    policy = benchmark.verified_policy(screen_files.policies / f"{REFERENCE}.json", screen_files.evidence, CANDIDATE)
    inputs = benchmark.promotion_inputs(
        screen_files.candidate, screen_files.baseline / "predictions.csv", policy, REFERENCE
    )
    assert policy["evidence"][name] is False
    assert benchmark.scoring.compare_models(inputs, REFERENCE, policy)["status"] == "incomplete"


def test_altered_reference_prediction_is_refused_before_scoring(screen_files: ScreenFiles) -> None:
    policy = json.loads((screen_files.policies / f"{REFERENCE}.json").read_text())
    reference_path = screen_files.baseline / "predictions.csv"
    predictions = pd.read_csv(reference_path, float_precision="round_trip")
    predictions.loc[0, "predicted"] = predictions.predicted.to_numpy(float)[0] + 1.0
    predictions.to_csv(reference_path, index=False)
    with pytest.raises(ValueError):
        benchmark.promotion_inputs(screen_files.candidate, reference_path, policy, REFERENCE)


@pytest.mark.parametrize("reference", REFERENCES)
def test_full_reference_repeats_pair_only_with_required_candidate_menus(
    screen_files: ScreenFiles, reference: str
) -> None:
    candidate = pd.read_csv(screen_files.candidate, float_precision="round_trip")
    pd.concat(
        [
            candidate,
            candidate.assign(target="outside_required_targets"),
            candidate.assign(repeat=1),
            candidate.assign(role="train"),
        ],
        ignore_index=True,
    ).to_csv(screen_files.candidate, index=False)
    policy = benchmark.verified_policy(screen_files.policies / f"{reference}.json", screen_files.evidence, CANDIDATE)
    inputs = benchmark.promotion_inputs(
        screen_files.candidate, screen_files.baseline / "predictions.csv", policy, reference
    )
    assert set(inputs.model) == {reference, CANDIDATE}
    assert set(inputs.repeat) == {0}
    assert set(inputs.role) == {"test", "tied_test"}
    assert inputs.groupby("model").size().to_dict() == {reference: 300, CANDIDATE: 300}
    result = benchmark.scoring.compare_models(inputs, reference, policy)
    assert result["status"] == "advance"
    assert len(result["candidates"]) == 1


def test_data_contract_drift_prevents_freezing_new_screen_policy(screen_files: ScreenFiles) -> None:
    benchmark.write_json(benchmark.PROTOCOL_DIR / "data_contract.json", {"expected_group_counts": {PANEL: 61}})
    output = screen_files.policies.parent / "after_drift"
    with pytest.raises(ValueError):
        benchmark.freeze_screen_policies(screen_files.prepared, [screen_files.baseline], output)
    assert not output.exists()


def test_insufficient_audited_fold_prevents_freezing_screen_policy(screen_files: ScreenFiles) -> None:
    audit_path = screen_files.prepared / PANEL / "split_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["cells"][0]["status"] = "insufficient"
    benchmark.write_json(audit_path, audit)
    manifest_path = screen_files.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][str(audit_path.relative_to(screen_files.prepared))] = benchmark.sha256(audit_path)
    benchmark.write_json(manifest_path, manifest)
    run_path = screen_files.baseline / "run.json"
    run = json.loads(run_path.read_text())
    run["prepared_manifest"] = benchmark.sha256(manifest_path)
    benchmark.write_json(run_path, run)
    output = screen_files.policies.parent / "insufficient"
    with pytest.raises(ValueError):
        benchmark.freeze_screen_policies(screen_files.prepared, [screen_files.baseline], output)
    assert not output.exists()


@pytest.mark.parametrize("defect", ["too_few_rows", "too_few_groups"])
def test_insufficient_prediction_menu_prevents_freeze_despite_sufficient_audit(
    screen_files: ScreenFiles, defect: str
) -> None:
    predictions_path = screen_files.baseline / "predictions.csv"
    predictions = pd.read_csv(predictions_path, float_precision="round_trip")
    menu = (
        predictions.model.eq(REFERENCE)
        & predictions.target.eq(benchmark.QUICK_TASKS[0])
        & predictions.repeat.eq(0)
        & predictions.outer_fold.eq(0)
        & predictions.role.eq("test")
    )
    if defect == "too_few_rows":
        predictions = predictions.drop(index=predictions.index[menu][0])
    else:
        predictions.loc[menu, "group_id"] = "one_group"
    predictions.to_csv(predictions_path, index=False)
    run_path = screen_files.baseline / "run.json"
    run = json.loads(run_path.read_text())
    run["outputs"]["predictions.csv"] = benchmark.sha256(predictions_path)
    benchmark.write_json(run_path, run)
    output = screen_files.policies.parent / "insufficient_predictions"
    with pytest.raises(ValueError):
        benchmark.freeze_screen_policies(screen_files.prepared, [screen_files.baseline], output)
    assert not output.exists()
