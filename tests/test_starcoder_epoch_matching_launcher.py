# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from levanter.main.train_lm import TrainLmConfig
from marin.execution.artifact import ArtifactRecord, FingerprintMismatchError, write_record
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.execution.step_status import STATUS_FAILED, STATUS_SUCCESS, StatusFile
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_epoch_matching as launcher


@pytest.fixture
def cached_metadata(monkeypatch: Any) -> None:
    def load_cache(_cls: type[launcher.TokenizedCache], path: str) -> launcher.TokenizedCache:
        cache = launcher.TokenizedCache(path=path)
        cache.__dict__["record"] = SimpleNamespace(
            config={
                "format": {"text_key": "text"},
                "tags": [],
                "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
            },
            source=None,
        )
        return cache

    monkeypatch.setattr(launcher.TokenizedCache, "raw_load", classmethod(load_cache))


def test_default_cli_emits_plan_without_reading_remote_caches(tmp_path: Path, monkeypatch: Any) -> None:
    def reject_cache_read(_cls: type[launcher.TokenizedCache], _path: str) -> launcher.TokenizedCache:
        raise AssertionError("A default dry-run must not read remote cache records")

    monkeypatch.setattr(launcher.TokenizedCache, "raw_load", classmethod(reject_cache_read))
    output = tmp_path / "plan.json"
    launcher.main(["--plan-path", str(output)])
    plan = json.loads(output.read_text())

    assert plan["new_training_runs"] == 10
    assert sum(row["arm"] == "target" for row in plan["runs"]) == 1
    assert all(row["fingerprint"] for row in plan["runs"])
    assert all(row["output_path"].startswith("gs://marin-us-central1/") for row in plan["runs"])


def test_stage_expansion_preserves_existing_training_identities() -> None:
    design = launcher.design_module.load_design()
    pilot_rows, pilot_steps = launcher.build_training_steps(design, "pilot")
    full_rows, full_steps = launcher.build_training_steps(design, "replicated")
    full = {row.run_name: step for row, step in zip(full_rows, full_steps, strict=True)}

    for row, step in zip(pilot_rows, pilot_steps, strict=True):
        assert full[row.run_name].path(launcher.base.DEFAULT_MARIN_PREFIX) == step.path(
            launcher.base.DEFAULT_MARIN_PREFIX
        )
        assert full[row.run_name].fingerprint() == step.fingerprint()


def test_endpoint_and_proxy_configs_keep_legacy_subset_keys(cached_metadata: None) -> None:
    design = launcher.design_module.load_design()
    rows, steps = launcher.build_training_steps(design, "pilot")
    endpoint = next((row, step) for row, step in zip(rows, steps, strict=True) if row.arm == "target")
    proxy = next(
        (row, step)
        for row, step in zip(rows, steps, strict=True)
        if row.arm == "matched" and row.starcoder_weight == 1.0
    )
    for row, step in (endpoint, proxy):
        pod = materialized_config(step, launcher.base.DEFAULT_MARIN_PREFIX)
        assert isinstance(pod, TrainLmOnPodConfig)
        config = pod.train_config
        assert isinstance(config, TrainLmConfig)
        shuffle_keys = config.data.train_component_shuffle_keys
        assert shuffle_keys is not None
        assert shuffle_keys["dolma/starcoder"] == (898005854, 446240491)
        assert len(shuffle_keys) == 7
        assert config.data.max_train_batches == {"dolma/starcoder": 1068 if row.arm == "target" else 40}
        assert config.data.experiment_budget is None
        assert config.data.target_budget is None
        assert config.data_seed == 20260711
        assert config.trainer.seed == 20260711
        assert config.optimizer.decay == row.total_steps - row.boundary_step
    launcher.audit_runtime_configs(design, rows, steps, marin_prefix=launcher.base.DEFAULT_MARIN_PREFIX)


def test_replicates_change_model_seed_without_changing_data_or_support(cached_metadata: None) -> None:
    design = launcher.design_module.load_design()
    rows, steps = launcher.build_training_steps(design, "replicated")
    selected = [
        (row, materialized_config(step, launcher.base.DEFAULT_MARIN_PREFIX).train_config)
        for row, step in zip(rows, steps, strict=True)
        if row.arm == "matched" and row.starcoder_weight == 0.5
    ]
    assert {config.trainer.seed for _, config in selected} == {20260711, 20260908, 20260909}
    assert {config.data_seed for _, config in selected} == {20260711}
    assert len({tuple(sorted(config.data.train_component_shuffle_keys.items())) for _, config in selected}) == 1
    assert all(config.data.max_train_batches == {"dolma/starcoder": 40} for _, config in selected)


def test_resume_requires_success_and_matching_fingerprint(tmp_path: Path) -> None:
    step = ArtifactStep(
        name="training",
        version=launcher.VERSION,
        artifact_type=LevanterCheckpoint,
        run=lambda _config: None,
        build_config=lambda _ctx: {"trainer_seed": 20260711},
    )
    path = step.path(str(tmp_path))
    status = StatusFile(path, worker_id="test")
    status.write_status(STATUS_FAILED)
    assert launcher.pending_training_steps((step,), marin_prefix=str(tmp_path)) == (step,)

    status.write_status(STATUS_SUCCESS)
    with pytest.raises(RuntimeError, match="no artifact fingerprint"):
        launcher.pending_training_steps((step,), marin_prefix=str(tmp_path))

    write_record(ArtifactRecord(output_path=path, fingerprint=step.fingerprint()))
    assert launcher.pending_training_steps((step,), marin_prefix=str(tmp_path)) == ()

    changed = replace(step, build_config=lambda _ctx: {"trainer_seed": 20260908})
    with pytest.raises(FingerprintMismatchError):
        launcher.pending_training_steps((changed,), marin_prefix=str(tmp_path))


def test_noncentral_deployment_is_rejected_before_planning(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    with pytest.raises(ValueError, match="frozen to us-central1"):
        launcher.main(["--marin-prefix", "gs://marin-us-east5", "--plan-path", str(output)])
    assert not output.exists()


def test_primary_release_waits_for_successful_pilot_artifacts(tmp_path: Path) -> None:
    design = launcher.design_module.load_design()
    _, pilot_steps = launcher.build_training_steps(design, "pilot")
    for step in pilot_steps:
        path = step.path(str(tmp_path))
        write_record(ArtifactRecord(output_path=path, fingerprint=step.fingerprint()))
        StatusFile(path, worker_id="test").write_status(STATUS_SUCCESS)
    unfinished_path = pilot_steps[-1].path(str(tmp_path))
    StatusFile(unfinished_path, worker_id="test").write_status(STATUS_FAILED)
    with pytest.raises(RuntimeError, match="Cannot submit primary"):
        launcher.require_previous_stage_complete(design, "primary", marin_prefix=str(tmp_path))

    StatusFile(unfinished_path, worker_id="test").write_status(STATUS_SUCCESS)
    # The gate concerns completion and identity, not measured gains or response direction.
    launcher.require_previous_stage_complete(design, "primary", marin_prefix=str(tmp_path))
    assert launcher.pending_training_steps(pilot_steps, marin_prefix=str(tmp_path)) == ()


@pytest.mark.parametrize("failure", [None, "stale_step", "conflicting_values", "wrong_fingerprint", "failed_status"])
def test_result_collection_requires_verified_final_evaluation(tmp_path: Path, failure: str | None) -> None:
    design = replace(launcher.design_module.load_design(), primary_metric="eval/test_fixture/bpb")
    requests, steps = launcher.build_training_steps(design, "pilot")
    request, step = requests[0], steps[0]
    artifact_path = step.path(str(tmp_path))
    fingerprint = "wrong" if failure == "wrong_fingerprint" else step.fingerprint()
    write_record(ArtifactRecord(output_path=artifact_path, fingerprint=fingerprint))
    StatusFile(artifact_path, worker_id="test").write_status(
        STATUS_FAILED if failure == "failed_status" else STATUS_SUCCESS
    )
    checkpoint_dir = Path(LevanterCheckpoint(path=artifact_path).checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    metric = design.primary_metric
    final_step = request.total_steps - 1
    records = [{"step": final_step - 1 if failure == "stale_step" else final_step, metric: 0.9}]
    if failure == "conflicting_values":
        records.append({"step": final_step, metric: 0.91})
    (checkpoint_dir / "eval_metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "measured.csv"

    if failure is not None:
        with pytest.raises((ValueError, RuntimeError, FingerprintMismatchError)):
            launcher.collect_results(design, (request,), (step,), marin_prefix=str(tmp_path), output_path=output)
        assert not output.exists()
        return

    assert launcher.collect_results(design, (request,), (step,), marin_prefix=str(tmp_path), output_path=output) == 1
    with output.open() as stream:
        (row,) = list(csv.DictReader(stream))
    assert row == {
        "run_name": request.run_name,
        "step": str(final_step),
        "metric": metric,
        "value": "0.9",
        "design_sha256": design.design_sha256,
        "config_fingerprint": step.fingerprint(),
        "status": "succeeded",
    }


def test_submission_requires_reuse_receipt_before_planning(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    output = tmp_path / "unsubmitted_plan.json"
    with pytest.raises(ValueError, match="requires --reuse-audit"):
        launcher.main(["--submit", "--plan-path", str(output)])
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"design_sha256": "stale"},
        {"status": "pending"},
        {"legacy_parent_match": False},
        {"nested_subset_match": False},
        {"historical_config_uri": "gs://marin-us-east5/unreviewed/config.json"},
        {"historical_config_sha256": "missing"},
        {"packed_starcoder_sequence_count": 1},
    ],
)
def test_reuse_receipt_rejects_stale_or_unverified_evidence(tmp_path: Path, change: dict) -> None:
    design = launcher.design_module.load_design()
    # Synthetic evidence exists only in this temporary validation test, never as an experiment receipt.
    receipt = {
        "design_sha256": design.design_sha256,
        "status": "passed",
        "historical_config_uri": "gs://marin-us-central1/synthetic-test-fixture/config.json",
        "historical_config_sha256": hashlib.sha256(b"synthetic config").hexdigest(),
        "parent_indices_sha256": hashlib.sha256(b"synthetic parent indices").hexdigest(),
        "matched_indices_sha256": hashlib.sha256(b"synthetic matched indices").hexdigest(),
        "packed_starcoder_sequence_count": 1_000_000,
        "legacy_parent_match": True,
        "nested_subset_match": True,
    }
    path = tmp_path / "synthetic_receipt.json"
    path.write_text(json.dumps(receipt))
    assert launcher.validate_reuse_audit(design, path) == receipt
    path.write_text(json.dumps({**receipt, **change}))
    with pytest.raises(ValueError):
        launcher.validate_reuse_audit(design, path)


@pytest.mark.parametrize("change", [{"max_train_batches_subset_seed": 11}, {"shuffle": True}])
def test_runtime_audit_rejects_changed_subset_permutation(cached_metadata: None, change: dict) -> None:
    design = launcher.design_module.load_design()
    requests, steps = launcher.build_training_steps(design, "pilot")
    request, step = requests[0], steps[0]
    pod = materialized_config(step, launcher.base.DEFAULT_MARIN_PREFIX)
    changed_config = replace(pod.train_config, data=replace(pod.train_config.data, **change))
    changed_pod = replace(pod, train_config=changed_config)
    changed_step = replace(step, build_config=lambda _ctx: changed_pod, expected_fingerprint=None)
    with pytest.raises(ValueError, match=r"subset permutation|block-shuffle geometry"):
        launcher.audit_runtime_configs(
            design, (request,), (changed_step,), marin_prefix=launcher.base.DEFAULT_MARIN_PREFIX
        )


def test_submission_plan_is_create_only_and_reusable_when_identical(tmp_path: Path) -> None:
    path = tmp_path / "durable" / "launch_plan.json"
    plan = {"design_sha256": "synthetic-test-design", "runs": [{"run_name": "test_run"}], "reuse_audit": {"test": True}}
    launcher.persist_submission_plan(plan, path.as_uri())
    original = path.read_bytes()
    assert json.loads(original) == plan
    launcher.persist_submission_plan(plan, path.as_uri())
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="refusing to overwrite"):
        launcher.persist_submission_plan({**plan, "runs": []}, path.as_uri())
    assert path.read_bytes() == original


def index_reference_receipt(design: launcher.design_module.ExperimentDesign) -> dict[str, Any]:
    # Full index hashes independently measured under the historical 0.10.1 runtime.
    return {
        "design_sha256": design.design_sha256,
        "status": "passed",
        "packed_starcoder_sequence_count": 105745752,
        "parent_indices_sha256": "24c5bdd4bd466de18bc56ba06ea92053e329ad749aadea354a983fb38c61c4c1",
        "matched_indices_sha256": "6c8ac23d240d4c99f88b2fa854cb7f7415b19e5387a68cf634f03dd73d2706c3",
        "legacy_parent_match": True,
        "nested_subset_match": True,
        "audit_runtime": {"jax_version": "0.10.1"},
    }


def test_runtime_index_audit_accepts_0111_with_matching_historical_hashes(monkeypatch: Any) -> None:
    original = launcher.design_module.load_design()
    alternate = replace(original, design_version="unit-test-alternate-path")
    payload = alternate.to_dict()
    payload.pop("design_sha256")
    design = replace(alternate, design_sha256=launcher.design_module.canonical_sha256(payload))
    monkeypatch.setattr(launcher.jax, "__version__", "0.11.1")
    receipt = index_reference_receipt(design)

    result = launcher.validate_runtime_indices(design, receipt)

    assert result["status"] == "passed"
    assert result["design_sha256"] == design.design_sha256
    assert result["parent_sequence_count"] == 136704
    assert result["matched_sequence_count"] == 5120
    assert result["parent_indices_sha256"] == receipt["parent_indices_sha256"]
    assert result["observed_environment"]["jax_version"] == "0.11.1"
    assert result["child_process_verified"] is False


@pytest.mark.parametrize("field", ["parent_indices_sha256", "matched_indices_sha256"])
def test_runtime_index_audit_rejects_wrong_reviewed_hash(field: str) -> None:
    design = launcher.design_module.load_design()
    receipt = index_reference_receipt(design)
    receipt[field] = "f" * 64
    with pytest.raises(ValueError, match=field):
        launcher.validate_runtime_indices(design, receipt)


def test_refinement_selects_only_authorized_reference_seed_additions() -> None:
    design = launcher.design_module.load_design()
    pilot = launcher.design_module.select_runs(design, "pilot")
    refinement = launcher.design_module.select_runs(design, "refinement")
    primary = launcher.design_module.select_runs(design, "primary")
    replicated = launcher.design_module.select_runs(design, "replicated")
    pilot_names = {run.run_name for run in pilot}
    additional = [run for run in refinement if run.run_name not in pilot_names]

    assert len(refinement) == 20
    assert {run.run_name for run in refinement} >= pilot_names
    assert {(run.arm, run.starcoder_weight, run.trainer_seed) for run in additional} == {
        (arm, weight, 20260711) for arm in ("matched", "unmatched") for weight in (0.2, 0.4, 0.5, 0.6, 0.9)
    }
    assert len(additional) == 10
    assert {run.run_name for run in primary} == {run.run_name for run in design.runs if run.trainer_seed == 20260711}
    assert len(primary) == 52
    assert replicated == design.runs
    assert len(replicated) == 154


def test_refinement_requires_pilot_and_skips_its_successful_artifacts(tmp_path: Path) -> None:
    design = launcher.design_module.load_design()
    pilot_rows, pilot_steps = launcher.build_training_steps(design, "pilot")
    refinement_rows, refinement_steps = launcher.build_training_steps(design, "refinement")
    with pytest.raises(RuntimeError, match="Cannot submit refinement"):
        launcher.require_previous_stage_complete(design, "refinement", marin_prefix=str(tmp_path))
    for step in pilot_steps:
        path = step.path(str(tmp_path))
        write_record(ArtifactRecord(output_path=path, fingerprint=step.fingerprint()))
        StatusFile(path, worker_id="test").write_status(STATUS_SUCCESS)

    launcher.require_previous_stage_complete(design, "refinement", marin_prefix=str(tmp_path))
    # Primary retains its original pilot gate even while refinement runs are unfinished.
    launcher.require_previous_stage_complete(design, "primary", marin_prefix=str(tmp_path))
    pending = launcher.pending_training_steps(refinement_steps, marin_prefix=str(tmp_path))
    pilot_names = {run.run_name for run in pilot_rows}
    expected = tuple(
        step for row, step in zip(refinement_rows, refinement_steps, strict=True) if row.run_name not in pilot_names
    )
    assert pending == expected
    assert len(pending) == 10
    plan = launcher.launch_plan(design, "refinement", refinement_rows, refinement_steps)
    assert plan["adaptive_release"]["based_on_stage"] == "pilot"
    assert plan["adaptive_release"]["additional_starcoder_weights"] == [0.2, 0.4, 0.5, 0.6, 0.9]
    assert "adaptive_release" not in launcher.launch_plan(design, "pilot", pilot_rows, pilot_steps)
