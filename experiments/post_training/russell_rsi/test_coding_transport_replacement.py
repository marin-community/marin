# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise replacement graphs with synthetic metadata and no model calls."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import EVALCHEMY, MARIN_SKYRL

from experiments.post_training.russell_rsi import test_retention_continuation
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_transport_replacement import (
    AMENDMENT_PROTOCOL,
    PROTOCOL,
    REPLACEMENT_SETTINGS,
    SELECTION_PROTOCOL,
    VERSION,
    prepare_coding_replacement,
    replacement_coding_attempt,
    replacement_selection_stages,
    run_replacement_coding,
)
from experiments.post_training.russell_rsi.retention_continuation import (
    FAILED_SOURCE_COMMIT,
    prepare_retention_continuation,
    submit_continuation_retention,
)

continuation_inputs = test_retention_continuation.continuation_inputs
study_inputs = test_retention_continuation.study_inputs
interrupted_study = test_retention_continuation.interrupted_study
completed_coding = test_retention_continuation.completed_coding


@pytest.fixture
def replacement_inputs(completed_coding, tmp_path):
    old = completed_coding
    failure_pin = old.pin(tmp_path / "retention-failure.json", old.failure)
    retained = {
        "protocol": test_retention_continuation.PROTOCOL,
        "version": test_retention_continuation.VERSION,
        "source_config_uri": old.config["source_config_uri"],
        "source_config_sha256": old.config["source_config_sha256"],
        "launch_failure_uri": failure_pin["uri"],
        "launch_failure_sha256": failure_pin["sha256"],
    }
    retain_pin = PinnedFile(**old.pin(tmp_path / "retain.json", retained))
    retention = prepare_retention_continuation(retained, old.source, old.original, retain_pin)
    attribution = {
        "protocol": "russell-rsi-coding-issuance-attribution-v1",
        "status": "root-reviewed",
        "benchmark_generation_requests": 0,
        "startup_probe_requests": 1,
        "scored_samples": 0,
        "calibration_status": "incomplete_infrastructure",
        "signal_gate_passed": None,
        "rl_authorized": False,
    }
    amendment = {
        "protocol": AMENDMENT_PROTOCOL,
        "calibration_status": "incomplete_infrastructure",
        "signal_gate_passed": None,
        "rl_authorized": False,
        "source": {
            "config": {"uri": retained["source_config_uri"], "sha256": retained["source_config_sha256"]},
            "source_commit": FAILED_SOURCE_COMMIT,
            "runtime_commit": MARIN_SKYRL.commit,
            "evalchemy_commit": EVALCHEMY.commit,
            "model_identity": artifact_identity(retention.model),
            "coding_identity": artifact_identity(retention.coding),
            "coding_panel_sha256": retention.selection.record.panel_sha256,
        },
        "original": {"status": "failed_infrastructure", "startup_probe_requests": 1, "benchmark_generation_requests": 0},
        "replacement": REPLACEMENT_SETTINGS,
        "retention": {"config": asdict(retain_pin), "fingerprint": retention.step.fingerprint()},
        "evidence": {
            name: old.pin(
                tmp_path / f"{name}.json", attribution if name == "issuance_attribution" else {"witness": name}
            )
            for name in ("terminal_inventory", "child_config", "server_metrics", "issuance_attribution")
        },
    }
    amendment_pin = old.pin(tmp_path / "coding-amendment.json", amendment)
    config = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "source_config_uri": retained["source_config_uri"],
        "source_config_sha256": retained["source_config_sha256"],
        "transport_amendment_uri": amendment_pin["uri"],
        "transport_amendment_sha256": amendment_pin["sha256"],
    }
    config_pin = PinnedFile(**old.pin(tmp_path / "coding-config.json", config))
    return old, retention, amendment, config, config_pin


def test_replacement_graph_schedules_only_coding_and_preserves_original(replacement_inputs):
    old, retention, _, config, config_pin = replacement_inputs
    original_identity = artifact_identity(retention.coding)
    prepared = prepare_coding_replacement(config, config_pin, retention)
    graph = graph_handles([prepared.coding])
    assert sum(step.run is run_replacement_coding for step in graph) == 1
    assert all(step.run is not submit_continuation_retention for step in graph)
    assert artifact_identity(retention.coding) == original_identity
    bound = prepared.coding.build_config(
        StepContext.for_run(
            "/metadata", "/outputs", deps=prepared.coding.deps, runtime_args=prepared.coding.runtime_args
        )
    )
    original = retention.coding.build_config(
        StepContext.for_run(
            "/metadata", "/outputs", deps=retention.coding.deps, runtime_args=retention.coding.runtime_args
        )
    )
    expected = asdict(original)
    expected["version"] = VERSION
    assert asdict(bound.evaluation) == expected
    assert bound.evaluation.model.identity == original.model.identity
    assert old.source["version"] != VERSION


def test_replacement_replay_and_incomplete_refusal_before_any_client(replacement_inputs, tmp_path):
    old, retention, _, config, config_pin = replacement_inputs
    prepared = prepare_coding_replacement(config, config_pin, retention)
    bound = prepared.coding.build_config(
        StepContext.for_run(
            str(tmp_path / "coding"),
            str(tmp_path),
            deps=prepared.coding.deps,
            runtime_args=prepared.coding.runtime_args,
        )
    )
    attempt = replacement_coding_attempt(bound)
    saved = {**old.saved, "path": bound.evaluation.artifact_path}

    async def finish():
        return saved

    asyncio.run(attempt.run(finish))
    assert run_replacement_coding(bound).results_paths == tuple(saved["results_paths"])
    assert attempt.binding["config"] == json.loads(json.dumps(asdict(bound.evaluation)))
    for evaluation in attempt.binding["transport_replacement"]["evaluations"]:
        assert evaluation["route"] == "capability"
        arguments = evaluation["executor_config"]["extra_model_args"]
        assert arguments["max_retries"] == 1
        assert arguments["transport_retry_budget"] == 900
    (Path(bound.evaluation.artifact_path) / "journal/coding/result.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        run_replacement_coding(bound)


@pytest.mark.parametrize("defect", ["issued", "model", "panel", "retention", "retry"])
def test_replacement_rejects_changed_science_or_issuance(replacement_inputs, tmp_path, defect):
    old, retention, amendment, config, _ = replacement_inputs
    if defect == "issued":
        amendment["original"]["benchmark_generation_requests"] = 1
    elif defect in {"model", "panel"}:
        amendment["source"]["model_identity" if defect == "model" else "coding_panel_sha256"] = "changed"
    elif defect == "retention":
        amendment["retention"]["fingerprint"] = "changed"
    else:
        amendment["replacement"] = {**REPLACEMENT_SETTINGS, "transport_retry_budget": 0}
    pin = old.pin(tmp_path / "changed-amendment.json", amendment)
    config = {**config, "transport_amendment_uri": pin["uri"], "transport_amendment_sha256": pin["sha256"]}
    config_pin = PinnedFile(**old.pin(tmp_path / "changed-config.json", config))
    with pytest.raises(ValueError, match="amendment changed"):
        prepare_coding_replacement(config, config_pin, retention)


def test_selection_requires_both_results_and_schedules_neither_eval(replacement_inputs, tmp_path):
    old, retention, _, config, config_pin = replacement_inputs
    prepared = prepare_coding_replacement(config, config_pin, retention)
    output = tmp_path / "replacement-completed"
    bound = prepared.coding.build_config(
        StepContext.for_run(
            str(output),
            old.source["recovery_artifact_prefix"],
            deps=prepared.coding.deps,
            runtime_args=prepared.coding.runtime_args,
        )
    )
    saved = {**old.saved, "path": str(output)}
    producer = {
        "name": prepared.coding.name,
        "version": VERSION,
        "fingerprint": prepared.coding.fingerprint(),
        "output_path": str(output),
        "config": asdict(bound),
        "result": {key: value for key, value in saved.items() if key != "path"},
    }
    result_pin = old.pin(output / ".artifact.json", producer)
    journal = replacement_coding_attempt(bound)

    async def completed():
        return saved

    asyncio.run(journal.run(completed))
    journal_pin = old.pin(output / "journal/coding/result.json", {"binding": journal.binding, "result": saved})
    selected = {
        "protocol": SELECTION_PROTOCOL,
        "version": VERSION,
        "coding_config_uri": config_pin.uri,
        "coding_config_sha256": config_pin.sha256,
        "retention_config_uri": retention.retention_config.uri,
        "retention_config_sha256": retention.retention_config.sha256,
        "coding_result_uri": result_pin["uri"],
        "coding_result_sha256": result_pin["sha256"],
        "coding_journal_result_uri": journal_pin["uri"],
        "coding_journal_result_sha256": journal_pin["sha256"],
    }
    with pytest.raises(ValueError, match="completed frozen retention"):
        replacement_selection_stages(selected, prepared)
    test_retention_continuation.completed_retention(retention, old.pin)
    with pytest.raises(ValueError, match="completed coding"):
        replacement_selection_stages(selected, prepared)
    (output / ".executor_status").write_text("SUCCESS")
    stages = replacement_selection_stages(selected, prepared)
    graph = graph_handles([stages["terminal"]])
    assert all(step.run is not run_replacement_coding for step in graph)
    assert all(step.run is not submit_continuation_retention for step in graph)
    adopted = stages["retention-sft"].adopt_config
    assert adopted is not None
    assert adopted["producer_identity"] == artifact_identity(retention.step)
    selection = stages["terminal"].build_config(
        StepContext.for_run(
            str(tmp_path / "selected"), old.source["recovery_artifact_prefix"], deps=stages["terminal"].deps
        )
    )
    assert selection.selection.selection.record.parent == old.selection.record.parent
    assert selection.retention_config == retention.retention_config
    Path(journal_pin["uri"]).write_text("changed")
    with pytest.raises(ValueError):
        replacement_selection_stages(selected, prepared)
