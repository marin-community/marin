# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind native worker inputs and retained attempt rows to the selected protocol."""

import hashlib
import json
from pathlib import Path

import pytest
from finestore.rl.difficulty import LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID, PROTOCOL_ID, Solver
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty import ledger_worker
from experiments.post_training.task_curation.difficulty.accounting import GpuAccounting
from experiments.post_training.task_curation.difficulty.campaign import identity_bundle
from experiments.post_training.task_curation.difficulty.importer import import_run, receipt_row
from experiments.post_training.task_curation.difficulty.ledger import digest, read_rows
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch
from experiments.post_training.task_curation.difficulty.state import Outcome
from experiments.post_training.task_curation.difficulty_config import CandidateRun
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource
from experiments.post_training.task_curation.difficulty_sampling import sampling_settings

V2 = LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID


def candidate(protocol_id=V2):
    return CandidateRun.model_validate(
        {
            "solver": "large",
            "solver_protocol_id": protocol_id,
            "identity": {
                "model": "Qwen/Qwen3.5-122B-A10B",
                "revision": "d" * 40,
                "tokenizer": "Qwen/Qwen3.5-122B-A10B",
                "tokenizer_revision": "d" * 40,
                "chat_template_sha256": "e" * 64,
                "serving_receipt": {"uri": "memory://serving", "sha256": "a" * 64},
            },
            "settings": sampling_settings(Solver.LARGE, [], protocol_id),
            "endpoint": "http://model.invalid/v1",
            "api_model": "frozen",
            "api_key_secret": None,
            "provider_limits": {"max_in_flight": 1, "attempts": 1, "request_timeout": 600},
            "environment_options": {},
            "judge": None,
            "workers": 1,
            "gpu_count": 0,
            "attempts_per_task": 1,
        }
    )


def test_frozen_sampling_bytes_and_campaign_identity_stay_unchanged():
    frozen = json.loads((Path(__file__).parent / "fixtures/difficulty-v1-sampling.json").read_text())
    assert {solver.value: sampling_settings(solver, []).model_dump(mode="json") for solver in Solver} == frozen[
        "sampling"
    ]
    assert digest(identity_bundle()) == frozen["campaign_identity_sha256"]
    old = candidate(PROTOCOL_ID)
    wire = old.model_dump(mode="json")
    assert "solver_protocol_id" not in wire and "dynamic_output" not in wire["settings"]
    assert CandidateRun.model_validate_json(old.model_dump_json()) == old
    new = candidate()
    assert CandidateRun.model_validate_json(new.model_dump_json()) == new
    assert new.settings.dynamic_output.output_capacity(12000) == 53472
    wire["settings"] = new.settings.model_dump(mode="json")
    with pytest.raises(ValueError, match="explicit solver protocol"):
        CandidateRun.model_validate(wire)


def test_worker_rejects_cross_protocol_action_before_creating_outputs(monkeypatch, tmp_path):
    config = tmp_path / "candidate.json"
    config.write_text(candidate().model_dump_json())
    sources = tmp_path / "sources.json"
    sources.write_text("[]")
    output, local = tmp_path / "output", tmp_path / "local"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ledger_worker",
            "--sources",
            str(sources),
            "--config",
            str(config),
            "--output-root",
            str(output),
            "--local-root",
            str(local),
            "--ledger",
            str(tmp_path / "ledger"),
            "--action-manifest",
            "action",
        ],
    )
    monkeypatch.setattr(ledger_worker, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(ledger_worker, "load_action", lambda _: AttemptBatch(Solver.LARGE, (), 1))
    with pytest.raises(ValueError, match="does not match"):
        ledger_worker.main()
    assert not local.exists() and not output.exists()


def test_scored_native_receipt_retains_v2_identity_through_worker_and_import(tmp_path):
    config = candidate()
    sample = {
        "source_id": "fixture",
        "release": {"repo": "fixture", "revision": "frozen", "url": "https://example.org/fixture"},
        "converter_version": "frozen",
        "converter_revision": "b" * 40,
        "task_specs_sha256": "c" * 64,
        "tasks": [{"task_id": "task", "task_sha256": "d" * 64}],
        "population_count": 1,
        "drop_manifest": {"uri": "memory://drops", "sha256": "e" * 64},
        "seed": 0,
        "pre_oracle_filter": False,
        "resample_after_oracle_filter": False,
    }
    receipt = tmp_path / "attempt.json"
    receipt.write_text(
        json.dumps({"task_id": "task", "status": "scored", "reward": 0.0, "reason": None, "solver_protocol_id": V2})
    )
    source = PreparedSource.model_validate(
        {"sample": sample, "sample_receipt": {"uri": "memory://sample", "sha256": "f" * 64}, "tasks": []}
    )
    meter = GpuAccounting(0, lambda: 0.0)
    meter.start("task")
    ledger = str(tmp_path / "ledger")
    ledger_worker.record_attempt(
        StoragePath(str(receipt)),
        source=source,
        config=config,
        output=StoragePath(str(tmp_path / "run/trials")),
        ledger=ledger,
        meter=meter,
        task_key="task",
    )
    row = read_rows(ledger)[0]
    assert row.solver_protocol_id == V2 and row.outcome == Outcome.SCORED and row.reward == 0.0
    imported = receipt_row(
        {"uri": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()},
        sample,
        Solver.LARGE,
        str(tmp_path / "run"),
        row.recorded_at,
    )
    assert imported == row


@pytest.mark.parametrize("receipt_protocol", [PROTOCOL_ID, V2])
def test_run_import_never_upgrades_an_old_score_to_the_new_serving_protocol(tmp_path, receipt_protocol):
    run, ledger = tmp_path / "run", str(tmp_path / "ledger")
    run.mkdir()
    (run / "serving.json").write_text(json.dumps({"solver": "large", "protocol": V2}))
    sample = {
        "source_id": "fixture",
        "release": {"repo": "fixture", "revision": "frozen"},
        "converter_version": "frozen",
        "converter_revision": "b" * 40,
        "task_specs_sha256": "c" * 64,
        "tasks": [{"task_id": "task", "task_sha256": "d" * 64}],
    }
    (run / "prepared.json").write_text(json.dumps({"sources": [{"sample": sample}], "source_queue_sha256": "e" * 64}))
    source = run / "trials" / hashlib.sha256(b"fixture").hexdigest()
    attempt = source / "attempts/task/00/attempt.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(
        json.dumps(
            {
                "task_id": "task",
                "status": "scored",
                "reward": 0.0,
                "reason": None,
                **({"solver_protocol_id": V2} if receipt_protocol == V2 else {}),
            }
        )
    )
    (source / "measurement.json").write_text(
        json.dumps(
            {
                "created_at": "2026-11-27T12:00:00+00:00",
                "measurement_id": "fixture",
                "status": "measured",
                "reason": None,
                "gpu_hours": 2.0,
            }
        )
    )
    if receipt_protocol == PROTOCOL_ID:
        with pytest.raises(ValueError, match="different solver protocols"):
            import_run(str(run), ledger)
        assert not read_rows(ledger)
    else:
        report = import_run(str(run), ledger)
        rows = read_rows(ledger)
        assert report["rows"] == 3
        assert all(row.solver_protocol_id == V2 for row in rows)
        assert sum(row.gpu_hours for row in rows) == 2.0
        assert next(row for row in rows if row.outcome == Outcome.SCORED).reward == 0.0
        assert import_run(str(run), ledger)["partition"] == report["partition"]
        assert read_rows(ledger) == rows
