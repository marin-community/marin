# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from marin.execution.artifact import ArtifactRecord, write_record
from marin.rl.skyrl import SkyRLRun
from rigging.event_log import EventLog
from rigging.filesystem.storage_path import StoragePath
from rigging.training_journal import TrainingJournal

from experiments.post_training.long_loop.runtime_timing import write_report


def test_producer_and_adopted_native_journals_are_reported(tmp_path):
    journal = tmp_path / "native.jsonl"
    log = EventLog(journal, "original-sft", "sft", "rank0")
    with log.span("compile"):
        pass
    eval_root = tmp_path / "eval"
    results = eval_root / "exports/dumped_evals/eval_only"
    results.mkdir(parents=True)
    row = {
        "data_source": "tiny",
        "verification": {"status": "verified", "score": 1},
        "response_token_ids": [1],
        "stop_reason": "stop",
    }
    (results / "results.jsonl").write_text(json.dumps(row))
    trained = tmp_path / "trained"
    trained.mkdir()
    native = TrainingJournal(str(trained / "telemetry"), "producer-sft", "sft", "rank-0")
    with native.events.span("steady_state"):
        native.log({"train/loss": 0.2}, step=1, commit=True)
    native.finish(status="completed")
    terminal = trained / "terminal.json"
    terminal.write_text(json.dumps({"result": {"state": "failed", "failure": "GroupAdmissionStalledError"}}))
    run = SkyRLRun(
        hf_model_uri="selected-best-export",
        global_step=16,
        tokenizer_uri="tokenizer",
        tokenizer_revision="pin",
        checkpoint_root="native-checkpoints",
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(terminal),
        iris_job_id="/user/rl",
    )
    write_record(ArtifactRecord(result=run.result_payload(), output_path=str(trained)))
    output = tmp_path / "report"
    output.mkdir()
    pin = {"source": str(journal), "sha256": hashlib.sha256(journal.read_bytes()).hexdigest()}
    write_report(
        output=str(output),
        stages=[str(trained), str(eval_root)],
        expected_rollouts=1,
        sft_geometry="CPU fixture",
        additional_timing=[pin],
        rl_terminal=str(trained),
    )
    assert json.loads((output / "rl-terminal.json").read_text()) == {
        "state": "failed",
        "failure": "GroupAdmissionStalledError",
    }
    sources = json.loads((output / "timing-sources.json").read_text())
    assert str(journal) in sources
    assert len(sources) == 2
    assert any(StoragePath(source).read_bytes() == native.events.path.read_bytes() for source in sources)
    journal.write_text(journal.read_text() + "\n")
    with pytest.raises(ValueError, match="Pinned native timing journal changed"):
        write_report(
            output=str(output),
            stages=[str(trained), str(eval_root)],
            expected_rollouts=1,
            sft_geometry="CPU fixture",
            additional_timing=[pin],
        )


@pytest.mark.parametrize("failure", ["accepted", "unaccepted", "missing_identity"])
def test_report_preserves_explicit_unscored_attempt_without_certifying_full_scoring(tmp_path, failure):
    evaluation = tmp_path / "eval"
    results = evaluation / "exports/dumped_evals/eval_only"
    results.mkdir(parents=True)
    rows = [
        {"uid": "0", "verification": {"status": "verified", "score": 1}},
        {"uid": "1", "verification": {"status": "error", "score": None}, "exception_type": "AddTestsDirError"},
    ]
    rows = [{**row, "data_source": "bench", "response_token_ids": [1], "stop_reason": "stop"} for row in rows]
    (results / "bench.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    selection = tmp_path / "selection"
    selection.mkdir()
    (selection / "selection.json").write_text(
        json.dumps(
            {
                "attempts": 2,
                "attempts_per_data_source": {"bench": 2},
                "identities": [
                    {"data_source": "bench", "dataset_row_uid": "0"},
                    {"data_source": "bench", "dataset_row_uid": "2" if failure == "missing_identity" else "1"},
                ],
            }
        )
    )
    output = tmp_path / "report"
    output.mkdir()
    accepted = [{"data_source": "bench", "uid": "1", "exception_type": "AddTestsDirError"}]
    arguments = dict(
        output=str(output),
        stages=[str(evaluation)],
        expected_rollouts=2,
        evaluation_selection=str(selection),
        sft_geometry="CPU fixture",
        accepted_unscored_attempts=[] if failure == "unaccepted" else accepted,
    )
    if failure != "accepted":
        with pytest.raises(ValueError, match="unaccepted unscored"):
            write_report(**arguments)
        return
    write_report(**arguments)
    summary = json.loads((output / "evaluation.json").read_text())
    assert not summary["all_verified"]
    assert summary["verified_rollouts"] == 1
    assert summary["benchmarks"][0]["mean_verified_reward"] == 1
    assert summary["unscored_attempts"] == summary["accepted_unscored_attempts"] == accepted
