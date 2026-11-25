# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from evalrecords.records import EvalRef, EvalRunRecord, HardwareRef, ModelRef, Provenance, RunStatus, read_record
from finestore.reader import ReadView
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.eval_archive import ROW_VERSION, publish_evaluation


def _record(tmp_path, results_path):
    return EvalRunRecord(
        run_id="smoke",
        group_id="smoke",
        created_at="2026-11-09T00:00:00Z",
        user="test",
        model=ModelRef(name="test", location="test", backend="vllm"),
        evaluation=EvalRef(name="smoke", mechanism="skyrl"),
        hardware=HardwareRef(platform="local", accelerator="none", region_or_cluster=None),
        status=RunStatus.FAILED,
        error="one verifier failed",
        results_path=str(results_path),
        metrics={},
        jobs={},
        log_tails={},
        provenance=Provenance(git_sha="a" * 40, launch_host="local"),
    )


def test_evaldash_archive_keeps_failed_trial_without_counting_it_as_zero_reward(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "uid": "a",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "verification": {"status": "verified", "score": 0.8, "passed": True},
        },
        {
            "uid": "b",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "verification": {"status": "error", "score": None},
        },
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (source / "bench.jsonl").write_bytes(raw)
    record = _record(tmp_path, tmp_path / "results")
    path = publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals"))
    assert publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals")) == path
    stored = read_record(path)
    assert stored.metrics == {"bench": {"reward": 0.8}}
    assert stored.coverage["bench"].n_attempted == 2
    assert stored.coverage["bench"].n_scored == 1
    assert stored.coverage["bench"].n_correct == 1
    assert stored.status == RunStatus.FAILED
    view = ReadView(stored.results_path)
    assert view.read_blob("sources/bench.jsonl") == raw
    assert view.is_sealed()


def test_evaldash_archive_fills_the_why_fields_from_the_trajectory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            # A completed trial: the response after the think block is the model's answer.
            "uid": "1",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "output_response": "<|start_think|>weighing the options<|end_think|>\n\nThe answer is 42.<|eot_id|>",
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "The answer is 42."}],
            "stop_reason": "stop",
            "verification": {"status": "verified", "score": 1.0, "passed": True},
        },
        {
            # Truncated mid-thinking: no final response, so the row carries a visible reason.
            "uid": "2",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "output_response": "<|start_think|>half a derivation",
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}],
            "stop_reason": "length",
            "verification": {"status": "verified", "score": 0.0, "passed": None},
        },
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (source / "bench.jsonl").write_bytes(raw)
    record = _record(tmp_path, tmp_path / "results")
    path = publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals"))
    stored = read_record(path)
    assert stored.metrics == {"bench": {"reward": 0.5}}
    view = ReadView(stored.results_path)
    samples = {row["doc_id"]: row for row in view.iter_rows("samples")}
    assert samples["1"]["output"] == "The answer is 42."
    assert samples["1"]["extracted"] == "The answer is 42."
    # The native dump carries no reference answer, so target_text stays unset.
    assert samples["1"]["target_text"] is None
    assert samples["2"]["output"] is None
    assert samples["2"]["extracted"] == "no answer (stop_reason=length)"


def test_evaldash_archive_re_archives_rows_from_an_older_row_contract(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "uid": "1",
            "data_source": "bench",
            "env_class": "taskcompendium",
            "output_response": "<|start_think|>thinking<|end_think|>The answer is 42.",
            "messages": [{"role": "assistant", "content": "The answer is 42."}],
            "stop_reason": "stop",
            "verification": {"status": "verified", "score": 1.0, "passed": True},
        }
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (source / "bench.jsonl").write_bytes(raw)
    record = _record(tmp_path, tmp_path / "results")
    path = publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals"))
    marker_path = record.results_path + "/publication.json"

    def read_marker():
        return json.loads(StoragePath(marker_path).read_bytes())

    assert read_marker()["row_version"] == ROW_VERSION
    # A marker from the older contract (no row_version) with the same input signature triggers a
    # deliberate re-archive instead of the stale-input error.
    old_marker = {key: value for key, value in read_marker().items() if key != "row_version"}
    StoragePath(marker_path).write_bytes(json.dumps(old_marker, sort_keys=True).encode())
    before = json.dumps(read_record(path).metrics, sort_keys=True)
    path = publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals"))
    assert read_marker()["row_version"] == ROW_VERSION
    # Scores and metrics stay byte-identical across the re-archive.
    assert json.dumps(read_record(path).metrics, sort_keys=True) == before
    view = ReadView(record.results_path)
    samples = {row["doc_id"]: row for row in view.iter_rows("samples")}
    assert samples["1"]["output"] == "The answer is 42."
    assert view.is_sealed()
    # The refreshed marker makes a further publish idempotent again.
    assert publish_evaluation(source=str(source), record=record, records_prefix=str(tmp_path / "evals")) == path
