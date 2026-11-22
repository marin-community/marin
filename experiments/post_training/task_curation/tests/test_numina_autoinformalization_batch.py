# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of the single-invocation Gemini scripts."""

import io
import json
import sys

import pytest

from experiments.post_training.task_curation.datasets.lean import (
    numina_autoinformalization_classify_answer_disclosure as classifier,
)
from experiments.post_training.task_curation.datasets.lean import (
    numina_autoinformalization_gemini_batch as batch,
)
from experiments.post_training.task_curation.datasets.lean import (
    numina_autoinformalization_golden_batch as golden,
)
from experiments.post_training.task_curation.datasets.lean import (
    numina_autoinformalization_quality as quality,
)
from experiments.post_training.task_curation.tests.test_numina_autoinformalization_quality import report_row


def test_batch_submits_waits_collects_and_resumes_without_another_submission(tmp_path, monkeypatch):
    requests = {"r": {"contents": []}}
    events = []
    statuses = iter(
        [{"name": "batches/test", "done": False}, {"done": True, "response": {"responsesFile": "files/output"}}]
    )
    monkeypatch.setenv("GEMINI_API_KEY", "offline")
    monkeypatch.setattr(batch, "upload_file", lambda *_: "files/input")
    monkeypatch.setattr(batch.time, "sleep", lambda _: events.append("wait"))

    def api(path, body=None):
        events.append(path)
        if path == "files/input":
            return {"state": "ACTIVE"}
        if body is not None:
            return {"name": "batches/test"}
        return next(statuses)

    monkeypatch.setattr(batch, "api_request", api)
    # Interrupt collection after submission and after waiting for completion.
    monkeypatch.setattr(batch, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("download")))
    with pytest.raises(TimeoutError):
        batch.run_batch(requests, "offline-model", tmp_path)
    monkeypatch.setattr(
        batch, "api_request", lambda *a, **kw: {"done": True, "response": {"responsesFile": "files/output"}}
    )
    monkeypatch.setattr(batch, "urlopen", lambda *a, **kw: io.BytesIO(b'{"key":"r","response":{}}\n'))
    result = batch.run_batch(requests, "offline-model", tmp_path)
    assert result == {"r": {"key": "r", "response": {}}}
    monkeypatch.setattr(batch, "api_request", lambda *a, **kw: pytest.fail("Completed batch used the network"))
    assert batch.run_batch(requests, "offline-model", tmp_path) == result
    assert events.count("models/offline-model:batchGenerateContent") == 1
    assert events.count("wait") == 1
    with pytest.raises(ValueError, match="different inputs"):
        batch.run_batch({"changed": {}}, "offline-model", tmp_path)


def test_uncertain_submission_never_resubmits(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline")
    monkeypatch.setattr(batch, "upload_file", lambda *_: "files/input")

    def api(path, body=None):
        if body is not None:
            raise TimeoutError("submission")
        return {"state": "ACTIVE"}

    monkeypatch.setattr(batch, "api_request", api)
    with pytest.raises(TimeoutError):
        batch.run_batch({"r": {}}, "offline-model", tmp_path)
    monkeypatch.setattr(batch, "api_request", lambda *_: pytest.fail("Automatic resubmission"))
    with pytest.raises(ValueError, match="Submission uncertain"):
        batch.run_batch({"r": {}}, "offline-model", tmp_path)


@pytest.mark.parametrize("module", [classifier, golden])
def test_cli_selects_raw_rows_and_writes_results_in_one_run(tmp_path, monkeypatch, module):
    row = report_row("0")
    monkeypatch.setattr(quality, "source_rows", lambda *_: [row])
    label = {"answer_disclosure": "no", "problem_lines": [1], "lean_lines": [1], "reason": "Given for proof."}
    text = json.dumps(label) if module is classifier else "Matches the theorem.\nPASS"
    seen = []

    def run(requests, *_):
        seen.extend(requests.values())
        return {
            "0": {
                "response": {
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 12},
                }
            }
        }

    monkeypatch.setattr(module, "run_batch", run)
    output = tmp_path / "results.jsonl"
    args = ["script", "source.parquet", "--prime-lists", str(tmp_path), "--output", str(output)]
    if module is golden:
        labels = tmp_path / "labels.jsonl"
        quality.write_records(labels, [dict(row="0", label="no")])
        args.extend(["--labels", str(labels), "--max-output-tokens", "2048"])
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    result = quality.read_records(output)[0]
    if module is classifier:
        assert result["label"] == "no"
    else:
        assert result["golden_control"] == "PASS"
        assert seen[0]["generationConfig"]["maxOutputTokens"] == 2048
        payload = json.loads(seen[0]["contents"][0]["parts"][0]["text"])
        assert payload == dict(question=row["formal"], reference=row["reference"], candidate=row["reference"])
    assert result["usage"]["promptTokenCount"] == 100


def test_golden_truncation_is_saved_without_a_second_request():
    assert golden.verdict({"response": {"candidates": [{"finishReason": "MAX_TOKENS"}]}})[0] == "MALFORMED"
