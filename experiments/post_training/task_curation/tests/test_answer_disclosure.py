# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for classifier sampling and spending controls."""

import argparse
import gzip
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.datasets.lean import classify_answer_disclosure as classifier
from experiments.post_training.task_curation.tests.lean_rows import numina_row


def test_prepare_exports_public_statement_without_source_proof_or_comments(tmp_path, monkeypatch):
    row = numina_row(formal_statement="import Mathlib\n/- SECRET -/\ntheorem t (n : Nat) : n + 0 = n := by")
    parquet = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([row, {**row, "author": "autoformalizer"}]), parquet)
    monkeypatch.setitem(classifier.MANIFEST, "input", {"sha256": hashlib.sha256(parquet.read_bytes()).hexdigest()})
    for name in classifier.PRIME_LISTS:
        (tmp_path / f"{name}.json.gz").write_bytes(gzip.compress(b"[]"))
    args = argparse.Namespace(
        parquet=parquet, prime_lists=tmp_path, output=tmp_path / "sample.jsonl", seed="s", limit=250
    )
    classifier.prepare(args)
    records = classifier.read_records(args.output)
    assert len(records) == 1
    assert "SECRET" not in records[0]["formal"]
    assert records[0]["formal"].endswith("n + 0 = n")
    assert "formal_ground_truth" not in records[0]
    assert records[0]["human_answer_disclosure"] == ""


def test_classify_reserves_before_request_and_counts_failed_attempt_on_resume(tmp_path, monkeypatch):
    source = tmp_path / "sample.jsonl"
    source.write_text(json.dumps({"row": "r", "problem": "Find n", "formal": "n = 12"}) + "\n")
    args = argparse.Namespace(
        input=source,
        output=tmp_path / "results.jsonl",
        model="gemini-test",
        input_rate=1.0,
        output_rate=1.0,
        budget=25,
        execute=False,
        limit=0,
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fail_request(*_args, **_kwargs):
        assert classifier.read_records(args.output)[0]["event"] == "reservation"
        raise TimeoutError("simulated interrupted request")

    monkeypatch.setattr(classifier, "urlopen", fail_request)
    classifier.classify(args)
    assert not args.output.exists()
    args.execute = True
    with pytest.raises(TimeoutError):
        classifier.classify(args)
    args.budget = classifier.read_records(args.output)[0]["reserved_usd"]
    classifier.classify(args)
    assert len(classifier.read_records(args.output)) == 1


def test_invalid_evidence_is_saved_as_failure_and_skipped_on_resume(tmp_path, monkeypatch):
    source = tmp_path / "sample.jsonl"
    row = {"row": "r", "problem": "Find n", "formal": "n = 12"}
    source.write_text(json.dumps(row) + "\n")
    label = {
        "answer_disclosure": "yes",
        "lean_lines": [99],
        "problem_lines": [1],
        "reason": "An explicit answer",
    }
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(label)}]}}]}
    calls = []

    def respond(*_args, **_kwargs):
        calls.append(1)
        return io.StringIO(json.dumps(response))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(classifier, "urlopen", respond)
    args = argparse.Namespace(
        input=source,
        output=tmp_path / "results.jsonl",
        model="gemini-test",
        input_rate=1.0,
        output_rate=1.0,
        budget=5,
        execute=True,
        limit=0,
    )
    classifier.classify(args)
    classifier.classify(args)
    assert len(calls) == 1
    saved = classifier.read_records(args.output)[1]
    assert saved["classification"] is None
    assert saved["processing_status"] == "invalid_evidence"
    assert saved["review_reason"]
    assert saved["response"] == response


@pytest.mark.parametrize(
    "label,status",
    [
        ({"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [2]}, "valid"),
        ({"answer_disclosure": "no", "problem_lines": [1], "lean_lines": [2]}, "valid"),
        ({"answer_disclosure": "uncertain", "problem_lines": [], "lean_lines": []}, "valid"),
        ({"answer_disclosure": "no", "problem_lines": [], "lean_lines": []}, "invalid_evidence"),
        ({"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [True]}, "invalid_response"),
        ({"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [0]}, "invalid_response"),
    ],
)
def test_line_citations_preserve_unicode_and_distinguish_abstention_from_failure(label, status):
    row = {"problem": "Find all n ∈ \u2115.", "formal": "import Mathlib\ntheorem t : n = 12"}
    payload = {**label, "reason": "A short explanation"}
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(payload)}]}}]}
    reviewed = classifier.reviewed_response(response, row)
    assert reviewed.status == status
    assert (reviewed.classification is not None) == (status == "valid")
    if reviewed.classification:
        assert reviewed.classification.answer_disclosure == label["answer_disclosure"]


@pytest.mark.parametrize("response", [{}, {"candidates": [{"finishReason": "MAX_TOKENS"}]}])
def test_blocked_and_truncated_outputs_do_not_become_semantic_labels(response):
    reviewed = classifier.reviewed_response(response, {"problem": "Find n", "formal": "n = 12"})
    assert reviewed.status == "incomplete_response"
    assert reviewed.classification is None


@pytest.mark.parametrize("model", [classifier.PILOT_MODEL, classifier.CARBON_MODEL])
def test_classify_saves_actual_cited_lines_and_sends_only_numbered_inputs(tmp_path, monkeypatch, model):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        json.dumps(
            {
                "row": "r",
                "problem": "Find n",
                "formal": "import Mathlib\ntheorem t : n = 12",
                "human_answer_disclosure": "PRIVATE REVIEW",
            }
        )
        + "\n"
    )
    label = {"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [2], "reason": "Supplies n."}

    def respond(request, **_kwargs):
        body = json.loads(request.data)
        text = body["contents"][0]["parts"][0]["text"]
        payload = json.loads(text.removeprefix(classifier.PROMPT))
        assert payload["lean"][1] == {"line": 2, "text": "theorem t : n = 12"}
        assert "PRIVATE REVIEW" not in text
        if model == classifier.CARBON_MODEL:
            assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
        else:
            assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
        return io.StringIO(
            json.dumps(
                {
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(label)}]}}],
                }
            )
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(classifier, "urlopen", respond)
    args = argparse.Namespace(
        input=source,
        output=tmp_path / "results.jsonl",
        model=model,
        input_rate=1.0,
        output_rate=1.0,
        budget=5,
        execute=True,
        limit=0,
    )
    classifier.classify(args)
    saved = classifier.read_records(args.output)[1]
    assert saved["processing_status"] == "valid"
    assert saved["classification"] == label
    assert saved["evidence"] == {"problem": ["Find n"], "lean": ["theorem t : n = 12"]}


def test_long_explanation_does_not_discard_a_valid_decision():
    label = {"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [1], "reason": "Explanation. " * 30}
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(label)}]}}]}
    reviewed = classifier.reviewed_response(response, {"problem": "Find n", "formal": "n = 12"})
    assert reviewed.status == "valid"
    assert reviewed.classification.reason == label["reason"]
