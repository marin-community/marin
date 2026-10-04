# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from experiments.post_training.russell_rsi import coding_eval_feedback as feedback_module
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CODING_ANALYSIS_CONTEXT_PROTOCOL,
    CODING_SUITES,
    CodingAnalysisConfig,
    CodingPanel,
    PanelItem,
    analyze_coding_failures,
    coding_analysis_request,
    coding_evaluation_context,
    coding_evidence_rows,
    protocol_digest,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256


def panel_fixture():
    records, archives, items = [], [], []
    for suite in CODING_SUITES:
        record = {
            "eval": {"name": suite, "source_digest": "dataset-pin", "tasks": [], "evalchemy": {"shots": 3}},
            "provenance": {"eval_runtime": "evaluator-pin"},
            "coverage": {
                suite: {
                    "n_benchmark": 32,
                    "n_attempted": 32,
                    "n_scored": 32,
                    "n_unanswered": 0,
                    "errors": {},
                }
            },
            "metrics": {suite: {"pass@1": 25 / 32, "scored_count": 32.0}},
            "model": {
                "config": {
                    "identity": "parent",
                    "tokenizer": "tokenizer",
                    "tokenizer_revision": "revision",
                    "serve": {},
                    "generation": {},
                }
            },
            "status": "succeeded",
            "error": None,
        }
        records.append(record)
        rows = []
        for index in range(32):
            prompt = f"{suite} prompt {index}"
            items.append(PanelItem(suite, str(index), hashlib.sha256(prompt.encode()).hexdigest()))
            rows.append(
                {
                    "task": suite,
                    "doc_id": str(index + 100),
                    "kind": "generation",
                    "prompt_text": prompt,
                    "output": f"actual answer {index}",
                    "extracted": None,
                    "doc": json.dumps({"task_id": index, "canonical_solution": "secret-gold", "test": "secret-test"}),
                    "metrics": [("pass_rate", 0.0 if index < 7 else 1.0)],
                    "grading": None,
                    "correct": None,
                    "filter": "none",
                    "trial_id": "",
                }
            )
        archives.append(rows)
    panel = CodingPanel(tuple(items), {record["eval"]["name"]: protocol_digest(record) for record in records})
    return tuple(records), tuple(archives), panel


def test_real_sample_shape_keeps_null_grading_and_excludes_gold_from_feedback():
    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    assert len(rows) == 64
    assert {suite: sum(row.pass_rate for row in rows if row.suite == suite) for suite in CODING_SUITES} == {
        "humanevalplus": 25,
        "mbppplus": 25,
    }
    context = coding_evaluation_context(records, rows)
    evidence = {
        "context_protocol": CODING_ANALYSIS_CONTEXT_PROTOCOL,
        "evaluation_context": context,
        "rows": [asdict(row) for row in rows],
    }
    request = coding_analysis_request(evidence)
    content = json.loads(request["messages"][1]["content"])
    failed = content["failures"]
    assert {row["suite"] for row in failed} == set(CODING_SUITES)
    serialized = json.dumps(request)
    assert "secret-gold" not in serialized and "secret-test" not in serialized
    assert content["context_protocol"] == CODING_ANALYSIS_CONTEXT_PROTOCOL
    assert content["failures"] == [asdict(row) for row in rows if row.pass_rate == 0]
    for suite in CODING_SUITES:
        suite_context = content["evaluation_context"]["suites"][suite]
        assert suite_context["run_status"] == "succeeded"
        assert suite_context["run_error"] is None
        assert suite_context["coverage_errors"] == {}
        assert suite_context["coverage"] == {
            "n_benchmark": 32,
            "n_attempted": 32,
            "n_scored": 32,
            "n_unanswered": 0,
        }
        assert suite_context["row_outcomes"] == {
            "passed": 25,
            "failed": 7,
            "null_grader_detail": {"passed": 25, "failed": 7},
        }


@pytest.mark.parametrize(
    ("record_field", "value_field", "value", "error"),
    [
        ("coverage", "n_scored", 31, "coverage differs"),
        ("metrics", "pass@1", 0.5, "score differs"),
    ],
)
def test_record_mismatch_cannot_become_evaluation_context(record_field, value_field, value, error):
    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    records[0][record_field]["humanevalplus"][value_field] = value
    with pytest.raises(ValueError, match=error):
        coding_evaluation_context(records, rows)


def test_ambiguous_provider_failure_never_issues_coding_request_again(tmp_path, monkeypatch):
    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    evidence = {
        "context_protocol": CODING_ANALYSIS_CONTEXT_PROTOCOL,
        "evaluation_context": coding_evaluation_context(records, rows),
        "rows": [asdict(row) for row in rows],
    }
    evidence_dir = tmp_path / "evidence"
    output_dir = tmp_path / "analysis"
    evidence_dir.mkdir()
    output_dir.mkdir()
    (evidence_dir / "coding-evidence.json").write_text(json.dumps(evidence))
    config = CodingAnalysisConfig(str(evidence_dir), "evidence-v1", "relay", str(output_dir))
    issued_path = output_dir / "private-analysis-issued.json"
    calls = []
    client_options = []

    class FailedClient:
        def __init__(self, **kwargs):
            client_options.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def create(self, **request):
            assert issued_path.exists()
            calls.append(request)
            raise ConnectionError("transport closed after request send")

    monkeypatch.setattr(feedback_module, "AsyncOpenAI", FailedClient)
    monkeypatch.setattr(feedback_module, "resolve_glm_base_url", lambda _job: "https://relay.test")
    monkeypatch.setenv(feedback_module.GLM_TOKEN_ENV, "test-token")
    expected_request = coding_analysis_request(evidence)

    with pytest.raises(ConnectionError, match="after request send"):
        asyncio.run(analyze_coding_failures(config))

    issued_bytes = issued_path.read_bytes()
    assert json.loads(issued_bytes) == {
        "request_sha256": compact_json_sha256(expected_request),
        "evidence_identity": config.evidence_identity,
    }
    with pytest.raises(ValueError, match="outcome is ambiguous"):
        asyncio.run(analyze_coding_failures(config))

    assert calls == [expected_request]
    assert len(client_options) == 1
    assert client_options[0]["max_retries"] == 0
    assert issued_path.read_bytes() == issued_bytes
    assert not (output_dir / "private-analysis.json").exists()
    assert not (output_dir / "capabilities.json").exists()


@pytest.mark.parametrize("change", ["prompt", "task", "missing", "duplicate", "model", "protocol", "metric"])
def test_panel_drift_and_incomplete_measurement_cannot_enter_feedback(change):
    records, archives, panel = panel_fixture()
    if change == "prompt":
        archives[0][0]["prompt_text"] += " changed"
    elif change == "task":
        archives[0][0]["doc"] = json.dumps({"task_id": "not-in-panel"})
    elif change == "missing":
        archives[0].pop()
    elif change == "duplicate":
        archives[0].append(archives[0][0])
    elif change == "model":
        records[0]["model"]["config"]["identity"] = "different-policy"
    elif change == "protocol":
        records[0]["eval"]["evalchemy"]["shots"] = 0
    else:
        archives[0][0]["metrics"] = []
    with pytest.raises(ValueError):
        coding_evidence_rows(records, archives, "parent", panel)
