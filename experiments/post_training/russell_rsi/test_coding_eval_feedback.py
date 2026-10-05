# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from experiments.post_training.russell_rsi import coding_eval_feedback as feedback_module
from experiments.post_training.russell_rsi.coding_analysis_recovery import (
    MERGE_RULE,
    PARTITION_ALGORITHM,
    PARTITION_PROTOCOL,
    PartitionedCodingAnalysisConfig,
    analyze_partitioned_coding_eval_failures,
    complete_failure_partitions,
    failure_key,
)
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
    coding_static_test_evidence,
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
                    "doc": json.dumps(
                        {
                            "task_id": index,
                            "canonical_solution": "secret-gold",
                            "test": "secret-test",
                            **({"entry_point": f"entry_{index}"} if suite == "humanevalplus" and index > 0 else {}),
                            **({"test_imports": ["import math"]} if suite == "mbppplus" and index > 0 else {}),
                        }
                    ),
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


def evidence_fixture(records, archives, panel):
    rows = coding_evidence_rows(records, archives, "parent", panel)
    return {
        "context_protocol": CODING_ANALYSIS_CONTEXT_PROTOCOL,
        "evaluation_context": coding_evaluation_context(records, rows),
        "static_test_evidence": coding_static_test_evidence(archives, rows),
        "rows": [asdict(row) for row in rows],
    }


def test_real_sample_shape_keeps_null_grading_and_excludes_gold_from_feedback():
    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    assert len(rows) == 64
    assert {suite: sum(row.pass_rate for row in rows if row.suite == suite) for suite in CODING_SUITES} == {
        "humanevalplus": 25,
        "mbppplus": 25,
    }
    evidence = evidence_fixture(records, archives, panel)
    request = coding_analysis_request(evidence)
    content = json.loads(request["messages"][1]["content"])
    failed = content["failures"]
    assert {row["suite"] for row in failed} == set(CODING_SUITES)
    serialized = json.dumps(request)
    assert "secret-gold" not in serialized
    assert content["context_protocol"] == CODING_ANALYSIS_CONTEXT_PROTOCOL
    assert content["failures"] == [asdict(row) for row in rows if row.pass_rate == 0]
    for row, item in zip(content["failures"], content["static_test_evidence"]["items"], strict=True):
        assert (item["suite"], item["benchmark_id"], item["source_sha256"]) == (
            row["suite"],
            row["benchmark_id"],
            row["source_sha256"],
        )
        assert item["status"] == "available"
        assert item["test_source"] == "secret-test"
        assert item["test_source_sha256"] == hashlib.sha256(b"secret-test").hexdigest()
    humaneval_metadata = next(
        item["harness_metadata"]
        for item in content["static_test_evidence"]["items"]
        if item["suite"] == "humanevalplus" and item["benchmark_id"] == "1"
    )
    assert humaneval_metadata == {
        "availability": {"entry_point": "available", "test_imports": "unavailable"},
        "entry_point": "entry_1",
    }
    mbpp_metadata = next(
        item["harness_metadata"]
        for item in content["static_test_evidence"]["items"]
        if item["suite"] == "mbppplus" and item["benchmark_id"] == "1"
    )
    assert mbpp_metadata == {
        "availability": {"entry_point": "unavailable", "test_imports": "available"},
        "test_imports": ["import math"],
    }
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
    evidence = evidence_fixture(records, archives, panel)
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


@pytest.mark.parametrize("test_value", ["absent", "null"])
def test_static_test_evidence_marks_missing_original_tests_unavailable(test_value):
    records, archives, panel = panel_fixture()
    doc = {"task_id": 0, "canonical_solution": "secret-gold"}
    if test_value == "null":
        doc["test"] = None
    archives[0][0]["doc"] = json.dumps(doc)
    evidence = evidence_fixture(records, archives, panel)
    failed = next(row for row in evidence["rows"] if row["suite"] == "humanevalplus" and row["benchmark_id"] == "0")
    item = next(
        item
        for item in evidence["static_test_evidence"]["items"]
        if item["suite"] == failed["suite"] and item["benchmark_id"] == failed["benchmark_id"]
    )
    assert item == {
        "suite": failed["suite"],
        "benchmark_id": failed["benchmark_id"],
        "source_sha256": failed["source_sha256"],
        "status": "unavailable",
        "reason": "not_in_original_archive",
        "harness_metadata": {"availability": {"entry_point": "unavailable", "test_imports": "unavailable"}},
    }
    request_content = json.loads(coding_analysis_request(evidence)["messages"][1]["content"])
    assert len(request_content["static_test_evidence"]["items"]) == len(request_content["failures"])
    assert item["harness_metadata"]["availability"] == {
        "entry_point": "unavailable",
        "test_imports": "unavailable",
    }


def test_static_test_evidence_rejects_missing_or_changed_archive_pairings():
    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    del archives[0][0]
    with pytest.raises(ValueError, match="missing from its original archive"):
        coding_static_test_evidence(archives, rows)

    records, archives, panel = panel_fixture()
    rows = coding_evidence_rows(records, archives, "parent", panel)
    archives[0][0]["doc"] = json.dumps({"task_id": 0, "test": "changed original source"})
    with pytest.raises(ValueError, match="source hash differs"):
        coding_static_test_evidence(archives, rows)


def test_oversized_archived_test_stops_before_api_issuance(tmp_path, monkeypatch):
    records, archives, panel = panel_fixture()
    evidence = evidence_fixture(records, archives, panel)
    baseline_bytes = len(coding_analysis_request(evidence)["messages"][1]["content"].encode())
    evidence["static_test_evidence"]["items"][0]["test_source"] = "x" * 1024
    evidence_dir = tmp_path / "evidence"
    output_dir = tmp_path / "analysis"
    evidence_dir.mkdir()
    output_dir.mkdir()
    (evidence_dir / "coding-evidence.json").write_text(json.dumps(evidence))

    class NoCallClient:
        def __init__(self, **_kwargs):
            pytest.fail("oversized evidence must stop before API construction")

    monkeypatch.setattr(feedback_module, "AsyncOpenAI", NoCallClient)
    config = CodingAnalysisConfig(
        str(evidence_dir), "evidence-v1", "relay", str(output_dir), maximum_evidence_bytes=baseline_bytes + 500
    )
    with pytest.raises(ValueError, match="Complete coding evidence exceeds"):
        asyncio.run(analyze_coding_failures(config))
    assert not (output_dir / "private-analysis-issued.json").exists()


def partitioned_analysis_fixture(tmp_path):
    records, archives, panel = panel_fixture()
    evidence = evidence_fixture(records, archives, panel)
    evidence["static_test_evidence"]["items"][0]["test_source"] = "x" * 600_000
    source = tmp_path / "evidence"
    source.mkdir()
    evidence_bytes = json.dumps(evidence).encode()
    (source / "coding-evidence.json").write_bytes(evidence_bytes)
    analysis = CodingAnalysisConfig(
        str(source), "complete-evidence-v1", "relay", str(tmp_path / "analysis"), maximum_evidence_bytes=1_048_576
    )
    full_request = coding_analysis_request(evidence, maximum_evidence_bytes=1_048_576)
    partitions = complete_failure_partitions(evidence, full_request)
    manifest = {
        "protocol": PARTITION_PROTOCOL,
        "algorithm": PARTITION_ALGORITHM,
        "merge_rule": MERGE_RULE,
        "evidence_identity": analysis.evidence_identity,
        "original_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "original_request_sha256": compact_json_sha256(full_request),
        "maximum_evidence_bytes": 1_048_576,
        "maximum_failed_rows": 64,
        "partitions": [],
    }
    requests = []
    for part, partition in enumerate(partitions, 1):
        request = coding_analysis_request(partition, maximum_evidence_bytes=1_048_576)
        requests.append(request)
        partition_sha = compact_json_sha256(partition)
        preflight = {
            "verified": True,
            "request_sha256": compact_json_sha256(request),
            "partition_evidence_sha256": partition_sha,
            "evidence_identity": analysis.evidence_identity,
            "evidence_sha256": manifest["original_evidence_sha256"],
            "served_model": request["model"],
            "prompt_tokens": 1000,
            "max_output_tokens": 2048,
            "context_limit": 4096,
            "tokenizer_evidence": {
                "method": "served_vllm_tokenize_and_chat_render",
                "token_ids_sha256": "b" * 64,
                "direct_server_evidence": {
                    "render_error": None,
                    "render_request_sha256": "a" * 64,
                    "render_response_sha256": "b" * 64,
                    "model_max_model_len": 4096,
                    "tokenizer_max_model_len": 4096,
                    "token_vectors": [{"equals_tokenize": True, "count": 1000, "sha256": "b" * 64}],
                },
            },
        }
        path = tmp_path / f"preflight-{part}.json"
        path.write_text(json.dumps(preflight))
        manifest["partitions"].append(
            {
                "part": part,
                "failure_keys": [failure_key(row) for row in partition["rows"]],
                "partition_evidence_sha256": partition_sha,
                "request_sha256": compact_json_sha256(request),
                "preflight_uri": str(path),
                "preflight_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = PartitionedCodingAnalysisConfig(analysis, str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    return evidence, config, requests, manifest


def partition_response_fixture(part):
    labels = (
        [("types", 0.8), ("boundaries", 0.93), ("state", 0.91), ("ordering", 0.8)]
        if part == 1
        else [("types", 0.96), ("parsing", 0.9), ("error_handling", 0.9), ("change_scope", 0.69)]
    )
    content = {
        "skills": [
            {"skill": label, "confidence": confidence, "evidence": f"private part-{part} citation"}
            for label, confidence in labels
        ]
    }
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def test_partitioned_analysis_keeps_complete_pairs_and_merges_four_raw_labels(tmp_path, monkeypatch):
    evidence, config, requests, _manifest = partitioned_analysis_fixture(tmp_path)
    full = json.loads(coding_analysis_request(evidence, maximum_evidence_bytes=1_048_576)["messages"][1]["content"])
    first, second = [json.loads(request["messages"][1]["content"]) for request in requests]
    assert first["failures"] == full["failures"][:1]
    assert second["failures"] == full["failures"][1:]
    assert (
        first["static_test_evidence"]["items"] + second["static_test_evidence"]["items"]
        == full["static_test_evidence"]["items"]
    )
    assert first["evaluation_context"] == second["evaluation_context"] == evidence["evaluation_context"]
    failed_dir = tmp_path / "original-analysis"
    failed_dir.mkdir()
    with pytest.raises(ValueError, match="Complete coding evidence exceeds"):
        asyncio.run(
            analyze_coding_failures(
                CodingAnalysisConfig(config.analysis.evidence_path, "original", "relay", str(failed_dir))
            )
        )
    requests_sent = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def create(self, **request):
            requests_sent.append(request)
            response = partition_response_fixture(len(requests_sent))
            return SimpleNamespace(model_dump=lambda **_kwargs: response)

    monkeypatch.setattr(feedback_module, "AsyncOpenAI", Client)
    monkeypatch.setattr(feedback_module, "resolve_glm_base_url", lambda _job: "https://relay.test")
    monkeypatch.setenv(feedback_module.GLM_TOKEN_ENV, "test-token")
    analyze_partitioned_coding_eval_failures(config)
    analyze_partitioned_coding_eval_failures(config)
    assert requests_sent == requests
    capabilities = json.loads((tmp_path / "analysis/capabilities.json").read_text())
    assert capabilities == {
        "skills": [
            {"label": label, "description": feedback_module.SKILL_DESCRIPTIONS[label]}
            for label in ("boundaries", "error_handling", "state", "types")
        ]
    }
    private = json.loads((tmp_path / "analysis/private-merged-analysis.json").read_text())
    assert private["review_status"] == "raw-unreviewed"
    assert len(private["entries"]) == 8
    assert private["ranked_selected_labels"] == ["types", "boundaries", "state", "error_handling"]
    assert {entry["disposition"] for entry in private["entries"]} == {
        "selected",
        "duplicate-label",
        "below-confidence-threshold",
        "four-label-limit",
    }
    assert not (failed_dir / "private-analysis-issued.json").exists()


@pytest.mark.parametrize(
    "mismatch", ["request_sha256", "evidence_sha256", "context_limit", "file_pin", "failure_keys", "token_vector"]
)
def test_second_partition_mismatch_stops_before_either_request(tmp_path, monkeypatch, mismatch):
    _evidence, config, _requests, manifest = partitioned_analysis_fixture(tmp_path)
    second = manifest["partitions"][1]
    path = tmp_path / "preflight-2.json"
    preflight = json.loads(path.read_text())
    if mismatch == "failure_keys":
        second["failure_keys"] = []
    elif mismatch == "token_vector":
        preflight["tokenizer_evidence"]["direct_server_evidence"]["token_vectors"][0]["equals_tokenize"] = False
    elif mismatch == "context_limit":
        preflight[mismatch] = preflight["max_output_tokens"]
    elif mismatch != "file_pin":
        preflight[mismatch] = "changed"
    path.write_text(json.dumps(preflight))
    if mismatch != "file_pin":
        second["preflight_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        path.write_text(path.read_text() + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = PartitionedCodingAnalysisConfig(
        config.analysis, str(manifest_path), hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )

    class NoCallClient:
        def __init__(self, **_kwargs):
            pytest.fail("Both partitions must pass preflight before either request")

    monkeypatch.setattr(feedback_module, "AsyncOpenAI", NoCallClient)
    with pytest.raises(ValueError):
        analyze_partitioned_coding_eval_failures(config)
    assert not (tmp_path / "analysis/capabilities.json").exists()
    assert not list((tmp_path / "analysis").rglob("private-analysis-issued.json"))


@pytest.mark.parametrize("failed_part", [1, 2])
def test_ambiguous_partition_never_reissues_or_merges_partial_feedback(tmp_path, monkeypatch, failed_part):
    _evidence, config, requests, _manifest = partitioned_analysis_fixture(tmp_path)
    calls = []

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def create(self, **request):
            calls.append(request)
            if len(calls) == failed_part:
                raise ConnectionError("transport closed after request send")
            response = partition_response_fixture(len(calls))
            return SimpleNamespace(model_dump=lambda **_kwargs: response)

    monkeypatch.setattr(feedback_module, "AsyncOpenAI", Client)
    monkeypatch.setattr(feedback_module, "resolve_glm_base_url", lambda _job: "https://relay.test")
    monkeypatch.setenv(feedback_module.GLM_TOKEN_ENV, "test-token")
    with pytest.raises(ConnectionError):
        analyze_partitioned_coding_eval_failures(config)
    with pytest.raises(ValueError, match="outcome is ambiguous"):
        analyze_partitioned_coding_eval_failures(config)
    assert calls == requests[:failed_part]
    assert not (tmp_path / "analysis/capabilities.json").exists()
    assert not (tmp_path / "analysis/private-merged-analysis.json").exists()
