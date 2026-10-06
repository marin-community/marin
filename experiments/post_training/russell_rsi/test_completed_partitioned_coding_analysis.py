# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles

from experiments.post_training.russell_rsi import (
    coding_eval_feedback,
    completed_coding_analysis,
    test_completed_coding_analysis,
)
from experiments.post_training.russell_rsi import completed_partitioned_coding_analysis as study
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_recovery import (
    MERGE_RULE,
    PARTITION_ALGORITHM,
    PARTITION_PROTOCOL,
    complete_failure_partitions,
    failure_key,
)
from experiments.post_training.russell_rsi.coding_analysis_worker import (
    run_regional_partitioned_coding_analysis,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import coding_analysis_request
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.test_coding_eval_feedback import partition_response_fixture

completed_evidence = test_completed_coding_analysis.completed_evidence


@pytest.fixture
def partitioned_completed(completed_evidence, tmp_path):
    previous, expected, pin, prefix = completed_evidence
    previous_pin = pin(tmp_path / "previous.json", previous)
    evidence = PinnedFile(**previous["evidence"]).read_json()
    identity = PinnedFile(**previous["decision"]).read_json()["evidence_identity"]
    request = coding_analysis_request(evidence, 64, 589824)
    manifest = {
        "protocol": PARTITION_PROTOCOL,
        "algorithm": PARTITION_ALGORITHM,
        "merge_rule": MERGE_RULE,
        "evidence_identity": identity,
        "original_evidence_sha256": previous["evidence"]["sha256"],
        "original_request_sha256": compact_json_sha256(request),
        "maximum_evidence_bytes": 589824,
        "maximum_failed_rows": 64,
        "partitions": [],
    }
    for part, payload in enumerate(complete_failure_partitions(evidence, request), 1):
        body = coding_analysis_request(payload, 64, 589824)
        preflight = {
            "verified": True,
            "request_sha256": compact_json_sha256(body),
            "partition_evidence_sha256": compact_json_sha256(payload),
            "evidence_identity": identity,
            "evidence_sha256": previous["evidence"]["sha256"],
            "served_model": body["model"],
            "prompt_tokens": 100,
            "max_output_tokens": 2048,
            "context_limit": 262144,
            "tokenizer_evidence": {
                "method": "served_vllm_tokenize_and_chat_render",
                "token_ids_sha256": "b" * 64,
                "direct_server_evidence": {
                    "render_error": None,
                    "render_request_sha256": "a" * 64,
                    "render_response_sha256": "b" * 64,
                    "model_max_model_len": 262144,
                    "tokenizer_max_model_len": 262144,
                    "token_vectors": [{"equals_tokenize": True, "count": 100, "sha256": "b" * 64}],
                },
            },
        }
        flight_pin = pin(tmp_path / f"preflight-{part}.json", preflight)
        manifest["partitions"].append(
            {
                "part": part,
                "failure_keys": [failure_key(row) for row in payload["rows"]],
                "partition_evidence_sha256": compact_json_sha256(payload),
                "request_sha256": compact_json_sha256(body),
                "preflight_uri": flight_pin["uri"],
                "preflight_sha256": flight_pin["sha256"],
            }
        )
    failure = {
        "original_config": previous_pin,
        "evidence_sha256": previous["evidence"]["sha256"],
        "request_sha256": compact_json_sha256(request),
        "status": "failed",
        "generation_requests": 0,
        "issuance_markers": 0,
        "response_records": 0,
    }
    predecessor = completed_coding_analysis.completed_evidence_analysis_stages(
        previous, PinnedFile(**previous_pin), expected
    )["terminal"]
    predecessor_path = predecessor.path(prefix)
    artifacts = {}
    for name in (
        ".executor_info",
        ".executor_status",
        "analysis-worker-provenance.json",
        "context-preflight/models.json",
        "worker-submission/reservation.json",
    ):
        path = Path(predecessor_path) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b"FAILED" if name == ".executor_status" else b"{}"
        path.write_bytes(raw)
        artifacts[name] = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    failure.update(
        {
            "protocol": "russell-rsi-coding-analysis-preflight-failure-v1",
            "error": {"type": "KeyError", "field": "max_model_len"},
            "failed_before": "private-analysis-issued.json",
            "source_head": "5489f732a9cd28e46543040db68f17588ea79603",
            "artifacts": artifacts,
            "terminal": pin(
                tmp_path / "terminal.json",
                {
                    "source_head": "5489f732a9cd28e46543040db68f17588ea79603",
                    "job": "/synthetic",
                    "state": "failed",
                    "failure_count": 1,
                    "preemption_count": 0,
                    "task_count": 1,
                    "completed_count": 0,
                    "exit_code": 0,
                    "tasks": [{"id": "/synthetic/0", "state": "failed", "exit_code": 1}],
                },
            ),
            "inventory": pin(
                tmp_path / "inventory.json",
                {
                    "prefix": predecessor_path,
                    "complete_recursive_listing": True,
                    "files": list(artifacts),
                    "artifacts": artifacts,
                    "issuance_markers": 0,
                    "response_records": 0,
                },
            ),
            "foreground_exit": pin(tmp_path / "exit.json", {"exit_code": 1}),
        }
    )
    oversize = {
        "verified": False,
        "analysis_requests_issued": 0,
        "evidence_identity": identity,
        "evidence_sha256": previous["evidence"]["sha256"],
        "request_sha256": compact_json_sha256(request),
        "served_model": request["model"],
        "max_output_tokens": 2048,
        "context_limit": 262144,
        "prompt_tokens": 309803,
        "tokenizer_evidence": {
            "method": "served_vllm_tokenize_and_chat_render",
            "direct_server_evidence": {"render_error": {"status": 400}},
        },
    }
    config = {
        "protocol": study.PROTOCOL,
        "version": study.VERSION,
        "maximum_analysis_calls": 2,
        "original_config": previous_pin,
        "failure": pin(tmp_path / "failure.json", failure),
        "oversize": pin(tmp_path / "oversize.json", oversize),
        "partition_manifest": pin(tmp_path / "manifest.json", manifest),
    }
    return config, expected, pin, prefix


def test_partition_graph_and_regional_worker_issue_only_two_then_resume_without_http(
    partitioned_completed, tmp_path, monkeypatch
):
    config, evidence_step, pin, prefix = partitioned_completed
    original_stages = completed_coding_analysis.completed_evidence_analysis_stages
    monkeypatch.setattr(
        study, "completed_coding_analysis_stages", lambda cfg, cfg_pin, _: original_stages(cfg, cfg_pin, evidence_step)
    )
    stages = study.completed_partitioned_analysis_stages(config, PinnedFile(**pin(tmp_path / "input.json", config)), {})
    terminal = stages["terminal"]
    assert {step.name for step in graph_handles([terminal])} == {terminal.name, stages["evidence"].name}
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "out"), prefix, deps=terminal.deps))
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def create(self, **body):
            calls.append(body)
            assert body["max_tokens"] == 2048
            assert body["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"
            return SimpleNamespace(model_dump=lambda **_: partition_response_fixture(len(calls)))

    monkeypatch.setattr(coding_eval_feedback, "AsyncOpenAI", Client)
    monkeypatch.setattr(coding_eval_feedback, "resolve_glm_base_url", lambda _: "https://synthetic.invalid/v1")
    monkeypatch.setenv(coding_eval_feedback.GLM_TOKEN_ENV, "synthetic")
    run_regional_partitioned_coding_analysis(bound)
    run_regional_partitioned_coding_analysis(bound)
    assert len(calls) == 2
    assert len(json.loads((tmp_path / "out/capabilities.json").read_text())["skills"]) == 4


@pytest.mark.parametrize("change", ["issued", "extra-file"])
def test_prior_issuance_or_incomplete_inventory_blocks_new_partition_graph(
    partitioned_completed, tmp_path, monkeypatch, change
):
    config, evidence_step, pin, _prefix = partitioned_completed
    failure = PinnedFile(**config["failure"]).read_json()
    inventory = PinnedFile(**failure["inventory"]).read_json()
    if change == "issued":
        inventory["issuance_markers"] = 1
    else:
        inventory["files"].append("private-analysis-issued.json")
    failure["inventory"] = pin(tmp_path / "changed-inventory.json", inventory)
    config["failure"] = pin(tmp_path / "changed-failure.json", failure)
    monkeypatch.setattr(
        study,
        "completed_coding_analysis_stages",
        lambda cfg, cfg_pin, _: completed_coding_analysis.completed_evidence_analysis_stages(
            cfg, cfg_pin, evidence_step
        ),
    )
    with pytest.raises(ValueError, match="inventory contradicts no issuance"):
        study.completed_partitioned_analysis_stages(
            config, PinnedFile(**pin(tmp_path / "refused-input.json", config)), {}
        )
