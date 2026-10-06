# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adopt real temporary archives without a model or relay connection."""

import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from finestore.eval import EvaluationStore, sample_from_archive_row
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvalRunRecord, record_path
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.experiment.cli import graph_handles

from experiments.post_training.russell_rsi import completed_coding_analysis as analysis
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_worker import WORKER_SETTINGS, submit_regional_coding_analysis
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingEvidenceConfig,
    analyze_coding_failures,
    coding_analysis_request,
    collect_coding_eval_evidence,
    protocol_digest,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.test_coding_eval_feedback import panel_fixture


@pytest.fixture(params=[False])
def completed_evidence(tmp_path, request):
    def pin(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    records, archives, panel = panel_fixture()
    if request.param:
        document = json.loads(archives[0][0]["doc"])
        document["test"] = "#" * 600000
        archives[0][0]["doc"] = json.dumps(document)
    prefix = str(tmp_path / "artifacts")
    records_prefix = str(tmp_path / "records")
    paths = []
    protocols = {}
    for index, (record, rows) in enumerate(zip(records, archives, strict=True)):
        path = str(tmp_path / f"archive-{index}")
        paths.append(path)
        with EvaluationStore.open(path, writer_id="synthetic") as store:
            for row in rows:
                store.add_sample(sample_from_archive_row(row), extraction_filter="none")
            store.seal()
        value = EvalRunRecord.model_validate(
            {
                **record,
                "run_id": str(index),
                "group_id": "synthetic",
                "created_at": "2026-10-06T00:00:00Z",
                "user": "synthetic",
                "model": {
                    "name": "parent",
                    "location": "/never-read",
                    "backend": "vllm",
                    "config": asdict(
                        ModelConfig(
                            "parent",
                            "/never-read",
                            identity="parent",
                            tokenizer="tokenizer",
                            tokenizer_revision="revision",
                        )
                    ),
                },
                "eval": {
                    "name": record["eval"]["name"],
                    "mechanism": "evalchemy",
                    "source_digest": "sha256:" + "0" * 64,
                    "evalchemy": {
                        "apply_chat_template": True,
                        "max_gen_toks": 8192,
                        "max_eval_instances": 32,
                        "num_concurrent": 16,
                        "batch_size": None,
                        "seed": 1234,
                    },
                },
                "hardware": {"platform": "coreweave", "accelerator": "H100x8", "region_or_cluster": "cw-us-east-02a"},
                "results_path": path,
                "jobs": {},
                "log_tails": {},
                "provenance": {**record["provenance"], "git_sha": "synthetic", "launch_host": "synthetic"},
            }
        ).model_dump(mode="json", by_alias=True)
        pin(record_path(records_prefix, str(index)), value)
        protocols[record["eval"]["name"]] = protocol_digest(value)
    cfg = CodingEvidenceConfig(
        records_prefix, ("0", "1"), tuple(paths), "parent", replace(panel, protocols=protocols), ""
    )
    step = ArtifactStep(
        name="documents/synthetic-completed-coding",
        version=analysis.EXTRACTION_VERSION,
        artifact_type=Artifact,
        deps=(),
        build_config=lambda ctx: replace(cfg, output_path=ctx.output_path),
        run=collect_coding_eval_evidence,
    )
    path = Path(step.path(prefix))
    path.mkdir(parents=True)
    bound = step.build_config(StepContext.for_run(str(path), prefix))
    collect_coding_eval_evidence(bound)
    record = {
        "name": step.name,
        "version": step.version,
        "fingerprint": step.fingerprint(),
        "output_path": str(path),
        "config": asdict(bound),
        "provenance": {"base_commit": analysis.EVIDENCE_SOURCE_HEAD[:9], "dirty": False},
    }
    producer_pin = pin(path / ".artifact.json", record)
    (path / ".executor_status").write_text("SUCCESS")
    source_pin = pin(tmp_path / "source.json", {"recovery_artifact_prefix": prefix})
    extraction_pin = pin(
        tmp_path / "extraction.json",
        {
            "protocol": analysis.EXTRACTION_PROTOCOL,
            "version": analysis.EXTRACTION_VERSION,
            "source_config": source_pin,
        },
    )
    evidence = json.loads((path / "coding-evidence.json").read_text())
    config = {
        "protocol": analysis.PROTOCOL,
        "version": analysis.VERSION,
        "extraction_config": extraction_pin,
        "evidence_producer": producer_pin,
        "evidence": pin(path / "coding-evidence.json", evidence),
        "relay_job": analysis.RELAY_JOB,
    }
    original_pin = pin(tmp_path / "original-input.json", config)
    request_body = coding_analysis_request(evidence, 64, 1048576)
    decision = {
        "protocol": analysis.DECISION_PROTOCOL,
        "original_input": original_pin,
        "evidence": config["evidence"],
        "evidence_identity": artifact_identity(step),
        "request_sha256": compact_json_sha256(request_body),
        "calibration_status": "incomplete_infrastructure",
        "signal_gate_passed": None,
        "rl_authorized": False,
        "analysis": {
            "maximum_failed_rows": 64,
            "maximum_evidence_bytes": 589824,
            "maximum_output_tokens": 2048,
            "reasoning_effort": "low",
            "model_retries": 0,
            "maximum_generation_requests": 1,
            "context_limit": 262144,
            "failed_rows": sum(row["pass_rate"] == 0 for row in evidence["rows"]),
            "evidence_rows": len(evidence["rows"]),
        },
        "worker": WORKER_SETTINGS,
    }
    config["decision"] = pin(tmp_path / "decision.json", decision)
    return config, step, pin, prefix


def test_analysis_adopts_only_completed_evidence_and_keeps_canonical_identity(completed_evidence, tmp_path):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    handles = graph_handles([stages["terminal"]])
    assert sum(handle.run is submit_regional_coding_analysis for handle in handles) == 1
    assert all(handle.run is not collect_coding_eval_evidence for handle in handles)
    assert len(handles) == 2
    bound = stages["terminal"].build_config(
        StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=stages["terminal"].deps)
    )
    assert bound.analysis.evidence_identity == artifact_identity(step)
    assert bound.analysis.evidence_identity != artifact_identity(stages["evidence"])


@pytest.mark.parametrize("defect", ["incomplete", "producer", "evidence"])
def test_analysis_refuses_incomplete_or_changed_evidence(completed_evidence, tmp_path, defect):
    config, step, pin, prefix = completed_evidence
    if defect == "incomplete":
        (Path(step.path(prefix)) / ".executor_status").write_text("FAILED")
    elif defect == "producer":
        target = config["evidence_producer"]
        record = PinnedFile(**target).read_json()
        record["config"]["model_identity"] = "another-model"
        target.update(pin(target["uri"], record))
    else:
        target = config["evidence"]
        evidence = PinnedFile(**target).read_json()
        evidence["rows"][0]["output"] = "substituted response"
        target.update(pin(target["uri"], evidence))
    with pytest.raises(ValueError):
        analysis.completed_evidence_analysis_stages(config, PinnedFile(**pin(tmp_path / "input.json", config)), step)


def test_existing_analyst_refuses_issued_request_and_replays_saved_response(completed_evidence, tmp_path):
    config, step, pin, prefix = completed_evidence
    stages = analysis.completed_evidence_analysis_stages(
        config, PinnedFile(**pin(tmp_path / "input.json", config)), step
    )
    bound = stages["terminal"].build_config(
        StepContext.for_run(str(tmp_path / "analysis"), prefix, deps=stages["terminal"].deps)
    )
    evidence = PinnedFile(**config["evidence"]).read_json()
    request = coding_analysis_request(
        evidence, bound.analysis.maximum_failed_rows, bound.analysis.maximum_evidence_bytes
    )
    binding = {"request_sha256": compact_json_sha256(request), "evidence_identity": artifact_identity(step)}
    issued = pin(tmp_path / "analysis/private-analysis-issued.json", binding)
    with pytest.raises(ValueError, match="ambiguous"):
        asyncio.run(analyze_coding_failures(bound.analysis))
    assert hashlib.sha256(Path(issued["uri"]).read_bytes()).hexdigest() == issued["sha256"]
    pin(
        tmp_path / "analysis/private-analysis.json",
        {
            **binding,
            "response": {"choices": [{"message": {"content": '{"skills":[]}'}}]},
        },
    )
    asyncio.run(analyze_coding_failures(bound.analysis))
    assert (tmp_path / "analysis/capabilities.json").exists()


@pytest.mark.parametrize("completed_evidence", [True], indirect=True)
def test_analysis_budget_hold_precedes_any_analyst_request(completed_evidence, tmp_path):
    config, step, pin, _ = completed_evidence
    with pytest.raises(ValueError, match="explicit analyst budget"):
        analysis.completed_evidence_analysis_stages(config, PinnedFile(**pin(tmp_path / "input.json", config)), step)
    assert not list(tmp_path.rglob("private-analysis-issued.json"))
