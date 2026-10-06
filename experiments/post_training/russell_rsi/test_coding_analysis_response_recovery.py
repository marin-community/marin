# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from pydantic import ValidationError

from experiments.post_training.russell_rsi import coding_analysis_response_recovery as recovery
from experiments.post_training.russell_rsi import (
    coding_eval_feedback,
    completed_coding_analysis,
    completed_partitioned_coding_analysis,
    test_completed_partitioned_coding_analysis,
)
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_recovery import (
    MERGE_RULE,
    partition_analysis_requests,
    partition_evidence_identity,
)
from experiments.post_training.russell_rsi.coding_analysis_response_recovery import (
    ResponseRecoveryConfig,
    run_response_recovery,
)
from experiments.post_training.russell_rsi.coding_analysis_worker import (
    WORKER_SETTINGS,
    RegionalPartitionedCodingAnalysisConfig,
    worker_source_files,
)
from experiments.post_training.russell_rsi.feedback import FeedbackAnalysis
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.test_coding_eval_feedback import partitioned_analysis_fixture


@pytest.fixture
def saved_long_partition(tmp_path):
    _evidence, partitioned, _requests, _manifest = partitioned_analysis_fixture(tmp_path)
    first, second = partition_analysis_requests(partitioned)
    source = tmp_path / "failed" / "part-1"
    source.mkdir(parents=True)
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "skills": [
                                {"skill": "change_scope", "confidence": 0.6, "evidence": "x" * 1163},
                            ]
                        }
                    )
                },
            }
        ],
        "usage": {"completion_tokens": 624},
    }
    values = {
        "coding-evidence.json": first.evidence,
        "private-request.json": {"binding": first.binding, "request": first.request},
        "private-analysis-issued.json": {
            "request_sha256": first.binding["request_sha256"],
            "evidence_identity": partition_evidence_identity(first),
        },
        "private-analysis.json": {
            "request": first.request,
            "request_sha256": first.binding["request_sha256"],
            "evidence_identity": partition_evidence_identity(first),
            "response": response,
        },
    }
    pins = {}
    for name, value in values.items():
        raw = (json.dumps(value) + "\n").encode()
        path = source / name
        path.write_bytes(raw)
        pins[name] = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    amendment = {
        "predecessor_output_uri": str(source.parent),
        "part1_files": pins,
        "predecessor_identity": "saved-v14",
        "predecessor_source_head": "857f8ac68",
        "bounds": {
            "prior_generation_requests": 1,
            "maximum_new_generation_requests": 1,
            "maximum_total_generation_requests": 2,
        },
    }
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment))
    pin = PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    worker = RegionalPartitionedCodingAnalysisConfig(
        analysis=partitioned.analysis,
        input_pin=pin,
        failure=pin,
        evidence=PinnedFile(
            str(Path(partitioned.analysis.evidence_path) / "coding-evidence.json"), first.binding["evidence_sha256"]
        ),
        source_files=worker_source_files(),
        manifest=PinnedFile(partitioned.manifest_path, partitioned.manifest_sha256),
    )
    return ResponseRecoveryConfig(worker, pin), first, second, values, pins


def test_saved_long_response_recovery_preserves_request_and_issues_only_part2(
    saved_long_partition, tmp_path, monkeypatch
):
    config, _first, second, _values, pins = saved_long_partition
    assert (
        compact_json_sha256(FeedbackAnalysis.model_json_schema())
        == "b56eaa3fdf8ea6f7e580caf5abd8aa6c6b633506aafa65a4ea9eddb9e5b2c9b8"
    )
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def create(self, **request):
            calls.append(request)
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"skills": [{"skill": "types", "confidence": 0.9, "evidence": "saved second partition"}]}
                            )
                        }
                    }
                ]
            }
            return SimpleNamespace(model_dump=lambda **_: response)

    monkeypatch.setattr(coding_eval_feedback, "AsyncOpenAI", Client)
    monkeypatch.setattr(coding_eval_feedback, "resolve_glm_base_url", lambda _: "https://synthetic.invalid/v1")
    monkeypatch.setenv(coding_eval_feedback.GLM_TOKEN_ENV, "synthetic")
    run_response_recovery(config)
    run_response_recovery(config)
    assert calls == [second.request]
    output = Path(config.worker.analysis.output_path)
    for name, pin in pins.items():
        assert (output / "part-1" / name).read_bytes() == Path(pin["uri"]).read_bytes()
    assert json.loads((output / "capabilities.json").read_text())["skills"] == [
        {"label": "types", "description": coding_eval_feedback.SKILL_DESCRIPTIONS["types"]}
    ]
    merged = json.loads((output / "private-merged-analysis.json").read_text())
    assert merged["entries"][0]["evidence"] == "x" * 1163
    assert merged["entries"][0]["confidence"] == 0.6
    assert merged["entries"][0]["disposition"] == "below-confidence-threshold"


@pytest.mark.parametrize("change", ["unknown-label", "extra", "invalid-confidence", "empty-evidence", "too-many"])
def test_saved_response_keeps_all_other_schema_guards_before_any_http(saved_long_partition, change, monkeypatch):
    calls = []

    def no_http(**kwargs):
        calls.append(kwargs)
        pytest.fail("Invalid saved response must fail before provider access")

    monkeypatch.setattr(coding_eval_feedback, "AsyncOpenAI", no_http)
    config, _first, _second, values, _pins = saved_long_partition
    saved = values["private-analysis.json"]
    content = json.loads(saved["response"]["choices"][0]["message"]["content"])
    entry = content["skills"][0]
    if change == "unknown-label":
        entry["skill"] = "unknown"
    elif change == "extra":
        entry["unexpected"] = True
    elif change == "invalid-confidence":
        entry["confidence"] = 1.1
    elif change == "empty-evidence":
        entry["evidence"] = ""
    else:
        content["skills"] *= 5
    saved["response"]["choices"][0]["message"]["content"] = json.dumps(content)
    amendment = config.amendment.read_json()
    pin = amendment["part1_files"]["private-analysis.json"]
    raw = (json.dumps(saved) + "\n").encode()
    Path(pin["uri"]).write_bytes(raw)
    pin.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    path = Path(config.amendment.uri)
    path.write_text(json.dumps(amendment))
    repaired_pin = PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValidationError):
        run_response_recovery(ResponseRecoveryConfig(config.worker, repaired_pin))
    assert calls == []


def test_issued_part2_without_response_blocks_all_http(saved_long_partition):
    config, _first, second, _values, _pins = saved_long_partition
    directory = Path(config.worker.analysis.output_path) / "part-2"
    directory.mkdir(parents=True)
    (directory / "private-analysis-issued.json").write_text(
        json.dumps(
            {
                "request_sha256": second.binding["request_sha256"],
                "evidence_identity": partition_evidence_identity(second),
            }
        )
    )
    with pytest.raises(ValueError, match="ambiguous"):
        run_response_recovery(config)


partitioned_completed = test_completed_partitioned_coding_analysis.partitioned_completed
completed_evidence = test_completed_partitioned_coding_analysis.completed_evidence


def test_recovery_graph_binds_historical_source_files_and_only_adopted_evidence(
    partitioned_completed, tmp_path, monkeypatch
):
    config, evidence_step, pin, prefix = partitioned_completed
    monkeypatch.setattr(
        completed_partitioned_coding_analysis,
        "completed_coding_analysis_stages",
        lambda cfg, cfg_pin, _: completed_coding_analysis.completed_evidence_analysis_stages(
            cfg, cfg_pin, evidence_step
        ),
    )
    previous_pin = PinnedFile(**pin(tmp_path / "v14.json", config))
    previous = completed_partitioned_coding_analysis.completed_partitioned_analysis_stages(config, previous_pin, {})[
        "terminal"
    ]
    frozen_files = {"historical-worker.py": "a" * 64}
    historical = replace(
        previous, build_config=lambda ctx: replace(previous.build_config(ctx), source_files=frozen_files)
    )
    path = Path(historical.path(prefix))
    bound = historical.build_config(StepContext.for_run(str(path), prefix, deps=historical.deps))
    first, _second = partition_analysis_requests(bound.partitioned())
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"skills": [{"skill": "change_scope", "confidence": 0.6, "evidence": "x" * 1163}]}
                    )
                },
            }
        ],
        "usage": {"completion_tokens": 624},
    }
    values = {
        ".executor_info": {},
        ".executor_status": "FAILED",
        "analysis-worker-provenance.json": {
            "source_files": frozen_files,
            "worker": WORKER_SETTINGS,
            "skyrl": {"direct_url": {"vcs_info": {"commit_id": MARIN_SKYRL.commit}}},
        },
        "worker-submission/reservation.json": asdict(bound),
        "part-1/coding-evidence.json": first.evidence,
        "part-1/private-request.json": {"binding": first.binding, "request": first.request},
        "part-1/private-analysis-issued.json": {
            "request_sha256": first.binding["request_sha256"],
            "evidence_identity": partition_evidence_identity(first),
        },
        "part-1/private-analysis.json": {
            "request": first.request,
            "request_sha256": first.binding["request_sha256"],
            "evidence_identity": partition_evidence_identity(first),
            "response": response,
        },
    }
    artifacts = {}
    for name, value in values.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = b"FAILED" if name == ".executor_status" else json.dumps(value).encode()
        target.write_bytes(raw)
        artifacts[name] = {"uri": str(target), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    terminal_pin = pin(
        tmp_path / "v14-terminal.json",
        {
            "job": "/saved-worker",
            "state": "failed",
            "failure_count": 1,
            "preemption_count": 0,
            "completed_count": 0,
            "task_count": 1,
            "source_head": recovery.PREDECESSOR_SOURCE,
            "tasks": [{"id": "/saved-worker/0", "state": "failed", "exit_code": 1}],
        },
    )
    inventory_pin = pin(
        tmp_path / "v14-inventory.json",
        {
            "complete_recursive_listing": True,
            "prefix": str(path),
            "files": list(values),
            "artifacts": artifacts,
            "issuance_per_part": {"1": 1, "2": 0},
            "response_per_part": {"1": 1, "2": 0},
        },
    )
    amendment = {
        "protocol": recovery.AMENDMENT_PROTOCOL,
        "predecessor_config": asdict(previous_pin),
        "predecessor_identity": artifact_identity(historical),
        "predecessor_output_uri": str(path),
        "predecessor_source_head": recovery.PREDECESSOR_SOURCE,
        "terminal": terminal_pin,
        "inventory": inventory_pin,
        "foreground_exit": pin(tmp_path / "v14-exit.json", {"exit_code": 1}),
        "part1_files": {
            name.removeprefix("part-1/"): entry for name, entry in artifacts.items() if name.startswith("part-1/")
        },
        "parser": {
            "request_schema_sha256": compact_json_sha256(FeedbackAnalysis.model_json_schema()),
            "response_evidence_min_length": 1,
            "response_evidence_max_length": None,
            "other_constraints": "unchanged",
        },
        "bounds": {
            "prior_generation_requests": 1,
            "maximum_new_generation_requests": 1,
            "maximum_total_generation_requests": 2,
            "max_tokens": 2048,
            "reasoning_effort": "low",
            "retries": 0,
            "merge_rule": MERGE_RULE,
        },
    }
    repaired = {
        "protocol": recovery.PROTOCOL,
        "version": recovery.VERSION,
        "predecessor_config": asdict(previous_pin),
        "parser_amendment": pin(tmp_path / "parser-amendment.json", amendment),
    }
    stages = recovery.response_recovery_stages(repaired, PinnedFile(**pin(tmp_path / "v16.json", repaired)), {})
    assert len(graph_handles([stages["terminal"]])) == 2
    assert all(step is not previous for step in graph_handles([stages["terminal"]]))
    new = stages["terminal"].build_config(
        StepContext.for_run(str(tmp_path / "fresh-v16"), prefix, deps=stages["terminal"].deps)
    )
    assert new.worker.source_files == worker_source_files()
    assert new.worker.source_files != frozen_files
    assert new.worker.analysis.output_path == str(tmp_path / "fresh-v16")
