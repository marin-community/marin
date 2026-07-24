# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``EvalRunRecord`` round-trip and the ``record.json`` wire-format contract.

The dashboard, the Postgres jsonb mirror, and every ``record.json`` already written to object
storage all agree on one shape: ``eval`` (not the Python attribute name ``evaluation``) nesting
``name``/``mechanism``/``tasks``, enum fields as plain strings, and ``log_tails`` as JSON arrays.
These tests pin that shape independently of the pydantic model that produces it.
"""

import json

from marin.evaluation.records import (
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    read_record,
    write_record,
)

_RECORD = EvalRunRecord(
    run_id="20260719-091431-qwen3-8b-gsm8k-7565",
    group_id="20260719-091431-qwen3-8b-7565",
    created_at="2026-07-19T09:14:31.123456+00:00",
    user="russell",
    version="2026.07.19",
    description="baseline sweep",
    model=ModelRef(name="qwen3-8b", location="gs://marin-models/qwen3-8b", backend="vllm"),
    evaluation=EvalRef(
        name="gsm8k",
        mechanism="evalchemy",
        tasks=(EvalTaskRef(name="gsm8k", num_fewshot=8),),
    ),
    hardware=HardwareRef(platform="tpu", accelerator="v6e-8", region_or_cluster="us-central2"),
    status=RunStatus.SUCCEEDED,
    error=None,
    results_path="gs://marin-eval-metadata/runs/20260719-091431-qwen3-8b-gsm8k-7565/results",
    metrics={"gsm8k": {"exact_match,none": 0.62, "exact_match_stderr,none": 0.01}},
    jobs={"orchestrator": "job/123", "serve": "job/124", "eval": "job/125"},
    log_tails={},
    provenance=Provenance(git_sha="abc123", eval_image="evalchemy:latest", launch_host="dev-box"),
)


def test_write_read_record_round_trip(tmp_path):
    path = write_record(_RECORD, str(tmp_path))

    reread = read_record(path)

    assert reread == _RECORD


def test_record_json_uses_eval_alias_and_plain_string_enum(tmp_path):
    """``evaluation`` serializes under the ``eval`` key, and ``status`` as its bare string value --
    the shape the dashboard's TS types and the Postgres ``record`` jsonb column both expect."""
    path = write_record(_RECORD, str(tmp_path))

    with open(path) as f:
        raw = json.load(f)

    assert "evaluation" not in raw
    assert raw["eval"] == {
        "name": "gsm8k",
        "mechanism": "evalchemy",
        "tasks": [{"name": "gsm8k", "num_fewshot": 8}],
        "harbor": None,
    }
    assert raw["status"] == "succeeded"
    assert isinstance(raw["status"], str)
    assert raw["version"] == "2026.07.19"
    assert raw["description"] == "baseline sweep"


def test_read_record_parses_a_previously_written_record_json(tmp_path):
    """A ``record.json`` written by a prior version of the module (plain dict, ``eval`` key, list-typed
    ``log_tails``) must still parse -- this is the on-disk shape of every record already in object
    storage, independent of whatever Python type produces or consumes it now."""
    legacy = {
        "run_id": "20260101-000000-llama-3-8b-mmlu-abcd",
        "group_id": "20260101-000000-llama-3-8b-abcd",
        "created_at": "2026-01-01T00:00:00+00:00",
        "user": "ci",
        "model": {"name": "llama-3-8b", "location": "gs://marin-models/llama-3-8b", "backend": "vllm"},
        "eval": {
            "name": "mmlu",
            "mechanism": "evalchemy",
            "tasks": [{"name": "mmlu", "num_fewshot": 5}],
        },
        "hardware": {"platform": "gpu", "accelerator": "H100x8", "region_or_cluster": None},
        "status": "infra_failed",
        "error": "RuntimeError: endpoint never came up",
        "results_path": "gs://marin-eval-metadata/runs/20260101-000000-llama-3-8b-mmlu-abcd/results",
        "metrics": {},
        "jobs": {"orchestrator": "job/1"},
        "log_tails": {"serve": ["boot failed", "OOM"]},
        "provenance": {"git_sha": "deadbeef", "eval_image": "evalchemy:old", "launch_host": "ci-runner"},
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(legacy))

    record = read_record(str(path))

    assert record.run_id == legacy["run_id"]
    assert record.evaluation.name == "mmlu"
    assert record.evaluation.tasks == (EvalTaskRef(name="mmlu", num_fewshot=5),)
    assert record.status is RunStatus.INFRA_FAILED
    assert record.log_tails == {"serve": ("boot failed", "OOM")}
    assert record.hardware.region_or_cluster is None
