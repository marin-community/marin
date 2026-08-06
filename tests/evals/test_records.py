# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``EvalRunRecord`` round-trip and the ``record.json`` wire-format contract.

The dashboard, the Postgres jsonb mirror, and every ``record.json`` already written to object
storage all agree on one shape: ``eval`` (not the Python attribute name ``evaluation``) nesting
``name``/``mechanism``/``tasks``, enum fields as plain strings, and ``log_tails`` as JSON arrays.
These tests pin that shape independently of the pydantic model that produces it.
"""

import json
from pathlib import Path

import pytest
from marin.evaluation.records import (
    EvalchemyRef,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HarborRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    RunTiming,
    ServingParams,
    read_record,
    read_records,
    scan_records,
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
    provenance=Provenance(git_sha="abc123", eval_runtime="evalchemy:latest", launch_host="dev-box"),
)


class _CachedListingFileSystem:
    def __init__(self):
        self._listings = {}

    def ls(self, path, detail):
        if path not in self._listings:
            self._listings[path] = [
                {"name": str(child), "type": "directory" if child.is_dir() else "file"} for child in Path(path).iterdir()
            ]
        return self._listings[path]

    def invalidate_cache(self, path):
        self._listings.pop(path, None)

    def unstrip_protocol(self, path):
        return path


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
        "tasks": [
            {
                "name": "gsm8k",
                "num_fewshot": 8,
                "task_alias": None,
                "generation": False,
                "unsafe_code": False,
                "completion_only": False,
            }
        ],
        "evalchemy": None,
        "harbor": None,
    }
    assert raw["status"] == "succeeded"
    assert isinstance(raw["status"], str)
    assert raw["version"] == "2026.07.19"
    assert raw["description"] == "baseline sweep"


def test_timing_round_trips_and_defaults_to_none(tmp_path):
    """``timing`` is optional: the base record omits it (``None``), and a record carrying a window
    round-trips through ``record.json`` and serializes under the ``timing`` key."""
    assert _RECORD.timing is None

    timed = _RECORD.model_copy(
        update={"timing": RunTiming(started_at="2026-07-19T09:06:31+00:00", finished_at="2026-07-19T09:14:31+00:00")}
    )
    path = write_record(timed, str(tmp_path))
    with open(path) as f:
        raw = json.load(f)
    assert raw["timing"] == {"started_at": "2026-07-19T09:06:31+00:00", "finished_at": "2026-07-19T09:14:31+00:00"}
    assert read_record(path) == timed


def test_serving_round_trips_and_defaults_to_none(tmp_path):
    """``serving`` is optional and round-trips: typed fields plus the free-form ``extra`` string map."""
    assert _RECORD.serving is None

    served = _RECORD.model_copy(
        update={
            "serving": ServingParams(
                tensor_parallel_size=8, max_model_len=4096, max_gen_tokens=2048, extra={"temperature": "0.0"}
            )
        }
    )
    path = write_record(served, str(tmp_path))
    with open(path) as f:
        raw = json.load(f)
    assert raw["serving"] == {
        "tensor_parallel_size": 8,
        "data_parallel_size": None,
        "max_model_len": 4096,
        "max_gen_tokens": 2048,
        "extra": {"temperature": "0.0"},
    }
    assert read_record(path) == served


def test_read_records_collects_parse_failures_without_dropping_good_ones(tmp_path):
    """A malformed record.json alongside a valid one is reported as a failure (path + error) rather
    than silently skipped, and the valid record still comes back."""
    write_record(_RECORD, str(tmp_path))
    broken_dir = tmp_path / "20260101-000000-broken-mmlu"
    broken_dir.mkdir()
    (broken_dir / "record.json").write_text(json.dumps({"run_id": "broken", "not": "a record"}))

    records, failures = read_records(str(tmp_path))

    assert [r.run_id for r in records] == [_RECORD.run_id]
    assert len(failures) == 1
    assert failures[0].path.endswith("20260101-000000-broken-mmlu/record.json")
    assert "ValidationError" in failures[0].error


def test_read_records_only_discovers_flat_eval_layout(tmp_path):
    evals = tmp_path / "evals"
    flat = _RECORD.model_copy(update={"run_id": "flat-run"})
    nested = _RECORD.model_copy(update={"run_id": "nested-run"})
    write_record(flat, str(evals))
    write_record(nested, str(evals / "model" / "suite" / "version"))

    records, failures = read_records(str(evals))

    assert [record.run_id for record in records] == ["flat-run"]
    assert failures == []


def test_scan_records_cache_detects_added_and_deleted_runs(tmp_path):
    evals = tmp_path / "evals"
    first = _RECORD.model_copy(update={"run_id": "first-run"})
    second = _RECORD.model_copy(update={"run_id": "second-run"})
    first_path = Path(write_record(first, str(evals)))
    initial = scan_records(str(evals))

    first_path.unlink()
    first_path.parent.rmdir()
    write_record(second, str(evals))
    refreshed = scan_records(str(evals), initial.records_by_path)

    assert [record.run_id for record in refreshed.records] == ["second-run"]


def test_scan_records_invalidates_filesystem_listing_cache(tmp_path, monkeypatch):
    evals = tmp_path / "evals"
    write_record(_RECORD.model_copy(update={"run_id": "first-run"}), str(evals))
    filesystem = _CachedListingFileSystem()
    monkeypatch.setattr(
        "marin.evaluation.records.url_to_fs",
        lambda prefix: (filesystem, prefix),
    )
    initial = scan_records(str(evals))

    write_record(_RECORD.model_copy(update={"run_id": "second-run"}), str(evals))
    refreshed = scan_records(str(evals), initial.records_by_path)

    assert [record.run_id for record in refreshed.records] == ["first-run", "second-run"]


def test_record_json_includes_harbor_policy_identity_and_effective_limit(tmp_path):
    record = _RECORD.model_copy(
        update={
            "evaluation": EvalRef(
                name="aime-policy",
                mechanism="harbor",
                harbor=HarborRef(
                    dataset="aime",
                    version="1.0",
                    agent="terminus-2",
                    env="daytona",
                    task_limit=2,
                    config_digest="sha256:" + "a" * 64,
                ),
            )
        }
    )

    path = write_record(record, str(tmp_path))
    with open(path) as f:
        raw = json.load(f)

    assert raw["eval"]["harbor"] == {
        "dataset": "aime",
        "version": "1.0",
        "agent": "terminus-2",
        "env": "daytona",
        "task_limit": 2,
        "config_digest": "sha256:" + "a" * 64,
    }


def test_record_json_includes_normalized_evalchemy_configuration(tmp_path):
    record = _RECORD.model_copy(
        update={
            "evaluation": EvalRef(
                name="ifeval",
                mechanism="evalchemy",
                tasks=(
                    EvalTaskRef(
                        name="ifeval",
                        num_fewshot=0,
                        task_alias="ifeval_0shot",
                        generation=True,
                    ),
                ),
                evalchemy=EvalchemyRef(
                    apply_chat_template=True,
                    max_gen_toks=2048,
                    max_eval_instances=64,
                    num_concurrent=16,
                    batch_size="1",
                    seed=1234,
                    extra_gen_kwargs={"temperature": "0"},
                    extra_model_args={"timeout": 900},
                    max_length=32768,
                ),
            )
        }
    )

    path = write_record(record, str(tmp_path))
    with open(path) as f:
        raw = json.load(f)

    assert raw["eval"]["evalchemy"] == {
        "apply_chat_template": True,
        "max_gen_toks": 2048,
        "max_eval_instances": 64,
        "num_concurrent": 16,
        "batch_size": "1",
        "seed": 1234,
        "extra_gen_kwargs": {"temperature": "0"},
        "extra_model_args": {"timeout": 900},
        "max_length": 32768,
    }


@pytest.mark.parametrize("runtime_key", ["evalchemy_image", "eval_image"])
def test_read_record_parses_a_previously_written_record_json(tmp_path, runtime_key):
    """Historical runtime field names normalize to the current record schema."""
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
        "provenance": {"git_sha": "deadbeef", runtime_key: "evalchemy:old", "launch_host": "ci-runner"},
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
    assert record.provenance.eval_runtime == "evalchemy:old"
    assert record.model_dump(mode="json", by_alias=True)["provenance"] == {
        "git_sha": "deadbeef",
        "eval_runtime": "evalchemy:old",
        "launch_host": "ci-runner",
    }
