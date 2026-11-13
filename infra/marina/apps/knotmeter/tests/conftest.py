# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from marina.db import UrlDatabase
from marina.testing import SourceServices
from marina.testing import test_database as disposable_database
from sqlalchemy import text

from infra.marina.apps.knotmeter.server.app import migrate


@pytest.fixture
def knotmeter_database():
    with disposable_database() as url:
        services = SourceServices(name="knotmeter", data_url="", database=UrlDatabase(url))
        engine = services.engine()
        with engine.begin() as connection:
            migrate(connection)

        def query(sql, parameters):
            with engine.begin() as connection:
                result = connection.execute(text(sql), parameters)
                return {
                    "rows": [dict(row) for row in result.mappings()] if result.returns_rows else [],
                    "row_count": result.rowcount,
                }

        try:
            yield services, query
        finally:
            engine.dispose()


@pytest.fixture
def bundle(tmp_path: Path):
    def build(hosts: int = 1):
        source = {
            "kind": "github-actions",
            "repository": "marin-community/marin",
            "run_id": 123,
            "run_attempt": 2,
            "head_sha": "source-commit",
        }
        gpu = "H100" if hosts == 1 else "GB200"
        devices = 8 if hosts == 1 else 4
        config = {
            "model": "example/model",
            "model_revision": "model-commit",
            "precision": "bf16",
            "profiles": {"test": {"gpu": gpu, "gpus": devices, "hosts": hosts}},
        }
        manifest: dict[str, Any] = {
            "source": source,
            "profile": "test",
            "concurrency": 8,
            "recipe_fingerprint": "recipe",
            "workload": {
                "key": "fixture-workload",
                "lineage": {"dataset": "example/corpus", "dataset_revision": "dataset-commit", "sessions": 393},
            },
        }
        receipt = {
            "status": "valid",
            "source": source,
            "attempt_uid": "native-attempt",
            "configuration": {"series": config},
            "validation": {"profiling_requests": 2, "coverage": {"expected_duration_seconds": 3600}},
        }
        aggregate = {
            "model": "example/model",
            "hw": "test",
            "conc": 8,
            "precision": "bf16",
            "framework": "vllm",
            "scenario_type": "agentic-coding",
            "image": "runtime@commit",
            "recipe_fingerprint": "recipe",
            "num_gpus": hosts * devices,
            "num_requests_total": 3,
            "num_requests_successful": 2,
            "request_metrics": {
                "throughput": {
                    "per_gpu": {"output_tput_tps": 40.5},
                    "output": {"tokens_per_second": 40.5 * hosts * devices},
                },
                "latency": {"ttft": {"p90": 1.25}},
            },
        }
        attempt = {
            "source": source,
            "server_parallelism": {"tp": 1, "dp": hosts * devices, "ep": hosts * devices, "pp": 1},
            "status": "succeeded",
            "iris_attempt_uid": "native-attempt",
            "recipe_fingerprint": "recipe",
        }
        base = 1793157479513724276
        records = []
        for index, (offset, phase) in enumerate([(12800, "profiling"), (0, "warmup"), (51, "profiling")]):
            records.append(
                {
                    "metadata": {
                        "x_request_id": str(index),
                        "conversation_id": "conversation::sa:1",
                        "source_trace_id": "conversation",
                        "source_outer_idx": 7,
                        "source_kind": "weka_subagent",
                        "turn_index": index,
                        "request_start_ns": base + offset,
                        "request_end_ns": base + offset + 123456789,
                        "benchmark_phase": phase,
                    },
                    "metrics": {
                        "time_to_first_token": {"value": 1.125, "unit": "ms"},
                        "input_sequence_length": {"value": 99},
                        "output_sequence_length": {"value": 12},
                    },
                }
            )
        files: dict[str, Any] = {
            "agentx-source.json": {"source": source, "points": [{"manifest": manifest, "receipt": receipt}]},
            "bmk_point/agg.json": aggregate,
            "point/attempt.json": attempt,
        }
        if hosts > 1:
            # Placement is an eligible pool; the gang selects only the requested hosts.
            manifest["placement"] = {"nodes": [str(host) for host in range(hosts + 2)]}
            files["point/gang.json"] = {
                "attempts": {f"task-{host}": f"attempt-{host}" for host in range(hosts)},
                "tasks": [
                    {
                        "status": "succeeded",
                        "source": source,
                        "recipe_fingerprint": "recipe",
                        "iris_node_name": str(host),
                        "iris_task_id": f"task-{host}:0",
                        "iris_attempt_uid": f"attempt-{host}",
                        "gpu_uuids": [f"{host}-{i}" for i in range(devices)],
                    }
                    for host in range(hosts)
                ],
            }
        for name, content in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
        raw = tmp_path / "point/aiperf_artifacts/profile_export.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("\n".join(json.dumps(r) for r in records))
        actual = source | {
            "id": 123,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-10-28T03:11:34Z",
        }
        return tmp_path, actual

    return build


@pytest.fixture
def calibration(bundle, tmp_path: Path):
    """Tiny native-format repeats; synthetic data stays in the test database."""
    original, _ = bundle(3)
    root = tmp_path / "calibration"
    root.mkdir()
    source = {"kind": "private-calibration", "commit": "private-source", "files": {"runner.py": "runner-hash"}}
    collector = json.loads((original / "agentx-source.json").read_text())
    config = collector["points"][0]["receipt"]["configuration"]["series"] | {"duration": 900}
    workload = collector["points"][0]["manifest"]["workload"]
    aggregate = json.loads((original / "bmk_point/agg.json").read_text()) | {"tp": 12}
    worker = "http://127.0.0.1:9000/metrics"
    aggregate["server_metrics"] = {
        "tokens": {"prompt_total": 600, "generation_total": 60, "requests_completed": 2},
        "sources": [
            {"id": f"combined|{endpoint}|engine={engine}", "prompt_tokens": prompt, "generation_tokens": generated}
            for endpoint in [worker, "http://127.0.0.1:9001/metrics"]
            for engine, prompt, generated in [(0, 100, 10), (1, 200, 20)]
        ],
    }
    topology = {"tp": 6, "dp": 2, "ep": 12, "pp": 1}
    rows, windows = [], []
    for index in range(2):
        job, uid = f"/test/calibration-repeat-{index}", f"native-{index}"
        task = job + "/0"
        processed = root / f"point-{index}"
        raw = processed / "restored-point"
        profile = raw / "aiperf_artifacts/profile_export.jsonl"
        profile.parent.mkdir(parents=True)
        profile.write_bytes((original / "point/aiperf_artifacts/profile_export.jsonl").read_bytes())
        native = {
            "start_time": f"2026-11-{10 + index}T12:00:00",
            "metadata": {
                "submission_valid": True,
                "metric_duration_coverage": [
                    {
                        "expected_duration_seconds": 900,
                        "required_ratio": 0.95,
                        "ttft_ratio": 1,
                        "inter_token_latency_ratio": 1,
                    }
                ],
            },
            "input_config": {"phases": [{"kind": "profiling", "duration": 900, "concurrency": 8}]},
            "request_count": {"avg": 2},
        }
        native_path = profile.with_name("profile_export_aiperf.json")
        native_path.write_text(json.dumps(native))
        phase_log = root / f"phase-{index}.log"
        phase_log.write_text("completed=2 cancelled=1 errors=0 sent=3")
        attempt = {
            "source": source,
            "status": "succeeded",
            "screening": False,
            "profile": "test",
            "iris_task_id": task + ":0",
            "iris_attempt_uid": uid,
            "gang_attempts": {task: uid},
            "recipe_fingerprint": "recipe",
            "all_allocated_gpus": 12,
            "server_parallelism": topology,
            "workload": workload,
            "server_command": ["vllm", "serve", "example/model", "--max-num-seqs", "128", "--port", str(8000 + index)],
            "server_environment": {},
            "worker_metrics_url": worker,
            "private_runtime": {"wheel": {"sha256": "runtime-pin"}},
        }
        files = {
            f"point-{index}/restored-point/attempt.json": attempt,
            f"point-{index}/aggregate/agg.json": aggregate,
            f"point-{index}/processor-command.json": {"source": "processor-pin"},
            f"point-{index}/summary.json": {"input_export_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
            f"submission-{index}.json": {
                "job_id": job,
                "series": config,
                "count": 12,
                "campaign": {"source": source, "manifest": {"workload": workload}},
            },
        }
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
        rows.append(
            {
                "job": job,
                "native_submission_valid": True,
                "screening": False,
                "tp": 6,
                "dp": 2,
                "ep": 12,
                "summary": f"point-{index}/summary.json",
                "submission": f"submission-{index}.json",
                "raw_archive": f"s3://test/point-{index}.tar.zst",
                "raw_archive_sha256": f"archive-{index}",
            }
        )
        windows.append(
            {
                "job": job,
                "native_phase_log": phase_log.name,
                "native_phase_log_sha256": hashlib.sha256(phase_log.read_bytes()).hexdigest(),
                "profiling_sent": 3,
                "profiling_requests": 2,
                "profiling_cancelled": 1,
                "profiling_errors": 0,
            }
        )
    rows.append({"job": "/test/invalid-screen", "native_submission_valid": False, "screening": True})
    (root / "curve-summary.json").write_text(json.dumps({"rows": rows}))
    (root / "window-effects.json").write_text(
        json.dumps({"rows": windows, "scope": "15-minute calibration; tail cancellations limit this window."})
    )
    return root
