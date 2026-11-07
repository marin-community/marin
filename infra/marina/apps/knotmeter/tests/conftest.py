# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
