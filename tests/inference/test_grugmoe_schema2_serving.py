# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the schema-2 serving qualification receipt."""

import json

from iris.client.client import TaskLogEntry
from iris.cluster.types import JobName
from marin.testing.inference.grugmoe_schema2_serving import (
    _runtime_evidence_logs,
    _validate_runtime_logs,
)
from rigging.timing import Timestamp


class _FakeRuntimeLogJob:
    """Small Finelog model: an unset limit is a 1,000-line server tail."""

    def __init__(self, entries: list[TaskLogEntry]):
        self._entries = entries

    def logs(
        self,
        *,
        max_lines: int = 0,
        substring: str = "",
        tail: bool = False,
    ) -> list[TaskLogEntry]:
        matching = [entry for entry in self._entries if substring in entry.data]
        limit = max_lines or 1_000
        return matching[-limit:] if tail else matching[:limit]


def _entry(sequence: int, data: str, *, task: int = 0) -> TaskLogEntry:
    task_id = JobName.from_string(f"/test/schema2-runtime/{task}")
    return TaskLogEntry(
        timestamp=Timestamp.from_ms(sequence),
        task_id=task_id,
        source="stdout",
        data=data,
        attempt_id=0,
        key=f"{task_id}:0",
    )


def test_runtime_evidence_survives_more_than_the_default_finelog_tail():
    pipeline_parallel_size = 3
    expected_wheel = {"source_commit": "3caca1d"}
    verified_wheel = {
        **expected_wheel,
        "compute_capability": "9.0",
        "extension_path": "/opt/venv/site-packages/vllm/_C.abi3.so",
    }
    lines: list[tuple[str, int]] = []
    for stage in range(pipeline_parallel_size):
        lines.extend(
            (
                (
                    "vLLM requested topology: tasks=3 GPUs/task=8 DP=8 EP=8 " f"PP=3 TP=1 task={stage}",
                    stage,
                ),
                ("MARIN_VLLM_WHEEL_VERIFIED=" + json.dumps(verified_wheel), stage),
                ("GPU KV cache size: 4,096 tokens", stage),
            )
        )
        for dp_rank in range(8):
            lines.extend(
                (
                    (
                        "Worker placement: "
                        f"process_rank={dp_rank * pipeline_parallel_size + stage} "
                        f"node_rank={stage} local_rank=0 "
                        f"DP={dp_rank}/8 EP={dp_rank}/8 "
                        f"PP={stage}/3 TP=0/1 GPU={dp_rank}",
                        stage,
                    ),
                    (
                        "GrugMoE effective config: "
                        f"TP=0/1 PP={stage}/3 layers=[{stage * 16},{(stage + 1) * 16}) "
                        f"DP={dp_rank}/8 EP={dp_rank}/8 use_ep=True experts=384 local=48",
                        stage,
                    ),
                )
            )
    lines.append(("load_format=dummy", 0))
    entries = [_entry(index, line, task=task) for index, (line, task) in enumerate(lines)]
    entries.extend(_entry(len(entries) + index, f"unrelated startup noise {index}") for index in range(1_500))
    job = _FakeRuntimeLogJob(entries)

    broad_tail = job.logs(max_lines=0, tail=True)
    assert not any("vLLM requested topology:" in entry.data for entry in broad_tail)

    logs = _runtime_evidence_logs(job, case="dummy_d6144")
    evidence = _validate_runtime_logs(
        logs,
        case="dummy_d6144",
        pipeline_parallel_size=pipeline_parallel_size,
        num_layers=48,
        num_experts=384,
        expected_wheel=expected_wheel,
    )

    assert evidence is not None
    assert evidence["worker_placement_count"] == 24
    assert evidence["runtime_topology_count"] == 24
    assert len(evidence["verified_wheels"]) == 3
    assert evidence["kv_cache_tokens"] == [4096, 4096, 4096]
    assert evidence["dummy_load_format_observed"] is True
    assert "unrelated startup noise" not in logs
