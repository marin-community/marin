# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover the forty frozen fc4 attempts that ran below original role budgets."""

import gzip
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.oracle_filter import ResourceCalibrationReceipt, recovered_source_runtime_requirements

from experiments.post_training.task_curation.oracle_filter import row_runtime, row_task, source_runtime_binding


def test_all_forty_original_fc4_attempts_keep_old_history_and_recover_exact_role_budgets():
    fixtures = Path(__file__).parent / "fixtures"
    parent_bytes = gzip.decompress((fixtures / "fc4-forty-parent-maps.json.gz").read_bytes())
    parents = json.loads(parent_bytes)
    rows = pq.read_table(
        io.BytesIO(gzip.decompress((fixtures / "fc4-forty-original-rows.parquet.gz").read_bytes()))
    ).to_pylist()
    recovered = []
    for row in rows:
        task = row_task(row)
        assert task is not None
        binding = source_runtime_binding(task, row_runtime(row))
        assert binding is not None
        mapping = parents[row["source_key"]][task.id]
        assert binding.task_sha256 == mapping["task_sha256"]
        assert binding.runtime_sha256 == mapping["runtime_sha256"]
        parent = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
        original = parent.model_dump(mode="json")
        shape, preserved, verifier = recovered_source_runtime_requirements(parent, binding)
        assert shape == ExecutionRequirements(
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
            agent_timeout_sec=parent.execution_requirements.agent_timeout_sec,
            verifier_timeout_sec=parent.execution_requirements.verifier_timeout_sec,
        )
        assert preserved == {**parent.preserved_requirements, "cpus": 2, "memory_mb": 4096, "storage_mb": 10240}
        assert verifier.fixed == {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}
        assert verifier.mode == parent.verifier.mode
        assert parent.execution_requirements == ExecutionRequirements(
            cpus=1,
            memory_mb=2048,
            storage_mb=2048,
            agent_timeout_sec=shape.agent_timeout_sec,
            verifier_timeout_sec=shape.verifier_timeout_sec,
        )
        assert parent.model_dump(mode="json") == original
        assert not parent.passed
        recovered.append((row["source_key"], task.id))
    assert len(set(recovered)) == 40
