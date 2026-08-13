# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET

import pytest
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)

from experiments.grug.moe_hero_ep.launch_mok_ep64_probe import (
    PROCESSES_PER_TASK,
    build_probe_request,
)
from experiments.grug.moe_hero_ep.mok_ep64_symmetric_memory_probe import (
    _fabric_xml_text,
    pattern_byte,
    probe_rank_from_env,
    sample_offsets,
)


def test_probe_rank_requires_one_gpu_per_supervised_process():
    env = {
        IRIS_MULTIGPU_PROCESS_INDEX_ENV: "63",
        IRIS_MULTIGPU_PROCESS_COUNT_ENV: "64",
        IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV: "3",
    }

    rank = probe_rank_from_env(env)

    assert rank.global_rank == 63
    assert rank.world_size == 64
    assert rank.local_device_id == 3

    env[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV] = "2,3"
    with pytest.raises(ValueError, match="one GPU per process"):
        probe_rank_from_env(env)


def test_probe_rank_rejects_missing_or_out_of_range_identity():
    with pytest.raises(ValueError, match="missing"):
        probe_rank_from_env({})

    env = {
        IRIS_MULTIGPU_PROCESS_INDEX_ENV: "64",
        IRIS_MULTIGPU_PROCESS_COUNT_ENV: "64",
        IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV: "0",
    }
    with pytest.raises(ValueError, match="Invalid supervised rank"):
        probe_rank_from_env(env)


def test_probe_patterns_cover_rank_32_boundary_and_multiple_generations():
    values = {pattern_byte(rank, iteration) for rank in (0, 31, 32, 63) for iteration in range(3)}

    assert 0 not in values
    assert len(values) == 12
    assert sample_offsets(4096) == (0, 1365, 2048, 4095)


def test_probe_rejects_too_small_arena():
    with pytest.raises(ValueError, match="at least 2"):
        sample_offsets(1)


def test_fabric_xml_reads_only_gpu_fabric_fields():
    root = ET.fromstring(
        """
        <nvidia_smi_log>
          <gpu>
            <display_active>Enabled</display_active>
            <gpu_fabric>
              <state>Completed</state>
              <status>Success</status>
              <clique_id>7</clique_id>
              <cluster_uuid>cluster-a</cluster_uuid>
            </gpu_fabric>
          </gpu>
        </nvidia_smi_log>
        """
    )

    assert _fabric_xml_text(root, "state") == ("Completed",)
    assert _fabric_xml_text(root, "status") == ("Success",)
    assert _fabric_xml_text(root, "cluster_uuid") == ("cluster-a",)


@pytest.mark.parametrize("num_nodes", [2, 16])
def test_probe_request_uses_one_process_per_gpu_and_no_retries(num_nodes: int):
    request = build_probe_request(run_id="test-probe", num_nodes=num_nodes)

    assert request.replicas == num_nodes
    assert request.processes_per_task == PROCESSES_PER_TASK
    assert request.max_retries_failure == 0
    assert request.max_retries_preemption == 0
    assert request.max_task_failures == 0
    assert request.environment is not None
    assert request.environment.extras == ["gpu"]
    assert request.environment.env_vars["TORCH_SYMM_MEM_DISABLE_MULTICAST"] == "1"
    assert request.environment.env_vars["TORCH_SYMMMEM_IMPLICIT_POOL"] == "0"
    callable_entrypoint = request.entrypoint.callable_entrypoint
    assert callable_entrypoint is not None
    config = callable_entrypoint.args[0]
    assert config.expected_world_size == num_nodes * PROCESSES_PER_TASK
