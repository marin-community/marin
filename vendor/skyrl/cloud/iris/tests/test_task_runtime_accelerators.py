# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for partial-host privileged NVIDIA jobs."""

from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

from cloud.iris import task_runtime


GPU_UUIDS = tuple(f"GPU-00000000-0000-0000-0000-{index:012x}" for index in range(8))
HOST_INVENTORY = tuple(f"00000000:{16 * (index + 1):02x}:00.0, {uid}" for index, uid in enumerate(GPU_UUIDS))
MEGATRON_SINGLE_RANK = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "expert_tensor_parallel_size": 1,
}


@pytest.fixture
def partial_host(monkeypatch, tmp_path):
    monkeypatch.setattr(task_runtime, "_NVIDIA_DEVICE_MOUNT", tmp_path)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", str(tmp_path))
    monkeypatch.setenv("IRIS_NUM_TASKS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    for uid in GPU_UUIDS[4:6]:
        (tmp_path / uid).touch()
    return OmegaConf.create(
        {
            "iris": {
                "cluster": "marin",
                "target_cluster": "cw-us-east-02a",
                "container_profile": "CONTAINER_PROFILE_PRIVILEGED",
                "allocation": {"num_nodes": 1, "gpus_per_node": 2, "gpu_variant": "H100"},
            },
            "run": {"mode": "train"},
            "runtime": {"profile": "megatron"},
            "skyrl": {
                "generator": {"backend": "vllm", "run_engines_locally": True},
                "trainer": {role: {"megatron_config": MEGATRON_SINGLE_RANK} for role in ("policy", "ref")},
            },
        }
    )


def test_gpu_mask_uses_unmasked_host_inventory_in_pci_order(partial_host, monkeypatch):
    query = Mock(return_value="\n".join(reversed(HOST_INVENTORY)))
    monkeypatch.setattr(task_runtime.subprocess, "check_output", query)
    task_runtime.pin_allocated_nvidia_gpus(partial_host)
    assert query.call_count == 1
    assert "CUDA_VISIBLE_DEVICES" not in query.call_args.kwargs["env"]
    assert task_runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert task_runtime.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (HOST_INVENTORY[4:6], "8 East02 H100 devices"),
        (
            (*HOST_INVENTORY[:5], f"00000000:60:00.0, {GPU_UUIDS[4]}", *HOST_INVENTORY[6:]),
            "repeats a PCI address or GPU UUID",
        ),
        (
            (*HOST_INVENTORY[:-1], f"0000:10:00.0, {GPU_UUIDS[-1]}"),
            "repeats a PCI address or GPU UUID",
        ),
        (
            (*HOST_INVENTORY[:5], f"00000000:60:00.0, GPU-ffffffff-ffff-ffff-ffff-ffffffffffff", *HOST_INVENTORY[6:]),
            "allocation does not match the host inventory",
        ),
        ((*HOST_INVENTORY[:-1], f"bad-bus, {GPU_UUIDS[-1]}"), "invalid PCI address or GPU UUID"),
    ],
    ids=["filtered-inventory", "duplicate-uuid", "duplicate-bdf", "missing-allocation", "invalid-bdf"],
)
def test_invalid_gpu_inventory_preserves_existing_mask(partial_host, monkeypatch, rows, error):
    monkeypatch.setattr(task_runtime.subprocess, "check_output", Mock(return_value="\n".join(rows)))
    with pytest.raises(RuntimeError, match=error):
        task_runtime.pin_allocated_nvidia_gpus(partial_host)
    assert task_runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert task_runtime.os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_gpu_mount_must_match_reserved_count(partial_host, tmp_path, monkeypatch):
    (tmp_path / GPU_UUIDS[5]).unlink()
    query = Mock()
    monkeypatch.setattr(task_runtime.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="reserved 2 GPUs.*mounted 1"):
        task_runtime.pin_allocated_nvidia_gpus(partial_host)
    query.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iris.target_cluster", "cw-us-east-08a"),
        ("iris.container_profile", "CONTAINER_PROFILE_DEFAULT"),
        ("iris.allocation.num_nodes", 2),
        ("iris.allocation.gpus_per_node", 8),
        ("iris.allocation.gpus_per_node", 0),
        ("skyrl.generator.run_engines_locally", False),
        ("skyrl.generator.inference_engine_tensor_parallel_size", 2),
        ("skyrl.generator.inference_engine_pipeline_parallel_size", 2),
        ("skyrl.generator.inference_engine_data_parallel_size", 2),
        ("skyrl.generator.inference_engine_expert_parallel_size", 2),
        ("skyrl.trainer.policy.megatron_config.tensor_model_parallel_size", 2),
        ("skyrl.trainer.ref.megatron_config.tensor_model_parallel_size", 2),
    ],
)
def test_other_topologies_keep_their_original_device_environment(partial_host, monkeypatch, field, value):
    OmegaConf.update(partial_host, field, value)
    query = Mock()
    monkeypatch.setattr(task_runtime.subprocess, "check_output", query)
    task_runtime.pin_allocated_nvidia_gpus(partial_host)
    query.assert_not_called()
    assert task_runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert task_runtime.os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


@pytest.mark.parametrize(
    ("env", "value"), [("IRIS_NUM_TASKS", "2"), ("NVIDIA_VISIBLE_DEVICES", GPU_UUIDS[4])]
)
def test_different_runtime_or_device_plugin_mode_keeps_existing_mask(partial_host, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    query = Mock()
    monkeypatch.setattr(task_runtime.subprocess, "check_output", query)
    task_runtime.pin_allocated_nvidia_gpus(partial_host)
    query.assert_not_called()
    assert task_runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
