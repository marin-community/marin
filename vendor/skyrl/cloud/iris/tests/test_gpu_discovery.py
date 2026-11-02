"""CUDA discovery follows the task's device-plugin allocation."""

import os

from marinskyrl import gpu_discovery
from marinskyrl.gpu_discovery import constrain_cuda_device_discovery


def test_task_allocation_survives_vllm_driver_visibility_reset(tmp_path, monkeypatch):
    for device in ("GPU-assigned-5", "GPU-assigned-2"):
        (tmp_path / device).touch()
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert constrain_cuda_device_discovery() == "GPU-assigned-2,GPU-assigned-5"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    constrain_cuda_device_discovery()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-assigned-2,GPU-assigned-5"


def test_unbounded_nvidia_setting_preserves_existing_ray_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gpu_discovery,
        "NVIDIA_DEVICE_ALLOCATION_DIRECTORY",
        tmp_path / "missing-device-plugin-allocation",
    )
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    assert constrain_cuda_device_discovery() is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_device_plugin_allocation_overrides_image_void_default(tmp_path, monkeypatch):
    for device in ("GPU-assigned-5", "GPU-assigned-2"):
        (tmp_path / device).touch()
    monkeypatch.setattr(gpu_discovery, "NVIDIA_DEVICE_ALLOCATION_DIRECTORY", tmp_path)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert constrain_cuda_device_discovery() == "GPU-assigned-2,GPU-assigned-5"
