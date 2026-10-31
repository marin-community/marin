"""Privileged task GPU discovery must stay inside the scheduler's allocation."""

import os

from marinskyrl import gpu_allocation
from marinskyrl.gpu_allocation import pin_task_cuda_devices


def test_plugin_allocation_survives_driver_visibility_reset(tmp_path, monkeypatch):
    for device in ("GPU-assigned-5", "GPU-assigned-2"):
        (tmp_path / device).touch()
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_task_cuda_devices() == "GPU-assigned-2,GPU-assigned-5"
    # The Ray/vLLM driver used to erase this mask and discover the whole host.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    pin_task_cuda_devices()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-assigned-2,GPU-assigned-5"


def test_unbounded_nvidia_setting_keeps_existing_ray_mask(monkeypatch):
    monkeypatch.setattr(gpu_allocation, "NVIDIA_DEVICE_ALLOCATION_DIRECTORY", gpu_allocation.Path("/no-device-plugin"))
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    pin_task_cuda_devices()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_image_void_default_does_not_hide_plugin_allocation(tmp_path, monkeypatch):
    for device in ("GPU-assigned-5", "GPU-assigned-2"):
        (tmp_path / device).touch()
    monkeypatch.setattr(gpu_allocation, "NVIDIA_DEVICE_ALLOCATION_DIRECTORY", tmp_path)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_task_cuda_devices() == "GPU-assigned-2,GPU-assigned-5"
