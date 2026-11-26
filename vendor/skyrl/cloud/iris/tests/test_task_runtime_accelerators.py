# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for partial-host privileged NVIDIA jobs."""

import pytest

from cloud.iris import task_runtime


def test_gpu_mask_follows_reserved_device_pci_order(tmp_path, monkeypatch):
    monkeypatch.setattr(task_runtime, "_NVIDIA_DEVICE_MOUNT", tmp_path)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    (tmp_path / "GPU-third").touch()
    (tmp_path / "GPU-fourth").touch()
    listing = """00000000:af:00.0, GPU-fourth
00000000:01:00.0, GPU-first
00000000:88:00.0, GPU-third
00000000:17:00.0, GPU-second
"""
    monkeypatch.setattr(task_runtime.subprocess, "check_output", lambda *args, **kwargs: listing)
    task_runtime.pin_allocated_nvidia_gpus(2)
    assert task_runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert task_runtime.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_wrong_gpu_reservation_cannot_fall_back_to_unrestricted_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(task_runtime, "_NVIDIA_DEVICE_MOUNT", tmp_path)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", str(tmp_path))
    (tmp_path / "GPU-missing").touch()
    monkeypatch.setattr(task_runtime.subprocess, "check_output", lambda *args, **kwargs: "00000000:af:00.0, GPU-other")
    with pytest.raises(RuntimeError, match="allocation does not match"):
        task_runtime.pin_allocated_nvidia_gpus(1)
