# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import pytest

from haliax.jax_utils import GpuFamily, device_kind, gpu_family


@dataclass(frozen=True)
class _StubDevice:
    device_kind: str
    compute_capability: str = ""


@pytest.fixture(autouse=True)
def _clear_device_caches():
    gpu_family.cache_clear()
    device_kind.cache_clear()
    yield
    gpu_family.cache_clear()
    device_kind.cache_clear()


def _stub_backend(monkeypatch, backend: str, device: _StubDevice) -> None:
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(jax, "devices", lambda *args, **kwargs: [device])


@pytest.mark.parametrize(
    ("backend", "device", "expected"),
    [
        ("gpu", _StubDevice("AMD Instinct MI355X", "gfx950"), GpuFamily.AMD_INSTINCT),
        ("gpu", _StubDevice("NVIDIA H100 80GB HBM3", "9.0"), GpuFamily.NVIDIA_HOPPER),
        ("gpu", _StubDevice("NVIDIA GB200", "10.0"), GpuFamily.NVIDIA_BLACKWELL),
        ("gpu", _StubDevice("NVIDIA GB10", "12.1"), GpuFamily.OTHER),
        ("gpu", _StubDevice("NVIDIA A100-SXM4-80GB", "8.0"), GpuFamily.OTHER),
        ("cpu", _StubDevice("cpu"), GpuFamily.OTHER),
    ],
)
def test_gpu_family_classifies_device(monkeypatch, backend, device, expected):
    _stub_backend(monkeypatch, backend, device)

    assert gpu_family() is expected
