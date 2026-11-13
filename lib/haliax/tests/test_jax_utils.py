# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import pytest

from haliax.jax_utils import GpuFamily, device_kind, gpu_family, is_amd_instinct


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


def _stub_backend(monkeypatch, backend: str, device: _StubDevice) -> list[str]:
    """Stub the default backend and device list; returns the log of ``jax.devices`` calls."""
    calls: list[str] = []

    def devices(*args, **kwargs):
        calls.append(args[0] if args else "default")
        return [device]

    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(jax, "devices", devices)
    return calls


@pytest.mark.parametrize(
    ("backend", "device", "expected"),
    [
        ("gpu", _StubDevice("AMD Instinct MI355X", "gfx950"), GpuFamily.AMD_INSTINCT),
        ("gpu", _StubDevice("AMD Instinct MI300X", "gfx942"), GpuFamily.AMD_INSTINCT),
        ("gpu", _StubDevice("NVIDIA H100 80GB HBM3", "9.0"), GpuFamily.NVIDIA_HOPPER),
        ("gpu", _StubDevice("NVIDIA GB200", "10.0"), GpuFamily.NVIDIA_BLACKWELL),
        ("gpu", _StubDevice("NVIDIA GB10", "12.1"), GpuFamily.OTHER),
        ("gpu", _StubDevice("NVIDIA A100-SXM4-80GB", "8.0"), GpuFamily.OTHER),
        ("tpu", _StubDevice("TPU v5 lite"), GpuFamily.OTHER),
        ("cpu", _StubDevice("cpu"), GpuFamily.OTHER),
    ],
)
def test_gpu_family_classifies_device(monkeypatch, backend, device, expected):
    _stub_backend(monkeypatch, backend, device)

    assert gpu_family() is expected
    assert is_amd_instinct() == (expected is GpuFamily.AMD_INSTINCT)


def test_gpu_family_and_device_kind_are_cached(monkeypatch):
    calls = _stub_backend(monkeypatch, "gpu", _StubDevice("AMD Instinct MI355X", "gfx950"))
    assert gpu_family() is GpuFamily.AMD_INSTINCT
    assert device_kind() == "AMD Instinct MI355X"

    _stub_backend(monkeypatch, "gpu", _StubDevice("NVIDIA H100 80GB HBM3", "9.0"))

    assert gpu_family() is GpuFamily.AMD_INSTINCT
    assert device_kind() == "AMD Instinct MI355X"
    assert calls == ["gpu", "default"]


def test_device_kind_is_empty_without_devices(monkeypatch):
    monkeypatch.setattr(jax, "devices", lambda *args, **kwargs: [])

    assert device_kind() == ""
