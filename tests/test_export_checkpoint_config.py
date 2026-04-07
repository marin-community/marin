# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for default resources used by the HF checkpoint export step."""

from fray.v1.cluster import CpuConfig

from marin.export.levanter_checkpoint import ConvertCheckpointStepConfig, _default_export_resources


def _parse_ram_gb(ram: str) -> float:
    ram = ram.strip().lower()
    if ram.endswith("g"):
        return float(ram[:-1])
    if ram.endswith("m"):
        return float(ram[:-1]) / 1024
    raise ValueError(f"Unsupported ram spec: {ram}")


def test_default_export_resources_have_enough_ram_for_large_models():
    # Levanter's `_restore_old_ts` uses concurrent_gb=300 during deserialization, and the export
    # step must additionally hold the full model in host RAM. Defaults must leave headroom for
    # 50B+ models to avoid OOM (see issue #4475).
    resources = _default_export_resources()
    assert isinstance(resources.device, CpuConfig)
    assert _parse_ram_gb(resources.ram) >= 300
    assert _parse_ram_gb(resources.disk) >= 200


def test_convert_checkpoint_step_config_uses_large_default_resources():
    config = ConvertCheckpointStepConfig(
        checkpoint_path="gs://fake/ckpt",
        trainer=None,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
    )
    assert _parse_ram_gb(config.resources.ram) >= 300
