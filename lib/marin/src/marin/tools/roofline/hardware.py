# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Editable hardware registry for roofline estimates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class Hardware:
    name: str
    devices_per_host: int
    bf16_peak_tflops_per_device: float
    fp32_peak_tflops_per_device: float
    hbm_bandwidth_gbps_per_device: float
    intra_host_collective_bandwidth_gbps: float
    inter_host_collective_bandwidth_gbps: float
    default_compute_efficiency: dict[str, float]
    default_comm_efficiency: dict[str, float]
    provenance: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Hardware:
        return cls(
            name=str(payload["name"]),
            devices_per_host=int(payload["devices_per_host"]),
            bf16_peak_tflops_per_device=float(payload["bf16_peak_tflops_per_device"]),
            fp32_peak_tflops_per_device=float(payload["fp32_peak_tflops_per_device"]),
            hbm_bandwidth_gbps_per_device=float(payload["hbm_bandwidth_gbps_per_device"]),
            intra_host_collective_bandwidth_gbps=float(payload["intra_host_collective_bandwidth_gbps"]),
            inter_host_collective_bandwidth_gbps=float(payload["inter_host_collective_bandwidth_gbps"]),
            default_compute_efficiency={str(k): float(v) for k, v in payload["default_compute_efficiency"].items()},
            default_comm_efficiency={str(k): float(v) for k, v in payload["default_comm_efficiency"].items()},
            provenance=str(payload["provenance"]) if payload.get("provenance") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "devices_per_host": self.devices_per_host,
            "bf16_peak_tflops_per_device": self.bf16_peak_tflops_per_device,
            "fp32_peak_tflops_per_device": self.fp32_peak_tflops_per_device,
            "hbm_bandwidth_gbps_per_device": self.hbm_bandwidth_gbps_per_device,
            "intra_host_collective_bandwidth_gbps": self.intra_host_collective_bandwidth_gbps,
            "inter_host_collective_bandwidth_gbps": self.inter_host_collective_bandwidth_gbps,
            "default_compute_efficiency": dict(self.default_compute_efficiency),
            "default_comm_efficiency": dict(self.default_comm_efficiency),
            "provenance": self.provenance,
        }


def load_hardware_registry() -> dict[str, Hardware]:
    registry_text = resources.files(__package__).joinpath("default_hardware.json").read_text(encoding="utf-8")
    payload = json.loads(registry_text)
    return {record["name"]: Hardware.from_dict(record) for record in payload["hardware"]}


def hardware_by_name(name: str) -> Hardware:
    registry = load_hardware_registry()
    try:
        return registry[name]
    except KeyError:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"Unknown hardware preset '{name}'. Available presets: {choices}") from None
