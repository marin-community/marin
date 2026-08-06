# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted calibration and matching artifacts for one-pair expert merging."""

import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import ml_dtypes
import numpy as np
from marin.execution.artifact import Artifact
from rigging.filesystem import StoragePath, prefix_join

from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCalibration,
    ExpertCostMatrix,
    ExpertProbeSet,
    ExpertReservoirCollection,
    ReservoirSample,
)

_CALIBRATION_MANIFEST = "calibration_manifest.json"
_MATCHING_MANIFEST = "matching_manifest.json"
_COST_MATRIX = "cost_matrix.npz"
_ASSIGNMENTS = "assignments.json"
_MATCHING_METRICS = "matching_metrics.json"
_FORMAT_VERSION = 1


class ExpertCalibrationArtifact(Artifact):
    """Per-layer, per-expert routed-state reservoirs."""


class ExpertMatchingArtifact(Artifact):
    """Probe sets, cost matrices, and assignment ablations derived from calibration."""


@dataclass(frozen=True)
class CalibrationArtifactManifest:
    source_checkpoint: str
    source_commit: str | None
    layers: tuple[int, ...]
    num_experts: int
    state_dim: int
    capacity_per_expert: int
    heldout_fraction: float
    calibration_tokens: int
    storage_dtype: str = "bfloat16"
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"layers must be distinct, got {self.layers}")
        if self.num_experts <= 0 or self.state_dim <= 0 or self.capacity_per_expert < 2:
            raise ValueError("calibration dimensions must be positive and capacity must be at least two")
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must lie strictly between zero and one")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        _storage_dtype(self.storage_dtype)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationArtifactManifest":
        format_version = int(payload["format_version"])
        if format_version != _FORMAT_VERSION:
            raise ValueError(f"unsupported calibration format {format_version}; expected {_FORMAT_VERSION}")
        return cls(
            source_checkpoint=str(payload["source_checkpoint"]),
            source_commit=payload.get("source_commit"),
            layers=tuple(int(layer) for layer in payload["layers"]),
            num_experts=int(payload["num_experts"]),
            state_dim=int(payload["state_dim"]),
            capacity_per_expert=int(payload["capacity_per_expert"]),
            heldout_fraction=float(payload["heldout_fraction"]),
            calibration_tokens=int(payload["calibration_tokens"]),
            storage_dtype=str(payload["storage_dtype"]),
            format_version=format_version,
        )


@dataclass(frozen=True)
class MatchingArtifactManifest:
    calibration_path: str
    representative_layer: int
    source_layer: int
    num_experts: int
    eta: float
    assignments: dict[AssignmentMode, tuple[int, ...]]
    prefit_checkpoint: str | None = None
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.representative_layer == self.source_layer:
            raise ValueError("representative_layer and source_layer must differ")
        expected = set(AssignmentMode)
        if set(self.assignments) != expected:
            raise ValueError(f"assignments must contain exactly {sorted(expected)}, got {sorted(self.assignments)}")
        target = np.arange(self.num_experts)
        for mode, assignment in self.assignments.items():
            if not np.array_equal(np.sort(np.asarray(assignment)), target):
                raise ValueError(f"{mode.value} assignment is not a bijection")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assignments"] = {mode.value: list(assignment) for mode, assignment in self.assignments.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MatchingArtifactManifest":
        format_version = int(payload["format_version"])
        if format_version != _FORMAT_VERSION:
            raise ValueError(f"unsupported matching format {format_version}; expected {_FORMAT_VERSION}")
        raw_assignments = payload["assignments"]
        if not isinstance(raw_assignments, Mapping):
            raise ValueError("assignments must be an object")
        return cls(
            calibration_path=str(payload["calibration_path"]),
            representative_layer=int(payload["representative_layer"]),
            source_layer=int(payload["source_layer"]),
            num_experts=int(payload["num_experts"]),
            eta=float(payload["eta"]),
            assignments={
                AssignmentMode(mode): tuple(int(index) for index in assignment)
                for mode, assignment in raw_assignments.items()
            },
            prefit_checkpoint=payload.get("prefit_checkpoint"),
            format_version=format_version,
        )


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _storage_dtype(name: str) -> np.dtype:
    if name == "bfloat16":
        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(name)


def _load_npz(path: str) -> dict[str, np.ndarray]:
    with StoragePath(path).open("rb") as source:
        with np.load(source, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}


def _restore_states(states: np.ndarray, storage_dtype: str) -> np.ndarray:
    if storage_dtype == "bfloat16" and states.dtype.kind == "V":
        states = states.view(ml_dtypes.bfloat16)
    return states.astype(np.float32)


def _expert_path(root: str, layer: int, expert: int) -> str:
    layer_root = prefix_join(prefix_join(root, "layers"), f"layer_{layer:02d}")
    return prefix_join(layer_root, f"expert_{expert:04d}.npz")


def _probe_path(root: str, expert: int) -> str:
    return prefix_join(prefix_join(root, "probes"), f"expert_{expert:04d}.npz")


def write_calibration_artifact(
    output_path: str,
    reservoirs_by_layer: Mapping[int, ExpertReservoirCollection],
    manifest: CalibrationArtifactManifest,
) -> None:
    """Write reservoirs and commit their manifest last."""
    if set(reservoirs_by_layer) != set(manifest.layers):
        raise ValueError("reservoir layers do not match the calibration manifest")
    storage_dtype = _storage_dtype(manifest.storage_dtype)
    for layer in manifest.layers:
        reservoirs = reservoirs_by_layer[layer]
        if (
            reservoirs.num_experts != manifest.num_experts
            or reservoirs.state_dim != manifest.state_dim
            or reservoirs.capacity_per_expert != manifest.capacity_per_expert
        ):
            raise ValueError(f"layer {layer} reservoir geometry does not match the manifest")
        for expert in range(manifest.num_experts):
            calibration = reservoirs.calibration(expert)
            StoragePath(_expert_path(output_path, layer, expert)).write_bytes(
                _npz_bytes(
                    train_states=calibration.train.states.astype(storage_dtype),
                    train_weights=calibration.train.weights.astype(np.float32),
                    heldout_states=calibration.heldout.states.astype(storage_dtype),
                    heldout_weights=calibration.heldout.weights.astype(np.float32),
                )
            )
    StoragePath(prefix_join(output_path, _CALIBRATION_MANIFEST)).write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True)
    )


def read_calibration_manifest(path: str) -> CalibrationArtifactManifest:
    payload = json.loads(StoragePath(prefix_join(path, _CALIBRATION_MANIFEST)).read_text())
    return CalibrationArtifactManifest.from_dict(payload)


def read_expert_calibration(
    path: str,
    layer: int,
    expert: int,
    *,
    manifest: CalibrationArtifactManifest | None = None,
) -> ExpertCalibration:
    """Read one expert reservoir, reusing a caller-cached manifest when supplied."""
    if manifest is None:
        manifest = read_calibration_manifest(path)
    if layer not in manifest.layers:
        raise ValueError(f"layer {layer} is not present in calibration artifact")
    if not 0 <= expert < manifest.num_experts:
        raise IndexError(f"expert must lie in [0, {manifest.num_experts}), got {expert}")
    arrays = _load_npz(_expert_path(path, layer, expert))
    return ExpertCalibration(
        train=ReservoirSample(
            states=_restore_states(arrays["train_states"], manifest.storage_dtype),
            weights=arrays["train_weights"].astype(np.float64),
        ),
        heldout=ReservoirSample(
            states=_restore_states(arrays["heldout_states"], manifest.storage_dtype),
            weights=arrays["heldout_weights"].astype(np.float64),
        ),
    )


def write_matching_artifact(
    output_path: str,
    probes: tuple[ExpertProbeSet, ...],
    costs: ExpertCostMatrix,
    manifest: MatchingArtifactManifest,
) -> None:
    """Write reusable probes, costs, assignments, then commit the manifest last."""
    if len(probes) != manifest.num_experts:
        raise ValueError(f"expected {manifest.num_experts} probe sets, got {len(probes)}")
    expected_cost_shape = (manifest.num_experts, manifest.num_experts)
    if any(matrix.shape != expected_cost_shape for matrix in (costs.native, costs.tangent, costs.total)):
        raise ValueError(f"cost matrices must all have shape {expected_cost_shape}")
    for expert, probe in enumerate(probes):
        StoragePath(_probe_path(output_path, expert)).write_bytes(
            _npz_bytes(
                ordinary_inputs=probe.ordinary_inputs,
                ordinary_weights=probe.ordinary_weights,
                centers=probe.centers,
                spectral_pairs=probe.spectral_pairs,
                input_directions=probe.input_directions,
                sensitivity_eigenvalues=probe.sensitivity_eigenvalues,
            )
        )
    StoragePath(prefix_join(output_path, _COST_MATRIX)).write_bytes(
        _npz_bytes(native=costs.native, tangent=costs.tangent, total=costs.total)
    )
    StoragePath(prefix_join(output_path, _ASSIGNMENTS)).write_text(
        json.dumps(
            {mode.value: list(assignment) for mode, assignment in manifest.assignments.items()},
            indent=2,
            sort_keys=True,
        )
    )
    assignment_metrics = {}
    for mode, assignment in manifest.assignments.items():
        matrix = costs.native if mode is AssignmentMode.NATIVE else costs.total
        selected = matrix[np.arange(manifest.num_experts), np.asarray(assignment)]
        counts, edges = np.histogram(selected, bins=min(20, manifest.num_experts))
        assignment_metrics[mode.value] = {
            "cost_mean": float(np.mean(selected)),
            "cost_histogram": {
                "counts": counts.tolist(),
                "bin_edges": edges.tolist(),
            },
        }
    StoragePath(prefix_join(output_path, _MATCHING_METRICS)).write_text(
        json.dumps(
            {
                "format_version": _FORMAT_VERSION,
                "merge/native_cost_mean": float(np.mean(costs.native)),
                "merge/spectral_cost_mean": float(np.mean(costs.total)),
                "merge/assignment_cost_histogram": assignment_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    StoragePath(prefix_join(output_path, _MATCHING_MANIFEST)).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )


def read_matching_manifest(path: str) -> MatchingArtifactManifest:
    payload = json.loads(StoragePath(prefix_join(path, _MATCHING_MANIFEST)).read_text())
    return MatchingArtifactManifest.from_dict(payload)


def read_expert_probe(path: str, expert: int) -> ExpertProbeSet:
    manifest = read_matching_manifest(path)
    if not 0 <= expert < manifest.num_experts:
        raise IndexError(f"expert must lie in [0, {manifest.num_experts}), got {expert}")
    arrays = _load_npz(_probe_path(path, expert))
    return ExpertProbeSet(
        ordinary_inputs=arrays["ordinary_inputs"],
        ordinary_weights=arrays["ordinary_weights"],
        centers=arrays["centers"],
        spectral_pairs=arrays["spectral_pairs"],
        input_directions=arrays["input_directions"],
        sensitivity_eigenvalues=arrays["sensitivity_eigenvalues"],
    )


def read_cost_matrix(path: str) -> ExpertCostMatrix:
    arrays = _load_npz(prefix_join(path, _COST_MATRIX))
    return ExpertCostMatrix(native=arrays["native"], tangent=arrays["tangent"], total=arrays["total"])


def read_matching_metrics(path: str) -> dict[str, Any]:
    return json.loads(StoragePath(prefix_join(path, _MATCHING_METRICS)).read_text())


__all__ = [
    "CalibrationArtifactManifest",
    "ExpertCalibrationArtifact",
    "ExpertMatchingArtifact",
    "MatchingArtifactManifest",
    "read_calibration_manifest",
    "read_cost_matrix",
    "read_expert_calibration",
    "read_expert_probe",
    "read_matching_manifest",
    "read_matching_metrics",
    "write_calibration_artifact",
    "write_matching_artifact",
]
