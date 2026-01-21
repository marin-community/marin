# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Union, Sequence, Tuple, Mapping, Dict

from draccus import field

from haliax.partitioning import ResourceMapping
import jax
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh

DEFAULT_DP_AXES = ("replica_dcn", "replica", "data")
DEFAULT_ICI_AXIS_SPEC = {"data": -1, "replica": 1, "model": 1}
DEFAULT_DCN_AXIS_SPEC = {"replica_dcn": -1}
DEFAULT_SHARED_MAPPING: Dict[str, str | Tuple[str, ...]] = {"mlp": "model", "heads": "model"}


@dataclass(frozen=True)
class MeshConfig:
    """
    Defines mesh axes and logical-to-physical mappings.
    axes: ICI sizes per axis (within a slice). -1 means absorb remaining ICI.
    dcn_axes: DCN sizes per axis (across slices). -1 means absorb remaining DCN.
    shared_mapping: common logical-axis defaults shared by both compute and parameter sharding.
    compute_mapping: logical -> physical axis (or axes) for compute (e.g., batch -> [replica_dcn, replica, data]).
    param_mapping: logical -> physical axis (or axes) for parameters and opt states
    """

    axes: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_ICI_AXIS_SPEC))
    dcn_axes: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_DCN_AXIS_SPEC))

    # Typically you should only set these fields in config, and only read the resolved_* properties.
    batch_axis_name: str | None = "batch"
    shared_mapping: Mapping[str, list[str] | str] = field(default_factory=lambda: {})
    compute_mapping: Mapping[str, list[str] | str] = field(default_factory=lambda: {})
    param_mapping: Mapping[str, list[str] | str] = field(default_factory=lambda: {"embed": "data"})

    @cached_property
    def resolved_compute_mapping(self) -> ResourceMapping:
        """
        Resolves the compute mapping by combining shared mappings and compute-specific mappings.
        """
        mapping = self._resolved_shared_axis_mapping()

        if self.batch_axis_name is not None and self.batch_axis_name not in mapping:
            mapping[self.batch_axis_name] = _norm(self.compute_mapping.get("batch", DEFAULT_DP_AXES))

        for logical, physical in self.compute_mapping.items():
            mapping[logical] = _norm(physical)

        return mapping

    @cached_property
    def resolved_param_mapping(self) -> ResourceMapping:
        # Parameter mapping should inherit shared defaults so parameters on those logical axes shard the
        # same way as compute unless explicitly overridden.
        mapping = self._resolved_shared_axis_mapping()
        for logical, physical in self.param_mapping.items():
            mapping[logical] = _norm(physical)
        return mapping

    def _resolved_shared_axis_mapping(self):
        mapping = dict(DEFAULT_SHARED_MAPPING)
        for logical, physical in self.shared_mapping.items():
            mapping[logical] = _norm(physical)

        return mapping

    def axis_shapes(self, num_devices: int, num_slices: int) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Computes the ICI and DCN axis sizes based on the configuration. num_devices is the total number of devices,
        which are split over num_slices slices.
        """
        if num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if num_devices % num_slices != 0:
            raise ValueError(f"num_devices ({num_devices}) must be divisible by num_slices ({num_slices})")

        per_slice = num_devices // num_slices

        default_axes = {"data": -1, "replica": 1, "model": 1}
        default_dcn_axes = {"replica_dcn": -1}

        axes = dict(default_axes)
        axes.update(self.axes)

        absorbers = [k for k, v in axes.items() if v == -1]
        if len(absorbers) > 1 and "data" in axes and "data" not in self.axes:
            axes["data"] = 1

        dcn_axes = dict(default_dcn_axes)
        dcn_axes.update(self.dcn_axes)

        dcn_absorbers = [k for k, v in dcn_axes.items() if v == -1]
        if len(dcn_absorbers) > 1 and "replica_dcn" in dcn_axes and "replica_dcn" not in self.dcn_axes:
            dcn_axes["replica_dcn"] = 1

        if set(axes.keys()) & set(dcn_axes.keys()):
            overlap = set(axes.keys()) & set(dcn_axes.keys())
            raise ValueError(f"Axis names cannot appear in both axes and dcn_axes: {sorted(overlap)}")

        unknown_ici = [n for n, v in axes.items() if v == -1]
        unknown_dcn = [n for n, v in dcn_axes.items() if v == -1]

        if len(unknown_ici) > 1:
            raise ValueError("Only one axis may have ici = -1.")
        if len(unknown_dcn) > 1:
            raise ValueError("Only one axis may have dcn = -1.")

        known_ici = prod(v for v in axes.values() if v != -1)
        known_dcn = prod(v for v in dcn_axes.values() if v != -1)

        if unknown_dcn:
            remaining = num_slices // known_dcn
            if remaining * known_dcn != num_slices:
                raise ValueError(f"DCN product {known_dcn} does not divide num_slices {num_slices}.")
            dcn_axes[unknown_dcn[0]] = remaining
            known_dcn *= remaining
        else:
            if known_dcn != num_slices:
                raise ValueError(f"DCN product {known_dcn} must equal num_slices {num_slices}.")

        if unknown_ici:
            remaining = per_slice // known_ici
            if remaining * known_ici != per_slice:
                raise ValueError(f"ICI product {known_ici} does not divide devices_per_slice {per_slice}.")
            axes[unknown_ici[0]] = remaining
            known_ici *= remaining
        else:
            if known_ici != per_slice:
                raise ValueError(f"ICI product {known_ici} must equal devices_per_slice {per_slice}.")

        return axes, dcn_axes


def create_mesh_from_axis_specs(
    *,
    ici_axes: Mapping[str, int],
    dcn_axes: Mapping[str, int],
    devices=None,
    allow_split_physical_axes: bool = True,
    axis_types: tuple[AxisType, ...] | None = None,
) -> Mesh:
    """
    Create a JAX mesh from ICI and DCN axis sizes. Supports both single-slice and multi-slice layouts.
    """
    axis_names = list(ici_axes.keys()) + [k for k in dcn_axes.keys() if k not in ici_axes]
    if not axis_names:
        raise ValueError("At least one axis is required to build a mesh.")
    overlapping = set(ici_axes.keys()) & set(dcn_axes.keys())
    if overlapping:
        raise ValueError(f"Axis names cannot appear in both ICI and DCN: {sorted(overlapping)}")

    if devices is None:
        devices = jax.devices()
    devices = list(devices)
    if not devices:
        raise ValueError("No devices available to build a mesh.")

    ici_mesh_shape = tuple(ici_axes.get(name, 1) for name in axis_names)
    dcn_mesh_shape = tuple(dcn_axes.get(name, 1) for name in axis_names)

    is_multislice = hasattr(devices[0], "slice_index")
    if is_multislice:
        device_mesh = mesh_utils.create_hybrid_device_mesh(
            mesh_shape=ici_mesh_shape,
            dcn_mesh_shape=dcn_mesh_shape,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        if any(d != 1 for d in dcn_mesh_shape):
            raise ValueError("Non-trivial DCN axis sizes require multi-slice hardware.")
        device_mesh = mesh_utils.create_device_mesh(
            ici_mesh_shape,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )

    if axis_types is not None and len(axis_types) != len(axis_names):
        raise ValueError(f"axis_types must match axis_names length: {len(axis_types)} != {len(axis_names)}")

    return Mesh(device_mesh, tuple(axis_names), axis_types=axis_types)


def _norm(v: Union[str, Sequence[str]]) -> Union[str, Tuple[str, ...]]:
    if isinstance(v, (list, tuple)):
        v = tuple(v)
        return v if len(v) > 1 else v[0]
    return v  # type: ignore[bad-return]
