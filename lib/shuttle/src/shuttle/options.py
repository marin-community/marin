# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed compiler options for the Shuttle MLIR pipeline."""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

SCHEMA_VERSION = 1
# Bump when unchanged JSON fields acquire new compiler semantics. This forces a
# distinct JAX/XLA cache identity even when the wire schema itself is stable.
PIPELINE_ABI_VERSION = 9
ENABLE_OPTION = "xla_shuttle_enable"
OPTIONS_OPTION = "xla_shuttle_options"
MAXIMUM_TENSOR_RANK = 8
MAXIMUM_CLUSTER_RANK = 3
MAXIMUM_NATIVE_INTEGER = 2**31 - 1


class Numerics(StrEnum):
    """Numerical rewrite policy used by Shuttle compilation."""

    SOURCE_ORDERED = "source_ordered"
    FAST = "fast"


class ExecutionMode(StrEnum):
    """Output language produced by the Shuttle compiler pipeline."""

    STABLEHLO_ROUND_TRIP = "stablehlo_round_trip"
    CPU_EXECUTABLE_BUNDLE = "cpu_executable_bundle"


class Materialization(StrEnum):
    """Generic preference for intermediate-buffer ownership."""

    AUTOMATIC = "automatic"
    PREFER_FUSION = "prefer_fusion"
    PREFER_MATERIALIZATION = "prefer_materialization"


@dataclass(frozen=True)
class Tuning:
    """Workload-independent bounds for physical candidate search."""

    tile_sizes: tuple[int, ...]
    cluster_shape: tuple[int, ...]
    pipeline_stages: int
    materialization: Materialization
    maximum_candidates: int

    def __post_init__(self) -> None:
        _require_bounded_shape("tile_sizes", self.tile_sizes, maximum_rank=MAXIMUM_TENSOR_RANK)
        _require_bounded_shape("cluster_shape", self.cluster_shape, maximum_rank=MAXIMUM_CLUSTER_RANK)
        _require_native_positive_integer("pipeline_stages", self.pipeline_stages)
        _require_native_positive_integer("maximum_candidates", self.maximum_candidates)
        if type(self.materialization) is not Materialization:
            raise TypeError("materialization must be a Materialization value")


@dataclass(frozen=True)
class Options:
    """Closed public configuration for one Shuttle compilation."""

    numerics: Numerics
    execution_mode: ExecutionMode
    tuning: Tuning

    def __post_init__(self) -> None:
        if type(self.numerics) is not Numerics:
            raise TypeError("numerics must be a Numerics value")
        if type(self.execution_mode) is not ExecutionMode:
            raise TypeError("execution_mode must be an ExecutionMode value")
        if type(self.tuning) is not Tuning:
            raise TypeError("tuning must be a Tuning value")


CompilerOptions = dict[str, bool | str]


def compiler_options(
    *,
    numerics: Numerics,
    tuning: Tuning,
    execution_mode: ExecutionMode = ExecutionMode.STABLEHLO_ROUND_TRIP,
) -> CompilerOptions:
    """Return canonical XLA compiler options for a Shuttle-enabled jaxlib."""
    options = Options(numerics=numerics, execution_mode=execution_mode, tuning=tuning)
    return {
        ENABLE_OPTION: True,
        OPTIONS_OPTION: canonical_options_json(options),
    }


def canonical_options_json(options: Options) -> str:
    """Serialize options into the closed cache-identity wire format."""
    payload = {
        "execution_mode": options.execution_mode.value,
        "numerics": options.numerics.value,
        "pipeline_abi_version": PIPELINE_ABI_VERSION,
        "schema_version": SCHEMA_VERSION,
        "tuning": {
            "cluster_shape": list(options.tuning.cluster_shape),
            "materialization": options.tuning.materialization.value,
            "maximum_candidates": options.tuning.maximum_candidates,
            "pipeline_stages": options.tuning.pipeline_stages,
            "tile_sizes": list(options.tuning.tile_sizes),
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def options_digest(
    *,
    numerics: Numerics,
    tuning: Tuning,
    execution_mode: ExecutionMode = ExecutionMode.STABLEHLO_ROUND_TRIP,
) -> str:
    """Return the stable digest used in compilation evidence."""
    options = Options(numerics=numerics, execution_mode=execution_mode, tuning=tuning)
    return hashlib.sha256(canonical_options_json(options).encode()).hexdigest()


def _require_bounded_shape(name: str, values: tuple[int, ...], *, maximum_rank: int) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    if len(values) > maximum_rank:
        raise ValueError(f"{name} supports at most {maximum_rank} entries")
    for value in values:
        _require_native_positive_integer(f"{name} entry", value)


def _require_native_positive_integer(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 < value <= MAXIMUM_NATIVE_INTEGER:
        raise ValueError(f"{name} must be between 1 and {MAXIMUM_NATIVE_INTEGER}")
