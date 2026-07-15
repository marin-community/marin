# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free coordinate manifest for the StarCoder WSD80 refinement."""

COARSE_COORDINATES = tuple(
    (p0, p1) for p0 in (0.05, 0.10, 0.15, 0.20, 0.25) for p1 in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
)
FINE_COORDINATES = tuple((p0, p1) for p0 in (0.125, 0.175) for p1 in (0.475, 0.525, 0.575))
LOW_P0_BRIDGE_COORDINATES = tuple((0.025, p1) for p1 in (0.45, 0.50, 0.55, 0.60, 0.65))
BOUNDARY_COORDINATES = ((0.0, 0.45), (0.0, 0.55))
DRIFT_ANCHOR_COORDINATE = (0.1452468603730965, 0.517364768878253)
REFINEMENT_COORDINATES = (
    *COARSE_COORDINATES,
    *FINE_COORDINATES,
    *LOW_P0_BRIDGE_COORDINATES,
    *BOUNDARY_COORDINATES,
)
