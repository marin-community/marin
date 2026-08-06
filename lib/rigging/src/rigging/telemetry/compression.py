# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compression policy for client payloads sent to Finelog."""

# Level 1 retains nearly all of zstd's default-level ratio for telemetry-shaped
# payloads while keeping compression off application hot paths.
FINELOG_ZSTD_LEVEL = 1
