# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU bad-node stderr signatures. Hits promote FAILED -> WORKER_FAILED."""

from collections.abc import Iterable

# Keep in sync with lib/iris/OPS.md bad-node triggers.
TPU_INIT_FAILURE_PATTERNS: tuple[str, ...] = (
    "Couldn't open iommu group",
    "open(/dev/vfio",
    "Failed to initialize TPU system",
    "TPU initialization failed",
    "No accelerator found",
)


def detect_tpu_init_failure(stderr_lines: Iterable[str]) -> str | None:
    """Return the first matching bad-node pattern found in ``stderr_lines``, or None."""
    for line in stderr_lines:
        if not line:
            continue
        for pattern in TPU_INIT_FAILURE_PATTERNS:
            if pattern in line:
                return pattern
    return None
