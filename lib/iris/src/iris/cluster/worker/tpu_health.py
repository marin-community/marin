# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU-level bad-node failure detection.

When a task container exits with a non-zero status, the worker normally marks
the task as ``TASK_STATE_FAILED`` (user-code failure). Some failure signatures
are actually signs that the underlying TPU VM is dirty — typically after a
preemption / teardown where ``/dev/vfio`` is still claimed by a previous
process. We need to promote those to ``TASK_STATE_WORKER_FAILED`` so the
controller treats the attempt as an infra preemption and retries it elsewhere.

Patterns are hard-coded on purpose: these signatures are stable strings
emitted by JAX / libtpu during TPU init, and OPS.md already documents them as
the manual trigger list for bad-node triage.
"""

from collections.abc import Iterable

# Substrings matched against container stderr tail. A single hit promotes the
# attempt from FAILED to WORKER_FAILED.
#
# Keep this list in sync with ``lib/iris/OPS.md`` bad-node triggers.
TPU_INIT_FAILURE_PATTERNS: tuple[str, ...] = (
    # /dev/vfio/<n> busy after a dirty preemption — the canonical case from #4783.
    "Couldn't open iommu group",
    "open(/dev/vfio",
    # libtpu / JAX surface when the device is held by another process.
    "Failed to initialize TPU system",
    "TPU initialization failed",
    # Host has no visible accelerator at all (VM came up without TPU attached).
    "No accelerator found",
)


def detect_tpu_init_failure(stderr_lines: Iterable[str]) -> str | None:
    """Return the first matching bad-node pattern found in ``stderr_lines``.

    ``stderr_lines`` is any iterable of stderr strings (typically the tail of
    the container log). Returns ``None`` if no pattern matches.

    Callers should pass a bounded tail (not the full log) — these signatures
    are emitted close to process exit, and scanning the full log wastes work.
    """
    for line in stderr_lines:
        if not line:
            continue
        for pattern in TPU_INIT_FAILURE_PATTERNS:
            if pattern in line:
                return pattern
    return None
