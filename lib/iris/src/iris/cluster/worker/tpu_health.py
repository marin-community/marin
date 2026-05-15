# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side stderr signatures that promote FAILED -> WORKER_FAILED.

Covers both TPU init-time bad-node patterns and run-time peer-loss patterns
(JAX-distributed-RPC unavailability, raw-signal-handler aborts) emitted when a
coscheduled sibling disappears mid-step. Run-time peer-loss exits are not
user-code failures, so we route them through the worker-failure budget rather
than counting them against ``max_retries_failure``. See issue #5753.
"""

from collections.abc import Iterable

# Keep in sync with lib/iris/OPS.md bad-node triggers.
TPU_INIT_FAILURE_PATTERNS: tuple[str, ...] = (
    # TPU init-time bad-node signatures.
    "Couldn't open iommu group",
    "open(/dev/vfio",
    "Failed to initialize TPU system",
    "TPU initialization failed",
    "No accelerator found",
    # JAX-distributed-RPC unavailability: a coscheduled sibling went away
    # mid-step, so this worker's RPC client times out. Not user-code failure.
    'grpc_message:"Socket closed"',
    "gRPC Socket closed",
    "Socket closed",
    # JAX raw-signal-handler aborts from peer loss / SIGSEGV / SIGABRT.
    "RAW: Raising signal",
    "Fatal Python error: Aborted",
)


def detect_tpu_init_failure(stderr_lines: Iterable[str]) -> str | None:
    """Return the first matching worker-failure pattern in ``stderr_lines``, or None."""
    for line in stderr_lines:
        if not line:
            continue
        for pattern in TPU_INIT_FAILURE_PATTERNS:
            if pattern in line:
                return pattern
    return None
