# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX/XLA hang-detection configuration for a GPU trainer subprocess.

Primary detection is XLA's per-execution deadman
(``--xla_gpu_execution_terminate_timeout``): it is armed at thunk dispatch and
released after stream sync, so it excludes compilation and autotune. On failure
it ends in ``LOG(FATAL)`` (process death), which the supervisor treats as a
recoverable crash. ``--xla_gpu_execution_progress_tracking`` adds kernel-level
attribution of the thunks that were pending when it fired, and
``--xla_gpu_nccl_termination_timeout_seconds`` bounds clique acquisition and the
init rendezvous (init-time hangs only; not in-flight collectives).

``ProgressWatchdog`` (built here via :func:`fallback_watchdog_config`) is the
belt-and-suspenders fallback for the case where the XLA deadman fails to fire. It
uses a fixed ``step_timeout`` only: EMA/dynamic step-time thresholds are unsafe
because PGLE recompiles cause legitimate step-time spikes.

This module is imported by the trainer subprocess (which creates a CUDA context)
and by the supervisor parent (which does not). Importing JAX does not by itself
create a context; the GPU backend initialises lazily on first device use. So
classification and env construction stay free of device work, and only
:func:`apply_in_process_jax_config` touches the live JAX config.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

import jax

from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.recovery.types import FaultClass


logger = logging.getLogger(__name__)


# XLA flag names (without the leading ``--``). See the module docstring for what
# each one bounds.
EXECUTION_TERMINATE_TIMEOUT_FLAG = "xla_gpu_execution_terminate_timeout"
EXECUTION_PROGRESS_TRACKING_FLAG = "xla_gpu_execution_progress_tracking"
NCCL_INIT_TERMINATION_TIMEOUT_FLAG = "xla_gpu_nccl_termination_timeout_seconds"
# Phase-2 recoverability recipe (default off): keep collectives non-blocking and
# async so a failed peer aborts the clique instead of wedging the whole mesh.
NCCL_BLOCKING_COMMUNICATORS_FLAG = "xla_gpu_nccl_blocking_communicators"
NCCL_ASYNC_EXECUTION_FLAG = "xla_gpu_nccl_async_execution"

# PJRT client env vars for the same recoverability recipe.
ABORT_COLLECTIVES_ON_FAILURE_ENV = "XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE"
USE_TFRT_GPU_CLIENT_ENV = "XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT"

# WARN surfaces the abort and transport errors without INFO's per-collective volume.
DEFAULT_NCCL_DEBUG = "WARN"

DEFAULT_EXECUTION_TERMINATE_TIMEOUT_SECONDS = 240.0
DEFAULT_PROGRESS_TRACKING = 8
DEFAULT_NCCL_INIT_TIMEOUT_SECONDS = 120

# The startup grace covers cold compile + autotune before the fallback watchdog
# arms; the diagnostic budget bounds the on-timeout diagnostic thread.
DEFAULT_FALLBACK_STARTUP_GRACE_SECONDS = 3600.0
DEFAULT_FALLBACK_DIAGNOSTIC_TIMEOUT_SECONDS = 20.0

# Substrings (matched case-insensitively) that identify a poisoned CUDA context.
# These faults leave the device unusable, so the supervisor must not warm-restart
# in the same process.
STICKY_SUBSTRINGS = (
    "cuda_error_illegal_instruction",
    "an illegal instruction",
    "uncorrectable",
    "misaligned address",
    "illegal memory access",
    "cuda_error_launch_failed",
)
# Substrings that identify a stall/timeout the supervisor can retry.
STALL_SUBSTRINGS = (
    "deadline_exceeded",
    "stuck in nccl",
    "terminating jobs stuck",
)
# A collective-timeout stall is reported with both words present but not adjacent.
STALL_COMBINED_SUBSTRINGS = ("collective", "timeout")


@dataclass(frozen=True)
class DetectionConfig:
    """XLA/JAX hang-detection knobs for one trainer subprocess.

    Attributes:
        execution_terminate_timeout_seconds: Per-execution deadman budget. Lowered
            into the XLA flag's ``"<int>s"`` duration string.
        progress_tracking: Verbosity of the deadman's pending-thunk attribution.
        nccl_init_timeout_seconds: Bound on clique acquisition / init rendezvous.
        enable_recoverability: Turn on the multi-process recoverability recipe
            (extra NCCL flags, PJRT client env vars, and the in-process
            ``jax_enable_recoverability`` config flag).
    """

    execution_terminate_timeout_seconds: float = DEFAULT_EXECUTION_TERMINATE_TIMEOUT_SECONDS
    progress_tracking: int = DEFAULT_PROGRESS_TRACKING
    nccl_init_timeout_seconds: int = DEFAULT_NCCL_INIT_TIMEOUT_SECONDS
    enable_recoverability: bool = False


def _merge_xla_flag(xla_flags: str, flag: str) -> str:
    """Append a ``--name=value`` flag to ``xla_flags`` unless its name is present.

    Comparison is on the part before ``=``, so a caller-supplied value for the
    same flag name is preserved and never duplicated.
    """
    name = flag.split("=", 1)[0]
    present = {part.split("=", 1)[0] for part in xla_flags.split()}
    if name in present:
        return xla_flags
    return f"{xla_flags} {flag}".strip()


def recovery_xla_env(cfg: DetectionConfig, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a new env dict with the hang-detection flags merged into ``XLA_FLAGS``.

    Copies ``base_env`` (or ``os.environ`` when ``None``) and never mutates its
    input. The deadman, progress-tracking, and NCCL init-timeout flags are merged
    idempotently: a flag name already in ``XLA_FLAGS`` keeps the caller's value.
    When ``cfg.enable_recoverability`` is set, the two ``XLA_PYTHON_CLIENT_*`` env
    vars and the two extra NCCL flags are added as well. ``NCCL_DEBUG`` defaults to
    ``WARN`` when unset.
    """
    env = dict(os.environ if base_env is None else base_env)

    terminate_duration = f"{int(cfg.execution_terminate_timeout_seconds)}s"
    flags = [
        f"--{EXECUTION_TERMINATE_TIMEOUT_FLAG}={terminate_duration}",
        f"--{EXECUTION_PROGRESS_TRACKING_FLAG}={cfg.progress_tracking}",
        f"--{NCCL_INIT_TERMINATION_TIMEOUT_FLAG}={cfg.nccl_init_timeout_seconds}",
    ]
    if cfg.enable_recoverability:
        flags.append(f"--{NCCL_BLOCKING_COMMUNICATORS_FLAG}=false")
        flags.append(f"--{NCCL_ASYNC_EXECUTION_FLAG}=true")
        env.setdefault(ABORT_COLLECTIVES_ON_FAILURE_ENV, "1")
        env.setdefault(USE_TFRT_GPU_CLIENT_ENV, "1")

    xla_flags = env.get("XLA_FLAGS", "")
    for flag in flags:
        xla_flags = _merge_xla_flag(xla_flags, flag)
    env["XLA_FLAGS"] = xla_flags

    env.setdefault("NCCL_DEBUG", DEFAULT_NCCL_DEBUG)
    return env


def apply_in_process_jax_config(cfg: DetectionConfig) -> None:
    """Enable ``jax_enable_recoverability`` in-process when the recipe is on.

    There is no reliable env form for this JAX config flag, so it must be set in
    the process after ``import jax`` and before ``jax.distributed.initialize``.
    No-op when ``cfg.enable_recoverability`` is false.
    """
    if not cfg.enable_recoverability:
        return

    logger.info("Enabling jax_enable_recoverability for multi-process collective recovery")
    jax.config.update("jax_enable_recoverability", True)


def _exception_chain_text(exc: BaseException) -> str:
    """Concatenate message + type name for ``exc`` and its cause/context chain.

    Prefers ``__cause__`` and falls back to ``__context__``; the visited set
    guards against reference cycles in the chain.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        parts.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return "\n".join(parts).lower()


def classify_exception(exc: BaseException) -> FaultClass:
    """Classify a trainer exception as ``STICKY``, ``STALL``, or ``HARD``.

    Walks the ``__cause__``/``__context__`` chain so a poisoned-context error
    wrapped inside a generic outer exception is still recognized. Sticky faults
    take precedence because they leave the CUDA context unusable.
    """
    text = _exception_chain_text(exc)
    if any(needle in text for needle in STICKY_SUBSTRINGS):
        return FaultClass.STICKY
    if any(needle in text for needle in STALL_SUBSTRINGS):
        return FaultClass.STALL
    if all(needle in text for needle in STALL_COMBINED_SUBSTRINGS):
        return FaultClass.STALL
    return FaultClass.HARD


def touch_heartbeat(path: str | os.PathLike, step: int) -> None:
    """Atomically write ``"<step> <time.time()>\\n"`` to ``path``.

    Writes a sibling temp file and ``os.replace``s it onto ``path`` so the
    supervisor can key its deadman off the file mtime and read the last completed
    step from the content without ever observing a torn write.
    """
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".heartbeat-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{step} {time.time()}\n")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def fallback_watchdog_config(
    step_timeout_seconds: float,
    *,
    startup_grace_seconds: float = DEFAULT_FALLBACK_STARTUP_GRACE_SECONDS,
    diagnostic_timeout_seconds: float | None = DEFAULT_FALLBACK_DIAGNOSTIC_TIMEOUT_SECONDS,
) -> ProgressWatchdogConfig:
    """Build the fixed-``step_timeout`` fallback watchdog config.

    Belt-and-suspenders for when the XLA deadman fails to fire. ``process_timeout``
    is left unset; only a fixed per-step deadline is used because PGLE recompiles
    make dynamic step-time thresholds unsafe.
    """
    return ProgressWatchdogConfig(
        step_timeout=timedelta(seconds=step_timeout_seconds),
        process_timeout=None,
        startup_grace_period=timedelta(seconds=startup_grace_seconds),
        diagnostic_timeout=(
            timedelta(seconds=diagnostic_timeout_seconds) if diagnostic_timeout_seconds is not None else None
        ),
    )
