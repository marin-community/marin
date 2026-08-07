# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fault injection for the GPU hang-recovery drills.

The trainer subprocess calls :func:`maybe_inject` once per training step. When the
configured step is reached on the configured JAX process, it injects one of the
:class:`~levanter.recovery.types.FaultKind` faults so the supervisor's
detection-and-recovery path can be exercised end to end.

The fault is selected per subprocess through environment variables. This is the
one place env-var configuration is correct: the ablation/test harness spawns each
trainer subprocess with a different fault wired in, and the subprocess reads its
assignment from the environment.

The pure decision path (:func:`should_fire`) is side-effect-free and touches no
device. Only the injectors call into JAX, and only when a fault actually fires;
importing this module never touches a device, since JAX initialises its GPU
backend lazily on first device use rather than at import.
"""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from levanter.recovery.types import FaultKind

logger = logging.getLogger(__name__)


# Environment variables the harness sets on each trainer subprocess.
ENV_FAULT_KIND = "LEVANTER_FAULT_KIND"
ENV_FAULT_STEP = "LEVANTER_FAULT_STEP"
ENV_FAULT_PROCESS = "LEVANTER_FAULT_PROCESS"

# Defaults when a variable is unset. A step of -1 never matches a real step index,
# so an unconfigured subprocess never fires.
DEFAULT_STEP = -1
DEFAULT_PROCESS = 0

# Cross-module contract: ``detection.classify_exception`` matches this substring to
# classify a raised error as a poisoned-CUDA-context (sticky) fault. Real drivers
# report the same code, so the synthetic message must carry it verbatim.
STICKY_CUDA_ERROR = "CUDA_ERROR_ILLEGAL_INSTRUCTION"


class InjectedStickyFault(RuntimeError):
    """Synthetic sticky CUDA fault raised by the STICKY injector.

    The message carries the :data:`STICKY_CUDA_ERROR` substring so
    ``detection.classify_exception`` recognises it as a poisoned-context fault,
    exactly as it would a real driver-reported sticky error.
    """


@dataclass(frozen=True)
class FaultConfig:
    """A per-subprocess fault assignment.

    Attributes:
        kind: Which fault to inject, or ``NONE`` to disable injection.
        step: The training step index at which to fire.
        process: The ``jax.process_index()`` that should fire; other processes ignore it.
    """

    kind: FaultKind
    step: int
    process: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FaultConfig:
        """Parse a fault assignment from the environment.

        Args:
            env: Mapping to read from; defaults to ``os.environ``.

        Returns:
            The parsed configuration.

        Raises:
            ValueError: If the kind string is not a known ``FaultKind`` or an
                integer field is malformed. Injection is a deliberate ablation
                knob, so a typo fails fast rather than silently disabling it.
        """
        if env is None:
            env = os.environ

        kind_raw = env.get(ENV_FAULT_KIND, FaultKind.NONE.value)
        try:
            kind = FaultKind(kind_raw)
        except ValueError as e:
            valid = ", ".join(k.value for k in FaultKind)
            raise ValueError(
                f"{ENV_FAULT_KIND}={kind_raw!r} is not a valid fault kind; expected one of: {valid}"
            ) from e

        return cls(
            kind=kind,
            step=int(env.get(ENV_FAULT_STEP, str(DEFAULT_STEP))),
            process=int(env.get(ENV_FAULT_PROCESS, str(DEFAULT_PROCESS))),
        )


def should_fire(cfg: FaultConfig, step: int, process_index: int) -> bool:
    """Decide whether a fault fires at this step on this process.

    Pure and side-effect-free: a fault fires only when a kind is configured and
    both the step and process index match the assignment.
    """
    return cfg.kind is not FaultKind.NONE and step == cfg.step and process_index == cfg.process


def maybe_inject(
    step: int,
    *,
    process_index: int | None = None,
    config: FaultConfig | None = None,
) -> None:
    """Per-step hook: inject the configured fault when this step should fire.

    Resolves ``config`` from the environment and ``process_index`` from
    ``jax.process_index()`` when not supplied. JAX is imported lazily, and the
    process index is resolved only when a fault is actually configured, so the
    no-fault path stays JAX-free.

    Depending on the fault kind this may raise, terminate, freeze, or wedge the
    calling process; see the module docstring.
    """
    cfg = config if config is not None else FaultConfig.from_env()

    if cfg.kind is FaultKind.NONE:
        return

    if process_index is None:
        process_index = jax.process_index()

    if not should_fire(cfg, step, process_index):
        return

    if cfg.kind is FaultKind.SIGKILL:
        _inject_sigkill()
    elif cfg.kind is FaultKind.SIGSTOP:
        _inject_sigstop()
    elif cfg.kind is FaultKind.STICKY:
        _inject_sticky()
    elif cfg.kind is FaultKind.SPIN:
        _inject_spin()
    else:
        raise AssertionError(f"unhandled fault kind: {cfg.kind}")


def _inject_sigkill() -> None:
    """Kill this process immediately, modelling abrupt peer death."""
    logger.warning("Injecting SIGKILL fault: killing pid %d (models abrupt peer death).", os.getpid())
    os.kill(os.getpid(), signal.SIGKILL)


def _inject_sigstop() -> None:
    """Freeze this process, modelling a host-side stall the deadman must clear.

    SIGSTOP suspends the process without killing it, so the supervisor's deadman
    timer is the only thing that can eventually reap it.
    """
    logger.warning(
        "Injecting SIGSTOP fault: freezing pid %d (models a host-side stall; the supervisor deadman must kill it).",
        os.getpid(),
    )
    os.kill(os.getpid(), signal.SIGSTOP)


def _inject_sticky() -> None:
    """Raise a synthetic sticky CUDA fault through the sentinel path."""
    logger.warning("Injecting STICKY fault: raising a synthetic poisoned-context error.")
    raise InjectedStickyFault(
        f"Injected fault: {STICKY_CUDA_ERROR}: an illegal instruction was encountered (synthetic)"
    )


def _inject_spin() -> None:
    """Wedge the calling host thread on an unterminating device kernel.

    Compiles a ``while_loop`` whose predicate is unconditionally true and blocks
    the host thread on the result. The loop takes no committed input and creates
    its carry from an on-device constant, so it dispatches on whatever mesh is
    active -- the trainer's full device mesh under ``set_mesh``, or the default
    device otherwise. Pinning the carry to a single device instead raises under
    an explicit multi-device mesh (the jit rejects a one-device argument against
    a many-device context mesh), so we let the ambient mesh place it. This models
    a silent NCCL/GPU hang: the device runs forever and the Python thread never
    returns.

    This function never returns by design.
    """
    logger.warning(
        "Injecting SPIN fault: launching an unterminating kernel on the active device mesh "
        "(models a silent GPU/NCCL hang; this call never returns)."
    )

    def _cond(carry: object) -> object:
        return jnp.bool_(True)

    def _body(carry: object) -> object:
        return carry + jnp.int32(1)

    @jax.jit
    def _spin() -> object:
        return jax.lax.while_loop(_cond, _body, jnp.int32(0))

    jax.block_until_ready(_spin())
