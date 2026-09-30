"""Harbor verifier-phase budgets for graders that run in a provider sandbox.

Harbor's verifier timeout (the lowered task.toml's, 600 s by default) is one
wall clock around the whole verifier phase.  For a sandboxed grader that phase
also contains the private sandbox's snapshot lookup, broker placement, host
create (with a cold image pull or snapshot build on that host), preparation and
uploads, and cleanup -- none of which the task author controls or declared.
The grader's own execution is separately bounded by its ContainerRuntime
``timeout`` inside the sandbox, so the phase budget only has to add the
provider's time on top of it.

Measured on catalog-full-construct-003 (2026-09-29, silo): successful sandboxed
verifier phases on shard-082 ran p50 163 s, p90 505 s, max 597 s against the
600 s budget while native verifiers peaked at 77 s; the silo broker's
placement wait alone is up to 480 s and observed sandbox creates took a median
333 s and up to 655 s.  248 runtime gates ended in VerifierTimeoutError, all on
sandboxed verifiers.  One full create (placement wait + host create) plus a
Retry-After is covered by the default provisioning allowance below.
"""

from __future__ import annotations

import math
import os
import tomllib
from pathlib import Path
from typing import Any

HARBOR_DEFAULT_VERIFIER_SECONDS = 600.0
SANDBOX_PROVISIONING_SECONDS = 900
# Preparation, three uploads, the payload, and confirmed cleanup.
SANDBOX_TRANSFER_SECONDS = 120


def provisioning_seconds(environ: dict[str, str] | None = None) -> int:
    """The provisioning allowance; an unusable override keeps the default."""
    environ = os.environ if environ is None else environ
    raw = environ.get("CAPABILITY_VERIFIER_PROVISIONING_SECONDS", "")
    try:
        value = int(raw) if raw.strip() else SANDBOX_PROVISIONING_SECONDS
    except ValueError:
        return SANDBOX_PROVISIONING_SECONDS
    return value if 0 <= value <= 7_200 else SANDBOX_PROVISIONING_SECONDS


def declared_verifier_seconds(package: Path) -> float:
    """The lowered task's own verifier budget (Harbor's default when absent)."""
    try:
        config = tomllib.loads((Path(package) / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return HARBOR_DEFAULT_VERIFIER_SECONDS
    value = (config.get("verifier") or {}).get("timeout_sec")
    declared = [value] if type(value) in (int, float) and math.isfinite(value) and value > 0 else []
    for step in config.get("steps") or []:
        step_value = ((step or {}).get("verifier") or {}).get("timeout_sec")
        if type(step_value) in (int, float) and math.isfinite(step_value) and step_value > 0:
            declared.append(step_value)
    return float(max(declared)) if declared else HARBOR_DEFAULT_VERIFIER_SECONDS


def container_step_seconds(runtime_timeout: float, *, provisioning: int) -> int:
    grader = float(runtime_timeout) if runtime_timeout and math.isfinite(runtime_timeout) else 0.0
    return math.ceil(grader) + provisioning + SANDBOX_TRANSFER_SECONDS


def composite_step_seconds(config: dict[str, Any], step_index: int, *, provisioning: int) -> int:
    """The calibrator's composite budget plus provisioning for each machine check."""
    from .composite_timeout import (
        MACHINE_ORCHESTRATION_SECONDS,
        composite_verifier_timeout,
    )

    checks = config["steps"][step_index]["machine_checks"]
    extra = max(0, provisioning + SANDBOX_TRANSFER_SECONDS - MACHINE_ORCHESTRATION_SECONDS)
    return composite_verifier_timeout(config, step_index) + extra * len(checks)


def verifier_override_seconds(
    package: Path,
    container_timeouts: dict[int, float],
    composite_config: dict[str, Any] | None,
    composite_step_indices: set[int] | frozenset[int] | tuple[int, ...] = (),
    environ: dict[str, str] | None = None,
) -> float | None:
    """Harbor ``verifier.override_timeout_sec`` for a trial, or None if no sandbox.

    Never lower than the task's declared budget; one value covers every step
    because Harbor applies the override to each step's verifier.
    """
    provisioning = provisioning_seconds(environ)
    budgets = [
        container_step_seconds(timeout, provisioning=provisioning)
        for index, timeout in container_timeouts.items()
        if index not in composite_step_indices
    ]
    declared = declared_verifier_seconds(package)
    for index in composite_step_indices if composite_config is not None else ():
        try:
            budgets.append(composite_step_seconds(composite_config, index, provisioning=provisioning))
        except (KeyError, IndexError, TypeError, ValueError):
            # A budget must never stop a gate: fall back to the declared budget
            # plus one provisioning allowance.
            budgets.append(math.ceil(declared) + provisioning + SANDBOX_TRANSFER_SECONDS)
    if not budgets:
        return None
    return float(max(declared, max(budgets)))
