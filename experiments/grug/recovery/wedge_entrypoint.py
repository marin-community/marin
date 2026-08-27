# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Child-side entrypoint that runs the #7344 wedge reproducer under the supervisor.

``GPUHangSupervisor`` spawns this via ``levanter.recovery.child``: it resolves
``run_wedge_repro`` by module + qualname and calls it with the pickled config. The
function brings up ``jax.distributed`` from the Iris task environment (the runtime
does not do it for a raw entrypoint) and then runs the unmodified reproducer.

Nothing here touches the model: the reproducer is verbatim from the standalone
#7344 branch. Detection is entirely environmental — the supervisor sets the XLA
per-execution deadman in this process's ``XLA_FLAGS`` before spawn, so when the
collective wedges the deadman ends the process in ``LOG(FATAL)`` and the
supervisor reads that as a crash. No heartbeat and no snapshot: this run only has
to reproduce the wedge and show the deadman firing.

Imports of the reproducer and the Iris runtime are deferred into the function so
the reproducer's module-level ``_apply_runtime_defaults`` (PGLE, cuda_async
allocator, command-buffer disable) runs before JAX initialises its backend, and
so the child's import graph stays minimal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WedgeReproConfig:
    """Arguments for one supervised reproducer run."""

    dp_racks: int
    num_steps: int


def run_wedge_repro(config: WedgeReproConfig) -> None:
    """Initialise the cross-node mesh, then run the reproducer to a wedge or completion."""
    # Deferred (not top-level) so the reproducer's module-level _apply_runtime_defaults
    # (PGLE / cuda_async allocator / command-buffer env) runs only in this child, not in
    # the CPU coordinator's import graph, and always before the JAX backend init in
    # initialize_jax() below. Matches minrepro_launch, which defers the same imports.
    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

    from experiments.grug.recovery.minimal_wedge_repro import main  # noqa: PLC0415

    initialize_jax()

    argv = [
        "--dp-racks",
        str(config.dp_racks),
        "--num-steps",
        str(config.num_steps),
        "--distributed",
        "off",  # jax.distributed is already up via initialize_jax()
    ]
    logger.info("running wedge reproducer: %s", " ".join(argv))
    returncode = main(argv)
    if returncode:
        raise SystemExit(returncode)
