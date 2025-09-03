"""Learner actor for Marin RL.

This provides a minimal asynchronous training loop interface that integrates
with the Orchestrator and the WeightTransferCoordinator. It mirrors the shape
of the synchronous example in ``simple_train.py`` but defers heavy training
details to future work.

API expected by Orchestrator:
- ``start()``: begins the learner loop
- ``get_step()``: returns current global step
- ``get_weight_broadcaster()``: returns the WeightTransferCoordinator actor

Notes
-----
This is intentionally lightweight so the Orchestrator can wire components
without requiring a full training stack. It tracks steps, emits basic metrics
via the Orchestrator, and exposes a weight broadcaster handle. The actual
model training and sampling from replay buffers will be implemented in a
subsequent iteration once the dataset path is finalized.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import ray
from ray.actor import ActorHandle

from .config import RlTrainingConfig
from .weight_transfer import WeightTransferCoordinator, instantiate_coordinator, start_transfer_server


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LoopConfig:
    """Internal loop controls for the minimal learner."""

    # How long to sleep per step to simulate work (seconds)
    step_sleep_s: float = 0.0
    # How often to publish a dummy weight version (steps); 0 disables
    publish_every_steps: int = 0


@ray.remote
class Learner:
    """Minimal Ray-based learner actor.

    Parameters
    ----------
    training_cfg:
        Basic training settings (steps, batch size). Mirrors RlTrainingConfig.
    orchestrator:
        Optional Orchestrator actor handle for logging metrics and step updates.
    loop_cfg:
        Optional controls for pacing and periodic publishing.
    """

    def __init__(
        self,
        training_cfg: RlTrainingConfig,
        *,
        orchestrator: Optional[ActorHandle] = None,
        loop_cfg: Optional[_LoopConfig] = None,
    ) -> None:
        self._training_cfg = training_cfg
        self._orchestrator = orchestrator
        self._loop_cfg = loop_cfg or _LoopConfig()

        self._step: int = 0
        self._running: bool = False

        # Weight transfer server + coordinator
        # In this minimal version, we start a local transfer server and create a coordinator
        # bound to this node so inference servers can pull weights later.
        self._transfer_server = start_transfer_server()
        self._broadcaster: ActorHandle = instantiate_coordinator(self._transfer_server)

        logger.info("Learner initialized: steps=%d batch_size=%d", training_cfg.num_steps, training_cfg.batch_size)

    # ------------------------------------------------------------------
    # Orchestrator API
    # ------------------------------------------------------------------

    def get_step(self) -> int:
        return self._step

    def get_weight_broadcaster(self) -> Optional[ActorHandle]:
        """Return the WeightTransferCoordinator actor, if available."""
        return self._broadcaster

    # ------------------------------------------------------------------
    # Training loop (minimal placeholder)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run a minimal async training loop.

        This loop increments the global step, emits basic metrics via the
        Orchestrator, and periodically publishes a new weight version if a
        broadcaster is available. Actual model updates will be integrated later.
        """
        if self._running:
            logger.info("Learner already running; ignoring start().")
            return

        self._running = True
        total_steps = int(self._training_cfg.num_steps)
        sleep_s = max(0.0, float(self._loop_cfg.step_sleep_s))
        publish_every = max(0, int(self._loop_cfg.publish_every_steps))

        logger.info("Learner loop starting for %d steps", total_steps)

        for step in range(self._step, total_steps):
            self._step = step

            # Simulate some work if requested
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            # Report step and dummy metrics
            if self._orchestrator is not None:
                ray.get(self._orchestrator.mark_step.remote(step))  # type: ignore[call-arg]
                self._orchestrator.log_metrics.remote(
                    {
                        "train/step": float(step),
                        # Placeholder metrics for now
                        "train/examples_per_batch": float(self._training_cfg.batch_size),
                    }
                )

            # Periodic publish hook
            if publish_every and step % publish_every == 0 and step > 0:
                try:
                    # In a full implementation, we would call
                    # process_weight_transfers(self._transfer_server, self._broadcaster, latest_weight_id, params)
                    # Here we simply nudge latest_weight_id forward by step.
                    latest_weight_id = step
                    # Fire-and-forget: the coordinator will accept the latest ID; the server would await pulls.
                    # We don't block here to keep the minimal loop simple.
                    self._broadcaster.poll_transfers.remote(latest_weight_id)  # type: ignore[attr-defined]
                    logger.info("Published weight version id=%d", latest_weight_id)
                except Exception:
                    logger.exception("Failed to publish weights.")

        logger.info("Learner loop finished at step=%d", self._step)

