"""Base environment interface for Marin RL.

An *environment* is a Ray actor that continuously produces
:class:`~marin.rl.types.RolloutGroup` objects and dispatches them to the
provided ``rollout_sink`` callback.

Concrete environments should inherit from :class:`AbstractMarinEnv` and
implement the :pymeth:`run` coroutine.
"""

import abc
import asyncio
import logging
from typing import Final

from .types import InferenceEndpoint, RolloutGroup, RolloutSink

logger: Final = logging.getLogger(__name__)


class AbstractMarinEnv(abc.ABC):
    """Base class for asynchronous Ray env actors.

    Concrete subclasses *must* implement :py:meth:`run` as ``async def`` and
    should periodically ``await`` to allow Ray to service other RPCs (including
    :py:meth:`stop`).
    """

    # Subclasses will be decorated with ``@ray.remote`` by their Config.

    def __init__(self, inference: InferenceEndpoint, rollout_sink: RolloutSink):
        self._inference = inference
        self._rollout_sink = rollout_sink
        self._stop_event: asyncio.Event = asyncio.Event()
        logger.info("Environment initialized with inference %s", inference.address)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Signal the event loop to terminate gracefully."""

        self._stop_event.set()
        logger.info("Stop signal received")

    @abc.abstractmethod
    async def run(self) -> None:  # pragma: no cover
        """
        Main loop that subclasses must implement.

        An environment is a Ray actor that continuously produces
        :class:`~marin.rl.types.RolloutGroup` objects and dispatches them to the
        provided ``rollout_sink`` callback.

        The environment should periodically check for a stop signal and terminate
        when it is received.  The environment should also call the ``rollout_sink``
        callback with a list of :class:`~marin.rl.types.RolloutGroup` objects as soon as it has generated them.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    async def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    async def shutdown(self) -> None:
        """Optional: release resources before shutdown."""

        logger.debug("%s closed", self.__class__.__name__)


class SimpleEnv(AbstractMarinEnv):
    """Concrete base that hides ``async`` details from subclasses."""

    def do_rollout(self) -> list[RolloutGroup]:  # pragma: no cover - abstract
        """Produce one or more rollout groups.

        Subclasses implement their rollout logic here as a regular function
        without needing to worry about ``async``/``await`` semantics.
        """

        raise NotImplementedError

    async def run(self) -> None:
        """Execute :py:meth:`do_rollout` in a loop and dispatch results."""

        while not await self._should_stop():
            groups = await asyncio.to_thread(self.do_rollout)
            if groups:
                self._rollout_sink(groups)
            await asyncio.sleep(0)  # yield to Ray scheduler
