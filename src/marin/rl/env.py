"""Base environment interface for Marin RL.

An *environment* is a Ray actor that continuously produces
:class:`~marin.rl.datatypes.RolloutGroup` objects and dispatches them to the
provided ``rollout_sink`` callback.

Concrete environments should inherit from :class:`AbstractMarinEnv` and
implement the :pymeth:`run` coroutine.
"""

import abc
import asyncio
import logging
from typing import Final

from .datatypes import InferenceEndpoint, RolloutGroup, RolloutSink

logger: Final = logging.getLogger(__name__)


class AbstractMarinEnv(abc.ABC):
    """Base class for asynchronous Ray env actors.

    Concrete subclasses must implement :py:meth:`run` as ``async def`` and
    should periodically ``await`` to allow Ray to service other RPCs (including
    pause, unpause, and shutdown).
    """

    # Subclasses will be decorated with ``@ray.remote`` by their Config.

    def __init__(self, inference: InferenceEndpoint, rollout_sink: RolloutSink):
        self._inference = inference
        self._rollout_sink = rollout_sink
        self._stop_event: asyncio.Event = asyncio.Event()
        self._paused: bool = False
        self._state_cond: asyncio.Condition = asyncio.Condition()
        logger.info("Environment initialized with inference %s", inference.address)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Signal the event loop to terminate gracefully.

        Default implementation sets the internal stop flag and calls
        :py:meth:`on_shutdown` for subclasses to release resources.
        """

        self._stop_event.set()
        async with self._state_cond:
            self._paused = False
            self._state_cond.notify_all()
        logger.info("Shutdown signal received")
        try:
            await self.on_shutdown()
        except Exception:
            logger.exception("Error during on_shutdown")

    async def pause(self) -> None:
        """Pause processing.

        Default behavior marks the environment paused; subclasses can
        override and should call ``await super().pause()`` to preserve signaling.
        """

        async with self._state_cond:
            self._paused = True
            self._state_cond.notify_all()
        try:
            await self.on_pause()
        except Exception:
            logger.exception("Error during on_pause")

    async def unpause(self) -> None:
        """Resume processing.

        Default behavior resumes processing by unpausing the loop. The
        main :py:meth:`run` coroutine must have been started already by
        the config's actor builder.
        """

        async with self._state_cond:
            self._paused = False
            self._state_cond.notify_all()
        try:
            await self.on_unpause()
        except Exception:
            logger.exception("Error during on_unpause")

    @abc.abstractmethod
    async def run(self) -> None:  # pragma: no cover
        """
        Main loop that subclasses must implement.

        An environment is a Ray actor that continuously produces
        :class:`~marin.rl.datatypes.RolloutGroup` objects and dispatches them to the
        provided ``rollout_sink`` callback.

        The environment should periodically check for a stop signal and terminate
        when it is received.  The environment should also call the ``rollout_sink``
        callback with a list of :class:`~marin.rl.datatypes.RolloutGroup` objects as soon as it has generated them.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    async def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    async def _wait_ready(self) -> bool:
        """Block while paused; return False if shutting down.

        Intended to be awaited at the top of each loop iteration in
        :py:meth:`run` implementations.
        """

        async with self._state_cond:
            while self._paused and not self._stop_event.is_set():
                await self._state_cond.wait()
        return not self._stop_event.is_set()

    async def on_shutdown(self) -> None:
        """Optional: release resources before shutdown."""

        logger.debug("%s closed", self.__class__.__name__)

    async def on_pause(self) -> None:
        """Optional: hook invoked on pause."""

        logger.debug("%s paused", self.__class__.__name__)

    async def on_unpause(self) -> None:
        """Optional: hook invoked on unpause."""

        logger.debug("%s unpaused", self.__class__.__name__)


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
            # Respect pause/shutdown signals
            if not await self._wait_ready():
                break
            groups = await asyncio.to_thread(self.do_rollout)
            if groups:
                self._rollout_sink(groups)
            await asyncio.sleep(0)  # yield to Ray scheduler
