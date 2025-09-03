"""Base environment interface for Marin RL (pull, no buffering).

Environments encapsulate only domain logic: deciding when to call the inference
engine and how to score results. They expose a single method ``step`` that
returns one batch (list) of :class:`~marin.rl.datatypes.Rollout` objects when
invoked. No internal queues or pause/unpause semantics.
"""

import abc
import asyncio
import logging
from typing import Final

from .datatypes import InferenceEndpoint, Rollout

logger: Final = logging.getLogger(__name__)


class AbstractMarinEnv(abc.ABC):
    """Base class for env actors that produce one batch per call."""

    # Subclasses will be decorated with ``@ray.remote`` by their Config.

    def __init__(self, inference: InferenceEndpoint):
        self._inference = inference

    @abc.abstractmethod
    async def step(self) -> list[Rollout]:  # pragma: no cover
        """Produce one batch of rollouts and return it."""
        raise NotImplementedError

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Release resources and terminate gracefully."""
        raise NotImplementedError("Subclasses must implement shutdown")



class SimpleEnv(AbstractMarinEnv):
    """Concrete base that hides ``async`` details from subclasses.

    Subclasses implement their rollout logic in a regular function
    ``do_rollout`` without needing to worry about ``async``/``await``.
    """

    def __init__(self, inference: InferenceEndpoint):
        super().__init__(inference)

    def do_rollout(self) -> list[Rollout]:  # pragma: no cover - abstract
        """Produce one or more rollouts."""
        raise NotImplementedError

    async def step(self) -> list[Rollout]:
        return await asyncio.to_thread(self.do_rollout)

    async def shutdown(self) -> None:
        logger.debug("%s shutdown", self.__class__.__name__)
