import time
from typing import List

import numpy as np
import ray

from .datatypes import Rollout
from marin.post_training.environments.marin_env import EnvStep, MarinEnv  # old-style

# ---------------------------------------------------------------------------
# Simple Ray actor that buffers RolloutGroups sent from async envs.
# ---------------------------------------------------------------------------


@ray.remote(max_concurrency=100)
class _GroupBuffer:
    """Async buffer so producers can push while consumers await fetch.

    Using an asyncio.Queue avoids blocking the entire actor thread, allowing
    push and fetch to interleave correctly under Ray's async actor model.
    """

    def __init__(self):
        import asyncio

        self._queue = asyncio.Queue()

    async def push(self, groups: list[RolloutGroup]):  # called from env actor
        for g in groups:
            await self._queue.put(g)

    async def fetch(self, min_count: int, *, timeout: float | None = None):
        """Return groups when available, awaiting until min_count are ready.

        Args:
            min_count: minimum number of groups to return.
            timeout: optional maximum time (seconds) to wait. If provided and the
                deadline is reached, returns whatever has been collected so far
                (possibly empty).
        """
        import asyncio

        # If we have enough items, return them immediately
        if self._queue.qsize() >= min_count:
            return [self._queue.get_nowait() for _ in range(min_count)]

        # Otherwise, await for items to arrive
        if timeout is not None:
            # Use asyncio.wait_for with the queue
            items = []
            remaining = min_count
            while remaining > 0:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    items.append(item)
                    remaining -= 1
                except asyncio.TimeoutError:
                    break
            return items
        else:
            # No timeout - await until we have enough
            items = []
            for _ in range(min_count):
                item = await self._queue.get()
                items.append(item)
            return items


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class NewStyleEnvWrapper(MarinEnv):
    """Expose a streaming RL environment via the synchronous MarinEnv API.

    Parameters
    ----------
    env_cfg:
        An :class:`marin.rl.config.AbstractEnvConfig` instance to build the
        streaming environment.
    inference:
        Inference endpoint object forwarded to *env_cfg.build*.
    batch_size:
        Minimum number of *RolloutGroup* objects that constitute one `step()`.
    replica_id:
        Passed through to *env_cfg.build* and can be used to vary seeds.
    """

    def __init__(self, env_cfg, inference, batch_size: int = 1, replica_id: int = 0, tokenizer=None):
        super().__init__()
        self._batch_size = batch_size
        self._tokenizer = tokenizer

        # Build the new-style pull-based environment actor
        self._actor = env_cfg.build(inference, replica_id)

    # --------------------------------------------------------------
    # MarinEnv interface
    # --------------------------------------------------------------

    def step(
        self,
        *,
        sampler=None,
        params=None,
        n_examples: int = 1,
        prng_key=None,
        mode: str = "train",
        n_generations: int = 1,
        **__,
    ) -> EnvStep:  # type: ignore[override]
        """Collect rollouts and reshape into (examples, generations) for GRPO-style grouping.

        This wrapper ignores sampler/params and simply forwards buffered rollouts
        from the async env, grouping them sequentially into ``n_examples`` each with
        ``n_generations`` responses when possible.
        """
        # Fetch exactly n_examples batches; each batch may contain multiple rollouts
        need_groups = max(1, int(n_examples))
        batches: List[list[Rollout]] = [ray.get(self._actor.step.remote()) for _ in range(need_groups)]

        gens = max(1, int(n_generations))
        examples: list[dict] = []
        responses: list[list[dict]] = []
        rewards: list[list[float]] = []

        for batch in batches:
            rollouts = batch
            if not rollouts:
                continue

            prompt = ""
            if rollouts and getattr(rollouts[0], "metadata", None):
                prompt = rollouts[0].metadata.get("prompt", "")  # type: ignore[assignment]
            examples.append({"prompt": prompt, "answer": None})

            gen_list: list[dict] = []
            reward_list: list[float] = []
            for r in rollouts[:gens]:
                text = ""
                if r.metadata is not None:
                    text = r.metadata.get("response", "")
                if self._tokenizer is not None and text:
                    toks = self._tokenizer.encode(text, add_special_tokens=False)
                else:
                    toks = []
                gen_list.append({"tokens": toks, "logprobs": [], "text": text})
                reward_list.append(float(r.reward or 0.0))

            while len(gen_list) < gens:
                gen_list.append({"tokens": [], "logprobs": [], "text": ""})
                reward_list.append(0.0)

            responses.append(gen_list)
            rewards.append(reward_list)

        rewards_arr = np.asarray(rewards, dtype=float)
        metrics = {
            "n_groups": len(batches),
            "n_rollouts": sum(len(b) for b in batches),
            "n_examples": len(examples),
            "n_generations": gens,
        }

        return EnvStep(examples=examples, responses=responses, rewards=rewards_arr, metrics=metrics)
