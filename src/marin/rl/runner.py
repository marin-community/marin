"""Minimal RL runner that launches env actors and writes rollouts to Parquet.

Usage (example):
  from marin.rl.config import MarinRlConfig
  from marin.rl.datatypes import InferenceEndpoint
  from marin.rl.envs.hello import HelloEnvConfig
  from marin.rl.sinks import make_parquet_sink
  import ray

  ray.init()
  cfg = MarinRlConfig(
      name="hello-parquet",
      envs=[HelloEnvConfig()],
      inference=InferenceEndpoint("https://api.openai.com/v1"),  # not used by Hello
      learner=None,  # not used in this runner
      learner_resources=None,  # not used in this runner
  )
  run_envs_to_parquet(cfg, parquet_root="/tmp/marin_rl_rollouts", num_replicas=2, max_runtime_s=10)
"""

import asyncio
import time

import ray

from marin.rl.config import AbstractEnvConfig, InferenceConfig, MarinRlConfig
from marin.rl.datatypes import InferenceEndpoint
from marin.rl.envs.hello import HelloEnvConfig
from marin.rl.sinks import make_parquet_sink
from marin.rl.parquet_store import iter_rollout_groups


async def _graceful_sleep(total_seconds: float, check_period: float = 0.25) -> None:
    """Sleep with periodic yields to the event loop."""

    start = time.time()
    while time.time() - start < total_seconds:
        await asyncio.sleep(min(check_period, total_seconds))


def run_envs_to_parquet(
    config: MarinRlConfig,
    parquet_root: str,
    *,
    num_replicas: int = 1,
    max_runtime_s: float | None = None,
) -> list[ray.actor.ActorHandle]:
    """Start environment actors that write rollouts to a Parquet dataset.

    Args:
        config: ``MarinRlConfig`` with at least ``envs`` and ``inference`` set.
        parquet_root: Output dataset root (local or cloud URI).
        num_replicas: Number of replicas per env config.
        max_runtime_s: If provided, keep actors alive for this duration then return.

    Returns:
        List of Ray actor handles for the launched envs.
    """

    # Initialize sink
    rollout_sink = make_parquet_sink(parquet_root)

    # Build all env replicas
    actors: list[ray.actor.ActorHandle] = []
    seed = 0
    for env_cfg in config.envs:
        assert isinstance(env_cfg, AbstractEnvConfig)
        for _ in range(num_replicas):
            actor = env_cfg.build(config.inference.endpoint, rollout_sink, seed=seed)
            actors.append(actor)
            seed += 1

    print("Hi", len(actors))

    # Optionally keep the process alive for a while to collect data
    if max_runtime_s is not None:
        try:
            asyncio.run(_graceful_sleep(max_runtime_s))
        finally:
            # Attempt to stop all actors politely
            for a in actors:
                # no strict interface yet; best-effort stop if available
                try:
                    a.stop.remote()  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Print a preview of the first few examples written
            _cat_first_examples(parquet_root, limit=20)

    return actors


def _cat_first_examples(root_path: str, *, limit: int = 20) -> None:
    """Print a brief preview of the first few examples from the Parquet dataset.

    For each rollout, prints the (optional) user prompt, last assistant answer (truncated), and reward.
    """

    count = 0
    for group in iter_rollout_groups(root_path):
        for rollout in group.rollouts:
            user_turns = [t for t in rollout.turns if t.role == "user"]
            assistant_turns = [t for t in rollout.turns if t.role == "assistant"]

            prompt = user_turns[0].message if user_turns else ""
            if assistant_turns:
                ans_turn = assistant_turns[-1]
                answer = ans_turn.message
                reward = ans_turn.reward
            else:
                answer = ""
                reward = None

            # Truncate long fields for readability
            def trunc(s: str, n: int = 200) -> str:
                return s if len(s) <= n else s[: n - 1] + "…"

            print(f"Example {count + 1}: reward={reward} prompt={trunc(prompt)!r} answer={trunc(answer)!r}")
            count += 1
            if count >= limit:
                return


if __name__ == "__main__":

    def main(config: MarinRlConfig):
        run_envs_to_parquet(config, parquet_root="/tmp/marin_rl_rollouts", num_replicas=2, max_runtime_s=10)

    # draccus.wrap()(main)()
    dummy = MarinRlConfig(
        name="hello-parquet",
        envs=[HelloEnvConfig()],
        inference=InferenceConfig(InferenceEndpoint("https://api.openai.com/v1")),  # not used by Hello
        learner=None,  # type: ignore[assignment] not used in this runner
        learner_resources=None,  # type: ignore[assignment] not used in this runner
    )

    main(dummy)
