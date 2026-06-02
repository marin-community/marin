# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal SkyRL-shaped workload, runnable on Iris via the fakeray shim.

This is NOT full SkyRL (which needs GPUs + torch + vLLM). It reproduces SkyRL's
*Ray usage pattern* with a tiny CPU example, exercising the shim's actor +
placement-group surface the way an RL post-training framework would:

  * ``placement_group([{CPU:1}] * world_size, strategy="PACK")`` + ``pg.ready()``
  * a ``@ray.remote`` worker class (à la SkyRL ``PolicyWorker``)
  * an actor group whose members are pinned to PG bundles via
    ``PlacementGroupSchedulingStrategy`` (à la ``PPORayActorGroup``)
  * driver coordination via the SkyRL ``async_run_ray_method`` idiom: invoke a
    method on every actor → list of ``ObjectRef`` → ``ray.get(...)``
  * a few PPO-ish steps: broadcast a weight, each worker computes, driver reduces

Run locally::

    PYTHONPATH=lib/fakeray/src python lib/fakeray/examples/skyrl_min.py

Run on Iris (from a bundle that includes fakeray + the marin libs)::

    iris ... job run --cpu 2 --memory 2g --no-preemptible --region europe-west4 \
        -- python skyrl_min.py --world-size 2 --steps 3 --region europe-west4

Verified SUCCEEDED on the marin cluster (europe-west4): the PolicyWorker actors
register as real Iris endpoints and the driver coordinates them through the shim.
"""

import argparse

import fakeray
from fakeray._scheduler import FakeRayConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    fakeray.set_config(FakeRayConfig(pool_size=args.world_size, device="cpu", region=args.region))
    fakeray.install()

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init()

    # one CPU bundle per worker (SkyRL: one GPU bundle per rank)
    pg = placement_group([{"CPU": 1}] * args.world_size, strategy="PACK")
    ray.get(pg.ready())

    @ray.remote
    class PolicyWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size
            self.weight = 0.0

        def init_model(self):
            return {"rank": self.rank, "ready": True}

        def set_weight(self, w):
            self.weight = w
            return self.rank

        def train_step(self, batch):
            return self.weight + self.rank + batch

    workers = [
        PolicyWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i)
        ).remote(i, args.world_size)
        for i in range(args.world_size)
    ]

    def async_run(method: str, *a):
        """SkyRL's async_run_ray_method: call `method` on every actor, return refs."""
        return [getattr(w, method).remote(*a) for w in workers]

    print(f"SKYRL_MIN init: {ray.get(async_run('init_model'))}", flush=True)

    weight = 1.0
    for step in range(args.steps):
        ray.get(async_run("set_weight", weight))
        contribs = ray.get(async_run("train_step", float(step)))
        weight = sum(contribs) / len(contribs)
        print(f"SKYRL_MIN step {step}: contribs={contribs} -> weight={weight}", flush=True)

    print("SKYRL_MIN_RESULT_BEGIN", flush=True)
    print(f"world_size={args.world_size} steps={args.steps} final_weight={weight}", flush=True)
    print("SKYRL_MIN_RESULT_END", flush=True)


if __name__ == "__main__":
    main()
