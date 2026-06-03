# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run SkyRL's REAL placement-group resolution orchestration via the shim.

Unlike skyrl_min.py (which mimics SkyRL idioms), this lifts SkyRL's *actual*
orchestration functions VERBATIM from
  novasky-ai/skyrl : skyrl/train/utils/utils.py
  (_probe_bundle_placement, ResolvedPlacementGroup, get_ray_pg_ready_with_timeout)
and runs them through fakeray. Those functions are pure-Ray (the surrounding
module is unimportable on CPU only because of torch/transformers imports), so
copying them is the cheapest way to exercise SkyRL's real Ray orchestration.

The one CPU concession: SkyRL's InfoActor is `@ray.remote(num_gpus=1)` and reads
`ray.get_gpu_ids()[0]`. We keep the exact structure but use a CPU bundle and a
GPU-id that falls back to the bundle index on CPU, so the orchestration path
(placement_group_table -> InfoActor per bundle -> get id -> kill -> sort) runs
unchanged. On a real GPU cluster the unmodified SkyRL code would run as-is.
"""

import argparse
import functools

import fakeray
from fakeray._scheduler import FakeRayConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-bundles", type=int, default=2)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    fakeray.set_config(FakeRayConfig(pool_size=args.num_bundles, device="cpu", region=args.region))
    fakeray.install()

    import ray
    from ray.util.placement_group import placement_group, placement_group_table
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init()

    # ---- BEGIN code lifted from SkyRL utils.py (verbatim except InfoActor cpu) ----
    @ray.remote
    class InfoActor:
        def get_gpu_id(self):
            gpus = ray.get_gpu_ids()
            return gpus[0] if gpus else 0  # CPU fallback; real SkyRL: gpus[0]

    def _probe_bundle_placement(pg):
        pg_data = placement_group_table(pg)
        num_bundles = len(pg_data["bundles"])
        bundle_to_node_ids = pg_data["bundles_to_node_id"]

        info_actors = []
        for i in range(num_bundles):
            info_actors.append(
                InfoActor.options(
                    num_cpus=0.01,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    ),
                ).remote()
            )

        gpu_ids = ray.get([actor.get_gpu_id.remote() for actor in info_actors])
        for actor in info_actors:
            ray.kill(actor)

        bundle_infos = [(i, bundle_to_node_ids[i], gpu_ids[i]) for i in range(num_bundles)]
        return sorted(bundle_infos, key=lambda x: (x[1], x[2]))

    class ResolvedPlacementGroup:
        def __init__(self, pg):
            self.pg = pg
            self._bundle_placement = None

        def _get_bundle_placement(self):
            if self._bundle_placement is None:
                self._bundle_placement = _probe_bundle_placement(self.pg)
            return self._bundle_placement

        @functools.cached_property
        def reordered_bundle_indices(self):
            return [info[0] for info in self._get_bundle_placement()]

        @functools.cached_property
        def bundle_node_ids(self):
            return [info[1] for info in self._get_bundle_placement()]

        @functools.cached_property
        def num_nodes(self):
            return len(set(self.bundle_node_ids))

    # ---- END lifted code ----

    pg = placement_group([{"CPU": 1}] * args.num_bundles, strategy="PACK")
    ray.get(pg.ready(), timeout=60)

    rpg = ResolvedPlacementGroup(pg)
    print("SKYRL_REAL_PG_BEGIN", flush=True)
    print(f"num_bundles={args.num_bundles}", flush=True)
    print(f"reordered_bundle_indices={rpg.reordered_bundle_indices}", flush=True)
    print(f"bundle_node_ids={rpg.bundle_node_ids}", flush=True)
    print(f"num_nodes={rpg.num_nodes}", flush=True)
    print("SKYRL_REAL_PG_END", flush=True)


if __name__ == "__main__":
    main()
