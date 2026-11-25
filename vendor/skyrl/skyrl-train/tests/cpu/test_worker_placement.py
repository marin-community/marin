from types import SimpleNamespace

import ray
from loguru import logger
from ray import cloudpickle
from ray.cluster_utils import Cluster

from skyrl_train.config.rollout_workers import RolloutWorkerPlacement
from skyrl_train.rollouts import workers


def test_rollout_actor_exports_with_a_runtime_log_file(tmp_path):
    sink = logger.add(tmp_path / "runtime.log")
    try:
        cloudpickle.dumps(workers.RolloutWorker.__ray_metadata__.modified_class)
    finally:
        logger.remove(sink)


@ray.remote
class _PlacementProbe:
    def __init__(self, spec, shard, sink, executor_threads):
        pass

    def node(self):
        return ray.get_runtime_context().get_node_id()


def test_spread_moves_worker_memory_off_the_driver(monkeypatch):
    cluster = Cluster()
    actors = []
    try:
        cluster.add_node(num_cpus=2)
        cluster.add_node(num_cpus=2)
        ray.init(address=cluster.address)
        monkeypatch.setattr(workers, "RolloutWorker", _PlacementProbe)
        for placement in RolloutWorkerPlacement:
            resources = workers.RolloutWorkerResources(
                num_workers=2,
                cpus_per_worker=1,
                executor_threads=1,
                progress_timeout_seconds=30,
                start_interval_seconds=0,
                placement=placement,
            )
            pool = workers.RolloutWorkerPool(SimpleNamespace(config=None), resources)
            actors = [pool._launch_worker(index) for index in range(2)]
            nodes = set(ray.get([actor.node.remote() for actor in actors], timeout=30))
            if placement is RolloutWorkerPlacement.DRIVER:
                assert nodes == {ray.get_runtime_context().get_node_id()}
            else:
                assert len(nodes) == 2
            for actor in actors:
                ray.kill(actor)
            actors = []
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()
        cluster.shutdown()
