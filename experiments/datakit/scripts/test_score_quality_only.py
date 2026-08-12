# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker and task sizing for the standalone quality-scoring entry point.

Zephyr defaults a task's cost to the whole worker, so asking for a node-sized
worker without stating the task cost reserves 115 cores to run one shard on two of
them. The sizing is the entire reason this entry point exists, so it is asserted
rather than left to a log line nobody reads.
"""

from fray.cluster import ResourceConfig
from marin.execution.step_spec import StepSpec

from experiments.datakit.cluster.quality.fast_transformer.score import TASK_RESOURCES, WORKER_RESOURCES
from experiments.datakit.scripts.score_quality_only import (
    DEFAULT_MAX_WORKERS,
    NODE_CPU,
    NODE_RAM_GB,
    quality_step,
    tasks_per_worker,
    worker_resources,
)


def _normalize_step() -> StepSpec:
    return StepSpec(name="datakit/normalized/x", hash_attrs={"v": 1}, output_path_prefix="s3://b/marin")


def test_a_worker_leaves_headroom_for_the_node_system_pods():
    """A request for the whole node is never admitted on a shared cluster."""
    worker = worker_resources()
    assert worker.cpu < NODE_CPU
    assert worker.cpu == int(0.8 * NODE_CPU)
    assert int(str(worker.ram).rstrip("g")) == int(0.8 * NODE_RAM_GB)


def test_a_worker_requests_no_accelerator():
    """The point of running here is to use idle CPU, not to hold a GB200 GPU."""
    assert worker_resources().device.kind == "cpu"


def test_a_fat_worker_packs_many_shards():
    """Without an explicit task cost this would be one shard per worker.

    That is the failure this entry point exists to avoid: 216 workers each running
    a single two-core shard, with the other 113 cores of every node idle.
    """
    packed = tasks_per_worker(worker_resources())
    assert packed >= 50, f"only {packed} shards per worker — the node-sized request buys nothing"
    assert packed == min(115 // TASK_RESOURCES.cpu, 768 // 8)


def test_task_cost_is_smaller_than_the_worker_it_runs_on():
    """A task costing the whole worker is what makes packing impossible."""
    worker = worker_resources()
    assert TASK_RESOURCES.cpu < worker.cpu
    assert int(str(TASK_RESOURCES.ram).rstrip("g")) < int(str(worker.ram).rstrip("g"))


def test_the_dedicated_pool_shape_is_unchanged():
    """A per-source pool sizes its worker to one task, and must keep doing so.

    The reference pipeline still builds a pool per source, where worker and task
    are the same shape; this entry point changes the shared case only.
    """
    assert tasks_per_worker(WORKER_RESOURCES) == 1


def test_worker_count_matches_the_fleet():
    """A node-sized worker means the 217th has nowhere to land."""
    assert DEFAULT_MAX_WORKERS == 216


def test_a_smaller_fraction_packs_proportionally_fewer():
    """The fraction is a real knob, not a constant baked into the sizing."""
    assert tasks_per_worker(worker_resources(0.4)) < tasks_per_worker(worker_resources(0.8))


def test_memory_binds_when_a_task_is_memory_hungry():
    """Whichever of CPU or memory runs out first has to be the one that binds."""
    hungry = ResourceConfig(cpu=1, ram="256g")
    assert tasks_per_worker(worker_resources(), hungry) == 3  # 768 / 256, not 115 / 1


def test_the_model_version_changes_the_output_directory():
    """Production location, different hash — two scorers must not collide in the store.

    The model directory is region-specific and deliberately not hashed; the version
    tag is what separates one scorer's output from another's.
    """
    norm = _normalize_step()
    deployed = quality_step("x", norm, "pooled-junkgate2", "s3://b/marin").output_path
    candidate = quality_step("x", norm, "glm52-v3", "s3://b/marin").output_path
    assert deployed != candidate
    assert deployed.startswith("s3://b/marin/datakit/quality/x_")
    assert candidate.startswith("s3://b/marin/datakit/quality/x_")


def test_the_path_is_the_pipeline_step_name():
    """Writing beside production output under a different name would defeat the point."""
    path = quality_step("arxiv", _normalize_step(), "glm52-v3", "s3://b/marin").output_path
    assert "/datakit/quality/arxiv_" in path


def test_a_changed_dependency_changes_the_output_directory():
    """The hash covers deps, so re-normalized input does not reuse stale scores."""
    a = quality_step(
        "x",
        StepSpec(name="datakit/normalized/x", hash_attrs={"v": 1}, output_path_prefix="s3://b/marin"),
        "glm52-v3",
        "s3://b/marin",
    ).output_path
    b = quality_step(
        "x",
        StepSpec(name="datakit/normalized/x", hash_attrs={"v": 2}, output_path_prefix="s3://b/marin"),
        "glm52-v3",
        "s3://b/marin",
    ).output_path
    assert a != b
