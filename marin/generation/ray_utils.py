import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def get_scheduling_strategy_fn(tensor_parallel_size: int):
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"TPU": 1, "CPU": 1}] * tensor_parallel_size,
            strategy="PACK",  # STRICT_PACK means same node, PACK means different node possible
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True))

    return scheduling_strategy_fn
