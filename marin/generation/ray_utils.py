import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


# Used in normal ray remote args scheduling strategies
def scheduling_strategy_fn(tensor_parallel_size: int):

    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{"TPU": 1, "CPU": 1}] * tensor_parallel_size,
        strategy="PACK",  # STRICT_PACK means same node
    )
    return PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True)


# Used in Ray Data pipelines
def get_ray_remote_args_scheduling_strategy_fn(tensor_parallel_size: int):
    def scheduling_strategy_dict_fn():
        return dict(scheduling_strategy=scheduling_strategy_fn(tensor_parallel_size))

    return scheduling_strategy_dict_fn
