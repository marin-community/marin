"""Fix vLLM parallel state PP group for TPU Ray workers.

On TPU Ray workers, vLLM distributed is initialized with world_size=1, rank=0.
This creates a PP group with rank_in_group=0, world_size=1 for ALL workers.
When the vLLM model calls make_layers -> get_pp_indices -> get_pp_group(),
all workers think they're PP rank 0 and create the wrong layer assignment.

Fix: After init_distributed_environment + ensure_model_parallel_initialized,
override the PP group's attributes to match the actual TPU PP rank.
This is safe because the TPU workers don't use torch distributed for PP
communication — they use JAX transfer servers instead.
"""

import os

PATH = "/workspace/tpu_inference/tpu_inference/worker/tpu_worker.py"

with open(PATH) as f:
    code = f.read()

# Find the ensure_model_parallel_initialized block and add PP group override after it
# Handle both original and previously-patched versions
target = """            ensure_model_parallel_initialized(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )"""

replacement = """            ensure_model_parallel_initialized(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
        # Override the vLLM PP group to match the actual TPU PP rank.
        # vLLM was initialized with world_size=1,rank=0 (single chip),
        # but the vLLM model's make_layers needs the real PP rank to
        # assign the correct layers to each worker.
        pp_size = self.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            from vllm.distributed.parallel_state import get_pp_group
            pp_group = get_pp_group()
            pp_group.rank = self.rank
            pp_group.ranks = list(range(pp_size))
            pp_group.rank_in_group = self.rank
            pp_group.world_size = pp_size
            logger.info(
                f"PP parallel state override: rank={self.rank}, "
                f"world_size={pp_size}, ranks={pp_group.ranks}")"""

if target in code:
    code = code.replace(target, replacement)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED tpu_worker.py: override vLLM PP group with actual TPU rank")
else:
    # Check if previous patch already modified this section
    if "VLLM_TPU_PP_RANK" in code or "PP_RANK_FIX" in code or "PP parallel state" in code:
        print("SKIP: previous PP patch detected, need clean file")
    else:
        print("SKIP: target not found")
