import os
import ray
import importlib
from loguru import logger
from ray.util.placement_group import placement_group
from omegaconf import DictConfig
import hydra
from functools import lru_cache

from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.utils import peer_access_supported, print_mem, validate_cfg


# Rank diagnostics file the DPO equivalence runner cats at the end of the job stream.
# run_equivalence.sh references the same filename; keep them in sync.
DPO_DIAGNOSTICS_LOG = "dpo-diagnostics.log"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.trainer.logger = "console"
        validate_cfg(cfg)

        return cfg


def get_rank_0_memory(actor_group, message: str):
    mem = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
    print_mem(message, mem)
    return mem["allocated"]


def import_worker(strategy: str, worker_type: str):
    if strategy != "megatron":
        raise ValueError(f"Unknown strategy type for {worker_type}: {strategy}")
    module_path = "skyrl_train.workers.megatron.megatron_worker"

    module = importlib.import_module(module_path)
    return getattr(module, f"{worker_type.capitalize()}Worker")


def init_worker_with_type(
    worker_type: str,
    shared_pg=None,
    colocate_all=False,
    num_gpus_per_node=1,
    num_nodes=1,
    cfg=None,
    num_training_steps: int | None = None,
) -> PPORayActorGroup:
    if cfg is None:
        cfg = get_test_actor_config()

    if shared_pg is not None:
        pg = shared_pg
        num_gpus_per_actor = 0.2
    else:
        bundles = [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node} for _ in range(num_nodes)]
        pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        num_gpus_per_actor = 0.75

    worker_cls = import_worker(cfg.trainer.strategy, worker_type)
    model = PPORayActorGroup(
        cfg,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
        colocate_all=colocate_all,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    # we use policy model path for all tests (regardless of actor type)
    init_kwargs = {} if num_training_steps is None else {"num_training_steps": num_training_steps}
    ray.get(model.async_init_model(cfg.trainer.policy.model.path, **init_kwargs))
    return model


def get_available_gpus():
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or all available GPUs"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES (can be comma-separated list)
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
        return gpu_ids
    else:
        # If not set, warn user but proceed with all GPUs
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_ids = list(range(gpu_count))
                print(f"CUDA_VISIBLE_DEVICES not set. Using all {gpu_count} GPUs: {gpu_ids}")
                print("This might conflict with other processes. Consider setting CUDA_VISIBLE_DEVICES explicitly.")
                return gpu_ids
            else:
                return []
        except Exception as e:
            print(f"Error getting available GPUs: {e}")
            return []


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


def ray_init_for_tests():
    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=4):
        log_once("Disabling NCCL P2P for test environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    # TODO (erictang000): refactor this to use the same prepare_runtime_environment function as in utils.py for tests
    # to remove duplicate code
    env_vars["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars["NVTE_FUSED_ATTN"] = "0"
    env_vars["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
    ray.init(runtime_env={"env_vars": env_vars})


# TODO: this is kind of messy. All these information are inside cfg but we are passing them in
# again. Make a global get_test_config function that is parametrized.
