"""Configuration-only training checks shared by launch composition and GPU entrypoints."""

from omegaconf import DictConfig
from marinskyrl.inference_placement import validate_expert_block_transport
from marinskyrl.runtime_options import GDNBackend, R3Transport, reference_model_required
from skyrl_train.config.placement_geometry import validate_colocated_engine_geometry


def validate_batch_invariant_config(cfg: DictConfig) -> None:
    """Validate that trainer and rollout kernels are controlled together."""

    if not cfg.trainer.algorithm.batch_invariant:
        return
    if cfg.generator.backend != "vllm":
        raise ValueError("trainer.algorithm.batch_invariant=true requires generator.backend='vllm'")
    if not cfg.generator.run_engines_locally:
        raise ValueError(
            "trainer.algorithm.batch_invariant=true cannot configure a remote inference server; "
            "run the vLLM engines locally so both rollout and trainer activation is guaranteed"
        )




def validate_megatron_config(cfg: DictConfig):
    # not yet supported + tested features
    assert cfg.generator.weight_sync_backend == "nccl", "only nccl is supported for megatron weight sync"
    assert cfg.trainer.critic.model.path is None, "only GRPO training is currently supported for megatron"

    worker_configs = [(cfg.trainer.policy, "policy"), (cfg.trainer.ref, "ref")]
    for config, worker_type in worker_configs:
        # Context parallel requires sample packing.
        if config.megatron_config.context_parallel_size > 1:
            assert cfg.trainer.use_sample_packing, "context parallel is only supported with sample packing"
        # check that sequence parallel is not configured outside of megatron
        assert config.sequence_parallel_size == 1, (
            f"found {worker_type}.sequence_parallel_size={config.sequence_parallel_size}, ulysses style sequence parallel is not supported for megatron"
        )
        # The fused router bypasses the replay hook entirely; catch it at config
        # validation on the launcher CPU instead of at model build on the GPUs.
        if config.megatron_config.get("moe_router_replay", False):
            for kwargs_name in ("transformer_config_kwargs", "model_config_kwargs"):
                for key_path, value in _iter_config_kwargs(config.megatron_config.get(kwargs_name, {})):
                    if key_path[-1] == "moe_router_fusion" and value:
                        raise ValueError(
                            f"trainer.{worker_type}.megatron_config.{kwargs_name}.{'.'.join(key_path)} "
                            "must stay false with moe_router_replay: the fused router bypasses the replay hook"
                        )


def _iter_config_kwargs(node):
    """Yield ``(key_path, value)`` for every leaf in a (possibly nested) kwargs mapping."""
    if not node:
        return
    for key, value in node.items():
        if isinstance(value, dict):
            for path, leaf in _iter_config_kwargs(value):
                yield (key, *path), leaf
        else:
            yield (key,), value


def validate_training_settings(cfg: DictConfig) -> None:
    if cfg.trainer.strategy != "megatron":
        raise ValueError(f"Unsupported training strategy: {cfg.trainer.strategy}")
    if cfg.trainer.critic.model.path:
        raise ValueError("Megatron does not support a critic worker")
    runtime_values = {
        "trainer.distributed.placement_group_timeout_seconds": cfg.trainer.distributed.placement_group_timeout_seconds,
        "trainer.distributed.worker_collective_timeout_seconds": (
            cfg.trainer.distributed.worker_collective_timeout_seconds
        ),
        "trainer.policy.host_memory_monitor.interval_seconds": cfg.trainer.policy.host_memory_monitor.interval_seconds,
        "trainer.model_load_retry.backoff_base_seconds": cfg.trainer.model_load_retry.backoff_base_seconds,
        "trainer.model_load_retry.backoff_cap_seconds": cfg.trainer.model_load_retry.backoff_cap_seconds,
        "trainer.progress.min_interval_seconds": cfg.trainer.progress.min_interval_seconds,
        "trainer.progress.heartbeat_seconds": cfg.trainer.progress.heartbeat_seconds,
        "trainer.progress.percent_step": cfg.trainer.progress.percent_step,
        "trainer.progress.count_step": cfg.trainer.progress.count_step,
        "generator.r3_dispatch_put_timeout_seconds": cfg.generator.r3_dispatch_put_timeout_seconds,
        "trajectory_runner.rollout_workers.num_workers": cfg.trajectory_runner.rollout_workers.num_workers,
        "trajectory_runner.rollout_workers.cpus_per_worker": cfg.trajectory_runner.rollout_workers.cpus_per_worker,
        "trajectory_runner.rollout_workers.executor_threads": cfg.trajectory_runner.rollout_workers.executor_threads,
        "trajectory_runner.rollout_workers.progress_timeout_seconds": (
            cfg.trajectory_runner.rollout_workers.progress_timeout_seconds
        ),
    }
    for path, value in runtime_values.items():
        if value <= 0:
            raise ValueError(f"{path} must be positive; got {value}")
    start_interval = cfg.trajectory_runner.rollout_workers.start_interval_seconds
    if start_interval < 0:
        raise ValueError(
            f"trajectory_runner.rollout_workers.start_interval_seconds must be non-negative; got {start_interval}"
        )
    if cfg.trainer.model_load_retry.max_retries < 0:
        raise ValueError(
            f"trainer.model_load_retry.max_retries must be non-negative; got {cfg.trainer.model_load_retry.max_retries}"
        )
    group_admission_stall_timeout = cfg.trainer.algorithm.group_admission.stall_timeout
    if group_admission_stall_timeout is not None and group_admission_stall_timeout <= 0:
        raise ValueError(
            "trainer.algorithm.group_admission.stall_timeout must be positive when set; got "
            f"{group_admission_stall_timeout}"
        )
    if cfg.trainer.algorithm.tis_lcs_alert_threshold < 0:
        raise ValueError(
            "trainer.algorithm.tis_lcs_alert_threshold must be non-negative; "
            f"got {cfg.trainer.algorithm.tis_lcs_alert_threshold}"
        )
    if cfg.trainer.progress.mode not in {"auto", "tqdm", "logging"}:
        raise ValueError(f"trainer.progress.mode must be one of auto, tqdm, logging; got {cfg.trainer.progress.mode!r}")
    if cfg.generator.r3_transport not in set(R3Transport):
        raise ValueError(
            f"generator.r3_transport must be one of by_value, resident, decentral; got {cfg.generator.r3_transport!r}"
        )
    if cfg.generator.gdn_backend not in set(GDNBackend):
        raise ValueError(f"generator.gdn_backend must be one of torch, flashqla; got {cfg.generator.gdn_backend!r}")
    use_ref_model = reference_model_required(cfg.trainer.algorithm)
    if cfg.trainer.policy.model.lora.rank > 0:
        raise ValueError("Megatron training does not support LoRA")

    if cfg.trainer.micro_forward_batch_size_per_gpu != cfg.trainer.micro_train_batch_size_per_gpu:
        raise ValueError(
            "Megatron recomputes old log-probs with micro_forward_batch_size_per_gpu="
            f"{cfg.trainer.micro_forward_batch_size_per_gpu} but trains with micro_train_batch_size_per_gpu="
            f"{cfg.trainer.micro_train_batch_size_per_gpu}; cuBLAS selects kernels per shape, so the two "
            "passes disagree and PPO clipping acts on kernel noise. Use equal sizes."
        )

    if cfg.trainer.offload_optimizer_during_rollouts and cfg.trainer.placement.colocate_all:
        raise ValueError(
            "trainer.offload_optimizer_during_rollouts applies to disaggregated runs only; "
            "colocated runs already offload the optimizer between policy updates"
        )

    # Validate placement
    validate_expert_block_transport(cfg)
    if cfg.trainer.placement.colocate_all and cfg.trainer.rollout_buffer.max_staleness_steps != 0:
        # Colocated engines sleep during training, so no rollout may run ahead of the trained policy.
        raise ValueError("colocate_all requires trainer.rollout_buffer.max_staleness_steps=0")
    if cfg.trainer.placement.colocate_all:
        tp_pp_size = (
            cfg.generator.inference_engine_tensor_parallel_size * cfg.generator.inference_engine_pipeline_parallel_size
        )
        gpus_per_node = cfg.trainer.placement.policy_num_gpus_per_node
        validate_colocated_engine_geometry(tensor_pipeline_size=tp_pp_size, gpus_per_node=gpus_per_node)
        num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
        num_rollout_gpus = (
            cfg.generator.num_inference_engines
            * cfg.generator.inference_engine_tensor_parallel_size
            * cfg.generator.inference_engine_pipeline_parallel_size
            * cfg.generator.inference_engine_data_parallel_size
        )
        assert num_policy_gpus == num_rollout_gpus, (
            f"num_policy_gpus ({num_policy_gpus}) and num_rollout_gpus ({num_rollout_gpus}) must be the same when colocating all models"
        )

    else:
        if cfg.trainer.placement.colocate_policy_ref and use_ref_model:
            assert cfg.trainer.placement.policy_num_nodes == cfg.trainer.placement.ref_num_nodes, (
                f"policy_num_nodes ({cfg.trainer.placement.policy_num_nodes}) and ref_num_nodes ({cfg.trainer.placement.ref_num_nodes}) must be the same when colocate policy and ref model."
            )
            assert cfg.trainer.placement.policy_num_gpus_per_node == cfg.trainer.placement.ref_num_gpus_per_node, (
                f"policy_num_gpus_per_node ({cfg.trainer.placement.policy_num_gpus_per_node}) and ref_num_gpus_per_node ({cfg.trainer.placement.ref_num_gpus_per_node}) must be the same when colocate policy and ref model."
            )

    if cfg.generator.engine_init_timeout_seconds <= 0:
        raise ValueError("generator.engine_init_timeout_seconds must be greater than zero")
