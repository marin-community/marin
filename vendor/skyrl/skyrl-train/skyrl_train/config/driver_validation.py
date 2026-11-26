"""Validate the composed configuration shared by SkyRL plans and drivers."""

import math
import os

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rigging.secrets import require_secret
from skyrlrecipe.declared import PromptAcquisition

from marinskyrl.distillation import (
    compile_distillation_plan_from_config,
    validate_distillation_runtime_support,
    validate_generation_logprobs,
)
from marinskyrl.rollout_grading import validate_nemotron_ultra_grading
from marinskyrl.runtime_options import (
    AdvantageEstimator,
    BatchBuilder,
    OUTCOME_ADVANTAGE_ESTIMATORS,
    PolicyLossType,
    reference_model_required,
)
from skyrl_train.config.behavior_logprobs import configure_behavior_logprob_sampling
from skyrl_train.config.callbacks import validate_hf_export_config
from skyrl_train.config.mismatch_probe import validate_mismatch_probe_config
from skyrl_train.config.objective_spec import BUILTIN_LOSS_SPECS, rollout_logprobs_required, validate_objective
from skyrl_train.config.progress import ProgressSettings
from skyrl_train.config.query_bias import resolve_grug_query_bias_update
from skyrl_train.config.reward_shaping import parse_trajectory_reward_shaping_config
from skyrl_train.config.rollout_validation import EntrypointOperation, validate_rollout_launch
from skyrl_train.config.sampling import validate_sampling_reversion
from skyrl_train.config.trajectory_selection import best_of_n_sample_count, optimization_samples_per_prompt
from skyrl_train.config.utils import generation_context_limit, model_expert_count, normalize_logging_backends
from skyrl_train.config.validation import (
    validate_batch_invariant_config,
    validate_megatron_config,
    validate_training_settings,
)
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.dynamic_sampling import resolve_dynamic_sampling_criteria
from skyrl_train.group_admission import (
    BUILTIN_GROUP_ADVANTAGE_CONTRACTS,
    NoGroupAdvantage,
    resolve_group_advantage_invariant,
)
from skyrl_train.trajectory_runners.trajectory_retention_config import parse_trajectory_retention_config


def moe_router_replay_requested(cfg: DictConfig, role: str = "policy") -> bool:
    return bool(cfg.trainer[role].megatron_config.get("moe_router_replay", False))


def validate_batch_sizes(cfg: DictConfig):
    """
    Validate configured batch sizes.

    Explanation of how batching operates:
    1. Each prompt creates `optimization_samples_per_prompt` learner rows after trajectory selection.
    2. During training, these rows are split across data parallel (DP) workers.
    3. Mini batches are normalized using that same post-selection row count.
    4. Per-gpu train batch size must be divisble by per-gpu mini batch size, otherwise the last mini batch will be incomplete.
    5. Per-gpu mini batch size must be divisible by per-gpu micro batch size, otherwise the last micro batch will be incomplete.
    """
    assert cfg.trainer.train_batch_size >= cfg.trainer.policy_mini_batch_size
    assert cfg.trainer.policy_mini_batch_size > 0, "policy_mini_batch_size must be greater than 0"
    assert cfg.trainer.micro_train_batch_size_per_gpu > 0, "micro_train_batch_size_per_gpu must be greater than 0"
    assert cfg.trainer.micro_forward_batch_size_per_gpu > 0, "micro_forward_batch_size_per_gpu must be greater than 0"

    # Validate policy mini batch size
    policy_world_size = cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node

    pp = cfg.trainer.policy.megatron_config.pipeline_model_parallel_size
    cp = cfg.trainer.policy.megatron_config.context_parallel_size
    tp = cfg.trainer.policy.megatron_config.tensor_model_parallel_size
    assert policy_world_size % (pp * cp * tp) == 0, (
        f"policy_world_size {policy_world_size} should be divisible by (pp * cp * tp) {pp * cp * tp}. "
        "This ensures that the data parallel size is an integer."
    )
    policy_dp_size = policy_world_size // (pp * cp * tp)

    assert cfg.trainer.train_batch_size % cfg.trainer.policy_mini_batch_size == 0, (
        f"train_batch_size {cfg.trainer.train_batch_size} should be divisible by policy_mini_batch_size {cfg.trainer.policy_mini_batch_size}"
    )
    optimization_group_size = optimization_samples_per_prompt(cfg)
    if (
        cfg.trainer.batch_builder == BatchBuilder.WORKER
        and cfg.trainer.train_batch_size * optimization_group_size % policy_dp_size
    ):
        raise ValueError("trainer.batch_builder=worker does not support rows requiring DP padding")
    policy_mini_batch_size_per_gpu = cfg.trainer.policy_mini_batch_size * optimization_group_size // policy_dp_size
    assert policy_mini_batch_size_per_gpu > 0, (
        f"Invalid policy_mini_batch_size_per_gpu: {policy_mini_batch_size_per_gpu}. "
        f"mini_batch_size={cfg.trainer.policy_mini_batch_size}, "
        f"optimization_samples_per_prompt={optimization_group_size}, "
        f"dp_size={policy_dp_size}"
    )
    assert policy_mini_batch_size_per_gpu % cfg.trainer.micro_train_batch_size_per_gpu == 0, (
        f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be divisible by micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    )
    assert policy_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu > 0, (
        f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be larger than micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    )
    policy_train_batch_size_per_gpu = cfg.trainer.train_batch_size * optimization_group_size // policy_dp_size

    # `train_batch_size_per_gpu` should be divisible by `policy_mini_batch_size_per_gpu`
    assert policy_train_batch_size_per_gpu % policy_mini_batch_size_per_gpu == 0, (
        f"normalized policy_train_batch_size_per_gpu (train_batch_size * optimization_samples_per_prompt // policy_dp_size) {policy_train_batch_size_per_gpu} should be divisible by policy_mini_batch_size_per_gpu (policy_mini_batch_size * optimization_samples_per_prompt // policy_dp_size) {policy_mini_batch_size_per_gpu}"
    )

    # Validate training batch size is larger than the least common multiple of the DP sizes of policy (and ref if used).
    lcm_dp_size = policy_dp_size

    use_ref_model = reference_model_required(cfg.trainer.algorithm)
    if use_ref_model:
        ref_world_size = cfg.trainer.placement.ref_num_nodes * cfg.trainer.placement.ref_num_gpus_per_node
        pp = cfg.trainer.ref.megatron_config.pipeline_model_parallel_size
        cp = cfg.trainer.ref.megatron_config.context_parallel_size
        tp = cfg.trainer.ref.megatron_config.tensor_model_parallel_size
        assert ref_world_size % (pp * cp * tp) == 0, (
            f"ref_world_size {ref_world_size} should be divisible by (pp * cp * tp) {pp * cp * tp}. "
            "This ensures that the data parallel size is an integer."
        )
        ref_dp_size = ref_world_size // (pp * cp * tp)
        lcm_dp_size = math.lcm(lcm_dp_size, ref_dp_size)

    assert cfg.trainer.train_batch_size >= lcm_dp_size, (
        f"train_batch_size ({cfg.trainer.train_batch_size}) should be larger than or equal to the least common multiple of the data parallel sizes of the enabled models: "
        f"policy_dp_size={policy_dp_size}, "
        f"ref_dp_size={ref_dp_size if use_ref_model else 'None'}, "
        f"lcm_dp_size={lcm_dp_size}"
    )


_OUTCOME_LOSSES = frozenset(
    {
        PolicyLossType.REGULAR,
        PolicyLossType.DUAL_CLIP,
        PolicyLossType.GSPO,
        PolicyLossType.CISPO,
        PolicyLossType.SAPO,
        PolicyLossType.IMPORTANCE_SAMPLING,
        PolicyLossType.BEHAVIOR_CLIP,
    }
)


def require_worker_batch_config(cfg: DictConfig, *, check_runtime_resources: bool = True) -> None:
    """Reject configuration settings unsupported by policy-worker assembly."""
    algorithm = cfg.trainer.algorithm
    shaping = parse_trajectory_reward_shaping_config(cfg.generator.get("trajectory_reward_shaping"))
    megatron = cfg.trainer.policy.megatron_config
    expert_tensor_parallel = megatron.get("expert_tensor_parallel_size") or megatron.tensor_model_parallel_size
    moe_etp = False
    if expert_tensor_parallel > 1:
        model = cfg.trainer.policy.model
        # The plan can check MoE expert parallelism when model metadata is already staged.
        if check_runtime_resources or os.path.isfile(os.path.join(model.path, "config.json")):
            from transformers import AutoConfig

            # Model registration imports utils while initializing the model package.
            from skyrl_train.models import register_local_models

            register_local_models()
            metadata = AutoConfig.from_pretrained(
                model.path,
                trust_remote_code=True,
                revision=model.get("revision"),
                local_files_only=not check_runtime_resources,
            )
            moe_etp = model_expert_count(metadata) is not None
    supported_loss = (
        algorithm.policy_loss_type in _OUTCOME_LOSSES and algorithm.advantage_estimator in OUTCOME_ADVANTAGE_ESTIMATORS
    ) or (
        algorithm.policy_loss_type == PolicyLossType.SFT and algorithm.advantage_estimator == AdvantageEstimator.UNIFORM
    )
    restrictions = [
        ("DPO and reference-model objectives", reference_model_required(algorithm)),
        ("the configured loss and estimator", not supported_loss),
        ("colocated placement", cfg.trainer.placement.colocate_all),
        ("distillation", algorithm.get("distillation") is not None),
        ("trajectory selection", cfg.trainer.trajectory_selector.type is not None),
        ("step-wise training", cfg.trainer.step_wise_training),
        ("advantage batch normalization", algorithm.advantage_batch_normalize),
        ("loop advantage credit", shaping.enabled and shaping.loop.advantage_penalty_per_token > 0),
        ("marker-target reasoning penalties", shaping.enabled and shaping.reasoning_span.charges_tokens),
        ("ETP greater than one", moe_etp),
        ("dense batch dumping", cfg.trainer.dump_data_batch),
    ]
    for reason, unsupported in restrictions:
        if unsupported:
            raise ValueError(f"trainer.batch_builder=worker does not support {reason}")


def validate_cfg(cfg: DictConfig, *, check_runtime_resources: bool = True):
    if BatchBuilder(cfg.trainer.batch_builder) is BatchBuilder.WORKER:
        require_worker_batch_config(cfg, check_runtime_resources=check_runtime_resources)
    validate_training_settings(cfg)
    validate_mismatch_probe_config(cfg)
    distillation_plan = compile_distillation_plan_from_config(cfg)
    validate_distillation_runtime_support(distillation_plan)
    validate_nemotron_ultra_grading(cfg, distillation_plan)
    uses_trajectory_selector = best_of_n_sample_count(cfg) is not None
    if uses_trajectory_selector:
        if cfg.trainer.step_wise_training:
            raise ValueError("best-of-N selection does not support step-wise training")
    resolve_dynamic_sampling_criteria(
        cfg.trainer.algorithm.dynamic_sampling.informative_on,
        float(cfg.trainer.algorithm.dynamic_sampling.min_reward_std),
        cfg.trainer.algorithm.dynamic_sampling.max_mean_reward,
    )
    if check_runtime_resources:
        from skyrl_train.utils.algorithm_registry import AdvantageEstimatorRegistry, PolicyLossRegistry

        available_policy_losses = PolicyLossRegistry.list_available()
        loss_spec = PolicyLossRegistry.spec
        group_contract = AdvantageEstimatorRegistry.group_contract
    else:
        available_policy_losses = [str(name) for name in BUILTIN_LOSS_SPECS]
        loss_spec = BUILTIN_LOSS_SPECS.__getitem__
        group_contract = BUILTIN_GROUP_ADVANTAGE_CONTRACTS.__getitem__
    assert available_policy_losses != [], "Policy loss registry is not populated."
    assert cfg.trainer.algorithm.policy_loss_type in available_policy_losses, (
        f"invalid policy_loss_type: {cfg.trainer.algorithm.policy_loss_type}. Must be one of {available_policy_losses}"
    )
    spec = loss_spec(cfg.trainer.algorithm.policy_loss_type)
    validate_objective(cfg, loss_spec=spec)
    resolve_weight_sync_pause_policy(cfg.generator)
    validate_generator_cfg(cfg, check_runtime_resources=check_runtime_resources)
    validate_batch_invariant_config(cfg)
    validate_hf_export_config(cfg)
    try:
        resolve_grug_query_bias_update(cfg.trainer.policy)
    except ValueError as error:
        raise AssertionError(str(error)) from error

    assert not (cfg.trainer.algorithm.use_kl_in_reward and cfg.trainer.algorithm.use_kl_loss), (
        "use_kl_in_reward and use_kl_loss should be mutually exclusive"
    )

    validate_batch_sizes(cfg)

    if cfg.trainer.max_ckpts_to_keep == 0:
        raise ValueError(
            "`max_ckpts_to_keep` must be greater than 0 to keep the last N checkpoints or negative to keep all checkpoints"
        )

    available_advantage_estimators = (
        AdvantageEstimatorRegistry.list_available()
        if check_runtime_resources
        else [str(name) for name in BUILTIN_GROUP_ADVANTAGE_CONTRACTS]
    )
    assert cfg.trainer.algorithm.advantage_estimator in available_advantage_estimators, (
        f"invalid advantage_estimator: {cfg.trainer.algorithm.advantage_estimator}. Must be one of {available_advantage_estimators}"
    )
    contract = group_contract(cfg.trainer.algorithm.advantage_estimator)
    if uses_trajectory_selector and not isinstance(contract, NoGroupAdvantage):
        raise ValueError("best-of-N selection requires a no-group advantage estimator")

    # add field to algorithm config needed for loss functions
    # create a new config to make it modifiable
    algorithm_config = OmegaConf.create(cfg.trainer.algorithm)
    group_advantage = resolve_group_advantage_invariant(
        advantage_estimator=str(algorithm_config.advantage_estimator),
        physical_group_size=optimization_samples_per_prompt(cfg),
        minimum_group_size=algorithm_config.group_advantage_min_size,
        contract=contract,
    )
    algorithm_config.resolved_group_advantage = group_advantage.to_config()
    # Sequence-normalized losses use the same bound as model generation.
    algorithm_config.max_seq_len = generation_context_limit(cfg.generator)

    cfg.trainer.algorithm = algorithm_config

    behavior_logprobs_required = rollout_logprobs_required(cfg.trainer.algorithm, loss_spec=spec) or bool(
        cfg.trainer.mismatch_probe.get("enabled", False)
    )
    if behavior_logprobs_required:
        if cfg.generator.sampling_params.logprobs is None:
            logger.warning(
                "The selected objective requires rollout logprobs; setting generator.sampling_params.logprobs=0."
            )
            cfg.generator.sampling_params.logprobs = 0
        configure_behavior_logprob_sampling(cfg.generator)


def validate_generator_cfg(cfg: DictConfig, *, check_runtime_resources: bool = True):
    """Validate generation settings before engine startup."""

    if cfg.generator.backend != "vllm":
        raise ValueError(f"generator.backend must be 'vllm', got {cfg.generator.backend!r}")

    parse_trajectory_reward_shaping_config(cfg.generator.get("trajectory_reward_shaping"))
    parse_trajectory_retention_config(cfg.generator.get("trajectory_retention"))

    if cfg.generator.max_turns == 1:
        assert cfg.generator.max_input_length == cfg.trainer.max_prompt_length, (
            "generator.max_input_length should be set equal to trainer.max_prompt_length for single-turn generation"
        )
    else:
        assert cfg.generator.max_input_length >= cfg.trainer.max_prompt_length, (
            "generator.max_input_length should be set greater than or equal to trainer.max_prompt_length for multi-turn generation"
        )

    if not cfg.generator.run_engines_locally:
        assert cfg.generator.num_inference_engines == len(cfg.generator.remote_inference_engine_urls), (
            "num_inference_engines should be equal to the number of remote_inference_engine_urls"
        )

    # W&B credentials are supplied by the launch environment.
    if check_runtime_resources and "wandb" in cfg.trainer.logger:
        require_secret("WANDB_API_KEY")

    if cfg.generator.override_existing_update_group == "auto":
        if cfg.generator.backend == "vllm" and not cfg.generator.run_engines_locally:
            # remote engines can be launched separately so we `enable` by default
            cfg.generator.override_existing_update_group = "enable"
        else:
            # Local engines manage their update group lifecycle.
            cfg.generator.override_existing_update_group = "disable"

    validate_generation_logprobs(cfg)

    validate_megatron_config(cfg)

    # Validate inference engine parallelism.
    ep_size = cfg.generator.inference_engine_expert_parallel_size
    dp_size = cfg.generator.inference_engine_data_parallel_size
    tp_size = cfg.generator.inference_engine_tensor_parallel_size
    if ep_size > 1:
        assert dp_size * tp_size == ep_size, (
            f"If inference expert parallel is enabled, data parallel size * tensor parallel size must equal expert parallel size. "
            f"Got dp_size={dp_size}, tp_size={tp_size}, ep_size={ep_size}"
        )

    # Validate inference engine Decode Context Parallel (DCP).
    _validate_dcp_cfg(cfg, local_files_only=not check_runtime_resources)


def _is_mla_arch(model_cfg) -> bool:
    """Whether an HF model config describes a Multi-head Latent Attention (MLA) arch.

    MLA models (DeepSeek-V2/V3, Kimi, etc.) compress the KV into a single shared latent
    rather than exposing `num_key_value_heads` GQA groups, so the per-head DCP bound does
    not apply the same way (DCP shards the latent; the effective kv-head count for the
    bound is 1 -> the bound relaxes to dcp <= tp). We detect MLA conservatively via the
    presence of the MLA-specific latent-rank fields that DeepSeek/Kimi configs carry.
    """
    if model_cfg is None:
        return False
    # DeepSeek/Kimi MLA configs carry the compressed-KV latent rank. Either field present
    # is a reliable MLA marker across the DeepSeek-V2/V3/Kimi family.
    for field in ("kv_lora_rank", "q_lora_rank"):
        val = getattr(model_cfg, field, None)
        if val is not None:
            return True
    # Fall back to architecture-name sniffing for forks that drop the latent-rank fields.
    archs = getattr(model_cfg, "architectures", None) or []
    if any(("deepseek" in a.lower() or "kimi" in a.lower()) for a in archs):
        return True
    return False


def _resolve_num_kv_heads(model_path: str, *, local_files_only: bool):
    """Resolve (num_kv_heads, is_mla, model_cfg) from an HF model config, offline-safe.

    Returns (None, False, None) if the config cannot be resolved without a network
    download (so the caller degrades gracefully to the cheap dcp<=tp bound + warning).
    For GQA/MHA models `num_kv_heads = num_key_value_heads` (falling back to
    `num_attention_heads` for pure MHA, which has no separate kv-head count). For MLA
    models the effective kv-head count is 1 (single shared latent).
    """
    try:
        from transformers import AutoConfig
    except Exception:  # transformers unavailable -> cannot resolve
        return None, False, None
    try:
        # Respect offline / cached configs: do not force a network download at
        # validate-time. If the config isn't locally resolvable, transformers raises
        # (offline) or we let any error fall through to the graceful-degrade path.
        model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)
    except Exception:
        return None, False, None

    is_mla = _is_mla_arch(model_cfg)
    if is_mla:
        return 1, True, model_cfg
    # text_config nesting (multimodal / wrapped configs) — prefer the inner text config.
    inner = getattr(model_cfg, "text_config", None) or model_cfg
    num_kv_heads = getattr(inner, "num_key_value_heads", None)
    if num_kv_heads is None:
        # Pure MHA: no separate kv-head count -> kv-heads == attention heads.
        num_kv_heads = getattr(inner, "num_attention_heads", None)
    if num_kv_heads is None:
        return None, False, model_cfg
    return int(num_kv_heads), False, model_cfg


def _assert_dcp_capable_arch(model_cfg, dcp: int):
    """(G3 f) Assert the model arch has a DCP-capable attention backend.

    DCP is supported for GQA models (default FlashAttention backend) and MLA models
    (FlashMLA / FlashAttnMLA). We keep the gate permissive: any standard decoder runs on
    FlashAttention's GQA path (DCP-capable), and MLA is explicitly capable. We reject only
    when the config positively declares a non-DCP-capable attention implementation
    (e.g. an arch forced onto an attn backend with no DCP path).
    """
    if model_cfg is None:
        # Could not resolve the config offline -> can't positively reject; rely on the
        # cheap bound + the vLLM init-time assert. (Handled by the caller's warning.)
        return
    if _is_mla_arch(model_cfg):
        return  # MLA is DCP-capable (FlashMLA / FlashAttnMLA).
    # GQA / dense decoders run on FlashAttention (DCP-capable). Reject only an explicit,
    # known-incapable attention implementation pinned in the config.
    attn_impl = getattr(model_cfg, "_attn_implementation", None) or getattr(model_cfg, "attn_implementation", None)
    incapable = {"eager", "sdpa"}
    if attn_impl is not None and str(attn_impl).lower() in incapable:
        raise AssertionError(
            f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) "
            f"requires a DCP-capable attention backend (FlashAttention for GQA, FlashMLA for MLA), "
            f"but the model config pins attn_implementation='{attn_impl}', which has no DCP path. "
            f"Use a GQA/MLA model on FlashAttention, or disable DCP."
        )


def _validate_dcp_cfg(cfg: DictConfig, *, local_files_only: bool):
    """Validate the vLLM Decode Context Parallel (DCP) generator config (Stages 0–2).

    vLLM DCP shards the KV cache along the token dim across `dcp` ranks *inside* the
    TP group during decode (it reuses the TP GPUs, adding no GPUs — G4). It is gated
    entirely off by default (`inference_engine_decode_context_parallel_size == 1`), in
    which case this function is a strict no-op and the run is byte-identical to today
    (G1). When enabled (`dcp > 1`) we fail-closed at SkyRL config-validate (G3) rather
    than deep in vLLM init, with clear messages:

      (a) tp % dcp == 0 (vLLM splits the TP group into dcp subgroups; parallel.py:474-478),
      (b) dcp <= tp // num_kv_heads(model) — the real upper bound; beyond tp/H the KV is
          merely duplicated and there is no non-attention work to shard (doc :27,29).
          Resolved model-aware from the HF config; degrades to the cheap dcp<=tp bound +
          a warning when the config is not offline-resolvable.
      (c) backend == "vllm",
      (e) NOT R3 router capture (vLLM rejects DCP + enable_return_routed_experts;
          vllm/config/vllm.py:1939-1944),
      (f) the arch is DCP-capable (GQA via FlashAttention, or MLA via FlashMLA).

    Remote engines (`run_engines_locally == False`): SkyRL does not launch the inference
    server, so the `-dcp N` flag must be set on the EXTERNAL `vllm serve` command. We run
    the geometry checks (a)/(b)/(f) where the config is resolvable and emit a warning
    reminding the operator to pass `-dcp` externally; we do not inject dcp into a server
    SkyRL doesn't spawn.

    See notes/RL/skyrl/vllm_dcp_rollout_stages/.
    """
    gen = cfg.generator
    dcp = gen.inference_engine_decode_context_parallel_size
    tp = gen.inference_engine_tensor_parallel_size
    assert dcp >= 1, f"generator.inference_engine_decode_context_parallel_size must be >= 1, got {dcp}"
    if dcp == 1:
        # DCP disabled -> strict no-op, no further constraints (G1).
        return

    # (a) vLLM splits the TP group into `dcp` subgroups -> tp must be divisible by dcp.
    assert tp % dcp == 0, (
        f"vLLM decode context parallel requires tensor_parallel_size % dcp == 0 "
        f"(DCP splits the TP group); got inference_engine_tensor_parallel_size={tp}, "
        f"inference_engine_decode_context_parallel_size={dcp}"
    )
    # (cheap bound) 1 <= dcp <= tp. Always enforced; the tighter kv-head bound below
    # supersedes it when the model config resolves offline.
    assert 1 <= dcp <= tp, (
        f"inference_engine_decode_context_parallel_size must satisfy 1 <= dcp <= "
        f"inference_engine_tensor_parallel_size; got dcp={dcp}, tp={tp}"
    )
    assert gen.backend == "vllm", (
        f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) "
        f"is only supported with generator.backend='vllm', got '{gen.backend}'"
    )

    # (b)/(f) model-aware bounds: resolve the HF config offline-safely.
    model_path = cfg.trainer.policy.model.path
    num_kv_heads, is_mla, model_cfg = _resolve_num_kv_heads(model_path, local_files_only=local_files_only)
    if num_kv_heads is None:
        # Could not resolve the config without a network download -> degrade gracefully:
        # keep the cheap dcp<=tp bound (already asserted) and warn, leaning on vLLM's own
        # init-time assert for the exact kv-head bound.
        logger.warning(
            f"DCP enabled (dcp={dcp}) but could not resolve the HF model config for "
            f"'{model_path}' offline -> skipping the model-aware kv-head bound "
            f"(dcp <= tp/num_kv_heads) and arch gate. The cheap bound (1 <= dcp <= tp) is "
            f"enforced; vLLM's engine init will enforce the exact dcp <= tp/num_kv_heads bound."
        )
    else:
        # (b) real upper bound: dcp <= tp // num_kv_heads. For MLA num_kv_heads==1 so this
        # relaxes to dcp <= tp (vLLM's own init enforces the exact MLA bound).
        max_dcp = tp // num_kv_heads
        kv_kind = "MLA latent (1 effective kv-head)" if is_mla else f"num_key_value_heads={num_kv_heads}"
        assert dcp <= max_dcp, (
            f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) "
            f"exceeds the kv-head bound dcp <= tp // num_kv_heads = {tp} // {num_kv_heads} = {max_dcp} "
            f"for model '{model_path}' ({kv_kind}). Beyond tp/num_kv_heads the KV cache is merely "
            f"duplicated and there is no non-attention work to shard; reduce dcp to <= {max_dcp}."
        )
        # (f) arch must be DCP-capable (GQA via FlashAttention, or MLA).
        _assert_dcp_capable_arch(model_cfg, dcp)

    # Remote path: SkyRL doesn't launch the server, so `-dcp` must be set externally.
    if not bool(gen.get("run_engines_locally", True)):
        logger.warning(
            f"DCP enabled (dcp={dcp}) with run_engines_locally=False: SkyRL does NOT launch "
            f"remote inference servers, so the external `vllm serve` command MUST be started "
            f"with `-dcp {dcp}` (and matching `--tensor-parallel-size {tp}`). SkyRL only carries "
            f"dcp as client metadata for geometry/GPU-accounting consistency."
        )
    # (e) vLLM rejects DCP together with R3 router capture (enable_return_routed_experts).
    # R3 capture is configured at the generator level (direct flag or engine_init_kwargs),
    # Training-side replay is configured through the Megatron policy config.
    #
    # Opt-in bypass: VLLM_ALLOW_ROUTED_EXPERTS_DCP=1 lifts this guard, mirroring the
    # identical env-var-gated bypass in the patched vLLM fork
    # (vllm/config/vllm.py:_validate_return_routed_experts). DCP+R3 is NOT a fundamental
    # incompatibility — MoE routing is a per-token function of each token's hidden state at
    # the MoE layers, independent of how DCP shards the attention decode-KV; each DCP rank
    # captures routing for its own tokens and the buffer is gathered like the output tokens.
    # Validated by the dcp=1-vs-dcp=2 routed-experts parity smoke (token mismatch 6.94% =
    # bf16-tie floor; job 905835 / DCP GQA-LSE fp32 fix). Fail-closed by default — without
    # the env var the original hard assertion stands. (Mirrors the vLLM guard so the two
    # never disagree; the SIF's vLLM honors the same flag.)
    r3_capture = (
        bool(gen.get("enable_return_routed_experts", False))
        or bool(gen.get("engine_init_kwargs", {}).get("enable_return_routed_experts", False))
        or moe_router_replay_requested(cfg)
    )
    allow_routed_experts_dcp = os.environ.get("VLLM_ALLOW_ROUTED_EXPERTS_DCP", "0") == "1"
    if r3_capture and allow_routed_experts_dcp:
        logger.warning(
            f"VLLM_ALLOW_ROUTED_EXPERTS_DCP=1: allowing decode context parallel "
            f"(inference_engine_decode_context_parallel_size={dcp}) together with R3 router "
            f"capture (enable_return_routed_experts / moe_router_replay). Out of vLLM "
            f"PR#39917's validated scope; relies on the dcp=1-vs-dcp=2 routed-experts parity "
            f"smoke (bf16-tie floor). Verify routed-experts capture parity before trusting results."
        )
    else:
        assert not r3_capture, (
            f"decode context parallel (inference_engine_decode_context_parallel_size={dcp}) is not "
            f"compatible with R3 router capture (enable_return_routed_experts / moe_router_replay); "
            f"vLLM rejects DCP + return_routed_experts. Disable one of them, or set "
            f"VLLM_ALLOW_ROUTED_EXPERTS_DCP=1 to opt into the parity-validated bypass."
        )

    # (G4) DCP rides the TP GPUs and must NOT enter rollout-GPU accounting. Assert the
    # rollout-GPU count is a function of tp*pp*dp only, independent of dcp.
    pp = gen.inference_engine_pipeline_parallel_size
    dp = gen.inference_engine_data_parallel_size
    per_engine_gpu_count = tp * pp * dp
    num_rollout_gpus = gen.num_inference_engines * per_engine_gpu_count
    assert num_rollout_gpus == gen.num_inference_engines * tp * pp * dp, (
        "DCP must not change rollout-GPU accounting (it reuses the TP GPUs); "
        f"num_rollout_gpus must equal num_inference_engines*tp*pp*dp regardless of dcp={dcp}"
    )


def _validate_paired_rollout_seed(cfg: DictConfig, operation: EntrypointOperation) -> None:
    seed = OmegaConf.select(cfg, "data.sampling.paired_rollout_seed")
    if seed is not None and (
        type(seed) is not int
        or seed < 0
        or operation is not EntrypointOperation.TRAIN
        or cfg.trainer.rollout_buffer.max_staleness_steps != 0
        or cfg.data.sampling.kind != PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE
        or cfg.generator.max_turns != 1
    ):
        raise ValueError(
            "data.sampling.paired_rollout_seed requires a nonnegative integer and synchronous one-turn source passes"
        )


def validate_driver_config(
    cfg: DictConfig, operation: EntrypointOperation, *, check_runtime_resources: bool = True
) -> None:
    """Apply the shared Ray driver's pre-start configuration checks."""
    if check_runtime_resources:
        from skyrl_train import objective  # noqa: F401 - register the native training losses

    normalize_logging_backends(cfg)
    _validate_paired_rollout_seed(cfg, operation)
    validate_rollout_launch(cfg, operation)
    if operation is EntrypointOperation.TRAIN:
        validate_sampling_reversion(cfg.data.sampling)
    validate_cfg(cfg, check_runtime_resources=check_runtime_resources)
    if check_runtime_resources:
        from skyrl_train.utils.progress import configure_progress

        configure_progress(cfg.trainer.progress)
    else:
        ProgressSettings.from_config(cfg.trainer.progress)


def validate_generation_driver_config(cfg: DictConfig, *, check_runtime_resources: bool = True) -> None:
    """Apply the evaluation-only driver's pre-start configuration checks."""
    normalize_logging_backends(cfg)
    _validate_paired_rollout_seed(cfg, EntrypointOperation.GENERATE)
    validate_generator_cfg(cfg, check_runtime_resources=check_runtime_resources)
