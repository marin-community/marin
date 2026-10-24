from pathlib import Path

import ray
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM, GrugMoeGatedNorm
from skyrl_train.trainer import policy_forward_keys
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.utils import get_test_actor_config

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def write_snowball_checkpoint(
    path: Path,
    *,
    num_hidden_layers: int = 4,
    num_local_experts: int = 256,
    router_weight_scale: float = 1.0,
    vocab_size: int | None = None,
) -> GrugMoeConfig:
    """Write a Grug model with Snowball's attention geometry, width, shared-expert width, norm epsilon and 2,048-token
    window, cut to ``num_hidden_layers`` layers with ``num_local_experts`` narrow experts, random weights and
    non-trivial gates; the router weights are scaled by ``router_weight_scale``, and ``vocab_size`` replaces
    Snowball's 128,256-entry vocabulary."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vocabulary = {} if vocab_size is None else {"vocab_size": vocab_size}
    config = GrugMoeConfig(
        num_hidden_layers=num_hidden_layers,
        num_local_experts=num_local_experts,
        intermediate_size=128,
        initializer_range=0.02,
        **vocabulary,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GrugMoeGatedNorm):
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
            # A decode-invariant engine multiplies the router weight in bf16 and refuses values bf16 cannot hold.
            layer.mlp.router.weight.copy_((layer.mlp.router.weight * router_weight_scale).to(torch.bfloat16))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return config


def exact_numerics_config(
    model_path: str, *, policy_gpus: int, pipeline_stages: int, expert_parallel: int, engine_ranks: int
) -> DictConfig:
    """An unvalidated exact-numerics run of the checkpoint at ``model_path``: a Megatron policy on ``policy_gpus``
    GPUs of one node with ``pipeline_stages`` stages and ``expert_parallel`` expert ranks, no reference model, and
    one decode-invariant engine of ``engine_ranks`` data- and expert-parallel ranks sampling at temperature 1."""
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.numerics.mode = "exact"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = policy_gpus
    cfg.trainer.placement.ref_num_gpus_per_node = policy_gpus
    megatron = cfg.trainer.policy.megatron_config
    megatron.tensor_model_parallel_size = 1
    megatron.pipeline_model_parallel_size = pipeline_stages
    megatron.context_parallel_size = 1
    megatron.expert_model_parallel_size = expert_parallel
    megatron.expert_tensor_parallel_size = 1
    megatron.ddp_config.overlap_param_gather = False
    cfg.generator.backend = "vllm"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = engine_ranks
    cfg.generator.inference_engine_expert_parallel_size = engine_ranks
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.engine_init_kwargs = {"moe_backend": "triton"}
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    return cfg


def policy_logprobs(policy: PPORayActorGroup, batch: TrainingInputBatch) -> torch.Tensor:
    """The ``[rows, response_length]`` float32 log-probabilities of ``batch``'s responses under ``policy``'s scoring
    forward, sent the fields the trainer's pre-update forward sends, on the CPU."""
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch.select(policy_forward_keys(batch))))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float().cpu()
