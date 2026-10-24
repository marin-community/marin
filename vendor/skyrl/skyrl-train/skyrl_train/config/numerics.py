"""How the trainer and the rollout engines compute, ``trainer.numerics``."""

from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from skyrl_train.inference_engines.vllm.decode_invariant.engine import decode_invariant_engine_init_kwargs

NUMERICS_KEY = "trainer.numerics"
# Engine kwargs a decode-invariant engine sets itself; its attention backend arrives through vllm_attention_backend.
DECODE_INVARIANT_ENGINE_KWARGS = (*decode_invariant_engine_init_kwargs(), "attention_backend")
# vLLM's default all2all_backend, whose expert-parallel combine runs NCCL's ring reduce-scatter.
ALLGATHER_REDUCESCATTER = "allgather_reducescatter"


class NumericsMode(StrEnum):
    """The kernels a run uses: ``native``, the trainer's and the engines' own; ``batch_invariant``, vLLM's
    batch-invariant kernels in both; ``exact``, decode-invariant vLLM engines and the trainer's own kernels."""

    NATIVE = "native"
    BATCH_INVARIANT = "batch_invariant"
    EXACT = "exact"


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept a known mode, ``batch_invariant`` and ``exact`` only with local vLLM engines, ``exact`` only with engine
    settings a decode-invariant engine runs on, and a check interval of at least 0."""
    numerics = cfg.trainer.numerics
    if numerics.mode not in set(NumericsMode):
        raise ValueError(f"{NUMERICS_KEY}.mode must be one of {', '.join(NumericsMode)}; got {numerics.mode!r}")
    if numerics.mode != NumericsMode.NATIVE and (
        cfg.generator.backend != "vllm" or not cfg.generator.run_engines_locally
    ):
        raise ValueError(
            f"{NUMERICS_KEY}.mode={numerics.mode} requires generator.backend=vllm and "
            "generator.run_engines_locally=true, so both rollout and trainer activation is guaranteed; got "
            f"generator.backend={cfg.generator.backend}, generator.run_engines_locally={cfg.generator.run_engines_locally}"
        )
    if numerics.mode == NumericsMode.EXACT:
        _validate_exact_numerics(cfg)
    if numerics.check.every_weight_syncs < 0:
        raise ValueError(
            f"{NUMERICS_KEY}.check.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}"
        )


def _validate_exact_numerics(cfg: DictConfig) -> None:
    """Raise unless the rollout engines of ``cfg`` can be decode-invariant and hand rollouts to the trainer in memory."""
    generator = cfg.generator
    kwargs = OmegaConf.to_container(generator.engine_init_kwargs, resolve=True)
    moe_backend = kwargs.get("moe_backend", (kwargs.get("kernel_config") or {}).get("moe_backend"))
    all2all_backend = kwargs.get("all2all_backend", ALLGATHER_REDUCESCATTER)
    # vLLM's model runner V2, which a decode-invariant engine runs, refuses logits processors.
    refused_kwargs = [
        key for key in (*DECODE_INVARIANT_ENGINE_KWARGS, "speculative_config", "logits_processors") if kwargs.get(key)
    ]
    requirements = {
        "trainer.rollout_buffer.object_store_root=null": cfg.trainer.rollout_buffer.object_store_root is None,
        "generator.speculative_decoding=null": generator.get("speculative_decoding") is None,
        "generator.vllm_attention_backend null or FLASH_ATTN": generator.vllm_attention_backend in (None, "FLASH_ATTN"),
        "generator.enforce_eager=false": not generator.enforce_eager,
        "generator.inference_engine_tensor_parallel_size=1": generator.inference_engine_tensor_parallel_size == 1,
        "generator.inference_engine_decode_context_parallel_size=1": (
            generator.inference_engine_decode_context_parallel_size == 1
        ),
        "engine_init_kwargs.moe_backend=triton": moe_backend == "triton",
        f"engine_init_kwargs.all2all_backend={ALLGATHER_REDUCESCATTER}": all2all_backend == ALLGATHER_REDUCESCATTER,
        f"engine_init_kwargs without {', '.join(refused_kwargs)}": not refused_kwargs,
    }
    unmet = [requirement for requirement, met in requirements.items() if not met]
    if unmet:
        raise ValueError(f"{NUMERICS_KEY}.mode=exact requires {'; '.join(unmet)}")
