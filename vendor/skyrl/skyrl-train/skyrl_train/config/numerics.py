"""Validation of ``trainer.numerics``, how the trainer and the rollout engines compute."""

from enum import StrEnum

from marinskyrl.runtime_options import NUMERICS_MODE_PATH, NUMERICS_PATH, NumericsMode
from omegaconf import DictConfig, OmegaConf

from skyrl_train.inference_engines.vllm.decode_invariant.engine import decode_invariant_engine_init_kwargs

# vLLM's default all2all_backend, whose expert-parallel combine runs NCCL's ring reduce-scatter.
ALLGATHER_REDUCESCATTER = "allgather_reducescatter"


class CheckFailure(StrEnum):
    """What the exactness check does when a token's log-probability differs: log it and continue, or stop the run."""

    LOG = "log"
    STOP = "stop"


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept a known mode; ``batch_invariant`` and ``exact`` only with local vLLM engines; ``exact`` only with a policy
    and the engine and trainer settings it runs on; a check interval of at least 0; and ``on_failure=stop`` only for
    ``exact``."""
    numerics = cfg.trainer.numerics
    if numerics.mode not in set(NumericsMode):
        raise ValueError(f"{NUMERICS_MODE_PATH} must be one of {', '.join(NumericsMode)}; got {numerics.mode!r}")
    if numerics.mode != NumericsMode.NATIVE and (
        cfg.generator.backend != "vllm" or not cfg.generator.run_engines_locally
    ):
        raise ValueError(
            f"{NUMERICS_MODE_PATH}={numerics.mode} requires generator.backend=vllm and "
            "generator.run_engines_locally=true, so both rollout and trainer activation is guaranteed; got "
            f"generator.backend={cfg.generator.backend}, generator.run_engines_locally={cfg.generator.run_engines_locally}"
        )
    if numerics.mode == NumericsMode.EXACT:
        _validate_exact_numerics(cfg)
    if numerics.check.every_weight_syncs < 0:
        raise ValueError(
            f"{NUMERICS_PATH}.check.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}"
        )
    if numerics.check.on_failure not in set(CheckFailure):
        raise ValueError(
            f"{NUMERICS_PATH}.check.on_failure must be one of {', '.join(CheckFailure)}; got {numerics.check.on_failure!r}"
        )
    if numerics.check.on_failure == CheckFailure.STOP and numerics.mode != NumericsMode.EXACT:
        raise ValueError(
            f"{NUMERICS_PATH}.check.on_failure=stop requires {NUMERICS_MODE_PATH}=exact; got "
            f"{NUMERICS_MODE_PATH}={numerics.mode}"
        )


def _validate_exact_numerics(cfg: DictConfig) -> None:
    """Raise unless the rollout engines of ``cfg`` can be decode-invariant and hand rollouts to the trainer in memory,
    and the exact trainer covers the policy and the trainer settings."""
    generator = cfg.generator
    kwargs = OmegaConf.to_container(generator.engine_init_kwargs, resolve=True)
    moe_backend = kwargs.get("moe_backend", (kwargs.get("kernel_config") or {}).get("moe_backend"))
    all2all_backend = kwargs.get("all2all_backend", ALLGATHER_REDUCESCATTER)
    # A decode-invariant engine sets its own kwargs and takes its attention backend from vllm_attention_backend. vLLM
    # runs logits processors only on its V1 model runner, which a decode-invariant engine refuses.
    refusable = (*decode_invariant_engine_init_kwargs(), "attention_backend", "speculative_config", "logits_processors")
    refused_kwargs = [key for key in refusable if kwargs.get(key)]
    requirements = {
        "trainer.rollout_buffer.object_store_root=null": cfg.trainer.rollout_buffer.object_store_root is None,
        "generator.speculative_decoding=null": generator.get("speculative_decoding") is None,
        "generator.vllm_attention_backend null or FLASH_ATTN": generator.vllm_attention_backend in (None, "FLASH_ATTN"),
        "generator.enforce_eager=false": not generator.enforce_eager,
        "generator.inference_engine_tensor_parallel_size=1": generator.inference_engine_tensor_parallel_size == 1,
        "generator.inference_engine_decode_context_parallel_size=1": (
            generator.inference_engine_decode_context_parallel_size == 1
        ),
        "generator.engine_init_kwargs.moe_backend=triton": moe_backend == "triton",
        f"generator.engine_init_kwargs.all2all_backend={ALLGATHER_REDUCESCATTER}": (
            all2all_backend == ALLGATHER_REDUCESCATTER
        ),
        f"generator.engine_init_kwargs without {', '.join(refused_kwargs)}": not refused_kwargs,
    }
    # Imported here: skyrl_train.models imports skyrl_train.utils, which imports this module.
    from skyrl_train.models.grug_exact.support import (  # noqa: PLC0415
        exact_model_problems,
        exact_trainer_problems,
        policy_model_config,
    )

    unmet = [requirement for requirement, met in requirements.items() if not met]
    unmet += exact_model_problems(policy_model_config(cfg)) + exact_trainer_problems(cfg)
    if unmet:
        raise ValueError(f"{NUMERICS_MODE_PATH}=exact requires {'; '.join(unmet)}")
