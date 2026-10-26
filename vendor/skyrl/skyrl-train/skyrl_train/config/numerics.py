"""Validation of ``trainer.numerics``, how the trainer and the rollout engines compute."""

from marinskyrl.runtime_options import NUMERICS_MODE_PATH, NUMERICS_PATH, NumericsMode
from omegaconf import DictConfig


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept a known mode, ``batch_invariant`` only with local vLLM engines, and a check interval of at least 0."""
    numerics = cfg.trainer.numerics
    if numerics.mode not in set(NumericsMode):
        raise ValueError(f"{NUMERICS_MODE_PATH} must be one of {', '.join(NumericsMode)}; got {numerics.mode!r}")
    if numerics.mode == NumericsMode.BATCH_INVARIANT and (
        cfg.generator.backend != "vllm" or not cfg.generator.run_engines_locally
    ):
        raise ValueError(
            f"{NUMERICS_MODE_PATH}=batch_invariant requires generator.backend=vllm and "
            "generator.run_engines_locally=true, so both rollout and trainer activation is guaranteed; got "
            f"generator.backend={cfg.generator.backend}, generator.run_engines_locally={cfg.generator.run_engines_locally}"
        )
    if numerics.check.every_weight_syncs < 0:
        raise ValueError(
            f"{NUMERICS_PATH}.check.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}"
        )
