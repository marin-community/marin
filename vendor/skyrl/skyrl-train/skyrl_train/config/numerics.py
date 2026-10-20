"""How the trainer and the rollout engines compute, ``trainer.numerics``."""

from enum import StrEnum

from omegaconf import DictConfig

NUMERICS_KEY = "trainer.numerics"


class NumericsMode(StrEnum):
    """The kernels a run uses: ``native``, the trainer's and the engines' own; ``batch_invariant``, vLLM's
    batch-invariant kernels in both."""

    NATIVE = "native"
    BATCH_INVARIANT = "batch_invariant"


def validate_numerics_config(cfg: DictConfig) -> None:
    """Validate the mode, local-engine requirement and nonnegative check interval."""
    numerics = cfg.trainer.numerics
    if numerics.mode not in set(NumericsMode):
        raise ValueError(f"{NUMERICS_KEY}.mode must be one of {', '.join(NumericsMode)}; got {numerics.mode!r}")
    if numerics.mode == NumericsMode.BATCH_INVARIANT and not cfg.generator.run_engines_locally:
        raise ValueError(
            f"{NUMERICS_KEY}.mode=batch_invariant cannot configure a remote inference server; run the vLLM "
            "engines locally so both rollout and trainer activation is guaranteed"
        )
    if numerics.check.every_weight_syncs < 0:
        raise ValueError(
            f"{NUMERICS_KEY}.check.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}"
        )
