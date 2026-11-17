"""Local model registration used by drivers and worker processes."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from skyrl_train.models.grug_moe import (
    GRUG_MOE_MODEL_TYPE,
    GrugMoeConfig,
    GrugMoeForCausalLM,
    GrugMoeModel,
)


def register_local_models() -> None:
    """Idempotently register models implemented inside MarinSkyRL."""

    AutoConfig.register(GRUG_MOE_MODEL_TYPE, GrugMoeConfig, exist_ok=True)
    AutoModel.register(GrugMoeConfig, GrugMoeModel, exist_ok=True)
    AutoModelForCausalLM.register(GrugMoeConfig, GrugMoeForCausalLM, exist_ok=True)


register_local_models()
