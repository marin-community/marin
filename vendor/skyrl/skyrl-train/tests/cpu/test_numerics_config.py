import pytest
from transformers import LlamaConfig

from skyrl_train.config.numerics import validate_numerics_config
from skyrl_train.config.utils import get_default_config
from skyrl_train.models.grug_moe import GrugMoeConfig


@pytest.mark.parametrize(
    ("model_config", "covered"),
    [
        (GrugMoeConfig(), True),
        (LlamaConfig(num_hidden_layers=1), False),
        (GrugMoeConfig(num_key_value_heads=4), False),
        (GrugMoeConfig(sconv=True, grugmoe_artifact_schema_version=2), False),
    ],
    ids=["Snowball's shape", "another model family", "another Grug shape", "Hero"],
)
def test_exact_numerics_validate_only_for_policies_the_kernels_cover(tmp_path, model_config, covered):
    model_config.save_pretrained(tmp_path)
    cfg = get_default_config()
    cfg.trainer.numerics.mode = "exact"
    cfg.trainer.policy.model.path = str(tmp_path)
    cfg.trainer.use_sample_packing = False
    cfg.generator.enforce_eager = False
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.engine_init_kwargs = {"moe_backend": "triton"}

    if covered:
        # Every other exact requirement holds, so the other rows fail on their policy alone.
        validate_numerics_config(cfg)
    else:
        with pytest.raises(ValueError):
            validate_numerics_config(cfg)
